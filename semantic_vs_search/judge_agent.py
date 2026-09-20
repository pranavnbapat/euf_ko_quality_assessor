"""
A relevance-judging agent, built to be checkable rather than merely confident.

The main comparison in this folder graded 5,925 pairs with a single LLM call per
batch of ten candidates, no reasoning, one model, one sample. That is enough to
rank systems roughly and not enough to trust a number. This module fixes the
four things wrong with it:

  batching      ten candidates in one prompt invites position and leniency
                drift; each document is graded on its own here
  no reasoning  a grade with no stated reason cannot be audited, and reasoning
                first measurably steadies the grade
  one model     a single model's blind spots become the ground truth; two
                families grade everything and their disagreement is recorded
  one sample    a lone sample hides the model's own uncertainty; contested
                rows are re-graded several times to see whether it is stable

What comes out is not "the LLM said 2". It is a grade, the reasons behind it,
whether two model families agreed, and whether the same model agreed with
itself - so the rows worth a human's attention identify themselves.

Typical use:

    python judge_agent.py grade --sheet judging.xlsx --out judged.xlsx
    python judge_agent.py agreement --llm judged.xlsx --human judging_human.xlsx
    python judge_agent.py export --sheet judged.xlsx --out judgments.json

The agreement step is the point of the whole exercise: until an LLM judge has
been measured against a human on the same rows, its grades are a proxy of
unknown quality, and scaling them only scales the uncertainty.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any

from llm_client import LLMClient, build_client_from_env, load_env_file

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("judge_agent")

GRADES = (0, 1, 2, 3)

SYSTEM_PROMPT = """You assess search relevance for EU-FarmBook, a multilingual \
European platform of agricultural knowledge objects: practice abstracts, project \
outputs, guides, datasets and reports aimed at farmers, advisors and researchers.

You judge one search and one result at a time. You are strict, consistent, and \
you judge what the searcher was trying to find - not whether the document is \
interesting, well written or generally about agriculture.

Queries arrive in any of 24 EU languages and a document may be written in a \
different language from the query. A document in another language is NOT less \
relevant for that reason alone: judge the meaning, not the language."""

# Anchors are what stop a grade scale drifting. They are written for this
# domain deliberately: "relevant" means something different for a farming
# practice abstract than for a news article.
FEW_SHOT = """Grade on this scale:

3 - ANSWERS IT. The document is about exactly what was searched for. Someone
    typing this query wanted this document.
    e.g. query "pocket digester manure" -> a practice abstract on small-scale
    anaerobic digestion of cattle manure on farms.

2 - CLEARLY USEFUL. Substantially about the query's topic and worth reading,
    but not squarely the thing asked for.
    e.g. query "pocket digester manure" -> a report on biogas yields from
    various manure types, without covering small-scale digesters.

1 - MARGINAL. Related subject area, would rarely help the searcher.
    e.g. query "pocket digester manure" -> a general guide to nutrient
    management on dairy farms.

0 - NOT RELEVANT. Different topic, or matches only on an incidental word.
    e.g. query "pocket digester manure" -> a study of pocket-sized soil
    sampling equipment.

A named project, organisation or place in the query must actually be the
subject of the document to score 3. A document that merely mentions it in
passing is at most 1."""

USER_TEMPLATE = """{few_shot}

SEARCH QUERY: {query}

CANDIDATE DOCUMENT
Title: {title}
Project: {project}
Description: {description}

Return ONLY JSON:
{{"reason": "<one sentence, max 25 words, saying why>", "grade": <0, 1, 2 or 3>}}"""


# ------------------------------------------------------------------- model ---

def build_panel(env_paths: list[str], cache_dir: str, models: list[str]) -> list[LLMClient]:
    """One client per judge.

    A panel of different model families is the point: two checkpoints of the
    same family tend to share their mistakes, so their agreement flatters both.
    Where families genuinely disagree is where a human should look.

    reasoning_effort is applied optimistically - the client drops it by itself
    for any model that refuses it, so no per-model configuration is needed.
    """
    merged: dict[str, str] = {}
    for path in env_paths:
        for key, value in load_env_file(path).items():
            merged.setdefault(key, value)
    url = merged.get("VLLM_URL") or merged.get("LLM_URL")
    key = merged.get("VLLM_API_KEY") or merged.get("LLM_API_KEY") or ""
    effort = (merged.get("LLM_REASONING_EFFORT") or "").strip() or None
    if not url:
        raise RuntimeError("No LLM endpoint configured in " + ", ".join(env_paths))
    return [LLMClient(url, key, model.strip(), cache_dir, reasoning_effort=effort)
            for model in models if model.strip()]


def fleiss_kappa(ratings: list[list[int]], categories=GRADES) -> float:
    """Chance-corrected agreement for more than two raters.

    Cohen's kappa only handles a pair. With a panel, Fleiss is the right
    statistic: it asks how much the judges agree beyond what their individual
    grade habits would produce by chance.
    """
    rows = [r for r in ratings if len(r) >= 2]
    if not rows:
        return 0.0
    n = len(rows[0])
    if any(len(r) != n for r in rows) or n < 2:
        # Ragged panels (a judge failed on some rows) are scored on the subset
        # every judge answered, rather than silently mixing panel sizes.
        width = min(len(r) for r in rows)
        rows = [r[:width] for r in rows]
        n = width
        if n < 2:
            return 0.0
    agreements = []
    category_totals = {c: 0 for c in categories}
    for row in rows:
        counts = Counter(row)
        for c in categories:
            category_totals[c] += counts.get(c, 0)
        agreements.append((sum(v * v for v in counts.values()) - n) / (n * (n - 1)))
    p_bar = sum(agreements) / len(rows)
    total = len(rows) * n
    p_e = sum((category_totals[c] / total) ** 2 for c in categories)
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def grade_one(client: LLMClient, query: str, doc: dict[str, str],
              temperature: float | None = None) -> dict[str, Any]:
    prompt = USER_TEMPLATE.format(
        few_shot=FEW_SHOT,
        query=query,
        title=doc.get("title") or "(no title)",
        project=doc.get("project") or "(no project)",
        description=(doc.get("description") or "(no description)")[:1200],
    )
    original = client.temperature
    if temperature is not None:
        client.temperature = temperature
    try:
        data = client.complete_json(prompt, max_tokens=180, system=SYSTEM_PROMPT)
    finally:
        client.temperature = original
    grade = data.get("grade") if isinstance(data, dict) else None
    try:
        grade = int(grade)
    except (TypeError, ValueError):
        grade = None
    if grade not in GRADES:
        raise ValueError(f"model returned an out-of-scale grade: {data!r}")
    return {"grade": grade, "reason": str((data or {}).get("reason") or "")[:200]}


# ------------------------------------------------------------------ grading ---

def read_sheet(path: str) -> list[dict[str, Any]]:
    from openpyxl import load_workbook

    ws = load_workbook(path, data_only=True)["judging"]
    rows: list[dict[str, Any]] = []
    query = lang = ""
    for excel_row, values in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        cells = (list(values) + [None] * 8)[:8]
        if cells[0]:
            query, lang = str(cells[0]).strip(), str(cells[1] or "")
        if not cells[7]:
            continue  # a "(no results returned)" marker row
        rows.append({"excel_row": excel_row, "query": query, "lang": lang,
                     "rank": cells[2], "title": cells[4], "project": cells[5],
                     "description": cells[6], "doc_id": str(cells[7])})
    return rows


def cmd_grade(args: argparse.Namespace) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill

    here = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.abspath(os.path.join(here, "..", ".."))
    env_paths = [os.path.join(workspace, "farm_assistant_um", ".env"),
                 os.path.join(workspace, "scout", ".env")]
    cache = os.path.join(args.cache_dir, "llm")

    models = [m for m in args.judges.split(",") if m.strip()]
    panel = build_panel(env_paths, cache, models)
    if not panel:
        sys.exit("No judges configured - pass --judges")
    log.info("panel: %s", ", ".join(c.model for c in panel))

    rows = read_sheet(args.sheet)
    log.info("%d rows x %d judges = %d calls", len(rows), len(panel), len(rows) * len(panel))

    votes: dict[int, dict[str, int]] = defaultdict(dict)
    reasons: dict[int, dict[str, str]] = defaultdict(dict)
    for client in panel:
        def worker(row, _c=client):
            try:
                return grade_one(_c, row["query"], row)
            except Exception as exc:  # noqa: BLE001
                log.debug("%s failed: %s", _c.model, exc)
                return None
        results = client.map_concurrent(rows, worker, workers=args.workers, label=client.model[:18])
        for i, result in enumerate(results):
            if result:
                votes[i][client.model] = result["grade"]
                reasons[i][client.model] = result["reason"]

    # Rows the panel splits on get resampled from the first judge at a non-zero
    # temperature: a grade that moves under resampling is uncertain, one that
    # holds is merely contested.
    contested_idx = [i for i in range(len(rows))
                     if votes[i] and (max(votes[i].values()) - min(votes[i].values())) >= args.disagreement]
    log.info("%d rows where the panel spans >= %d grades", len(contested_idx), args.disagreement)

    if contested_idx and args.resamples > 0:
        jobs = [(i, s) for i in contested_idx for s in range(args.resamples)]

        def resample(job):
            i, _ = job
            try:
                return i, grade_one(panel[0], rows[i]["query"], rows[i],
                                    temperature=args.resample_temperature)["grade"]
            except Exception:  # noqa: BLE001
                return i, None
        for item in panel[0].map_concurrent(jobs, resample, workers=args.workers, label="resample"):
            if item and item[1] is not None:
                votes[item[0]][f"resample_{len(votes[item[0]])}"] = item[1]

    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        cast = list(votes[i].values())
        if not cast:
            out.append({**row, "grade": None, "status": "failed", "spread": None,
                        "per_model": {}, "reason": ""})
            continue
        tally = Counter(cast)
        top = max(tally.values())
        # Ties break downward: a panel that cannot choose between "useful" and
        # "answers it" has not established that it answers it.
        grade = min(g for g, n in tally.items() if n == top)
        spread = max(cast) - min(cast)
        status = "unanimous" if spread == 0 else ("minor" if spread == 1 else "CONTESTED")
        out.append({**row, "grade": grade, "spread": spread, "status": status,
                    "per_model": dict(votes[i]),
                    "reason": next(iter(reasons[i].values()), "")})

    # ---- write a sheet that shows its own uncertainty ----
    wb = load_workbook(args.sheet)
    ws = wb["judging"]
    headers = ["llm_grade", "spread", "status"] + [c.model[:16] for c in panel] + ["llm_reason"]
    start = ws.max_column + 1
    for offset, name in enumerate(headers):
        cell = ws.cell(row=1, column=start + offset, value=name)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F5C4D")
    fills = {"CONTESTED": PatternFill("solid", fgColor="FCE4E4"),
             "minor": PatternFill("solid", fgColor="FFF7E0")}
    for item in out:
        r = item["excel_row"]
        cells = [item["grade"], item["spread"], item["status"]]
        cells += [item["per_model"].get(c.model) for c in panel]
        cells += [item["reason"]]
        for offset, value in enumerate(cells):
            ws.cell(row=r, column=start + offset, value=value)
        fill = fills.get(item["status"])
        if fill:
            for offset in range(len(headers)):
                ws.cell(row=r, column=start + offset).fill = fill
    for offset, width in enumerate([10, 8, 12] + [11] * len(panel) + [60]):
        ws.column_dimensions[ws.cell(row=1, column=start + offset).column_letter].width = width
    wb.save(args.out)

    # ---- report ----
    counts = Counter(i["status"] for i in out)
    graded = [i for i in out if i["grade"] is not None]
    complete = [[i["per_model"][c.model] for c in panel] for i in out
                if all(c.model in i["per_model"] for c in panel)]
    log.info("Wrote %s", args.out)
    print()
    print(f"  rows graded    : {len(graded)}/{len(out)}")
    print(f"  unanimous      : {counts['unanimous']}  ({100*counts['unanimous']/max(1,len(out)):.0f}%)")
    print(f"  differ by one  : {counts['minor']}")
    print(f"  CONTESTED      : {counts['CONTESTED']}  <- the rows worth a human's time")
    print(f"  failed         : {counts['failed']}")
    print(f"  grades         : {dict(sorted(Counter(i['grade'] for i in graded).items()))}")
    if complete:
        print(f"  Fleiss kappa   : {fleiss_kappa(complete):.2f}  (agreement among the {len(panel)} judges)")
    print()
    print("  per-judge grade distribution and pairwise agreement:")
    for c in panel:
        dist = Counter(i["per_model"][c.model] for i in out if c.model in i["per_model"])
        n = sum(dist.values())
        mean = sum(g * v for g, v in dist.items()) / max(1, n)
        print(f"    {c.model:32} n={n:4} mean={mean:.2f} {dict(sorted(dist.items()))}")
    for a_i in range(len(panel)):
        for b_i in range(a_i + 1, len(panel)):
            ma, mb = panel[a_i].model, panel[b_i].model
            pairs = [(i["per_model"][ma], i["per_model"][mb]) for i in out
                     if ma in i["per_model"] and mb in i["per_model"]]
            if pairs:
                k = cohens_kappa([x for x, _ in pairs], [y for _, y in pairs])
                agree = 100 * sum(1 for x, y in pairs if x == y) / len(pairs)
                print(f"    {ma[:18]:20} vs {mb[:18]:20} exact {agree:4.0f}%  kappa {k:5.2f}")
    print()
    for c in panel:
        print(f"  {c.model:32} {c.usage_summary()}")
    print()
    print(f"Next: have a human grade ONLY the {counts['CONTESTED']} contested rows, then")
    print(f"  python judge_agent.py agreement --llm {args.out} --human <their file>")


# ---------------------------------------------------------------- agreement ---

def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Chance-corrected agreement. Raw percentage flatters a skewed scale."""
    if not a or len(a) != len(b):
        return 0.0
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[g] / n) * (cb[g] / n) for g in set(a) | set(b))
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def cmd_agreement(args: argparse.Namespace) -> None:
    from openpyxl import load_workbook

    def grades_by_doc(path: str, column_name: str) -> dict[tuple[str, str], int]:
        ws = load_workbook(path, data_only=True)["judging"]
        header = [c.value for c in ws[1]]
        try:
            col = header.index(column_name) + 1
        except ValueError:
            sys.exit(f"{path} has no column named {column_name!r} (found: {header})")
        out: dict[tuple[str, str], int] = {}
        query = ""
        for row in ws.iter_rows(min_row=2):
            values = [c.value for c in row]
            if values[0]:
                query = str(values[0]).strip()
            doc_id = values[7]
            grade = ws.cell(row=row[0].row, column=col).value
            if not doc_id or grade is None or str(grade).strip() == "":
                continue
            try:
                out[(query, str(doc_id))] = int(str(grade).strip())
            except ValueError:
                continue
        return out

    llm = grades_by_doc(args.llm, args.llm_column)
    human = grades_by_doc(args.human, args.human_column)
    shared = sorted(set(llm) & set(human))
    if not shared:
        sys.exit("No rows graded by both. Check that the two files describe the same sheet.")

    a = [llm[k] for k in shared]
    b = [human[k] for k in shared]
    exact = sum(1 for x, y in zip(a, b) if x == y) / len(shared)
    within1 = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(shared)
    kappa = cohens_kappa(a, b)
    bias = statistics.mean(x - y for x, y in zip(a, b))

    print(f"Rows graded by both : {len(shared)}")
    print(f"Exact agreement     : {exact:.0%}")
    print(f"Within one grade    : {within1:.0%}")
    print(f"Cohen's kappa       : {kappa:.2f}")
    print(f"LLM bias vs human   : {bias:+.2f} grades ({'more generous' if bias > 0 else 'stricter'})")
    print()
    print("Confusion (rows = LLM, columns = human):")
    print("        " + "".join(f"{g:>6}" for g in GRADES))
    for g in GRADES:
        row = "".join(f"{sum(1 for x,y in zip(a,b) if x==g and y==h):>6}" for h in GRADES)
        print(f"  LLM {g} {row}")
    print()
    if kappa >= 0.8:
        verdict = ("Near-human. Scaling this judge to thousands of queries is defensible.")
    elif kappa >= 0.6:
        verdict = ("Substantial agreement - the usual bar for an LLM judge. Scale it, but keep "
                   "reporting the kappa alongside any number it produces.")
    elif kappa >= 0.4:
        verdict = ("Moderate. Usable for ranking systems against each other, too weak to quote "
                   "as an absolute quality score.")
    else:
        verdict = ("Poor. This judge should not stand in for human judgement; fix the prompt or "
                   "the scale before trusting any figure built on it.")
    print(f"VERDICT: {verdict}")
    if abs(bias) >= 0.3:
        print(f"         The {abs(bias):.2f}-grade bias is large enough to correct for before use.")


# ------------------------------------------------------------------- export ---

def cmd_export(args: argparse.Namespace) -> None:
    from openpyxl import load_workbook

    ws = load_workbook(args.sheet, data_only=True)["judging"]
    header = [c.value for c in ws[1]]
    try:
        col = header.index(args.column) + 1
    except ValueError:
        sys.exit(f"No column named {args.column!r} (found: {header})")

    judgments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in ws.iter_rows(min_row=2):
        values = [c.value for c in row]
        if values[0]:
            current = {"query": str(values[0]).strip(), "relevant": {}}
            judgments.append(current)
        doc_id = values[7]
        grade = ws.cell(row=row[0].row, column=col).value
        if current is None or not doc_id or grade is None:
            continue
        try:
            value = int(str(grade).strip())
        except ValueError:
            continue
        if value > 0:
            current["relevant"][str(doc_id)] = value

    judgments = [j for j in judgments if j["relevant"]]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(judgments, fh, ensure_ascii=False, indent=2)
    sizes = [len(j["relevant"]) for j in judgments] or [0]
    print(f"Wrote {args.out}: {len(judgments)} queries, "
          f"mean {sum(sizes)/len(sizes):.1f} relevant each")
    print("\nNow run:")
    print("  python ../which_fields_to_choose/evaluate_field_bundles.py \\")
    print(f"      --input ../input/<export>.json --judgments {args.out} --id-field _id")


# --------------------------------------------------------------------- main ---

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("grade", help="Grade a judging sheet with two models")
    p.add_argument("--sheet", default="judging.xlsx")
    p.add_argument("--out", default="judged.xlsx")
    p.add_argument("--judges",
                   default="qwen3.5-397b-a17b,gpt-oss-120b,mistral-medium-3.5-128b,glm-5.2",
                   help="Comma-separated judge models, ideally from different families")
    p.add_argument("--disagreement", type=int, default=2,
                   help="Grade gap that marks a row contested")
    p.add_argument("--resamples", type=int, default=3,
                   help="Extra samples for contested rows (0 disables)")
    p.add_argument("--resample-temperature", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-dir", default=os.path.join(here, ".cache"))
    p.set_defaults(func=cmd_grade)

    p = sub.add_parser("agreement", help="Measure the judge against human grades")
    p.add_argument("--llm", required=True)
    p.add_argument("--human", required=True)
    p.add_argument("--llm-column", default="llm_grade")
    p.add_argument("--human-column", default="grade")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("export", help="Turn a graded sheet into judgments JSON")
    p.add_argument("--sheet", default="judged.xlsx")
    p.add_argument("--out", default="judgments.json")
    p.add_argument("--column", default="llm_grade")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
