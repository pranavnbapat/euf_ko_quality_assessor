# assess_ko_quality/legacy/metadata_llm.py  (was stages/metadata_llm.py)
"""
Superseded on 24 September 2026; nothing imports it. The KO score now compares the
contributor's fields with an LLM version, and the LLM's opinion is the fourth pillar of
each field score, one call per field per version (stages/llm_score.py). Kept for
reference.

Stage 5: LLM-generated metadata, measured and judged against the contributor's.

For each KO:

  1. GENERATE  each generator model writes a new title, subtitle, description and
               keywords from the KO's summary (ko_content_flat_summarised), under the
               published rules. The default generator, "glm@blind", sees the summary
               and the rules only - not the contributor's fields - so the LLM version
               is written from the summary alone. One call per KO per generator.
  2. MEASURE   every version - the contributor's, each generator's, and the *_llm
               fields already in the export when --variants asks for them - is scored
               by stage 4 on all ten sub-pillars, against the reference text (the
               summary by default). No LLM.
  3. JUDGE     each judge model sees, per field, the contributor's and the generated
               versions as A / B / C in a random order - never labelled - and rates
               each on the same properties (1-5), then names the best and why. With the
               position check on, it is asked again with the order reversed; a verdict
               counts only if it names the same version both times.
  4. JUDGE SCORE  each judge's ratings, averaged over both orders, become a 0-100 score
               per field and version, aggregated exactly like the measured field score
               (same pillars, same weights, same parts left out; an empty field scores
               0), and a KO judge score per version. It is reported next to the
               measured pillars and never enters the metadata score.

Why the judge is blinded and asked twice: LLM judges favour the option shown first
(Wang et al., ACL 2024; Zheng et al., NeurIPS 2023) and their own model's writing
(Panickssery et al., NeurIPS 2024). Running two model families in both roles measures
both effects instead of assuming them away; by default the judge is a model that wrote
none of the versions it compares.

What is measured, not judged: length, noise, formatting and findability are counted
exactly by stage 4; an LLM's rating of them is recorded only to see whether it agrees.
The recommended version is the one with the best measured field score that is not less
faithful to the reference text than the original (reward over-optimisation, Gao et al.,
ICML 2023); the judges' verdicts and scores are reported next to it, with how often they
agree.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import metadata_metrics as mm
from core.corpus_index import FIELDS, CorpusIndex, Embedder, field_text, reference_text, summary_text
from stages.metadata_score import score_variant

# The measured sub-pillars, as a judge is asked to rate them. Findability is one
# question for the judge; the scorer measures it twice (keyword and meaning search).
PARAMETERS = {
    "length": "within the published limits: title at most 150 characters not counting spaces; description 50-5000 characters",
    "completeness": "present, and not a placeholder, a file name, an internal project code or a copy of another field",
    "noise": "free of markup, URLs, encoding debris and stray symbols",
    "formatting": "normal capitalisation; no stray line breaks; keywords one concept each, without duplicates",
    "consistency": "everything it says is supported by the file, and it agrees with the other fields",
    "density": "uses words specific to this file rather than generic ones",
    "clarity": "easily understood: not cut off, no unexplained acronyms",
    "usefulness": "covers what the file is mainly about, and adds to the title rather than repeating it",
    "findability": "someone searching for this file's topic would find it with these words",
}

# The judge's properties grouped as the measured sub-pillars are, so a judge score is
# aggregated exactly like the measured field score it is reported next to.
JUDGE_PILLARS = {
    "structural": ("length", "completeness", "noise", "formatting"),
    "semantic": ("consistency", "density", "clarity", "usefulness"),
    "functional": ("findability",),
}

GEN_SYSTEM = (
    "You write metadata for entries on a multilingual European platform of practical "
    "knowledge on agriculture, forestry, livestock, food and rural life. Base everything "
    "on the file summary you are given; never add facts that are not in it."
)
JUDGE_SYSTEM = (
    "You are an impartial reviewer of metadata for a European agricultural knowledge "
    "platform. You compare anonymous versions of the same field. Judge only against the "
    "file summary; ignore which version is longer and the order in which they appear."
)

LANGUAGE_POLICY = {
    "keep": "Write each field in the same language as the current version of that field.",
    "source": "Write each field in the language of the file.",
    "english": "Write each field in English.",
}


# ------------------------------------------------------------------ inputs
def file_summary(rec: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[str, str]:
    """
    The text the LLM sees instead of the file: the summary. Only when the reference is
    the file itself (reference.source "file") does a KO without a usable summary fall
    back to the start of the file's text; with the summary as the reference, generator
    and judge see exactly what the fields are measured against.
    """
    cap = int(spec["llm_review"]["max_file_chars"])
    summary = summary_text(rec)
    if spec["reference"].get("source", "file") == "summary" or len(summary) >= 200:
        return summary[:cap], "summary"
    return reference_text(rec, spec["reference"]["max_chars"], "file")[:cap], "file_text"


def _field_view(f: str, value: Any) -> str:
    text = field_text(f, value)
    return text if text else "(empty)"


# ------------------------------------------------------------------ generation
def generation_prompt(rec: Dict[str, Any], spec: Dict[str, Any], language: str,
                      mode: str = "edit") -> str:
    """
    mode "edit": the model sees the contributor's current fields and improves them.
    mode "blind": it sees only the summary and the rules and writes the fields from
    scratch - an independent version, not anchored to the contributor's wording.
    """
    summary, _ = file_summary(rec, spec)
    current = [] if mode == "blind" else [
        "=== CURRENT METADATA ===",
        *[f"{f.upper()}: {_field_view(f, rec.get(f))}" for f in FIELDS], ""]
    return "\n".join([
        "=== WHAT THE FILE SAYS (summary) ===", summary, "",
        *current,
        "=== RULES ===",
        "- title: short and easily understandable, one key sentence, at most 150 characters "
        "not counting spaces; not an internal project code (like \"D3.1\" or \"Task 4\") and "
        "not a file name",
        "- subtitle: one sentence that adds to the title without repeating it",
        "- description: an account of the file - its topic, purpose, key results and who it "
        "is useful for (farmers, advisors); 50-5000 characters; explain technical terms and "
        "avoid unexplained abbreviations",
        "- keywords: words or short phrases, each one concept, specific to this file, without "
        "duplicates",
        "- say nothing the summary does not support",
        *([LANGUAGE_POLICY["source"]] if mode == "blind" else
          [LANGUAGE_POLICY[language],
           "- a field shown as (empty) is missing: create it, in the language of the file"]), "",
        "Write the best version of all four fields for this file." if mode != "blind" else
        "Write a title, subtitle, description and keywords for this file.",
        'Return JSON only: {"title": "...", "subtitle": "...", "description": "...", '
        '"keywords": ["...", "..."]}',
    ])


def _parse_generation(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Any] = {}
    for f in FIELDS:
        v = raw.get(f)
        if f == "keywords" and isinstance(v, str):
            v = [x.strip() for x in v.replace(";", ",").split(",") if x.strip()]
        out[f] = v if v else None
    return out if any(out.values()) else None


# ------------------------------------------------------------------ judging
def judge_prompt(rec: Dict[str, Any], spec: Dict[str, Any],
                 versions_per_field: Dict[str, List[Tuple[str, Any]]]) -> str:
    """versions_per_field: field -> [(label, value)] in the order to show them."""
    summary, _ = file_summary(rec, spec)
    lines = ["=== WHAT THE FILE SAYS (summary) ===", summary, "",
             "=== PROPERTIES (rate each 1-5; 5 = fully meets it) ==="]
    lines += [f"- {k}: {v}" for k, v in PARAMETERS.items()]
    lines += ["", "=== VERSIONS TO COMPARE ==="]
    for f, versions in versions_per_field.items():
        lines.append(f"[{f.upper()}]")
        for label, value in versions:
            lines.append(f"  {label}: {_field_view(f, value)}")
    labels = sorted({lab for vs in versions_per_field.values() for lab, _ in vs})
    lines += [
        "",
        "For each field, rate every version on every property, then name the best version "
        "and give one sentence explaining why. An empty version scores 1 on everything.",
        "Return JSON only, shaped like:",
        '{"title": {"ratings": {"' + labels[0] + '": {"length": 5, "completeness": 5, ...}, ...}, '
        '"best": "' + labels[0] + '", "why": "..."}, "subtitle": {...}, "description": {...}, "keywords": {...}}',
    ]
    return "\n".join(lines)


def _parse_judgement(raw: Any, labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for f in FIELDS:
        e = raw.get(f)
        if not isinstance(e, dict):
            continue
        ratings: Dict[str, Dict[str, float]] = {}
        for lab in labels:
            r = (e.get("ratings") or {}).get(lab)
            if isinstance(r, dict):
                ratings[lab] = {k: float(r[k]) for k in PARAMETERS
                                if isinstance(r.get(k), (int, float))}
        best = str(e.get("best") or "").strip().upper()[:1]
        out[f] = {"ratings": ratings, "best": best if best in labels else None,
                  "why": str(e.get("why") or "")[:400]}
    return out


def judge_field_score(ratings: Optional[Dict[str, float]], measured: Optional[Dict[str, Any]],
                      spec: Dict[str, Any]) -> Optional[float]:
    """
    One judge's 0-100 score for one version of one field.

    Its 1-5 ratings (already averaged over the presentation orders) are mapped linearly
    to [0, 1], averaged per pillar and combined with the spec's pillar weights - the
    aggregation of the measured field score - leaving out the properties the
    measurement left out (no published length for subtitle or keywords, say). An empty
    field scores 0, as it does when measured. None when there is nothing to score.
    """
    if measured is None:
        return None
    if not measured.get("text"):
        return 0.0
    if not ratings:
        return None
    skip = set(measured.get("not_evaluable") or [])
    pillars: Dict[str, Optional[float]] = {}
    for pillar, props in JUDGE_PILLARS.items():
        vals = []
        for k in props:
            left_out = ({"bm25_findability", "semantic_findability"} <= skip) if k == "findability" else k in skip
            if not left_out and k in ratings:
                vals.append((min(5.0, max(1.0, float(ratings[k]))) - 1.0) / 4.0)
        pillars[pillar] = sum(vals) / len(vals) if vals else None
    weights = spec["pillars"]
    den = sum(weights[p] for p, v in pillars.items() if v is not None)
    if not den:
        return None
    return round(100.0 * sum(weights[p] * v for p, v in pillars.items() if v is not None) / den, 1)


def judge_ko_score(field_scores: Dict[str, Optional[float]], spec: Dict[str, Any]) -> Optional[float]:
    """A KO's judge score for one version: the fields' judge scores, weighted as in stage 4."""
    have = {f: s for f, s in field_scores.items() if s is not None}
    den = sum(spec["fields"][f]["weight"] for f in have)
    return round(sum(spec["fields"][f]["weight"] * s for f, s in have.items()) / den, 1) if den else None


# ------------------------------------------------------------------ driver
def _complete(client, prompt: str, system: str, max_tokens: int) -> Any:
    budget = max_tokens
    for _ in range(2):
        try:
            raw = client.complete_json(prompt, max_tokens=budget, system=system)
            if raw is not None:
                return raw
        except ValueError:
            pass                       # truncated JSON: retry once with more room
        budget *= 2
    return None


def compare(
    records: Sequence[Dict[str, Any]],
    original_scores: Dict[str, Dict[str, Any]],
    existing_llm_scores: Dict[str, Optional[Dict[str, Any]]],
    idx: CorpusIndex,
    emb: Embedder,
    nli: mm.NLI,
    spec: Dict[str, Any],
    generators: Sequence[str],
    judges: Sequence[str],
    language: str = "keep",
    position_check: bool = True,
    workers: int = 4,
    log=print,
) -> Dict[str, Any]:
    """Generate, measure and judge. Returns per-KO results and the model names used."""
    from core.corpus_index import ko_id
    from core.llm import get_client

    cfg = spec["llm_review"]
    modes = {g: ("blind" if g.endswith("@blind") else "edit") for g in generators}
    gen_clients = {g: get_client(model=g.split("@")[0]) for g in generators}
    judge_clients = {j: get_client(model=j) for j in judges}
    missing = [m for m, c in {**gen_clients, **judge_clients}.items() if c is None]
    if missing:
        raise RuntimeError(f"no LLM available for: {missing}")
    names = {m: c.model for m, c in {**gen_clients, **judge_clients}.items()}
    for g in generators:
        if modes[g] == "blind":
            names[g] = names[g] + "@blind"
    n_calls = len(records) * (len(generators) + len(judges) * (2 if position_check else 1))
    log(f"[llm] {len(records)} KOs x ({len(generators)} generators + {len(judges)} judges"
        f"{' x 2 orders' if position_check else ''}) = {n_calls} calls at most (cached ones are free)")

    # 1. generate
    generated: Dict[str, List[Optional[Dict[str, Any]]]] = {}
    for g, client in gen_clients.items():
        prompts = [generation_prompt(r, spec, language, modes[g]) for r in records]
        raws = client.map_concurrent(
            prompts, lambda p, c=client: _complete(c, p, GEN_SYSTEM, int(cfg["max_tokens"])),
            workers=workers, label=f"generate:{names[g]}")
        generated[g] = [_parse_generation(x) for x in raws]
        log(f"[llm] {names[g]}: {sum(x is not None for x in generated[g])}/{len(records)} generated")

    # 2. measure every version against the file
    results = []
    for i, rec in enumerate(records):
        kid = ko_id(rec)
        ctx = mm.build_context(rec, idx, emb, spec)
        versions: Dict[str, Dict[str, Any]] = {"original": {f: rec.get(f) for f in FIELDS}}
        scores: Dict[str, Optional[Dict[str, Any]]] = {"original": original_scores.get(kid)}
        if existing_llm_scores.get(kid):
            versions["existing_llm"] = {f: rec.get(f"{f}_llm") for f in FIELDS}
            scores["existing_llm"] = existing_llm_scores[kid]
        for g in generators:
            if generated[g][i]:
                v = f"gen:{names[g]}"
                versions[v] = generated[g][i]
                scores[v] = score_variant(rec, v, ctx, idx, emb, nli, spec, values=generated[g][i])
        results.append({"_orig_id": kid, "summary_source": file_summary(rec, spec)[1],
                        "versions": versions, "scores": scores, "judgements": {}})

    # 3. judge: contributor's vs each generator's, blinded, random order (+ reversed)
    judged = ["original"] + [f"gen:{names[g]}" for g in generators]
    jobs = []
    for i, (rec, res) in enumerate(zip(records, results)):
        present = [v for v in judged if v in res["versions"]]
        order = present[:]
        random.Random(f"{res['_orig_id']}|order").shuffle(order)
        orders = [order, list(reversed(order))] if position_check else [order]
        for j in judges:
            for o in orders:
                mapping = {chr(65 + n): v for n, v in enumerate(o)}
                vpf = {f: [(lab, res["versions"][v].get(f)) for lab, v in mapping.items()] for f in FIELDS}
                jobs.append((i, j, mapping, judge_prompt(rec, spec, vpf)))
    for j, client in judge_clients.items():
        mine = [x for x in jobs if x[1] == j]
        raws = client.map_concurrent(
            [x[3] for x in mine],
            lambda p, c=client: _complete(c, p, JUDGE_SYSTEM, int(cfg["judge_max_tokens"])),
            workers=workers, label=f"judge:{names[j]}")
        for (i, _, mapping, _p), raw in zip(mine, raws):
            parsed = _parse_judgement(raw, list(mapping))
            for f, e in parsed.items():                     # labels back to version names
                e["ratings"] = {mapping[lab]: r for lab, r in e["ratings"].items()}
                e["best"] = mapping.get(e["best"]) if e["best"] else None
                e["first_shown"] = mapping["A"]
            results[i]["judgements"].setdefault(names[j], []).append(parsed)
        log(f"[llm] judge {names[j]}: {len(mine)} calls")

    # 4. per field: measured best, recommendation, verdicts and agreement
    tol = float(cfg["accuracy_tolerance"])
    for res in results:
        res["fields"] = {}
        for f in FIELDS:
            row: Dict[str, Any] = {"measured": {}, "judges": {}}
            for v, sc in res["scores"].items():
                fr = ((sc or {}).get("fields") or {}).get(f)
                if fr:
                    acc = ((fr.get("evidence") or {}).get("consistency") or {}).get("parts", {}).get("accuracy")
                    row["measured"][v] = {"score": fr.get("score"), "subpillars": fr.get("subpillars"),
                                          "accuracy": acc, "text": fr.get("text")}
            # Candidates: the contributor's, the *_llm already in the export, and each new
            # generation. The faithfulness bar is the contributor's field; when that is
            # missing, the existing *_llm one - so an invented field cannot win on score
            # alone while saying more than the file supports.
            cand = {v: m for v, m in row["measured"].items()
                    if (v in judged or v == "existing_llm") and m["score"] is not None}
            row["measured_best"] = max(cand, key=lambda v: cand[v]["score"]) if cand else None
            # An empty contributor field is scored (0) since version 8, so test its text.
            ref = "original" if (row["measured"].get("original") or {}).get("text") else "existing_llm"
            o_acc = (row["measured"].get(ref) or {}).get("accuracy")
            ok = {v: m for v, m in cand.items()
                  if v == ref or o_acc is None or m["accuracy"] is None or m["accuracy"] >= o_acc - tol}
            row["recommended"] = max(ok, key=lambda v: ok[v]["score"]) if ok else None
            for jname, calls in res["judgements"].items():
                entries = [c.get(f) or {} for c in calls]
                picks = [e.get("best") for e in entries]
                consistent = len(picks) > 1 and picks[0] is not None and len(set(picks)) == 1
                verdict = picks[0] if (len(picks) == 1 or consistent) else None
                mean_ratings: Dict[str, Dict[str, float]] = {}
                for v in judged:
                    vals = [e["ratings"][v] for e in entries if v in (e.get("ratings") or {})]
                    # Averaged over the orders it was rated in, property by property; a
                    # property the judge did not rate is missing, not a zero.
                    means: Dict[str, float] = {}
                    for k in PARAMETERS:
                        xs = [x[k] for x in vals if k in x]
                        if xs:
                            means[k] = round(sum(xs) / len(xs), 2)
                    if means:
                        mean_ratings[v] = means
                # A judge call that failed rates nothing: no score, rather than a 0 that
                # would read as a verdict on the empty fields only.
                scores = {v: (judge_field_score(mean_ratings.get(v),
                                                ((res["scores"].get(v) or {}).get("fields") or {}).get(f), spec)
                              if mean_ratings else None)
                          for v in judged if v in res["versions"]}
                row["judges"][jname] = {
                    "verdict": verdict, "picks": picks,
                    "position_consistent": consistent if len(picks) > 1 else None,
                    "first_shown": [e.get("first_shown") for e in entries],
                    "agrees_with_measurement": (verdict == row["measured_best"]) if verdict else None,
                    "why": next((e.get("why") for e in entries if e.get("why")), ""),
                    "ratings": mean_ratings, "scores": scores}
            res["fields"][f] = row
        res["judge_scores"] = {
            jname: {v: judge_ko_score({f: ((res["fields"][f]["judges"].get(jname) or {}).get("scores") or {}).get(v)
                                       for f in FIELDS}, spec)
                    for v in judged if v in res["versions"]}
            for jname in res["judgements"]}
    return {"results": results, "models": names, "generators": [names[g] for g in generators],
            "judges": [names[j] for j in judges], "position_check": position_check}


def summarise(out: Dict[str, Any]) -> Dict[str, Any]:
    """Run-level answers: which generator writes better, which judge is more reliable."""
    from statistics import mean

    def _means(by_version: Dict[str, List[Optional[float]]]) -> Dict[str, float]:
        out_: Dict[str, float] = {}
        for v, vals in by_version.items():
            xs = [x for x in vals if x is not None]
            if xs:
                out_[v] = round(mean(xs), 1)
        return out_

    res = out["results"]
    s: Dict[str, Any] = {"kos": len(res), "versions": {}, "judges": {}}
    gens = [f"gen:{g}" for g in out["generators"]]
    for v in ["original", "existing_llm"] + gens:
        cells = [r["fields"][f]["measured"][v] for r in res for f in FIELDS if v in r["fields"][f]["measured"]]
        scores = [c["score"] for c in cells if c["score"] is not None]
        accs = [c["accuracy"] for c in cells if c["accuracy"] is not None]
        ko = [(r["scores"].get(v) or {}).get("score") for r in res]
        ko = [x for x in ko if x is not None]
        s["versions"][v] = {
            "mean_ko_score": round(mean(ko), 1) if ko else None,
            "mean_field_score": round(mean(scores), 1) if scores else None,
            "mean_accuracy": round(mean(accs), 3) if accs else None,
            "measured_best_fields": sum(1 for r in res for f in FIELDS if r["fields"][f]["measured_best"] == v),
            "recommended_fields": sum(1 for r in res for f in FIELDS if r["fields"][f]["recommended"] == v),
        }
    for j in out["judges"]:
        rows = [r["fields"][f]["judges"].get(j) for r in res for f in FIELDS]
        rows = [x for x in rows if x]
        pc = [x["position_consistent"] for x in rows if x["position_consistent"] is not None]
        ag = [x["agrees_with_measurement"] for x in rows if x["agrees_with_measurement"] is not None]
        first = [p == fs for x in rows for p, fs in zip(x["picks"], x["first_shown"]) if p]
        wins = {v: sum(1 for x in rows if x["verdict"] == v) for v in ["original"] + gens}
        ko_scores = {v: [((r.get("judge_scores") or {}).get(j) or {}).get(v) for r in res]
                     for v in ["original"] + gens}
        field_scores = {v: [(x.get("scores") or {}).get(v) for x in rows] for v in ["original"] + gens}
        s["judges"][j] = {
            "field_verdicts": len(rows),
            "position_consistency": round(sum(pc) / len(pc), 3) if pc else None,
            "picked_first_shown": round(sum(first) / len(first), 3) if first else None,
            "chance_of_first": round(1 / (1 + len(gens)), 3),
            "agreement_with_measurement": round(sum(ag) / len(ag), 3) if ag else None,
            "consistent_wins": wins,
            "mean_ko_score": _means(ko_scores),
            "mean_field_score": _means(field_scores),
        }
    if len(out["judges"]) > 1 and len(gens) > 1:
        # Of the verdicts that went to a generated version, the share each judge gave to
        # each generator: a judge favouring its own model's text shows up as a diagonal.
        s["self_preference"] = {}
        for j in out["judges"]:
            w = s["judges"][j]["consistent_wins"]
            tot = sum(w[g] for g in gens) or 1
            s["self_preference"][j] = {g: round(w[g] / tot, 3) for g in gens}
    return s


__all__ = ["compare", "summarise", "generation_prompt", "judge_prompt", "judge_field_score",
           "judge_ko_score", "PARAMETERS", "JUDGE_PILLARS", "LANGUAGE_POLICY"]
