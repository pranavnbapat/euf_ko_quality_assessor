#!/usr/bin/env python3
# assess_ko_quality/pipeline.py
"""
The KO quality pipeline: one entry point, four stages.

    KO
     |
     +-- STAGE 0  GATE ........ domain relevance. Off-domain KOs are never scored,
     |                          because relevance is a precondition and not a quality
     |                          you can trade against good formatting. Three
     |                          well-written off-topic documents (Bach, async servers,
     |                          atrial fibrillation) scored 75.2-75.8 on the old
     |                          four-pillar total, above every real agricultural KO in
     |                          the same run, with the domain pillar switched on.
     |
     +-- STAGE 1  COMPLIANCE .. obligations, as a pass/fail checklist. Kept out of the
     |                          score: as a scored component, completeness correlated
     |                          with reviewer judgement at -0.217, the wrong direction.
     |
     +-- STAGE 2  SCORE ....... the one number validated against human review, built
     |                          from the only two measurements that survived that
     |                          validation (content depth and lexical variety, both
     |                          continuous - banding content length destroyed 53% of
     |                          its signal).
     |
     +-- STAGE 3  DIAGNOSTICS . everything else, reported as unusual-for-this-corpus
                                rather than as faults, because none of it predicts
                                reviewer judgement. Wording comes from the LLM judge.

Usage:
    python pipeline.py --input input/kos.json
    python pipeline.py --input input/kos.json --judge          # add the LLM judge
    python pipeline.py --input input/kos.json --learn          # refit profiles first
    python pipeline.py --input input/kos.json --no-gate        # score everything

Run from inside assess_ko_quality/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.io_utils import _latest_json_file, _read_json_any, _unique_outfile, ensure_directory
from core.text_utils import _ensure_str_list, detect_lang_safe, mtld, norm_text, strip_stops, tokens
from core.structural import structural_scores
from core.semantic import semantic_scores
from core.functional import functional_scores
from stages import compliance as compliance_stage
from stages import diagnostics as diagnostics_stage
from stages import scoring as scoring_stage


# --------------------------------------------------------------------------- input
def read_records(path: Path) -> List[Dict[str, Any]]:
    """
    Read a KO export.

    Exports come either as a bare array or wrapped as {meta, counts, docs:[...]}. The
    wrapped shape is the current one and the old reader silently scored the wrapper as
    a single record, so both are handled explicitly here.
    """
    raw = list(_read_json_any(path))
    if len(raw) == 1 and isinstance(raw[0], dict) and isinstance(raw[0].get("docs"), list):
        return [d for d in raw[0]["docs"] if isinstance(d, dict)]
    return raw


# --------------------------------------------------------------------------- selection
# A KO's file type is recorded in three different places and none is complete:
# category is the platform's own taxonomy, the extension is missing for 16.7% of
# records, and the mimetype for 10.6%. Matching across all three means "--file-type pdf"
# and "--file-type Video" both work without the caller having to know which field
# carries the answer.
_TYPE_FIELDS = ("category", "ko_object_extension", "ko_object_mimetype", "subcategories")


def _values(rec: Dict[str, Any], field: str) -> List[str]:
    v = rec.get(field)
    if v is None:
        return []
    items = v if isinstance(v, (list, tuple)) else [v]
    return [str(x).strip().lower().lstrip(".") for x in items if str(x).strip()]


def matches_type(rec: Dict[str, Any], wanted: Sequence[str]) -> bool:
    if not wanted:
        return True
    have = {v for f in _TYPE_FIELDS for v in _values(rec, f)}
    flat = " ".join(have)
    for w in wanted:
        w = w.strip().lower().lstrip(".")
        if w in have or w in flat:
            return True
    return False


def primary_language(rec: Dict[str, Any]) -> str:
    """
    A KO's language is the first one listed.

    About a fifth of records carry more than one. Taking the first keeps filtering and
    counting consistent: asking for French returns KOs whose language is French, and
    every KO belongs to exactly one bucket. A KO listed as English then Spanish is
    treated as English and will not come back from a Spanish filter.
    """
    langs = rec.get("languages") or []
    if isinstance(langs, str):
        langs = [langs]
    names = [str(x).strip() for x in langs if str(x).strip()]
    return names[0] if names else "unknown"


def select_records(
    records: Sequence[Dict[str, Any]],
    limit: int = 0,
    min_chars: int = 500,
    file_types: Sequence[str] = (),
    languages: Sequence[str] = (),
    lang_balance: str = "equal",
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Choose which KOs to assess, and say plainly what was left out.

    `lang_balance="equal"` spreads a limited run round-robin across the languages
    present, so a 100-record run is not 57% English simply because the platform is.
    Use "proportional" to mirror the corpus instead.

    Excluded counts are reported rather than swallowed. `min_chars` in particular
    removes about 36% of this corpus - the KOs whose text never extracted - and those
    are a real finding about ingestion, not records that do not matter.
    """
    import random

    rng = random.Random(seed)
    report: Dict[str, Any] = {"input": len(records), "excluded": {}}

    kept = list(records)
    if min_chars > 0:
        before = len(kept)
        kept = [r for r in kept if len(norm_text(r.get("ko_content_flat"))) >= min_chars]
        report["excluded"]["below_min_chars"] = before - len(kept)
    if file_types:
        before = len(kept)
        kept = [r for r in kept if matches_type(r, file_types)]
        report["excluded"]["wrong_file_type"] = before - len(kept)
    if languages:
        asked = {l.strip().lower() for l in languages if l.strip()}
        if asked and "all" not in asked:
            before = len(kept)
            kept = [r for r in kept if primary_language(r).lower() in asked]
            report["excluded"]["wrong_language"] = before - len(kept)

    report["eligible"] = len(kept)

    if limit and limit < len(kept):
        by_lang: Dict[str, List[Dict[str, Any]]] = {}
        for r in kept:
            by_lang.setdefault(primary_language(r), []).append(r)
        for pool in by_lang.values():
            rng.shuffle(pool)

        if lang_balance == "equal":
            # Round-robin, so languages with few records contribute what they have and
            # the rest is shared out rather than the largest language absorbing it.
            picked: List[Dict[str, Any]] = []
            order = sorted(by_lang, key=lambda k: len(by_lang[k]))
            while len(picked) < limit:
                progressed = False
                for lang in order:
                    if by_lang[lang] and len(picked) < limit:
                        picked.append(by_lang[lang].pop())
                        progressed = True
                if not progressed:
                    break
            kept = picked
        else:
            kept = rng.sample(kept, limit)

    report["selected"] = len(kept)
    report["by_language"] = dict(sorted(
        {l: sum(1 for r in kept if primary_language(r) == l) for l in
         {primary_language(r) for r in kept}}.items(), key=lambda kv: -kv[1]))
    report["by_category"] = dict(sorted(
        {c: sum(1 for r in kept if c in _values(r, "category")) for c in
         {v for r in kept for v in _values(r, "category")}}.items(), key=lambda kv: -kv[1]))
    return kept, report


# --------------------------------------------------------------------------- stages
def measure(ko: Dict[str, Any]) -> Dict[str, Any]:
    """Run the metric modules over one KO and return the merged measurements."""
    title = norm_text(ko.get("title"))
    subtitle = norm_text(ko.get("subtitle"))
    desc = norm_text(ko.get("description"))
    content = norm_text(ko.get("ko_content_flat"))
    keywords = [norm_text(x) for x in _ensure_str_list(ko.get("keywords")) if norm_text(x)]

    probe = " ".join([title, subtitle, desc, content[:500]]).strip()
    lang = detect_lang_safe(probe) if len(probe) >= 50 else "unknown"

    struct = structural_scores(title, subtitle, desc, content, keywords)
    sem = semantic_scores(title, subtitle, desc, content, keywords)
    func = functional_scores(title, desc, content, keywords)

    content_tokens = struct.get("Structural_metrics_content_words") or 0
    lexical = mtld(strip_stops(tokens(content))) if content else 0.0

    return {
        "lang_detected": lang,
        "content_tokens": content_tokens,
        "mtld": lexical,
        **struct, **sem, **func,
    }


def assess(
    ko: Dict[str, Any],
    gate_result: Optional[Dict[str, Any]] = None,
    judge: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assess one KO through every stage. The gate result is supplied by the caller."""
    ko_id = ko.get("_orig_id") or ko.get("_id") or ko.get("@id") or ""
    row: Dict[str, Any] = {"_orig_id": ko_id, "title": norm_text(ko.get("title"))[:300]}

    # --- Stage 0: gate -----------------------------------------------------
    if gate_result:
        row.update(gate_result)
    else:
        row["gate_probability"] = None
        row["gate_decision"] = "not_evaluated"

    # An off-domain KO is not scored. Scoring it would invite exactly the comparison
    # the gate exists to prevent: a polished off-topic upload outranking real content.
    if row["gate_decision"] == "reject":
        row.update({
            "quality_score": None,
            "quality_score_skipped": "off_domain",
            "compliance_status": "not_evaluated",
        })
        return row

    metrics = measure(ko)
    row["lang_detected"] = metrics["lang_detected"]

    # --- Stage 1: compliance ----------------------------------------------
    comp = compliance_stage.check(ko, profile=profile)
    row.update({k: v for k, v in comp.items()
                if k not in ("compliance_checks", "compliance_vocab_drift")})
    row["compliance_checks"] = json.dumps(comp["compliance_checks"], separators=(",", ":"))
    row["compliance_vocab_drift"] = json.dumps(comp.get("compliance_vocab_drift") or {},
                                               ensure_ascii=False, separators=(",", ":"))

    # --- Stage 2: score ----------------------------------------------------
    # The judge's overall verdict is a scored input when a judge ran; without one the
    # two-feature model is used, so a validated score is still produced offline.
    judge_overall = None
    if judge is not None and judge.get("llm_ok"):
        answers = judge.get("llm_answers") or {}
        judge_overall = next((v.get("answer") for k, v in answers.items()
                              if "recommend" in k and v.get("answer") is not None), None)
    scored = scoring_stage.score(metrics["content_tokens"], metrics["mtld"],
                                 judge_overall=judge_overall)
    row.update({k: v for k, v in scored.items() if k != "quality_score_contributions"})
    row["quality_score_contributions"] = json.dumps(
        scored.get("quality_score_contributions", {}), separators=(",", ":"))
    row.update(scoring_stage.retrieval_readiness(metrics))

    # --- Stage 3: diagnostics ---------------------------------------------
    diag = diagnostics_stage.diagnose(
        metrics,
        compliance=comp,
        judge=judge,
        extraction_suspect=scored["quality_extraction_suspect"],
        thresholds=thresholds,
    )
    row["diagnostics_unusual"] = json.dumps(diag["diagnostics_unusual"], separators=(",", ":"))
    row["diagnostics_unusual_count"] = diag["diagnostics_unusual_count"]
    row["diagnostics_notes"] = " | ".join(diag["diagnostics_notes"])[:2000]

    if judge is not None:
        row["judge_model"] = judge.get("llm_model")
        row["judge_ok"] = judge.get("llm_ok")
        answers = judge.get("llm_answers") or {}
        row["judge_answers"] = json.dumps(
            {k: v.get("answer") for k, v in answers.items()}, separators=(",", ":"))
        overall = next((v.get("answer") for k, v in answers.items() if "recommend" in k), None)
        row["judge_overall"] = overall

    return row


# --------------------------------------------------------------------------- driver
def run(
    records: List[Dict[str, Any]],
    use_gate: bool = True,
    use_gate_llm: bool = True,
    use_judge: bool = False,
    judge_model: str = "",
    workers: int = 8,
    checkpoint: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    profile = compliance_stage.load_profile()
    thresholds = diagnostics_stage.load_thresholds()
    if profile is None:
        print("[WARN] no compliance profile; run --learn. Only vocabulary checks will apply.")
    if thresholds is None:
        print("[WARN] no diagnostic thresholds; run --learn. Nothing will be flagged unusual.")

    # Gate: classifier for every KO in one batched pass; an LLM only for the band where
    # the classifier is unsure. On real KOs that band is nearly empty, so the LLM sees a
    # small minority of records - the ones where its judgement changes the outcome.
    gate_results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    if use_gate:
        try:
            from stages import gate as gate_stage
            print(f"[INFO] gate: classifying {len(records)} records ...", flush=True)
            gate_results = gate_stage.apply(records, use_llm=use_gate_llm, workers=workers)
            by_llm = [g for g in gate_results if g and g.get("gate_decided_by") == "llm"]
            # "Reversed" compares against the classifier's own lean, not its three-way
            # decision: a classifier "review" is not an answer the LLM can agree with, so
            # counting every resolved review as a reversal overstated the LLM's effect.
            reversed_ = sum(1 for g in by_llm
                            if (g["gate_probability"] >= gate_stage.ACCEPT_AT)
                            != (g["gate_decision"] == "accept"))
            band = f"[{gate_stage.LLM_BAND_LOW:.2f}, {gate_stage.LLM_BAND_HIGH:.2f})"
            print(f"[INFO] gate: {len(by_llm)} of {len(records)} fell in the uncertain band "
                  f"{band} and went to the LLM; it reversed the classifier's lean on "
                  f"{reversed_}", flush=True)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}; skipping stage 0.")
        except Exception as exc:
            print(f"[WARN] gate unavailable ({type(exc).__name__}: {exc}); skipping stage 0.")

    # Judge: only for KOs the gate did not reject, so no tokens are spent off-domain.
    judgements: List[Optional[Dict[str, Any]]] = [None] * len(records)
    if use_judge:
        from stages.gate import decide
        from stages.llm_judge import judge_many, load_rubric
        wanted = [i for i, g in enumerate(gate_results)
                  if g is None or g.get("gate_decision") != "reject"]
        if wanted:
            print(f"[INFO] judge: {len(wanted)} KOs to judge ...", flush=True)
            rubric = load_rubric(Path("data/2024_KOs for assessment all collating.xlsx"))
            results = judge_many([records[i] for i in wanted], rubric,
                                 model=judge_model, workers=workers)
            for i, res in zip(wanted, results):
                judgements[i] = res
            ok = sum(1 for r in results if r.get("llm_ok"))
            # calls made vs results returned: a judge that silently fails on long or
            # non-English KOs would bias everything downstream of it.
            print(f"[INFO] judge: {ok}/{len(wanted)} returned answers", flush=True)

    # Rows are flushed as they are produced. A long run that dies at hour three should
    # not lose everything before it, and the judge calls behind it are already paid for.
    handle = checkpoint.open("w", encoding="utf-8") if checkpoint else None
    rows = []
    try:
        for i, ko in enumerate(records):
            try:
                row = assess(ko, gate_result=gate_results[i], judge=judgements[i],
                             profile=profile, thresholds=thresholds)
            except Exception as exc:
                row = {"_orig_id": ko.get("_orig_id") or ko.get("_id") or f"row_{i}",
                       "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            if handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            if (i + 1) % 250 == 0:
                print(f"[INFO] scored {i + 1}/{len(records)}", flush=True)
    finally:
        if handle:
            handle.close()
    return rows


def learn(records: List[Dict[str, Any]]) -> None:
    """Refit the corpus-derived profiles: what is expected, and what counts as unusual."""
    print(f"[INFO] learning compliance profile from {len(records)} records ...")
    profile = compliance_stage.learn_profile(records)
    compliance_stage.save_profile(profile)
    print(f"[OK] {len(profile['expected_fields'])} contributor obligations "
          f"-> {compliance_stage.DEFAULT_PROFILE}")

    print(f"[INFO] learning diagnostic thresholds ...")
    metric_rows = [measure(r) for r in records]
    thresholds = diagnostics_stage.learn_thresholds(metric_rows)
    diagnostics_stage.save_thresholds(thresholds)
    print(f"[OK] {len(thresholds['cuts'])} metrics with learned tails "
          f"-> {diagnostics_stage.DEFAULT_THRESHOLDS}")


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    ensure_directory(out_dir)
    jsonl = _unique_outfile(out_dir, stem="ko_quality", ext=".jsonl")
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    columns: List[str] = []
    for row in rows:
        for k in row:
            if k not in columns:
                columns.append(k)
    tsv = jsonl.with_suffix(".tsv")
    with tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t",
                           quoting=csv.QUOTE_MINIMAL, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in columns})
    print(f"[OK] {len(rows)} rows -> {jsonl}\n          -> {tsv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None, help="KO export JSON (default: newest in input/)")
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--learn", action="store_true", help="Refit corpus profiles from this input first")
    ap.add_argument("--judge", action="store_true", help="Run the LLM judge (diagnostics only)")
    ap.add_argument("--judge-model", default="", help="Model preference, e.g. 'gpt-oss'")
    ap.add_argument("--no-gate", action="store_true", help="Score every KO, including off-domain")
    ap.add_argument("--no-gate-llm", action="store_true",
                    help="Classifier-only gate: no LLM calls for uncertain records")
    ap.add_argument("--limit", type=int, default=0, help="Assess at most N records (0 = all)")
    ap.add_argument("--min-chars", type=int, default=500,
                    help="Skip KOs with less extracted text than this (default 500; "
                         "0 to include failed extractions)")
    ap.add_argument("--file-type", default="",
                    help="Comma list, matched against category, extension, mimetype and "
                         "subcategory: 'pdf', 'mp4', 'Video', 'Factsheet'")
    ap.add_argument("--languages", default="",
                    help="Comma list of languages to keep, e.g. 'English,French'. "
                         "Empty or 'all' keeps every language")
    ap.add_argument("--lang-balance", choices=("equal", "proportional"), default="equal",
                    help="How --limit is spread across languages (default equal)")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    path = args.input or _latest_json_file("input")
    print(f"[INFO] input: {path}")
    records = read_records(Path(path))
    print(f"[INFO] {len(records)} records in file")

    records, selection = select_records(
        records,
        limit=args.limit,
        min_chars=args.min_chars,
        file_types=[t for t in args.file_type.split(",") if t.strip()],
        languages=[l for l in args.languages.split(",") if l.strip()],
        lang_balance=args.lang_balance,
        seed=args.seed,
    )
    for reason, n in selection["excluded"].items():
        if n:
            print(f"[INFO] excluded {n} ({reason})")
    print(f"[INFO] eligible {selection['eligible']} -> assessing {selection['selected']}")
    if selection["by_language"]:
        print(f"[INFO] languages: {selection['by_language']}")
    if selection["by_category"]:
        print(f"[INFO] categories: {selection['by_category']}")
    if not records:
        sys.exit("Nothing selected. Loosen --min-chars, --file-type or --languages.")

    if args.learn:
        learn(records)

    ensure_directory(args.output_dir)
    checkpoint = args.output_dir / "ko_quality_partial.jsonl"
    t0 = time.time()
    rows = run(records, use_gate=not args.no_gate, use_gate_llm=not args.no_gate_llm,
               use_judge=args.judge,
               judge_model=args.judge_model, workers=args.workers,
               checkpoint=checkpoint)
    print(f"[INFO] assessed in {time.time() - t0:.0f}s")

    gated = sum(1 for r in rows if r.get("gate_decision") == "reject")
    scored = sum(1 for r in rows if r.get("quality_score") is not None)
    incomplete = sum(1 for r in rows if r.get("compliance_status") == "incomplete")
    print(f"[INFO] gate rejected {gated} | scored {scored} | compliance incomplete {incomplete}")
    write_outputs(rows, args.output_dir)
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=1))
    if checkpoint.exists():
        checkpoint.unlink()      # final files written; the partial is redundant


if __name__ == "__main__":
    main()
