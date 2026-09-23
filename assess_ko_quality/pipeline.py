#!/usr/bin/env python3
# assess_ko_quality/pipeline.py
"""
The KO quality pipeline, end to end.

    select ─► 1 GATE ─► 2 COMPLIANCE ─► 3 CONTENT ─► 4 METADATA SCORE ─► 5 LLM REVIEW ─► 6 ANALYSIS
              domain     obligations     the file       title, subtitle,    optional:        why, and how
              relevance  (reported)      itself         description,        verdict +        much the
                                         (reported,     keywords            proposal per     scores depend
                                         never scored)  (THE score)         field, accepted  on our choices
                                                                            only if measured
                                                                            better

  select ......... which KOs: --limit, --file-type, --languages (first listed language),
                   --lang-balance, --min-chars
  1 gate ......... is the KO about agriculture in the broad sense? A classifier decides
                   the clear cases; an LLM only the uncertain band. Rejected KOs stop here.
  2 compliance ... fields the upload form requires, controlled-vocabulary conformance
                   (drift reported separately), age. Pass/fail; never in the score.
  3 content ...... the published web-text quality filters (Gopher, C4, FineWeb, FineWeb2),
                   lexical diversity, readability, declared vs detected language,
                   provenance cues. The file is the author's work, so this is reported and
                   never scored.
  4 metadata ..... the score: title, subtitle, description and keywords measured against
                   the file on three pillars and ten sub-pillars, 0-100. Scored for the
                   contributor's metadata and for the LLM-improved *_llm fields already in
                   the export, on identical criteria.
  5 llm review ... (--llm-review) one call per KO: is each field correct? If not, a
                   proposal - kept only if it scores higher without losing faithfulness.
  6 analysis ..... the weakest sub-pillars of each KO with their evidence; rank stability
                   under alternative weights; redundant sub-pillars; length bias;
                   (--selftest) perturbation tests of every sub-pillar.

Methods and sources: METHODS.md and LITERATURE_REVIEW.md. Settings with provenance:
validation/metadata_spec.json.

Usage (from inside assess_ko_quality/):
    python pipeline.py --limit 20
    python pipeline.py --limit 200 --languages English --file-type pdf
    python pipeline.py --limit 20 --selftest                  # also run the perturbation tests
    python pipeline.py --limit 10 --llm-review                # 10 LLM calls (plus the gate's few)
    python pipeline.py --no-gate-llm                          # zero LLM calls anywhere

The first run on an export builds the corpus index (about 5 minutes, plus about 15 the
very first time to embed every KO) and caches it under output/_corpus_cache/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.io_utils import _latest_json_file, _read_json_any, ensure_directory
from core.text_utils import norm_text
from stages import compliance as compliance_stage



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
def run_gate(records: List[Dict[str, Any]], use_gate: bool, use_llm: bool,
             workers: int) -> List[Optional[Dict[str, Any]]]:
    """Stage 1. Classifier for every KO in one batch; an LLM only in the uncertain band."""
    if not use_gate:
        return [None] * len(records)
    try:
        from stages import gate as gate_stage

        print(f"[gate] classifying {len(records)} KOs ...", flush=True)
        results = gate_stage.apply(records, use_llm=use_llm, workers=workers)
        by_llm = [g for g in results if g and g.get("gate_decided_by") == "llm"]
        # "Reversed" compares with the classifier's own lean, not its three-way decision.
        reversed_ = sum(1 for g in by_llm
                        if (g["gate_probability"] >= gate_stage.ACCEPT_AT)
                        != (g["gate_decision"] == "accept"))
        band = f"[{gate_stage.LLM_BAND_LOW:.2f}, {gate_stage.LLM_BAND_HIGH:.2f})"
        rejected = sum(1 for g in results if g and g.get("gate_decision") == "reject")
        print(f"[gate] {len(by_llm)} in the uncertain band {band} went to the LLM "
              f"(reversed {reversed_}); rejected {rejected}", flush=True)
        return results
    except Exception as exc:
        print(f"[gate] unavailable ({type(exc).__name__}: {exc}); every KO passes", flush=True)
        return [None] * len(records)


def assess_one(rec: Dict[str, Any], gate: Optional[Dict[str, Any]], profile, idx, emb, nli,
               spec: Dict[str, Any], variants: Sequence[str]) -> Dict[str, Any]:
    """Stages 2-4 and the per-KO part of 6 for one KO."""
    from core import content_report
    from core.corpus_index import ko_id
    from stages import analysis
    from stages.metadata_score import score_ko

    out: Dict[str, Any] = {"_orig_id": ko_id(rec), "title": norm_text(rec.get("title"))[:200],
                           "declared_language": primary_language(rec),
                           "category": rec.get("category")}
    out["gate"] = gate or {"gate_decision": "not_evaluated"}
    if (gate or {}).get("gate_decision") == "reject":
        out["skipped"] = "off-domain (gate)"
        return out

    out["compliance"] = compliance_stage.check(rec, profile=profile)
    pos = idx.pos.get(ko_id(rec))
    lang = idx.langs[pos] if pos is not None else "unknown"
    out["content"] = content_report.report(rec.get("ko_content_flat"), rec.get("languages"), lang)
    out["metadata"] = score_ko(rec, idx, emb, nli, spec, variants=variants)
    out["weakest"] = analysis.explain(out["metadata"]["variants"].get("original"))
    return out


def run_analysis(results: List[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 6, corpus part: robustness, redundancy and length bias over this run."""
    from core.metadata_metrics import SUBPILLARS
    from stages import analysis

    scored = [r for r in results if (r.get("metadata") or {}).get("variants", {}).get("original")]
    pillars = list(SUBPILLARS)
    P = [[(r["metadata"]["variants"]["original"]["pillars"].get(p) or float("nan")) / 100.0
          for p in pillars] for r in scored]
    nominal_p = [spec["pillars"][p] for p in pillars]

    # Sub-pillar level: each KO's mean over its fields; nominal weights are what the
    # hierarchy implies (pillar weight shared equally among the pillar's sub-pillars).
    subs = analysis.SUBS
    S = []
    for r in scored:
        fields = r["metadata"]["variants"]["original"]["fields"].values()
        row = []
        for s in subs:
            vals = [f["subpillars"].get(s) for f in fields if f.get("subpillars", {}).get(s) is not None]
            row.append(sum(vals) / len(vals) if vals else float("nan"))
        S.append(row)
    total_p = sum(nominal_p)
    nominal_s = [spec["pillars"][p] / total_p / len(names)
                 for p, names in SUBPILLARS.items() for _ in names]

    field_rows = [{"subpillars": f["subpillars"],
                   "words": len((f.get("text") or "").split())}
                  for r in scored for f in r["metadata"]["variants"]["original"]["fields"].values()
                  if f.get("subpillars")]
    return {
        "robustness_pillars": analysis.robustness(P, pillars, nominal_p),
        "robustness_subpillars": analysis.robustness(S, subs, nominal_s),
        "redundant_subpillars": analysis.redundancy(S, subs),
        "length_bias": analysis.length_bias(field_rows),
    }


# --------------------------------------------------------------------------- outputs
def _pillar(res: Optional[Dict[str, Any]], p: str) -> Optional[float]:
    return None if not res else res["pillars"].get(p)


def _field(res: Optional[Dict[str, Any]], f: str) -> Optional[float]:
    return None if not res else (res["fields"].get(f) or {}).get("score")


def write_outputs(out_dir: Path, results: List[Dict[str, Any]], reviews: List[Dict[str, Any]],
                  analysis_out: Dict[str, Any], selftest: List[Dict[str, Any]],
                  run_info: Dict[str, Any]) -> None:
    from core.corpus_index import FIELDS
    from core.metadata_metrics import SUBPILLARS
    from stages.analysis import SUBS

    ensure_directory(out_dir)
    with (out_dir / "details.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    review_by_id = {r["_orig_id"]: r for r in reviews}
    cols = ["_orig_id", "title", "declared_language", "detected_language", "gate_decision",
            "gate_probability", "gate_decided_by", "compliance_status", "compliance_failed",
            "vocab_drift_fields", "content_words", "content_filters_raised",
            "content_language_mismatch", "metadata_score", "metadata_score_llm_fields", "delta"]
    cols += [f"{p}" for p in SUBPILLARS] + [f"{f}_score" for f in FIELDS]
    cols += ["weakest", "llm_review_accepted"]
    with (out_dir / "kos.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in results:
            g = r.get("gate") or {}
            c = r.get("compliance") or {}
            ct = r.get("content") or {}
            v = ((r.get("metadata") or {}).get("variants") or {})
            o, l = v.get("original"), v.get("llm")
            os_, ls_ = (o or {}).get("score"), (l or {}).get("score")
            w.writerow([
                r["_orig_id"], r.get("title"), r.get("declared_language"),
                ct.get("content_language_detected"), g.get("gate_decision"),
                g.get("gate_probability"), g.get("gate_decided_by"),
                c.get("compliance_status"), ";".join(c.get("compliance_failed") or []),
                ";".join(sorted((c.get("compliance_vocab_drift") or {}).keys())),
                ct.get("content_words"), ";".join(ct.get("content_filters_raised") or []),
                ct.get("content_language_mismatch"), os_, ls_,
                None if os_ is None or ls_ is None else round(ls_ - os_, 1),
                *[_pillar(o, p) for p in SUBPILLARS], *[_field(o, f) for f in FIELDS],
                " | ".join(f"{x['field']}.{x['subpillar']}={x['value']:.2f} {x['why']}".strip()
                           for x in r.get("weakest") or []),
                ";".join((review_by_id.get(r["_orig_id"]) or {}).get("accepted_fields") or []),
            ])

    with (out_dir / "fields.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["_orig_id", "variant", "field", "score", *SUBPILLARS, *SUBS, "text"])
        for r in results:
            for var, vres in (((r.get("metadata") or {}).get("variants")) or {}).items():
                for f, fr in vres["fields"].items():
                    w.writerow([r["_orig_id"], var, f, fr.get("score"),
                                *[fr["pillars"].get(p) for p in SUBPILLARS],
                                *[fr["subpillars"].get(s) for s in SUBS],
                                (fr.get("text") or "").replace("\t", " ")])

    if reviews:
        with (out_dir / "llm_review.tsv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["_orig_id", "field", "verdict", "problems", "decision", "original",
                        "original_score", "proposal", "proposal_score", "gain",
                        "accuracy_original", "accuracy_proposal", "disagreement"])
            for rv in reviews:
                for f, e in (rv.get("fields") or {}).items():
                    w.writerow([rv["_orig_id"], f, e.get("verdict"), ";".join(e.get("problems") or []),
                                e.get("decision"), e.get("original"), e.get("original_score"),
                                e.get("proposal"), e.get("proposal_score"), e.get("gain"),
                                e.get("accuracy_original"), e.get("accuracy_proposal"),
                                e.get("disagreement")])

    if selftest:
        with (out_dir / "selftest.tsv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(selftest[0]), delimiter="\t")
            w.writeheader()
            w.writerows(selftest)

    (out_dir / "analysis.json").write_text(json.dumps(analysis_out, indent=1, default=str))
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=1, default=str))


def print_summary(results, reviews, analysis_out, selftest, out_dir: Path) -> None:
    from statistics import mean, median

    from core.corpus_index import FIELDS
    from core.metadata_metrics import SUBPILLARS

    def m(xs):
        xs = [x for x in xs if x is not None]
        return f"{mean(xs):5.1f} (median {median(xs):5.1f}, n={len(xs)})" if xs else "  -"

    scored = [r for r in results if r.get("metadata")]
    rejected = [r for r in results if r.get("skipped")]
    o = [r["metadata"]["variants"].get("original") for r in scored]
    l = [r["metadata"]["variants"].get("llm") for r in scored]
    print(f"\n=== {len(results)} KOs: {len(rejected)} rejected by the gate, {len(scored)} assessed")
    incomplete = sum(1 for r in scored if r["compliance"]["compliance_status"] == "incomplete")
    flagged = sum(1 for r in scored if r["content"]["content_filters_raised"])
    mism = sum(1 for r in scored if r["content"]["content_language_mismatch"])
    print(f"    compliance incomplete {incomplete} | content flagged by a web-text filter {flagged}"
          f" | declared language differs from detected {mism}")
    print(f"\n  metadata score           {'contributor':>30}   {'existing *_llm fields':>30}")
    print(f"  overall                  {m([x and x['score'] for x in o]):>30}   {m([x and x['score'] for x in l]):>30}")
    for p in SUBPILLARS:
        print(f"  {p:24s} {m([_pillar(x, p) for x in o]):>30}   {m([_pillar(x, p) for x in l]):>30}")
    for f in FIELDS:
        print(f"  {f:24s} {m([_field(x, f) for x in o]):>30}   {m([_field(x, f) for x in l]):>30}")

    rb = analysis_out.get("robustness_pillars", {}).get("rank_change")
    if rb:
        print("\n  rank stability, pillar weights (Spearman with equal weights / mean rank shift):")
        for k, v in rb.items():
            print(f"    {k:8s} {v['spearman_vs_nominal']:+.3f} / {v['mean_rank_shift']:.1f}")
    red = analysis_out.get("redundant_subpillars") or {}
    if red.get("pairs"):
        print(f"  redundant sub-pillars (|rho| >= 0.9): {red['pairs']}")
    if red.get("constant_in_this_sample"):
        print(f"  constant across this sample: {red['constant_in_this_sample']}")
    lb = {k: v for k, v in (analysis_out.get("length_bias") or {}).items() if v is not None and abs(v) >= 0.5}
    if lb:
        print(f"  sub-pillars tracking length (|rho| >= 0.5): {lb}")

    if reviews:
        acc = sum(len(r.get("accepted_fields") or []) for r in reviews)
        total = sum(len(r.get("fields") or {}) for r in reviews)
        dis = sum(1 for r in reviews for e in (r.get("fields") or {}).values() if e.get("disagreement"))
        print(f"\n  LLM review: {len(reviews)} KOs, {acc} of {total} fields replaced; "
              f"{dis} fields where the model said 'correct' but the measurement says weak")
    if selftest:
        from collections import Counter
        c = Counter((s["sub"], s["outcome"]) for s in selftest)
        subs = sorted({s["sub"] for s in selftest})
        drop = sum(v for (s, o_), v in c.items() if o_ == "dropped")
        print(f"\n  self-test: {drop} of {len(selftest)} deliberate defects detected")
        for s in subs:
            n = sum(v for (x, _), v in c.items() if x == s)
            print(f"    {s:24s} {c[(s, 'dropped')]:3d}/{n}")
    print(f"\noutputs -> {out_dir}")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None, help="KO export (default: newest in ../input)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: output/run_<timestamp>")
    # selection
    ap.add_argument("--limit", type=int, default=0, help="Assess at most N KOs (0 = all)")
    ap.add_argument("--min-chars", type=int, default=500,
                    help="Skip KOs with less extracted text (default 500; 0 keeps failed extractions)")
    ap.add_argument("--file-type", default="", help="Comma list: 'pdf', 'mp4', 'Video', 'Factsheet'")
    ap.add_argument("--languages", default="", help="Comma list, e.g. 'English,French'; empty = all")
    ap.add_argument("--lang-balance", choices=("equal", "proportional"), default="equal")
    ap.add_argument("--seed", type=int, default=42)
    # stages
    ap.add_argument("--no-gate", action="store_true", help="Assess every KO, including off-domain")
    ap.add_argument("--no-gate-llm", action="store_true", help="Classifier-only gate: no LLM calls")
    ap.add_argument("--variants", default="original,llm",
                    help="Metadata to score: 'original' (contributor), 'llm' (existing *_llm fields)")
    ap.add_argument("--llm-review", action="store_true",
                    help="Stage 5: one LLM call per KO to review and, if justified, rewrite fields")
    ap.add_argument("--llm-limit", type=int, default=10,
                    help="Review at most this many KOs (default 10; 0 = every assessed KO)")
    ap.add_argument("--llm-model", default="", help="Model preference for the review, e.g. 'glm'")
    ap.add_argument("--llm-language", choices=("keep", "source", "english"), default="keep",
                    help="Language of proposals: same as the field (default), the file's, or English")
    ap.add_argument("--selftest", action="store_true", help="Also run the perturbation tests")
    ap.add_argument("--learn", action="store_true",
                    help="Relearn the compliance profile from this export first (1 LLM call)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from core import metadata_metrics as mm
    from core.corpus_index import CorpusIndex, Embedder, load_spec

    t0 = time.time()
    export = Path(args.input or _latest_json_file(str(Path(__file__).resolve().parent.parent / "input")))
    print(f"[load] {export}", flush=True)
    records = read_records(export)
    print(f"[load] {len(records)} records", flush=True)

    spec = load_spec()
    emb = Embedder(spec["models"]["embedding"])
    nli = mm.NLI(spec["models"]["nli"])
    idx = CorpusIndex.load_or_build(records, export, spec, emb, log=lambda m: print(m, flush=True))

    if args.learn:
        profile = compliance_stage.learn_profile(records)
        compliance_stage.save_profile(profile)
        print(f"[compliance] relearned: {len(profile['expected_fields'])} required fields", flush=True)
    profile = compliance_stage.load_profile()

    selected, selection = select_records(
        records, limit=args.limit, min_chars=args.min_chars,
        file_types=[t for t in args.file_type.split(",") if t.strip()],
        languages=[l for l in args.languages.split(",") if l.strip()],
        lang_balance=args.lang_balance, seed=args.seed)
    for reason, n in selection["excluded"].items():
        if n:
            print(f"[select] excluded {n} ({reason})", flush=True)
    print(f"[select] {selection['selected']} of {selection['eligible']} eligible: "
          f"{selection['by_language']}", flush=True)
    if not selected:
        sys.exit("Nothing selected. Loosen --min-chars, --file-type or --languages.")

    gates = run_gate(selected, not args.no_gate, not args.no_gate_llm, args.workers)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out_dir = args.output_dir or (Path(__file__).resolve().parent / "output" /
                                  f"run_{datetime.now():%Y%m%d_%H%M%S}")
    ensure_directory(out_dir)
    results: List[Dict[str, Any]] = []
    t1 = time.time()
    # Flushed as produced: a long run that dies late keeps what it finished.
    with (out_dir / "partial.jsonl").open("w", encoding="utf-8") as part:
        for i, (rec, gate) in enumerate(zip(selected, gates), 1):
            try:
                r = assess_one(rec, gate, profile, idx, emb, nli, spec, variants)
            except Exception as exc:
                r = {"_orig_id": rec.get("_orig_id"), "error": f"{type(exc).__name__}: {exc}"}
            results.append(r)
            part.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            part.flush()
            if i % 10 == 0 or i == len(selected):
                print(f"[assess] {i}/{len(selected)}  {time.time() - t1:.0f}s", flush=True)

    reviews: List[Dict[str, Any]] = []
    if args.llm_review:
        from stages.metadata_llm import review

        todo = [(rec, r) for rec, r in zip(selected, results)
                if (r.get("metadata") or {}).get("variants", {}).get("original")]
        if args.llm_limit:
            todo = todo[:args.llm_limit]
        reviews = review([rec for rec, _ in todo],
                         {r["_orig_id"]: r["metadata"]["variants"]["original"] for _, r in todo},
                         idx, emb, nli, spec, model=args.llm_model,
                         language=args.llm_language, workers=min(4, args.workers),
                         log=lambda m: print(m, flush=True))

    selftest: List[Dict[str, Any]] = []
    if args.selftest:
        from tools.score_metadata import run_selftest

        print("[selftest] one deliberate defect per sub-pillar per KO ...", flush=True)
        ok = [(rec, r["metadata"]) for rec, r in zip(selected, results) if r.get("metadata")]
        try:
            selftest = run_selftest(records, [x for x, _ in ok], [m for _, m in ok],
                                    idx, emb, nli, spec, args.seed)
        except Exception as exc:
            print(f"[selftest] failed: {type(exc).__name__}: {exc}", flush=True)

    try:
        analysis_out = run_analysis(results, spec)
    except Exception as exc:            # never lose a finished run to the summary step
        analysis_out = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"[analysis] failed: {analysis_out['error']}", flush=True)
    analysis_out["selftest"] = selftest and {
        "defects": len(selftest),
        "detected": sum(1 for s in selftest if s["outcome"] == "dropped")}
    run_info = {
        "export": str(export), "args": {k: str(v) for k, v in vars(args).items()},
        "selection": selection, "spec_version": spec["version"], "models": spec["models"],
        "corpus_fingerprint": CorpusIndex.fingerprint(export, spec), "corpus": idx.meta,
        "llm_review_kos": len(reviews), "seconds": round(time.time() - t0, 1),
    }
    write_outputs(out_dir, results, reviews, analysis_out, selftest, run_info)
    (out_dir / "partial.jsonl").unlink(missing_ok=True)
    print_summary(results, reviews, analysis_out, selftest, out_dir)


if __name__ == "__main__":
    main()
