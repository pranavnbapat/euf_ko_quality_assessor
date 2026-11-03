#!/usr/bin/env python3
# assess_ko_quality/pipeline.py
"""
The KO quality pipeline, end to end.

    select ─► 1 GATE ─► 2 COMPLIANCE ─► 3 CONTENT ─► 4 METADATA SCORE ─► 5 ANALYSIS
              domain     obligations     the file       title, subtitle,     why, and how
              relevance  (reported)      itself         description,         much the
                                         (reported,     keywords vs the      scores depend
                                         never scored)  summary: 3 measured  on our choices
                                                        pillars + the LLM
                                                        score (THE score)

  select ......... which KOs: --limit, --file-type, --languages (first listed language,
                   English by default), --licenses / --exclude-licenses, --lang-balance,
                   --min-summary-words
  1 gate ......... is the KO about agriculture in the broad sense? A classifier decides
                   the clear cases; an LLM only the uncertain band. Rejected KOs stop here.
  2 compliance ... fields the upload form requires, controlled-vocabulary conformance
                   (drift reported separately), age. Pass/fail; never in the score.
  3 content ...... the published web-text quality filters (Gopher, C4, FineWeb, FineWeb2),
                   lexical diversity, readability, declared vs detected language,
                   provenance cues. The file is the author's work, so this is reported and
                   never scored.
  4 metadata ..... the score: title, subtitle, description and keywords, the contributor's
                   version and an LLM version written fresh from the KO's summary (one
                   call per KO, stages/generate_metadata.py; --use-existing-llm scores
                   the export's *_llm fields instead), measured against that summary
                   (ko_content_flat_summarised; `reference.source` in the spec). With
                   --resummarise the summary is written fresh from the file itself, in
                   the document's own language (stages/resummarise.py), and the
                   summaries used are stored in the outputs.
                   Per field: three measured pillars (ten sub-pillars, no LLM) and an LLM's
                   0-100 score of the field against the summary (one call per field per
                   version), averaged with equal weights; an empty field scores 0. Per
                   field, the better version is recommended: the higher score, unless less
                   faithful to the summary than the contributor's.
  5 analysis ..... the weakest sub-pillars of each KO with their evidence; rank stability
                   under alternative weights; redundant sub-pillars; length bias;
                   (--selftest) perturbation tests of every sub-pillar.

Methods and sources: METHODS.md and LITERATURE_REVIEW.md. Settings with provenance:
validation/metadata_spec.json.

Usage (from inside assess_ko_quality/; every language by default):
    python pipeline.py --limit 10                             # up to 9 LLM calls per KO
    python pipeline.py --limit 10 --no-llm-score --no-gate-llm --use-existing-llm
                                                              # no LLM calls at all
    python pipeline.py --limit 200 --file-type pdf
    python pipeline.py --limit 20 --selftest                  # also run the perturbation tests
    python pipeline.py --languages English --limit 20         # one language only

The LLM version of the fields costs one call per KO (--use-existing-llm skips it and
scores the export's *_llm fields); the LLM score one call per field that has text, per
version: up to 8 per KO. Every number is printed before any call is made; answers are
cached, so a rerun is free. The gate adds a call for each KO in its uncertain band
unless --no-gate-llm. The first run on an export builds the corpus index (about 5
minutes, plus about 15 the very first time to embed every KO) and caches it under
output/_corpus_cache/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.llm import merged_env

# Hugging Face, set before anything imports it: the HF_TOKEN from the same .env files the
# LLM client reads, so model requests are authenticated, and no progress bars, since the
# models load from the local cache on every run. An exported variable wins over both.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
if not os.environ.get("HF_TOKEN") and merged_env().get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = merged_env()["HF_TOKEN"]

from core.corpus_index import summary_text  # noqa: E402
from core.io_utils import _latest_json_file, _read_json_any, ensure_directory  # noqa: E402
from core.text_utils import norm_text  # noqa: E402
from stages import compliance as compliance_stage  # noqa: E402



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


def _norm_licence(value: Any) -> str:
    return " ".join(re.sub(r"[_\-]+", " ", str(value)).lower().split())


def licences_of(rec: Dict[str, Any]) -> List[str]:
    v = rec.get("license")
    return [_norm_licence(x) for x in (v if isinstance(v, list) else [v]) if x]


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
    min_summary_words: int = 50,
    file_types: Sequence[str] = (),
    languages: Sequence[str] = (),
    lang_balance: str = "equal",
    licenses: Sequence[str] = (),
    exclude_licenses: Sequence[str] = (),
    seed: int = 42,
    upload_sources: Sequence[str] = (),
    ids: Sequence[str] = (),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Choose which KOs to assess, and say plainly what was left out.

    `lang_balance="equal"` spreads a limited run round-robin across the languages
    present, so a 100-record run is not 57% English simply because the platform is.
    Use "proportional" to mirror the corpus instead.

    Excluded counts are reported rather than swallowed. `min_summary_words` keeps the
    KOs whose summary (ko_content_flat_summarised) has at least that many words. Below 50
    words the export's *_llm fields are copies of the contributor's (584 of 584 KOs in
    the June 2026 export), and the summary is also what the fields are measured against
    and what the LLM reads; 6.3% of that export falls below.
    """
    import random

    rng = random.Random(seed)
    report: Dict[str, Any] = {"input": len(records), "excluded": {}}

    kept = list(records)
    if ids:
        # Named KOs (debugging, reproducing one case); the other filters still apply.
        from core.corpus_index import ko_id

        wanted = {str(i).strip() for i in ids if str(i).strip()}
        before = len(kept)
        kept = [r for r in kept if ko_id(r) in wanted]
        report["excluded"]["not_in_ids"] = before - len(kept)
    if min_summary_words > 0:
        before = len(kept)
        kept = [r for r in kept if len(summary_text(r).split()) >= min_summary_words]
        report["excluded"]["summary_under_min_words"] = before - len(kept)
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
    if upload_sources:
        # How the KO reached the platform: upload_form, public_api, ai_uploader, ...
        wanted_src = {s.strip().lower() for s in upload_sources if s.strip()}
        before = len(kept)
        kept = [r for r in kept if str(r.get("ko_upload_source") or "").strip().lower() in wanted_src]
        report["excluded"]["wrong_upload_source"] = before - len(kept)

    if licenses:
        wanted = {_norm_licence(l) for l in licenses if l.strip()}
        before = len(kept)
        kept = [r for r in kept if set(licences_of(r)) & wanted]
        report["excluded"]["wrong_license"] = before - len(kept)
    if exclude_licenses:
        banned = {_norm_licence(l) for l in exclude_licenses if l.strip()}
        before = len(kept)
        kept = [r for r in kept if not (set(licences_of(r)) & banned)]
        report["excluded"]["excluded_license"] = before - len(kept)

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
def _model_family(name: str) -> str:
    """The leading letters of a model id: qwen3.5-397b-a17b and qwen3.8-27b are both
    "qwen". Used to warn when the generator and the judge share a family."""
    m = re.match(r"[a-z]+", str(name).lower())
    return m.group(0) if m else str(name).lower()


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
               spec: Dict[str, Any], variants: Sequence[str],
               llm: Optional[Dict[str, Dict[str, Any]]] = None,
               check_summary: bool = True) -> Dict[str, Any]:
    """Stages 2-4 and the per-KO part of 5 for one KO. `llm`: this KO's LLM scores."""
    from core import content_report
    from core.corpus_index import ko_id
    from stages import analysis, summary_check
    from stages.metadata_score import recommend, score_ko

    out: Dict[str, Any] = {"_orig_id": ko_id(rec), "title": norm_text(rec.get("title"))[:200],
                           "declared_language": primary_language(rec),
                           "category": rec.get("category")}
    out["_licence"] = ";".join(str(x) for x in (rec.get("license") if isinstance(rec.get("license"), list)
                                               else [rec.get("license")]) if x)
    out["gate"] = gate or {"gate_decision": "not_evaluated"}
    if (gate or {}).get("gate_decision") == "reject":
        out["skipped"] = "off-domain (gate)"
        return out

    out["compliance"] = compliance_stage.check(rec, profile=profile)
    pos = idx.pos.get(ko_id(rec))
    lang = idx.langs[pos] if pos is not None else "unknown"
    out["content"] = content_report.report(rec.get("ko_content_flat"), rec.get("languages"), lang)
    if check_summary:
        out["summary_check"] = summary_check.check(rec, idx, emb, nli, spec)
    # The reference summary itself, stored with every run: what the fields were
    # measured against. With --resummarise also its English version (the corpus row)
    # and the platform's own summary it replaced.
    summary = summary_text(rec)
    out["summary"] = {"source": "resummarised" if rec.get("_resummarised_model") else "export",
                      "language": rec.get("_summary_language"),
                      "words": len(summary.split()), "text": summary}
    if rec.get("_resummarised_model"):
        if rec.get("_summary_en") and rec["_summary_en"] != summary:
            out["summary"]["english"] = rec["_summary_en"]
        out["summary"]["platform"] = rec.get("_summary_platform")
        # _resummarised_from: {"from": "file"|"export_text", "kind", pages, chunks,
        # calls, findings, ...} - the basis of this KO's summary (stages/resummarise.py)
        out["resummarised"] = {"model": rec["_resummarised_model"],
                               **(rec.get("_resummarised_from") or {}),
                               "platform_summary_words": len(str(rec.get("_summary_platform") or "").split())}
    # Which fields the "llm" variant scores: the ones written this run, or - when
    # generation was off (--use-existing-llm) or failed for this KO - the export's.
    if rec.get("_generated_llm_model"):
        out["generated_llm"] = {"model": rec["_generated_llm_model"],
                                "language": rec.get("_generated_llm_language")}
    out["metadata"] = score_ko(rec, idx, emb, nli, spec, variants=variants, llm_scores=llm)
    out["weakest"] = analysis.explain(out["metadata"]["variants"].get("original"),
                                      reference=f"the {spec['reference'].get('source', 'file')}")
    out["recommended"] = recommend(out["metadata"]["variants"],
                                   float(spec["recommendation"]["accuracy_tolerance"]),
                                   llm_name="generated_llm" if rec.get("_generated_llm_model")
                                   else "existing_llm")
    return out


def run_analysis(results: List[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 5, corpus part: robustness, redundancy and length bias over this run."""
    from core.metadata_metrics import SUBPILLARS
    from stages import analysis
    from stages.metadata_score import PILLARS

    scored = [r for r in results if (r.get("metadata") or {}).get("variants", {}).get("original")]
    # The LLM pillar joins the robustness check only when this run scored it.
    pillars = [p for p in PILLARS if p != "llm" or any(
        r["metadata"]["variants"]["original"]["pillars"].get("llm") is not None for r in scored)]
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
    total_m = sum(spec["pillars"][p] for p in SUBPILLARS)     # the measured pillars only
    nominal_s = [spec["pillars"][p] / total_m / len(names)
                 for p, names in SUBPILLARS.items() for _ in names]

    field_rows = [{"subpillars": f["subpillars"],
                   "words": len((f.get("text") or "").split())}
                  for r in scored for f in r["metadata"]["variants"]["original"]["fields"].values()
                  if f.get("subpillars") and f.get("text")]
    return {
        "robustness_pillars": analysis.robustness(P, pillars, nominal_p),
        "robustness_subpillars": analysis.robustness(S, subs, nominal_s),
        "redundant_subpillars": analysis.redundancy(S, subs),
        "length_bias": analysis.length_bias(field_rows),
        # Attaches corpus-relative flags to each KO's summary_check in place.
        "reference_check": analysis.reference_flags(results),
        "llm_vs_measured": analysis.llm_agreement(results),
    }


# --------------------------------------------------------------------------- outputs
def _pillar(res: Optional[Dict[str, Any]], p: str) -> Optional[float]:
    return None if not res else res["pillars"].get(p)


def _field(res: Optional[Dict[str, Any]], f: str) -> Optional[float]:
    return None if not res else (res["fields"].get(f) or {}).get("score")


def write_outputs(out_dir: Path, results: List[Dict[str, Any]],
                  analysis_out: Dict[str, Any], selftest: List[Dict[str, Any]],
                  run_info: Dict[str, Any]) -> Path:
    """
    One file to read - report.xlsx (sheets Run, KOs, Fields, Self-test) - and
    details.jsonl, the raw archive with every piece of evidence, for code.
    """
    from stages import report

    ensure_directory(out_dir)
    with (out_dir / "details.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        fh.write(json.dumps({"_run": run_info, "_analysis": analysis_out,
                             "_selftest": selftest}, ensure_ascii=False, default=str) + "\n")
    return report.write(out_dir, results, analysis_out, selftest, run_info)


def print_summary(results, analysis_out, selftest, out_dir: Path,
                  reference: str = "file", llm_info: Optional[Dict[str, Any]] = None,
                  gen_info: Optional[Dict[str, Any]] = None) -> None:
    from statistics import mean, median

    from core.corpus_index import FIELDS
    from stages.metadata_score import PILLARS

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
    # Language cross-check against the file itself, available when --resummarise read
    # the pages; the stored text's langdetect stays in details.jsonl only.
    flangs = [((r.get("resummarised") or {}).get("file_language"), r.get("declared_language"))
              for r in scored]
    fl_known = sum(1 for f, _ in flangs if f)
    fl_mism = sum(1 for f, d in flangs
                  if f and d and f.strip().lower() != str(d).strip().lower())
    print(f"    compliance incomplete {incomplete} | content flagged by a web-text filter {flagged}"
          + (f" | file language differs from declared {fl_mism} of {fl_known}" if fl_known else ""))
    checked = [r for r in scored if r.get("summary_check")]
    if checked:
        unreliable = sum(1 for r in checked if r["summary_check"].get("reference_reliability") != "ok")
        print(f"    reference summary checked on {len(checked)}: {unreliable} flagged "
              f"(their scores read against a questionable reference)")
    has_llm = any(l)

    def line(label: str, a: str, b: str) -> None:
        print(f"  {label:24s} {a:>30}" + (f"   {b:>30}" if has_llm else ""))

    print(f"\n  metadata score, measured against the {reference}")
    if has_llm:
        print(f"  *_llm fields: written by {gen_info['model']} from the summary, "
              f"{gen_info['calls']} calls"
              + (f", {gen_info['failed']} failed (those KOs keep the export's)"
                 if gen_info.get("failed") else "")
              if gen_info and gen_info.get("model")
              else "  *_llm fields: the export's (--use-existing-llm)")
    if llm_info and llm_info.get("model"):
        print(f"  LLM pillar: {llm_info['model']}, {llm_info['calls']} calls"
              + (f", {llm_info['failed']} without a usable score" if llm_info.get("failed") else ""))
        agree = analysis_out.get("llm_vs_measured") or {}
        if any(v.get("spearman") is not None for v in agree.values() if isinstance(v, dict)):
            print("  code vs LLM per pillar (Spearman): "
                  + ", ".join(f"{p} {v['spearman']:+.2f} (n={v['n']})"
                              for p, v in agree.items()
                              if isinstance(v, dict) and v.get("spearman") is not None))
    else:
        print("  LLM pillar: not scored (--no-llm-score); field scores average the three measured pillars")
    line("", "contributor",
         "generated *_llm fields" if gen_info and gen_info.get("model") else "existing *_llm fields")
    line("overall", m([x and x["score"] for x in o]), m([x and x["score"] for x in l]))
    for p in PILLARS:
        line(p, m([_pillar(x, p) for x in o]), m([_pillar(x, p) for x in l]))
    for f in FIELDS:
        line(f, m([_field(x, f) for x in o]), m([_field(x, f) for x in l]))

    # Per-language view: scores are not comparable across languages (non-English
    # fields are scored on fewer instruments, and cross-lingual readings run lower),
    # so the pooled table above is never the whole story on a mixed run.
    from stages.analysis import file_language_of

    def mv(xs):
        xs = [x for x in xs if x is not None]
        return f"{sum(xs) / len(xs):5.1f}" if xs else "    -"

    groups: Dict[str, List[Any]] = {}
    for r in scored:
        groups.setdefault(file_language_of(r), []).append(r)
    if len(groups) > 1:
        print("\n  by file language (not comparable across languages: instrument coverage differs)")
        for lang in sorted(groups, key=lambda k: (-len(groups[k]), k)):
            rs = groups[lang]
            go = [r["metadata"]["variants"].get("original") for r in rs]
            gl = [r["metadata"]["variants"].get("llm") for r in rs]
            print(f"    {lang:14s} n={len(rs):3d}   contributor {mv([x and x['score'] for x in go])}"
                  + (f"   *_llm {mv([x and x['score'] for x in gl])}" if has_llm else ""))

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
    ap.add_argument("--min-summary-words", type=int, default=50,
                    help="Skip KOs whose summary is shorter (default 50). The summary is what the "
                         "fields are measured against; below 50 words the export's *_llm fields "
                         "are copies of the contributor's. 0 = no filter")
    ap.add_argument("--file-type", default="", help="Comma list: 'pdf', 'mp4', 'Video', 'Factsheet'")
    ap.add_argument("--languages", default="all",
                    help="Comma list of first-listed languages, e.g. 'English,French'; 'all' (the "
                         "default) = every language. Results are reported per language, never pooled")
    ap.add_argument("--lang-balance", choices=("equal", "proportional"), default="equal")
    ap.add_argument("--licenses", default="",
                    help="Keep only these licences, comma list, e.g. 'CC BY,CC BY-SA'")
    ap.add_argument("--exclude-licenses", default="",
                    help="Drop these licences, comma list, e.g. 'CC BY-NC-ND'")
    ap.add_argument("--upload-sources", default="",
                    help="Keep only KOs with these ko_upload_source values, comma list, e.g. "
                         "'upload_form' (others: public_api, ai_uploader)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ids", default="",
                    help="Comma list of _orig_id values to assess (debugging, reproducing one "
                         "case); the other filters still apply")
    # stages
    ap.add_argument("--no-gate", action="store_true", help="Assess every KO, including off-domain")
    ap.add_argument("--no-gate-llm", action="store_true", help="Classifier-only gate: no LLM calls")
    ap.add_argument("--variants", default="original,llm",
                    help="Metadata to score: 'original' (contributor) and 'llm' (written fresh "
                         "from the summary by the current model; the export's *_llm fields with "
                         "--use-existing-llm)")
    ap.add_argument("--use-existing-llm", action="store_true",
                    help="Score the export's *_llm fields as the LLM variant instead of writing "
                         "them fresh from the summary (one call per KO fewer; with --no-gate-llm "
                         "and --no-llm-score no LLM is called at all)")
    ap.add_argument("--no-llm-score", action="store_true",
                    help="Leave the LLM pillar out: no scoring calls, and each field score is the "
                         "mean of the three measured pillars (not comparable with a full run)")
    ap.add_argument("--resummarise", action="store_true",
                    help="Write each assessed KO's reference summary fresh from its file, in the "
                         "document's own language. The file is fetched from its @id URL and read "
                         "as it is: pages as images (a PDF over 100 pages as its own text, a PDF over "
                         "100 MB not downloaded), office files converted first, a video's sound "
                         "transcribed and its frames shown. Long files are summarised in parts and "
                         "merged, every draft is checked against its pages and revised, and a "
                         "non-English summary is also translated to English for the search index. "
                         "About 2-4 calls per short document, more per further 12 pages. A file "
                         "that cannot be fetched or read falls back to the export's text (marked "
                         "per KO). Not comparable with runs on the export's summaries")
    ap.add_argument("--no-summary-check", action="store_true",
                    help="Skip the summary reference check (stage 3b): no compression/coverage/"
                         "NLI-support reading on the reference summary")
    ap.add_argument("--selftest", action="store_true", help="Also run the perturbation tests")
    ap.add_argument("--learn", action="store_true",
                    help="Relearn the compliance profile from this export first (1 LLM call)")
    # Per-run model overrides, resolved against the provider's live /v1/models (name a
    # model in full: the shortest matching id wins). Defaults come from the spec; an
    # empty spec model means the client's default (LLM_MODEL in .env).
    ap.add_argument("--generate-model", default="",
                    help="Model that writes the *_llm fields (default: generate_metadata.model "
                         "in the spec, '' = the client's default model). Keep it a different "
                         "family from the judge")
    ap.add_argument("--judge-model", default="",
                    help="Model for the LLM pillar (default: llm_score.model in the spec). "
                         "Never the generator's family: LLM judges favour their own model's text")
    ap.add_argument("--resummarise-model", default="",
                    help="Model for --resummarise (default: resummarise.model in the spec, "
                         "'' = the client's default model)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from core import metadata_metrics as mm
    from core.corpus_index import CorpusIndex, Embedder, ko_id, load_spec

    t0 = time.time()
    export = Path(args.input or _latest_json_file(str(Path(__file__).resolve().parent.parent / "input")))
    print(f"[load] {export}", flush=True)
    records = read_records(export)
    print(f"[load] {len(records)} records", flush=True)

    spec = load_spec()
    # Model overrides from the command line; run_info["args"] records them.
    if args.generate_model:
        spec["generate_metadata"]["model"] = args.generate_model
    if args.judge_model:
        spec["llm_score"]["model"] = args.judge_model
    if args.resummarise_model:
        spec["resummarise"]["model"] = args.resummarise_model
    emb = Embedder(spec["models"]["embedding"])
    nli = mm.NLI(spec["models"]["nli"])
    idx = CorpusIndex.load_or_build(records, export, spec, emb, log=lambda m: print(m, flush=True))

    if args.learn:
        profile = compliance_stage.learn_profile(records)
        compliance_stage.save_profile(profile)
        print(f"[compliance] relearned: {len(profile['expected_fields'])} required fields", flush=True)
    profile = compliance_stage.load_profile()

    selected, selection = select_records(
        records, limit=args.limit, min_summary_words=args.min_summary_words,
        file_types=[t for t in args.file_type.split(",") if t.strip()],
        languages=[l for l in args.languages.split(",") if l.strip()],
        lang_balance=args.lang_balance, seed=args.seed,
        licenses=[l for l in args.licenses.split(",") if l.strip()],
        exclude_licenses=[l for l in args.exclude_licenses.split(",") if l.strip()],
        upload_sources=[u for u in args.upload_sources.split(",") if u.strip()],
        ids=[i for i in args.ids.split(",") if i.strip()])
    for reason, n in selection["excluded"].items():
        if n:
            print(f"[select] excluded {n} ({reason})", flush=True)
    print(f"[select] {selection['selected']} of {selection['eligible']} eligible: "
          f"{selection['by_language']}", flush=True)
    if not selected:
        sys.exit("Nothing selected. Loosen --min-summary-words, --file-type or --languages.")

    gates = run_gate(selected, not args.no_gate, not args.no_gate_llm, args.workers)
    passed = [rec for rec, g in zip(selected, gates) if (g or {}).get("gate_decision") != "reject"]

    # --resummarise: fresh reference summaries, in the document's own language, for the
    # KOs that will be scored - written into the in-memory records (the export file is
    # untouched) and, as their English version, into the index's own-row rankings. Must
    # come before the LLM pillar and the generation, which read the summary.
    resum_info: Optional[Dict[str, Any]] = None
    if args.resummarise:
        from core.corpus_index import SUMMARY_FIELD
        from stages import resummarise

        new_sums, resum_info, basis = resummarise.generate(
            passed, spec, workers=args.workers, log=lambda m: print(m, flush=True))
        for rec in passed:
            s = new_sums.get(ko_id(rec))
            if not s:
                continue
            rec["_summary_platform"] = rec.get(SUMMARY_FIELD)
            rec["_resummarised_model"] = resum_info["model"]
            rec["_resummarised_from"] = basis.get(ko_id(rec))
            rec[SUMMARY_FIELD] = s["summary"]             # the reference: document language
            rec["_summary_language"] = s["language"]
            rec["_summary_en"] = s["summary_en"]           # the corpus row; secondary reference
            pos = idx.pos.get(ko_id(rec))
            if pos is not None:
                # The index is built on English summaries, so the KO's own row stays
                # English too: a lone row in another language would have no competition.
                idx.override_reference(pos, s["summary_en"][:int(spec["reference"]["max_chars"])],
                                       emb, spec)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    # The "llm" variant is written here, fresh from the reference summary (one call per
    # KO), unless --use-existing-llm takes the export's *_llm fields instead. Must come
    # after --resummarise, so the fields are
    # written from the summary they will be measured against, and before the LLM
    # pillar, which reads them. The export file itself is untouched.
    gen_info: Optional[Dict[str, Any]] = None
    if "llm" in variants and not args.use_existing_llm:
        from core.corpus_index import FIELDS
        from stages import generate_metadata

        for rec in passed:                 # the detected-code fallback for target_language
            pos = idx.pos.get(ko_id(rec))
            rec["_detected_lang_code"] = idx.langs[pos] if pos is not None else ""
        new_fields, gen_info, gen_langs = generate_metadata.generate(
            passed, spec, workers=args.workers, log=lambda m: print(m, flush=True))
        for rec in passed:
            vals = new_fields.get(ko_id(rec))
            if not vals:
                continue
            rec["_existing_llm"] = {f: rec.get(f"{f}_llm") for f in FIELDS}
            rec["_generated_llm_model"] = gen_info["model"]
            rec["_generated_llm_language"] = gen_langs.get(ko_id(rec))
            for f in FIELDS:
                rec[f"{f}_llm"] = vals.get(f)
        judge = str(spec.get("llm_score", {}).get("model", ""))
        if not args.no_llm_score and judge and \
                _model_family(gen_info["model"]) == _model_family(judge):
            print(f"[generate] WARNING: the generator ({gen_info['model']}) and the judge "
                  f"({judge}) are the same model family; LLM judges favour their own "
                  "model's text", flush=True)

    # The LLM pillar: one call per field with text, per version, for every KO the gate
    # let through - all made up front and concurrently, and counted before they start.
    llm_scores: Dict[str, Any] = {}
    llm_info: Dict[str, Any] = {"model": None, "calls": 0, "failed": 0}
    if not args.no_llm_score:
        from stages import llm_score

        llm_scores, llm_info = llm_score.score_all(passed, variants, spec, workers=args.workers,
                                                   log=lambda m: print(m, flush=True))
    out_dir = args.output_dir or (Path(__file__).resolve().parent / "output" /
                                  f"run_{datetime.now():%Y%m%d_%H%M%S}")
    ensure_directory(out_dir)
    results: List[Dict[str, Any]] = []
    t1 = time.time()
    # Flushed as produced: a long run that dies late keeps what it finished.
    with (out_dir / "partial.jsonl").open("w", encoding="utf-8") as part:
        for i, (rec, gate) in enumerate(zip(selected, gates), 1):
            try:
                r = assess_one(rec, gate, profile, idx, emb, nli, spec, variants,
                               llm=None if args.no_llm_score else llm_scores.get(ko_id(rec), {}),
                               check_summary=not args.no_summary_check)
            except Exception as exc:
                r = {"_orig_id": rec.get("_orig_id"), "error": f"{type(exc).__name__}: {exc}"}
            results.append(r)
            part.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            part.flush()
            if i % 10 == 0 or i == len(selected):
                print(f"[assess] {i}/{len(selected)}  {time.time() - t1:.0f}s", flush=True)

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
        "reference_source": spec["reference"].get("source", "file"),
        "corpus_fingerprint": CorpusIndex.fingerprint(export, spec), "corpus": idx.meta,
        "llm_score": llm_info, "resummarise": resum_info, "generate_metadata": gen_info,
        "seconds": round(time.time() - t0, 1),
    }
    write_outputs(out_dir, results, analysis_out, selftest, run_info)
    (out_dir / "partial.jsonl").unlink(missing_ok=True)
    print_summary(results, analysis_out, selftest, out_dir,
                  spec["reference"].get("source", "file"), llm_info, gen_info)


if __name__ == "__main__":
    main()
