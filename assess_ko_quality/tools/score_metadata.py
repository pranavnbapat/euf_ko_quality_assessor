#!/usr/bin/env python3
# assess_ko_quality/tools/score_metadata.py
"""
Score KO metadata - the contributor's and the LLM's proposal - on the same criteria.

For each selected KO this scores the original title / subtitle / description /
keywords and the *_llm versions already in the export, against the same reading of
the file, and reports which is better, field by field and pillar by pillar. No LLM is
called: the models are local (multilingual E5 for embeddings, XLM-R XNLI for
entailment) and the corpus statistics come from the export.

--selftest also damages each KO's metadata on purpose, once per sub-pillar, and checks
that the sub-pillar notices. A metric that does not drop when its property is broken
is not measuring that property, whatever its name says.

    cd assess_ko_quality
    python -m tools.score_metadata --limit 20 --selftest
    python -m tools.score_metadata --limit 200 --languages English

The first run on an export builds the corpus index (about 20 minutes on CPU, mostly
embedding every KO once) and caches it under output/_corpus_cache/.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

from core.corpus_index import (  # noqa: E402
    CorpusIndex, Embedder, FIELDS, field_text, field_value, ko_id, load_spec, project_key,
    split_keywords, word_tokens,
)
from core.io_utils import _latest_json_file  # noqa: E402
from core import metadata_metrics as mm  # noqa: E402
from stages.metadata_score import score_ko, subpillar_calls  # noqa: E402

SUBS = [s for names in mm.SUBPILLARS.values() for s in names]


# --------------------------------------------------------------------- self-test
def _donor(records: Sequence[Dict[str, Any]], rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """A KO from another project with all three main fields filled, chosen reproducibly."""
    mine = project_key(rec)
    for _ in range(1000):
        d = records[rng.randrange(len(records))]
        if project_key(d) != mine and all(field_text(f, d.get(f)) for f in ("title", "description", "keywords")):
            return d
    raise RuntimeError("no donor KO found")


def perturbations(rec: Dict[str, Any], donor: Dict[str, Any], ctx: mm.KOContext,
                  idx: CorpusIndex) -> List[Dict[str, Any]]:
    """
    One deliberate defect per sub-pillar. Each targets the property the sub-pillar
    claims to measure and nothing cleverer: if the metric cannot see THIS, it cannot
    be trusted with subtler cases.
    """
    title = field_text("title", rec.get("title"))
    desc = field_text("description", rec.get("description"))
    words = title.split()
    dwords = desc.split()
    generic = idx.generic_words(ctx.lang)
    return [
        {"sub": "length", "field": "title", "what": "title padded past the platform's length limit",
         "value": " ".join([title] * max(2, 320 // max(1, len(title.replace(" ", ""))) + 1))},
        {"sub": "completeness", "field": "title", "what": "title replaced by a filename",
         "value": "_".join(w.strip(".,:;") for w in words[:3]) + "_final.pdf"},
        {"sub": "noise", "field": "description", "what": "markup, a URL and encoding debris appended",
         "value": desc + " &amp; <br/> http://example.org/x �� ####"},
        {"sub": "formatting", "field": "title", "what": "title set in capitals",
         # the raw value, so the defect adds one fault and cleans none away
         "value": str(rec.get("title") or "").upper()},
        {"sub": "consistency", "field": "description", "what": "description taken from another project's KO",
         "value": donor.get("description")},
        {"sub": "consistency", "field": "title", "what": "title taken from another project's KO",
         "value": donor.get("title")},
        {"sub": "density", "field": "title", "what": "title rewritten in the language's commonest words",
         "value": " ".join(generic[:max(3, len(words))])},
        {"sub": "clarity", "field": "description", "what": "description cut off mid-sentence with an ellipsis",
         "value": " ".join(dwords[:max(3, int(0.6 * len(dwords)))]).rstrip(".!?;:,") + " ..."},
        {"sub": "usefulness", "field": "keywords", "what": "keywords taken from another project's KO",
         "value": donor.get("keywords")},
        {"sub": "usefulness", "field": "keywords", "what": "keywords padded with case variants of themselves",
         "value": split_keywords(rec.get("keywords")) + [t.upper() for t in split_keywords(rec.get("keywords"))]},
        {"sub": "bm25_findability", "field": "title", "what": "title taken from another project's KO",
         "value": donor.get("title")},
        {"sub": "semantic_findability", "field": "title", "what": "title taken from another project's KO",
         "value": donor.get("title")},
    ]


def run_selftest(records, selected, results, idx, emb, nli, spec, seed) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_id = {r["_orig_id"]: r for r in results}
    for i, rec in enumerate(selected):
        res = by_id.get(ko_id(rec))
        orig = (res or {}).get("variants", {}).get("original")
        if not orig:
            continue
        rng = random.Random(seed * 1000 + i)
        donor = _donor(records, rec, rng)
        ctx = mm.build_context(rec, idx, emb, spec)
        for p in perturbations(rec, donor, ctx, idx):
            base_field = orig["fields"].get(p["field"])
            if not base_field:
                continue
            before = base_field["subpillars"].get(p["sub"])
            values = {f: field_value(rec, f, "original") for f in FIELDS}
            values[p["field"]] = p["value"]
            after, _ = subpillar_calls(p["field"], p["value"], values, "original",
                                       ctx, idx, emb, nli, spec)[p["sub"]]()
            if before is None or after is None:
                outcome = "not_evaluable"
            elif after < before - 1e-9:
                outcome = "dropped"
            elif abs(after - before) <= 1e-9:
                outcome = "unchanged_at_floor" if before <= 1e-9 else "unchanged"
            else:
                outcome = "rose"
            rows.append({"_orig_id": ko_id(rec), "sub": p["sub"], "field": p["field"],
                         "defect": p["what"], "before": before,
                         "after": None if after is None else round(float(after), 4),
                         "outcome": outcome})
    return rows


# --------------------------------------------------------------------- output
def _fmt(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.1f}"


def write_outputs(out_dir: Path, results: List[Dict[str, Any]], selftest: List[Dict[str, Any]],
                  run_info: Dict[str, Any], records_by_id: Dict[str, Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "details.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    field_cols = (["_orig_id", "variant", "field", "score"] + list(mm.SUBPILLARS) + SUBS + ["text"])
    with (out_dir / "fields.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(field_cols)
        for r in results:
            for var, vres in r["variants"].items():
                for f, fres in vres["fields"].items():
                    w.writerow([r["_orig_id"], var, f, fres["score"]]
                               + [fres["pillars"].get(p) for p in mm.SUBPILLARS]
                               + [fres["subpillars"].get(s) for s in SUBS]
                               + [fres["text"].replace("\t", " ")])

    ko_cols = ["_orig_id", "declared_language", "reference_language", "reference_chars",
               "score_original", "score_llm", "delta"]
    ko_cols += [f"{p}_{v}" for p in mm.SUBPILLARS for v in ("original", "llm")]
    ko_cols += [f"{f}_{v}" for f in FIELDS for v in ("original", "llm")]
    with (out_dir / "kos.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(ko_cols + ["title_original", "title_llm"])
        for r in results:
            o = r["variants"].get("original") or {}
            l = r["variants"].get("llm") or {}
            rec = records_by_id.get(r["_orig_id"], {})
            langs = rec.get("languages") or []
            row = [r["_orig_id"], (langs[0] if isinstance(langs, list) and langs else langs),
                   r["reference_language"], r["reference_chars"], o.get("score"), l.get("score"),
                   (round(l["score"] - o["score"], 1) if o.get("score") is not None and l.get("score") is not None else None)]
            row += [(x.get("pillars") or {}).get(p) for p in mm.SUBPILLARS for x in (o, l)]
            row += [((x.get("fields") or {}).get(f) or {}).get("score") for f in FIELDS for x in (o, l)]
            row += [field_text("title", rec.get("title")), field_text("title", rec.get("title_llm"))]
            w.writerow(row)

    if selftest:
        with (out_dir / "selftest.tsv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(selftest[0]), delimiter="\t")
            w.writeheader()
            w.writerows(selftest)

    (out_dir / "run.json").write_text(json.dumps(run_info, indent=1, default=str))


def summarise(results: List[Dict[str, Any]], selftest: List[Dict[str, Any]]) -> Dict[str, Any]:
    both = [r for r in results if "original" in r["variants"] and "llm" in r["variants"]
            and r["variants"]["original"]["score"] is not None and r["variants"]["llm"]["score"] is not None]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 1) if xs else None

    s: Dict[str, Any] = {"kos": len(results), "kos_with_both": len(both)}
    if both:
        o = [r["variants"]["original"]["score"] for r in both]
        l = [r["variants"]["llm"]["score"] for r in both]
        s["score"] = {"original_mean": mean(o), "llm_mean": mean(l),
                      "original_median": round(statistics.median(o), 1),
                      "llm_median": round(statistics.median(l), 1),
                      "llm_better": sum(b > a + 1 for a, b in zip(o, l)),
                      "original_better": sum(a > b + 1 for a, b in zip(o, l)),
                      "within_1_point": sum(abs(a - b) <= 1 for a, b in zip(o, l))}
        s["pillars"] = {p: {v: mean([r["variants"][v]["pillars"].get(p) for r in both])
                            for v in ("original", "llm")} for p in mm.SUBPILLARS}
        s["fields"] = {f: {v: mean([(r["variants"][v]["fields"].get(f) or {}).get("score") for r in both])
                           for v in ("original", "llm")} for f in FIELDS}
        s["subpillars"] = {}
        for sub in SUBS:
            s["subpillars"][sub] = {v: mean([100 * x for r in both for fr in r["variants"][v]["fields"].values()
                                             for x in [fr["subpillars"].get(sub)] if x is not None])
                                    for v in ("original", "llm")}
    if selftest:
        agg: Dict[str, Dict[str, int]] = {}
        for row in selftest:
            key = f"{row['sub']} ({row['field']})"
            agg.setdefault(key, {}).setdefault(row["outcome"], 0)
            agg[key][row["outcome"]] += 1
        s["selftest"] = agg
    return s


def print_summary(s: Dict[str, Any], out_dir: Path) -> None:
    print(f"\n=== metadata score: {s['kos']} KOs, {s['kos_with_both']} with both original and LLM metadata")
    if "score" in s:
        sc = s["score"]
        print(f"  overall   original {sc['original_mean']:>5} (median {sc['original_median']})"
              f"   llm {sc['llm_mean']:>5} (median {sc['llm_median']})")
        print(f"  llm better on {sc['llm_better']}, original better on {sc['original_better']},"
              f" within 1 point on {sc['within_1_point']}")
        print(f"\n  {'':24s} {'original':>9} {'llm':>9}")
        for p, v in s["pillars"].items():
            print(f"  {p:24s} {_fmt(v['original']):>9} {_fmt(v['llm']):>9}")
        print()
        for f, v in s["fields"].items():
            print(f"  {f:24s} {_fmt(v['original']):>9} {_fmt(v['llm']):>9}")
        print()
        for sub, v in s["subpillars"].items():
            print(f"  {sub:24s} {_fmt(v['original']):>9} {_fmt(v['llm']):>9}")
    if s.get("selftest"):
        print("\n=== self-test: does each sub-pillar drop when its property is broken?")
        cols = ["dropped", "unchanged_at_floor", "unchanged", "rose", "not_evaluable"]
        print(f"  {'sub-pillar (field)':36s} " + " ".join(f"{c[:9]:>9}" for c in cols))
        for key, counts in s["selftest"].items():
            print(f"  {key:36s} " + " ".join(f"{counts.get(c, 0):>9}" for c in cols))
    print(f"\noutputs -> {out_dir}")


# --------------------------------------------------------------------- main
def main() -> None:
    import pipeline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None, help="KO export (default: newest in ../input)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--file-type", nargs="*", default=[])
    ap.add_argument("--languages", nargs="*", default=[])
    ap.add_argument("--lang-balance", choices=["equal", "proportional"], default="equal")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true", help="also run the perturbation tests")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    export = args.input or _latest_json_file(str(BASE.parent / "input"))
    print(f"[load] {export}", flush=True)
    records = pipeline.read_records(Path(export))
    print(f"[load] {len(records)} records in {time.time() - t0:.0f}s", flush=True)

    spec = load_spec()
    emb = Embedder(spec["models"]["embedding"])
    nli = mm.NLI(spec["models"]["nli"])
    idx = CorpusIndex.load_or_build(records, Path(export), spec, emb,
                                    log=lambda m: print(m, flush=True))

    selected, sel_report = pipeline.select_records(
        records, limit=args.limit, min_chars=args.min_chars, file_types=args.file_type,
        languages=args.languages, lang_balance=args.lang_balance, seed=args.seed)
    print(f"[select] {sel_report['selected']} of {sel_report['eligible']} eligible "
          f"({', '.join(f'{k} {v}' for k, v in sel_report['by_language'].items())})", flush=True)

    results = []
    t1 = time.time()
    for i, rec in enumerate(selected, 1):
        results.append(score_ko(rec, idx, emb, nli, spec))
        if i % 5 == 0 or i == len(selected):
            print(f"[score] {i}/{len(selected)}  {time.time() - t1:.0f}s", flush=True)

    selftest = []
    if args.selftest:
        print("[selftest] damaging each KO's metadata once per sub-pillar ...", flush=True)
        selftest = run_selftest(records, selected, results, idx, emb, nli, spec, args.seed)

    summary = summarise(results, selftest)
    out_dir = args.out_dir or (BASE / "output" / f"metadata_{datetime.now():%Y%m%d_%H%M%S}")
    run_info = {
        "export": str(export), "args": {k: str(v) for k, v in vars(args).items()},
        "selection": sel_report, "spec_version": spec["version"],
        "corpus_fingerprint": CorpusIndex.fingerprint(Path(export), spec),
        "corpus": idx.meta, "llm_calls": 0, "seconds": round(time.time() - t0, 1),
        "summary": summary,
    }
    write_outputs(out_dir, results, selftest, run_info, {ko_id(r): r for r in selected})
    print_summary(summary, out_dir)


if __name__ == "__main__":
    main()
