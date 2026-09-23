#!/usr/bin/env python3
# assess_ko_quality/tools/compare_nli.py
"""
Choose the NLI model for the accuracy check by measurement, not by reputation.

For each selected KO, every candidate model scores the KO's own title and description
against its file, and a title and description taken from another project's KO. A model
that measures accuracy must rank the true field above the swapped one. The comparison
reports, per model:

    AUC     P(true field scores higher than the swapped one); 0.5 is chance
    gap     mean score difference, true minus swapped
    speed   NLI pairs per second on this machine

split by whether the KO's language is one the model was trained on (XNLI languages) or
reached only by cross-lingual transfer, because that is where multilingual NLI models
are known to differ.

    cd assess_ko_quality
    python -m tools.compare_nli --limit 40
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

from core.corpus_index import CorpusIndex, Embedder, field_text, ko_id, load_spec  # noqa: E402
from core.io_utils import _latest_json_file  # noqa: E402
from core import metadata_metrics as mm  # noqa: E402
from tools.score_metadata import _donor  # noqa: E402

CANDIDATES = [
    "joeddav/xlm-roberta-large-xnli",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
]
# The fifteen XNLI languages (Conneau et al. 2018).
XNLI = {"en", "fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"}


def auc(pos: List[float], neg: List[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p, n in zip(pos, neg))
    return wins / len(pos) if pos else float("nan")


def main() -> None:
    import pipeline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7, help="differs from the scoring run's 42 on purpose")
    ap.add_argument("--models", nargs="*", default=CANDIDATES)
    ap.add_argument("--out", type=Path, default=BASE / "validation" / "runs" / "nli_model_comparison.json")
    args = ap.parse_args()

    export = Path(args.input or _latest_json_file(str(BASE.parent / "input")))
    records = pipeline.read_records(export)
    spec = load_spec()
    emb = Embedder(spec["models"]["embedding"])
    idx = CorpusIndex.load_or_build(records, export, spec, emb, log=lambda m: print(m, flush=True))
    selected, rep = pipeline.select_records(records, limit=args.limit, min_chars=500, seed=args.seed)
    print(f"[select] {rep['selected']} KOs: {rep['by_language']}", flush=True)

    # Contexts and donors are shared by every model, so each model sees identical inputs.
    cases = []
    for i, rec in enumerate(selected):
        ctx = mm.build_context(rec, idx, emb, spec)
        if not ctx.evaluable:
            continue
        donor = _donor(records, rec, random.Random(args.seed * 1000 + i))
        cases.append((rec, donor, ctx))

    results: Dict[str, Any] = {"n_kos": len(cases), "seed": args.seed, "models": {}}
    for name in args.models:
        nli = mm.NLI(name)
        nli.probs([("warm up", "warm up")])
        rows = []
        t0 = time.time()
        counter = {"pairs": 0}
        real_probs = nli.probs

        def counted(pairs, batch=16, _f=real_probs):
            counter["pairs"] += len(pairs)
            return _f(pairs, batch)
        nli.probs = counted
        for rec, donor, ctx in cases:
            for field in ("title", "description"):
                true_text = field_text(field, rec.get(field))
                fake_text = field_text(field, donor.get(field))
                if not true_text or not fake_text:
                    continue
                t, _ = mm._accuracy(field, rec.get(field), true_text, ctx, emb, nli, spec)
                f, _ = mm._accuracy(field, donor.get(field), fake_text, ctx, emb, nli, spec)
                rows.append({"field": field, "lang": ctx.lang, "true": t, "swapped": f})
        secs = time.time() - t0
        summary: Dict[str, Any] = {"pairs_per_second": round(counter["pairs"] / secs, 2)}
        for label, keep in [("all", lambda r: True),
                            ("xnli_languages", lambda r: r["lang"] in XNLI),
                            ("other_languages", lambda r: r["lang"] not in XNLI)]:
            for field in ("title", "description", "both"):
                sel = [r for r in rows if keep(r) and (field == "both" or r["field"] == field)]
                if not sel:
                    continue
                summary[f"{label}/{field}"] = {
                    "n": len(sel),
                    "auc": round(auc([r["true"] for r in sel], [r["swapped"] for r in sel]), 3),
                    "gap": round(sum(r["true"] - r["swapped"] for r in sel) / len(sel), 3),
                }
        results["models"][name] = {"summary": summary, "rows": rows}
        print(f"\n{name}  ({summary['pairs_per_second']} pairs/s)", flush=True)
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"  {k:32s} n={v['n']:3d}  AUC {v['auc']:.3f}  gap {v['gap']:+.3f}", flush=True)

    args.out.write_text(json.dumps(results, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
