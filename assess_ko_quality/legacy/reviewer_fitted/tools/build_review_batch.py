#!/usr/bin/env python3
# assess_ko_quality/tools/build_review_batch.py
"""
Build a stratified human-review batch, in the format the 2024 reviewers already used.

WHY THIS EXISTS

Every weight in the score is fitted on the 91 KOs covered by the 2024 review, and that
sample does not look like the platform it is applied to:

    content length, median chars        3,428 sample   vs   1,807 corpus
    failed extractions (< 400 chars)     12.0% sample   vs    35.8% corpus
    English                              76.1% sample   vs    59.9% corpus
    Forestry                              8.7% sample   vs    17.6% corpus
    Economics                            37.0% sample   vs    20.6% corpus

Measured on the sample's own percentile scale, the median corpus KO sat at the 19th
percentile. Reference quantiles were moved to the corpus to fix the scale, but that does
not fix the fitting: whether the relationship holds on short, non-English, Forestry
content is simply untested.

Reviewing another 100 KOs drawn the same way would reproduce the same blind spot. This
tool draws a batch that matches the corpus instead, so the next round tells us what we
do not already know.

HOW IT SAMPLES

Strata are content-length band x language, allocated proportionally to the corpus.
Length is the dimension that matters most because it is a score input, and the
extraction-failure band is kept as its own stratum rather than being swallowed into a
quartile - a third of the platform lives there and the review has barely seen it.

ANCHORS

A few KOs from the 2024 round are included again. Without overlap there is no way to
tell whether a shift in scores means the content changed or the reviewers did. They are
listed in the key file, not marked in the workbook.

    python -m tools.build_review_batch --input ../input/<export>.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

from core.text_utils import norm_text  # noqa: E402

REVIEWED = BASE / "validation" / "runs" / "human_reviewed_92_kos.json"
WORKBOOK = BASE / "data" / "2024_KOs for assessment all collating.xlsx"

# Fields a reviewer needs to judge the KO. The internal bookkeeping columns of the 2024
# sheet are dropped: the parsers locate the id column and the questions by content, so
# nothing downstream depends on the exact column layout.
META_FIELDS = [
    ("_id", "_id"),
    ("@id", "@id"),
    ("title", "title"),
    ("subtitle", "subtitle"),
    ("description", "description"),
    ("keywords", "keywords"),
    ("creators", "creators"),
    ("languages", "languages"),
    ("date_of_completion", "date_of_completion"),
    ("intended_purposes", "intended_purposes"),
    ("locations", "locations"),
    ("category", "category"),
    ("subcategories", "subcategory"),
    ("topics", "topics"),
    ("themes", "themes"),
    ("license", "license"),
    ("project_name", "project_name"),
    ("project_type", "project type"),
]

CONTRIB_URL = "https://eufarmbook.eu/en/contributions/{}"

# Length bands. The first is the extraction-failure band, kept separate on purpose.
BANDS: List[Tuple[str, int, int]] = [
    ("A_no_text", 0, 400),
    ("B_short", 400, 1800),
    ("C_medium", 1800, 5000),
    ("D_long", 5000, 10 ** 9),
]


def band_of(chars: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= chars < hi:
            return name
    return BANDS[-1][0]


def as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if x is not None and str(x).strip()]
    return [str(v)]


def cell(v: Any) -> str:
    return "; ".join(as_list(v)) if not isinstance(v, str) else v.strip()


def profile(doc: Dict[str, Any]) -> Dict[str, Any]:
    chars = len(norm_text(doc.get("ko_content_flat")))
    langs = as_list(doc.get("languages"))
    return {
        "id": (doc.get("_orig_id") or doc.get("_id") or "").lower(),
        "chars": chars,
        "band": band_of(chars),
        "english": "English" in langs,
        "topics": as_list(doc.get("topics")),
        "doc": doc,
    }


def allocate(counts: Dict[Tuple[str, bool], int], total: int, minimum: int = 2) -> Dict[Tuple[str, bool], int]:
    """
    Proportional allocation with a floor, adjusted so the parts sum to `total`.

    The floor matters: a stratum that is 3% of the corpus still needs enough rows to say
    anything about, and those are exactly the cells the 2024 round under-sampled.
    """
    grand = sum(counts.values())
    if not grand:
        return {}
    alloc = {k: max(minimum, round(total * n / grand)) for k, n in counts.items() if n}
    # Trim or pad from the largest strata so the batch is exactly `total`.
    while sum(alloc.values()) != total:
        diff = total - sum(alloc.values())
        order = sorted(alloc, key=lambda k: counts[k], reverse=(diff > 0))
        for k in order:
            if diff > 0:
                alloc[k] += 1
                diff -= 1
            elif alloc[k] > minimum:
                alloc[k] -= 1
                diff += 1
            if diff == 0:
                break
        else:
            break          # every stratum is at the floor; accept the overshoot
    return alloc


def build(
    docs: Sequence[Dict[str, Any]],
    size: int,
    anchors: int,
    seed: int,
    already: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    seen = {a.lower() for a in already}

    profiles = [profile(d) for d in docs]
    fresh = [p for p in profiles if p["id"] and p["id"] not in seen]
    prior = [p for p in profiles if p["id"] in seen]

    corpus_counts: Counter = Counter((p["band"], p["english"]) for p in fresh)
    alloc = allocate(dict(corpus_counts), size)

    by_cell: Dict[Tuple[str, bool], List[Dict[str, Any]]] = {}
    for p in fresh:
        by_cell.setdefault((p["band"], p["english"]), []).append(p)

    picked: List[Dict[str, Any]] = []
    for key, n in sorted(alloc.items()):
        pool = by_cell.get(key, [])
        picked.extend(rng.sample(pool, min(n, len(pool))))

    anchor_pick = rng.sample(prior, min(anchors, len(prior))) if prior else []
    batch = picked + anchor_pick
    rng.shuffle(batch)        # reviewers should not see the strata in the row order

    def dist(items: Sequence[Dict[str, Any]], key) -> Dict[str, float]:
        if not items:
            return {}
        c: Counter = Counter()
        for it in items:
            v = key(it)
            for x in (v if isinstance(v, list) else [v]):
                c[str(x)] += 1
        return {k: round(100 * v / len(items), 1) for k, v in sorted(c.items())}

    report = {
        "built": datetime.now(timezone.utc).date().isoformat(),
        "seed": seed,
        "corpus_size": len(profiles),
        "eligible_unreviewed": len(fresh),
        "batch_size": len(batch),
        "anchors_from_2024": [p["id"] for p in anchor_pick],
        "allocation": {f"{b}|{'EN' if e else 'non-EN'}": n for (b, e), n in sorted(alloc.items())},
        "match": {
            "length_band": {"corpus": dist(fresh, lambda p: p["band"]),
                            "batch": dist(picked, lambda p: p["band"])},
            "english": {"corpus": dist(fresh, lambda p: p["english"]),
                        "batch": dist(picked, lambda p: p["english"])},
            "topics": {"corpus": dist(fresh, lambda p: p["topics"]),
                       "batch": dist(picked, lambda p: p["topics"])},
        },
    }
    return batch, report


def write_workbook(batch: Sequence[Dict[str, Any]], workbook: Path, out: Path) -> None:
    """Write the batch using the 2024 review form, read from the original workbook."""
    import pandas as pd
    from openpyxl import Workbook

    src = pd.read_excel(workbook, sheet_name="CREA")
    questions: List[Tuple[str, str, str]] = []      # (group label, question, answer type)
    for col in src.columns:
        q, a = src.loc[0, col], src.loc[1, col]
        has_question = isinstance(q, str) and len(q.strip()) >= 25
        has_type = isinstance(a, str) and bool(a.strip())
        # The overall recommendation carries question text but no answer-type cell, so
        # filtering on the type alone silently drops the one question the score is
        # validated against. Keep anything that asks something.
        if not (has_question or has_type):
            continue
        group = str(col) if not str(col).startswith("Unnamed") else ""
        qtext = q.strip() if isinstance(q, str) else ""
        atype = a.strip() if has_type else ("1-5" if "recommend" in qtext.lower() else "")
        questions.append((group, qtext, atype))
    if not questions:
        raise ValueError(f"No review questions found in {workbook}")

    wb = Workbook()
    ws = wb.active
    ws.title = "review"

    headers = ["contribution_url"] + [label for _, label in META_FIELDS] + [g for g, _, _ in questions]
    row_q = [""] * (1 + len(META_FIELDS)) + [q for _, q, _ in questions]
    row_a = [""] * (1 + len(META_FIELDS)) + [a for _, _, a in questions]
    ws.append(headers)
    ws.append(row_q)
    ws.append(row_a)

    for p in batch:
        d = p["doc"]
        row = [CONTRIB_URL.format(p["id"])]
        for field, _ in META_FIELDS:
            row.append(cell(d.get(field)))
        row.extend([""] * len(questions))
        ws.append(row)

    widths = {1: 46, 2: 26, 4: 52, 6: 60}
    for idx, w in widths.items():
        ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = w
    ws.freeze_panes = "A4"

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="KO export JSON")
    ap.add_argument("--size", type=int, default=100, help="New KOs to review")
    ap.add_argument("--anchors", type=int, default=10,
                    help="KOs from the 2024 round to repeat, for cross-round calibration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=BASE / "data" / "review_batch_2026.xlsx")
    args = ap.parse_args()

    import orjson
    raw = orjson.loads(args.input.read_bytes())
    docs = raw["docs"] if isinstance(raw, dict) and "docs" in raw else raw
    already = []
    if REVIEWED.exists():
        already = [k.get("_orig_id", "") for k in orjson.loads(REVIEWED.read_bytes())]

    batch, report = build(docs, args.size, args.anchors, args.seed, already)
    write_workbook(batch, WORKBOOK, args.out)

    key = args.out.with_suffix(".key.json")
    key.write_text(json.dumps({**report, "ids": [p["id"] for p in batch]}, indent=1))

    print(f"corpus {report['corpus_size']}, unreviewed {report['eligible_unreviewed']}")
    print(f"batch  {report['batch_size']}  ({args.size} new + {len(report['anchors_from_2024'])} anchors)")
    print(f"\n{'stratum':22s} {'n':>4}")
    for k, n in report["allocation"].items():
        print(f"  {k:20s} {n:4d}")
    print(f"\n{'':22s} {'corpus':>9} {'batch':>9}")
    for dim in ("length_band", "english", "topics"):
        print(f"{dim}:")
        m = report["match"][dim]
        for k in sorted(set(m["corpus"]) | set(m["batch"])):
            print(f"  {k:20s} {m['corpus'].get(k, 0):8.1f}% {m['batch'].get(k, 0):8.1f}%")
    print(f"\nworkbook -> {args.out}")
    print(f"key      -> {key}")


if __name__ == "__main__":
    main()
