# assess_ko_quality/ko_content_quality_analysis.py

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple


def ensure_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError("Input JSON must be a list or an object")


def ko_label(obj: Dict[str, Any]) -> str:
    title = (obj.get("title") or "").strip()
    oid = obj.get("_orig_id") or ""
    kid = obj.get("@id") or ""
    label = title if title else "[no-title]"
    if len(label) > 80:
        label = label[:77] + "…"
    suffix = oid or kid
    return f"{label}  [{suffix}]" if suffix else label


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Analyse KO summary quality JSON")
    parser.add_argument("--input", required=True, help="Path to JSON with quality metrics")
    parser.add_argument("--text-field", default="ko_content_flat", help="Base field name")
    parser.add_argument("--topk", type=int, default=10, help="How many to show in Top lists")
    args = parser.parse_args()

    qkey = f"{args.text_field}_summary_quality"

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = ensure_list(data)

    rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    skipped = 0

    for obj in items:
        q = obj.get(qkey)
        if not isinstance(q, dict):
            skipped += 1
            continue
        s = q.get("summary_quality_score_0_100")
        if not isinstance(s, (int, float)):
            skipped += 1
            continue
        rows.append((obj, q))

    total = len(rows)

    print("\n==============================")
    print("KO SUMMARY QUALITY — SUMMARY")
    print("==============================")
    print(f"Input file: {args.input}")
    print(f"KOs with usable quality metrics: {total}")
    print(f"KOs skipped (missing/invalid): {skipped}")

    # Grade counts
    grade_counts: Dict[str, int] = {}
    scores: List[float] = []
    flags_count: Dict[str, int] = {}

    for _, q in rows:
        g = q.get("summary_quality_grade") or "?"
        grade_counts[g] = grade_counts.get(g, 0) + 1
        scores.append(float(q["summary_quality_score_0_100"]))
        for fl in q.get("summary_quality_flags", []) or []:
            flags_count[fl] = flags_count.get(fl, 0) + 1

    print("\n--- Grade counts ---")
    for g in ["A", "B", "C", "D", "?"]:
        c = grade_counts.get(g, 0)
        pct = (c / total * 100.0) if total else 0.0
        print(f"{g:>2}: {c:>6}  ({pct:5.1f}%)")

    print("\n--- Score stats ---")
    if scores:
        scores_sorted = sorted(scores)
        p50 = scores_sorted[int(0.50 * (len(scores_sorted) - 1))]
        p10 = scores_sorted[int(0.10 * (len(scores_sorted) - 1))]
        p90 = scores_sorted[int(0.90 * (len(scores_sorted) - 1))]
        print(f"mean={mean(scores):.1f}  p10={p10:.1f}  p50={p50:.1f}  p90={p90:.1f}")
    else:
        print("No scores.")

    print("\n--- Most common flags ---")
    top_flags = sorted(flags_count.items(), key=lambda x: x[1], reverse=True)[:15]
    if not top_flags:
        print("(none)")
    else:
        for fl, c in top_flags:
            print(f"{fl:>28}: {c}")

    # Worst summaries
    topk = args.topk
    rows_by_score_asc = sorted(rows, key=lambda x: float(x[1]["summary_quality_score_0_100"]))
    print(f"\n--- Bottom {topk} summaries (lowest quality score) ---")
    for i, (obj, q) in enumerate(rows_by_score_asc[:topk], start=1):
        sc = q.get("summary_quality_score_0_100")
        cr = q.get("compression_ratio")
        er = q.get("entity_recall")
        sim = q.get("semantic_similarity")
        fl = ",".join((q.get("summary_quality_flags") or [])[:5])
        print(f"{i:>2}. score={sc:>5.1f}  ratio={cr!s:>6}  entR={er!s:>6}  sim={sim!s:>6}  {ko_label(obj)}  flags={fl}")

    # Best summaries
    rows_by_score_desc = sorted(rows, key=lambda x: float(x[1]["summary_quality_score_0_100"]), reverse=True)
    print(f"\n--- Top {topk} summaries (highest quality score) ---")
    for i, (obj, q) in enumerate(rows_by_score_desc[:topk], start=1):
        sc = q.get("summary_quality_score_0_100")
        cr = q.get("compression_ratio")
        er = q.get("entity_recall")
        sim = q.get("semantic_similarity")
        print(f"{i:>2}. score={sc:>5.1f}  ratio={cr!s:>6}  entR={er!s:>6}  sim={sim!s:>6}  {ko_label(obj)}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
