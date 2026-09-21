# assess_ko_quality/ko_content_quality_analysis.py

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple


def ensure_list(data: Any) -> List[Dict[str, Any]]:
    """
    Extract list of KO objects from JSON data.
    Handles:
    - Old format: direct list of KOs
    - New format: {meta, stats, docs: [...]} where docs contains KOs
    - Single object: wrap in list
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        # New format: check for 'docs' field containing the KOs
        if "docs" in data and isinstance(data["docs"], list):
            return [x for x in data["docs"] if isinstance(x, dict)]
        # Single object (legacy single KO format)
        return [data]
    raise ValueError("Input JSON must be a list, an object, or an object with 'docs' field")


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
    parser = argparse.ArgumentParser(description="Analyse KO field content quality JSON")
    parser.add_argument("--input", required=True, help="Path to JSON with quality metrics")
    parser.add_argument("--text-field", default="ko_content_flat", help="Base field name")
    parser.add_argument("--topk", type=int, default=10, help="How many to show in Top lists")
    args = parser.parse_args()

    qkey = f"{args.text_field}_metrics"

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = ensure_list(data)

    rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    skipped = 0

    # Show sample keys from first KO to help debug
    if items:
        sample_keys = list(items[0].keys())
        metrics_keys = [k for k in sample_keys if k.endswith("_metrics")]
        print(f"\n[DEBUG] First KO has keys: {sample_keys[:10]}...")
        if metrics_keys:
            print(f"[DEBUG] Found metrics keys: {metrics_keys}")
        else:
            print(f"[DEBUG] Looking for metrics key: '{qkey}'")
    elif isinstance(data, dict) and "docs" in data:
        print(f"\n[DEBUG] Input has 'docs' field with {len(data.get('docs', []))} items, but no valid KO objects found.")

    for obj in items:
        q = obj.get(qkey)
        if not isinstance(q, dict):
            skipped += 1
            continue
        s = q.get("field_quality_score_0_100")
        if not isinstance(s, (int, float)):
            skipped += 1
            continue
        rows.append((obj, q))

    total = len(rows)

    if total == 0 and items:
        print(f"\n[WARNING] No KOs with valid '{qkey}' metrics found.")
        print("[HINT] Make sure you've run ko_content_quality_diagnostics.py on this file first.")
        print(f"[HINT] Expected field key pattern: '{{field}}_metrics' (e.g., 'ko_content_flat_metrics')")

    print("\n==============================")
    print("KO CONTENT QUALITY — SUMMARY")
    print("==============================")
    print(f"Input file: {args.input}")
    print(f"Input format: {'wrapped (meta/stats/docs)' if isinstance(data, dict) and 'docs' in data else 'direct list'}")
    print(f"Field analyzed: {args.text_field}")
    print(f"KOs with usable quality metrics: {total}")
    print(f"KOs skipped (missing/invalid): {skipped}")

    # Grade counts
    grade_counts: Dict[str, int] = {}
    scores: List[float] = []
    flags_count: Dict[str, int] = {}

    for _, q in rows:
        g = q.get("field_quality_grade") or "?"
        grade_counts[g] = grade_counts.get(g, 0) + 1
        scores.append(float(q["field_quality_score_0_100"]))
        for fl in q.get("field_quality_flags", []) or []:
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

    # Token stats
    print("\n--- Token stats ---")
    tokens = [q.get("tokens") for _, q in rows if isinstance(q.get("tokens"), (int, float))]
    if tokens:
        print(f"mean={mean(tokens):.0f}  min={min(tokens)}  max={max(tokens)}")
    else:
        print("No token counts.")

    # Worst content
    topk = args.topk
    rows_by_score_asc = sorted(rows, key=lambda x: float(x[1]["field_quality_score_0_100"]))
    print(f"\n--- Bottom {topk} (lowest quality score) ---")
    for i, (obj, q) in enumerate(rows_by_score_asc[:topk], start=1):
        sc = q.get("field_quality_score_0_100")
        tk = q.get("tokens")
        ent = q.get("entity_mentions")
        ttr = q.get("ttr")
        fl = ",".join((q.get("field_quality_flags") or [])[:5])
        print(f"{i:>2}. score={sc:>5.1f}  tokens={tk!s:>6}  ents={ent!s:>6}  ttr={ttr!s:>6}  {ko_label(obj)}  flags={fl}")

    # Best content
    rows_by_score_desc = sorted(rows, key=lambda x: float(x[1]["field_quality_score_0_100"]), reverse=True)
    print(f"\n--- Top {topk} (highest quality score) ---")
    for i, (obj, q) in enumerate(rows_by_score_desc[:topk], start=1):
        sc = q.get("field_quality_score_0_100")
        tk = q.get("tokens")
        ent = q.get("entity_mentions")
        ttr = q.get("ttr")
        fl = ",".join((q.get("field_quality_flags") or [])[:5])
        print(f"{i:>2}. score={sc:>5.1f}  tokens={tk!s:>6}  ents={ent!s:>6}  ttr={ttr!s:>6}  {ko_label(obj)}  flags={fl}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
