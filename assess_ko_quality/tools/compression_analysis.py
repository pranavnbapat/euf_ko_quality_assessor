# assess_ko_quality/ko_compression_analysis.py

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


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def ko_label(obj: Dict[str, Any]) -> str:
    """Human-friendly identifier for printing."""
    title = obj.get("title") or ""
    oid = obj.get("_orig_id") or ""
    kid = obj.get("@id") or ""
    # Keep it short and stable
    label = title.strip()
    if len(label) > 80:
        label = label[:77] + "…"
    suffix = oid or kid
    if suffix:
        return f"{label}  [{suffix}]"
    return label or "[no-title]"


def token_bucket(tok: int) -> str:
    """Token bucket labels."""
    if tok < 300:
        return "<300"
    if tok < 1000:
        return "300–999"
    if tok < 3000:
        return "1k–2.9k"
    if tok < 8000:
        return "3k–7.9k"
    return ">=8k"


def mean(nums: List[float]) -> float:
    return sum(nums) / len(nums) if nums else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Analyse KO compression diagnostics JSON")
    parser.add_argument("--input", required=True, help="Path to diagnostics JSON")
    parser.add_argument("--text-field", default="ko_content_flat", help="Text field used in diagnostics")
    parser.add_argument("--topk", type=int, default=10, help="How many to show in the Top lists")
    args = parser.parse_args()

    metrics_key = f"{args.text_field}_metrics"

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = ensure_list(data)

    rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    skipped = 0

    for obj in items:
        m = obj.get(metrics_key)
        if not isinstance(m, dict):
            skipped += 1
            continue
        # Require token_count at minimum
        tok = m.get("token_count")
        if not isinstance(tok, int):
            skipped += 1
            continue
        rows.append((obj, m))

    total = len(rows)

    print("\n==============================")
    print("KO COMPRESSION DIAGNOSTICS — SUMMARY")
    print("==============================")
    print(f"Input file: {args.input}")
    print(f"KOs with usable metrics: {total}")
    print(f"KOs skipped (missing/invalid metrics): {skipped}")

    # ----------------------------
    # Counts per token bucket
    # ----------------------------
    bucket_counts: Dict[str, int] = {}
    for _, m in rows:
        b = token_bucket(int(m["token_count"]))
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    print("\n--- Counts per token bucket ---")
    for b in ["<300", "300–999", "1k–2.9k", "3k–7.9k", ">=8k"]:
        c = bucket_counts.get(b, 0)
        pct = (c / total * 100.0) if total else 0.0
        print(f"{b:>8}: {c:>6}  ({pct:5.1f}%)")

    # ----------------------------
    # Counts per model class
    # ----------------------------
    model_counts: Dict[str, int] = {}
    for _, m in rows:
        mc = m.get("suggested_model_class") or "unknown"
        model_counts[mc] = model_counts.get(mc, 0) + 1

    print("\n--- Counts per suggested model class ---")
    for k in sorted(model_counts.keys()):
        c = model_counts[k]
        pct = (c / total * 100.0) if total else 0.0
        print(f"{k:>8}: {c:>6}  ({pct:5.1f}%)")

    # ----------------------------
    # Mean CDS per bucket
    # ----------------------------
    cds_by_bucket: Dict[str, List[float]] = {}
    for _, m in rows:
        b = token_bucket(int(m["token_count"]))
        cds = m.get("compression_difficulty_score_0_1")
        if isinstance(cds, (int, float)):
            cds_by_bucket.setdefault(b, []).append(float(cds))

    print("\n--- Mean CDS per token bucket ---")
    for b in ["<300", "300–999", "1k–2.9k", "3k–7.9k", ">=8k"]:
        vals = cds_by_bucket.get(b, [])
        if vals:
            print(f"{b:>8}: mean={mean(vals):.3f}  n={len(vals)}")
        else:
            print(f"{b:>8}: mean=NA     n=0")

    # ----------------------------
    # Top 10 longest KOs
    # ----------------------------
    rows_by_len = sorted(rows, key=lambda x: int(x[1]["token_count"]), reverse=True)
    topk = args.topk

    print(f"\n--- Top {topk} longest KOs (by token_count) ---")
    for i, (obj, m) in enumerate(rows_by_len[:topk], start=1):
        tok = int(m["token_count"])
        cds = m.get("compression_difficulty_score_0_1")
        mc = m.get("suggested_model_class")
        band = m.get("difficulty_band")
        print(f"{i:>2}. tok={tok:>6}  cds={cds!s:>6}  band={band!s:>6}  model={mc!s:>6}  {ko_label(obj)}")

    # ----------------------------
    # Top 10 highest CDS KOs
    # ----------------------------
    rows_by_cds = sorted(
        rows,
        key=lambda x: float(x[1].get("compression_difficulty_score_0_1", -1.0))
        if isinstance(x[1].get("compression_difficulty_score_0_1"), (int, float))
        else -1.0,
        reverse=True,
    )

    print(f"\n--- Top {topk} highest CDS KOs (by compression_difficulty_score_0_1) ---")
    for i, (obj, m) in enumerate(rows_by_cds[:topk], start=1):
        tok = int(m["token_count"])
        cds = m.get("compression_difficulty_score_0_1")
        mc = m.get("suggested_model_class")
        band = m.get("difficulty_band")
        print(f"{i:>2}. cds={cds!s:>6}  tok={tok:>6}  band={band!s:>6}  model={mc!s:>6}  {ko_label(obj)}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
