# assess_ko_quality/compare_quality_tsv.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GRADE_ORDER = {"D": 0, "C": 1, "B": 2, "A": 3}


def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"id": str})
    # Normalise expected columns
    if "score_0_100" not in df.columns:
        raise ValueError(f"{path} missing required column: score_0_100")
    if "grade" not in df.columns:
        raise ValueError(f"{path} missing required column: grade")
    if "id" not in df.columns:
        raise ValueError(f"{path} missing required column: id")

    # Clean up types
    df["score_0_100"] = pd.to_numeric(df["score_0_100"], errors="coerce")
    df["grade"] = df["grade"].astype(str).str.strip()
    df["tokens"] = pd.to_numeric(df.get("tokens"), errors="coerce")
    return df


def grade_counts(df: pd.DataFrame) -> pd.Series:
    # Keep consistent order A/B/C/D
    counts = df["grade"].value_counts(dropna=False)
    for g in ["A", "B", "C", "D"]:
        if g not in counts:
            counts[g] = 0
    return counts[["A", "B", "C", "D"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KO quality TSVs (original vs summarised).")
    parser.add_argument("--orig", required=True, help="Path to original TSV (e.g. ko_content_flat)")
    parser.add_argument("--summ", required=True, help="Path to summarised TSV (e.g. ko_content_flat_summarised)")
    parser.add_argument("--out", default=None, help="Optional path to write a merged TSV with deltas")
    parser.add_argument("--show-top", type=int, default=10, help="Show top N improved and worsened IDs")
    args = parser.parse_args()

    orig_path = Path(args.orig)
    summ_path = Path(args.summ)

    df_o = read_tsv(orig_path).rename(
        columns={
            "score_0_100": "score_orig",
            "grade": "grade_orig",
            "tokens": "tokens_orig",
        }
    )
    df_s = read_tsv(summ_path).rename(
        columns={
            "score_0_100": "score_summ",
            "grade": "grade_summ",
            "tokens": "tokens_summ",
        }
    )

    # Keep only the columns we care about to avoid merge collisions
    keep_o = ["id", "title", "field", "score_orig", "grade_orig", "tokens_orig"]
    keep_s = ["id", "score_summ", "grade_summ", "tokens_summ"]
    df_o = df_o[[c for c in keep_o if c in df_o.columns]]
    df_s = df_s[[c for c in keep_s if c in df_s.columns]]

    merged = df_o.merge(df_s, on="id", how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping IDs between the two TSV files.")

    merged["delta_score"] = merged["score_summ"] - merged["score_orig"]
    merged["delta_tokens"] = merged["tokens_summ"] - merged["tokens_orig"]
    merged["compression_ratio"] = merged["tokens_summ"] / merged["tokens_orig"]

    # Grade delta (bucket transitions)
    merged["grade_orig_rank"] = merged["grade_orig"].map(GRADE_ORDER)
    merged["grade_summ_rank"] = merged["grade_summ"].map(GRADE_ORDER)
    merged["delta_grade_rank"] = merged["grade_summ_rank"] - merged["grade_orig_rank"]

    # ----------------------------
    # Report
    # ----------------------------
    n = len(merged)
    print("\n==============================")
    print("KO QUALITY COMPARISON REPORT")
    print("==============================")
    print(f"Original TSV:   {orig_path}")
    print(f"Summarised TSV: {summ_path}")
    print(f"Matched KOs:    {n}")

    # Basic score stats
    print("\n--- Score stats (0..100) ---")
    print(f"Original:   mean={merged['score_orig'].mean():.3f}  median={merged['score_orig'].median():.3f}")
    print(f"Summarised: mean={merged['score_summ'].mean():.3f}  median={merged['score_summ'].median():.3f}")
    print(f"Delta:      mean={merged['delta_score'].mean():.3f}  median={merged['delta_score'].median():.3f}")

    improved = (merged["delta_score"] > 0).sum()
    worsened = (merged["delta_score"] < 0).sum()
    unchanged = (merged["delta_score"] == 0).sum()

    print("\n--- Per-KO delta counts ---")
    print(f"Improved:  {improved} ({improved/n*100:.1f}%)")
    print(f"Worsened:  {worsened} ({worsened/n*100:.1f}%)")
    print(f"Unchanged: {unchanged} ({unchanged/n*100:.1f}%)")

    # Grade counts + transitions
    print("\n--- Grade counts ---")
    print("Original:\n", grade_counts(merged.rename(columns={"grade_orig": "grade"})))
    print("Summarised:\n", grade_counts(merged.rename(columns={"grade_summ": "grade"})))

    print("\n--- Grade transitions (orig -> summ) ---")
    transitions = pd.crosstab(merged["grade_orig"], merged["grade_summ"], dropna=False)
    # Ensure consistent row/col order
    transitions = transitions.reindex(index=["A", "B", "C", "D"], columns=["A", "B", "C", "D"], fill_value=0)
    print(transitions)

    # Grade movement summary
    up = (merged["delta_grade_rank"] > 0).sum()
    down = (merged["delta_grade_rank"] < 0).sum()
    same = (merged["delta_grade_rank"] == 0).sum()
    print("\n--- Grade movement ---")
    print(f"Upgraded:   {up} ({up/n*100:.1f}%)")
    print(f"Downgraded: {down} ({down/n*100:.1f}%)")
    print(f"Same grade: {same} ({same/n*100:.1f}%)")

    # Top movers
    topn = max(0, int(args.show_top))
    if topn:
        print(f"\n--- Top {topn} improvements (by delta_score) ---")
        cols = ["id", "delta_score", "score_orig", "score_summ", "grade_orig", "grade_summ", "compression_ratio", "title"]
        print(merged.sort_values("delta_score", ascending=False)[cols].head(topn).to_string(index=False))

        print(f"\n--- Top {topn} regressions (by delta_score) ---")
        print(merged.sort_values("delta_score", ascending=True)[cols].head(topn).to_string(index=False))

    # Optional output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
        print(f"\nWrote merged TSV with deltas: {out_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
