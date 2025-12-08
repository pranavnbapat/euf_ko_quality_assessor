# assess_ko_quality/stats_ko_quality.py

"""
Compute descriptive statistics for the KO quality TSV file.

Usage:
    python stats_ko_quality.py path/to/file.tsv
"""

from pathlib import Path
import sys
import pandas as pd


def load_tsv(path: Path) -> pd.DataFrame:
    """Load the TSV file into a pandas DataFrame."""
    df = pd.read_csv(path, sep="\t")
    return df


def print_basic_info(df: pd.DataFrame) -> None:
    """Print basic dataset information."""
    print("\n=== BASIC INFO ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumn dtypes:")
    print(df.dtypes)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for all numeric columns.
    Returns a DataFrame with one row per numeric column.
    """
    num_df = df.select_dtypes(include="number")
    desc = num_df.describe().T  # transpose so each row is a column
    # Add missing-value counts
    desc["missing"] = num_df.isna().sum()
    # Add % missing for quick inspection
    desc["missing_pct"] = desc["missing"] / len(df) * 100
    return desc


def categorical_summary(df: pd.DataFrame, max_unique: int = 20) -> None:
    """
    Print value counts for non-numeric columns with limited cardinality.
    Only columns with <= max_unique distinct values are expanded.
    """
    cat_df = df.select_dtypes(exclude="number")

    print("\n=== CATEGORICAL COLUMNS (limited to low-cardinality) ===")
    for col in cat_df.columns:
        nunique = cat_df[col].nunique(dropna=True)
        print(f"\nColumn: {col} (unique values: {nunique})")
        if nunique <= max_unique:
            print(cat_df[col].value_counts(dropna=False))
        else:
            print("  (Too many distinct values; skipping detailed counts.)")


def quality_buckets(df: pd.DataFrame, col: str = "Total_Quality_0_100") -> pd.DataFrame:
    """
    Compute counts and percentages for quality bands on the given column.
    Bands: <50, 50-69, 70-84, >=85
    Returns a DataFrame with counts and percentages.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    s = df[col]

    # Define bands using pandas.cut
    bins = [-float("inf"), 50, 70, 85, float("inf")]
    labels = ["<50", "50-69", "70-84", ">=85"]

    # Categorise each value into a band
    bucketed = pd.cut(s, bins=bins, labels=labels, right=False)

    # Count how many rows fall into each band
    counts = bucketed.value_counts().reindex(labels, fill_value=0)
    result = pd.DataFrame({"count": counts})
    result["pct"] = (result["count"] / len(df) * 100).round(2)

    return result


def score_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Compute correlations between Total_Quality_0_100 and all *_Score_0_25 columns.
    Returns a Series sorted by correlation strength.
    """
    # Find all sub-score columns
    score_cols = [c for c in df.columns if c.endswith("Score_0_25")]

    # Ensure Total_Quality_0_100 is included if present
    if "Total_Quality_0_100" in df.columns:
        score_cols.append("Total_Quality_0_100")

    # Keep only numeric columns from this list (defensive)
    numeric_score_cols = [c for c in score_cols if pd.api.types.is_numeric_dtype(df[c])]
    score_df = df[numeric_score_cols]

    # Compute correlation matrix
    corr = score_df.corr()

    if "Total_Quality_0_100" not in corr.columns:
        raise KeyError("Total_Quality_0_100 not in correlation matrix (is it numeric?)")

    # Correlation of each score with Total_Quality_0_100
    corr_with_total = corr["Total_Quality_0_100"].sort_values(ascending=False)
    return corr_with_total


def binary_flag_stats(df: pd.DataFrame, flag_cols: list[str]) -> pd.DataFrame:
    """
    For each binary flag column (0/1), compute:
    - percentage of rows where flag == 1
    - mean Total_Quality_0_100 when flag == 1 and flag == 0
    """
    results = []
    for col in flag_cols:
        if col not in df.columns:
            continue

        series = df[col]
        # Only proceed if column is numeric-like (0/1 typically)
        if not pd.api.types.is_numeric_dtype(series):
            continue

        total = len(df)
        ones = (series == 1).sum()
        pct_ones = ones / total * 100

        if "Total_Quality_0_100" in df.columns:
            mean_when_1 = df.loc[series == 1, "Total_Quality_0_100"].mean()
            mean_when_0 = df.loc[series == 0, "Total_Quality_0_100"].mean()
        else:
            mean_when_1 = float("nan")
            mean_when_0 = float("nan")

        results.append(
            {
                "flag": col,
                "pct_1": pct_ones,
                "mean_total_when_1": mean_when_1,
                "mean_total_when_0": mean_when_0,
            }
        )

    return pd.DataFrame(results)


def text_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple text-length stats for some key text columns.
    Currently: title, Notes.
    """
    text_cols = [c for c in ["title", "Notes"] if c in df.columns]
    rows = []

    for col in text_cols:
        # Compute character length for non-null values
        lengths = df[col].dropna().astype(str).str.len()
        rows.append(
            {
                "column": col,
                "mean_len": lengths.mean(),
                "median_len": lengths.median(),
                "min_len": lengths.min(),
                "max_len": lengths.max(),
            }
        )

    return pd.DataFrame(rows)


def main(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = load_tsv(path)

    # 1. Basic info
    print_basic_info(df)

    # 2. Numeric summary
    num_stats = numeric_summary(df)
    print("\n=== NUMERIC SUMMARY (per column) ===")
    print(num_stats)
    # Save for further inspection in e.g. Excel
    num_stats.to_csv(path.with_suffix(".numeric_summary.csv"))

    # 3. Categorical summary (only low-cardinality columns)
    categorical_summary(df, max_unique=20)

    # 4. Quality buckets
    if "Total_Quality_0_100" in df.columns:
        buckets = quality_buckets(df, col="Total_Quality_0_100")
        print("\n=== TOTAL_QUALITY_0_100 BUCKETS ===")
        print(buckets)
        buckets.to_csv(path.with_suffix(".quality_buckets.csv"))
    else:
        print("\nWARNING: Column 'Total_Quality_0_100' not found; skipping buckets.")

    # 5. Correlations of sub-scores with Total_Quality_0_100
    try:
        corr_with_total = score_correlations(df)
        print("\n=== CORRELATION WITH Total_Quality_0_100 ===")
        print(corr_with_total)
        corr_with_total.to_csv(path.with_suffix(".score_correlations_with_total.csv"))
    except KeyError as e:
        print(f"\nSkipping correlation analysis: {e}")

    # 6. Binary flag stats (adapt this list as needed)
    flag_candidates = [
        "Domain_in_title",
        "Domain_in_keywords",
        "Domain_consistency",
        "Functional_BM25_readiness",
        "Functional_embedding_readiness",
        "Functional_RAG_readiness",
        "Functional_keyword_indexability",
    ]
    flag_stats = binary_flag_stats(df, flag_candidates)
    if not flag_stats.empty:
        print("\n=== BINARY FLAG STATS ===")
        print(flag_stats)
        flag_stats.to_csv(path.with_suffix(".binary_flag_stats.csv"), index=False)

    # 7. Text length stats
    text_stats = text_length_stats(df)
    if not text_stats.empty:
        print("\n=== TEXT LENGTH STATS ===")
        print(text_stats)
        text_stats.to_csv(path.with_suffix(".text_length_stats.csv"), index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stats_ko_quality.py path/to/file.tsv", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
