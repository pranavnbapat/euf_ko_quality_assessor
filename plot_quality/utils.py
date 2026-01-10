# plot_quality/utils.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Calibrated weights (sum to 100)
WEIGHTS = {
    "Structural_Score_0_25": 30,
    "Semantic_Score_0_25": 35,
    "Functional_Score_0_25": 25,
    "Domain_Score_0_25": 10,
}

TOTAL_COL = "Total_Quality_weighted_0_100"

# ---- Configuration container ----
@dataclass(frozen=True)
class Cols:
    id_cols: tuple[str, ...] = ("_orig_id", "title", "lang_meta_detected")
    pillar_cols: tuple[str, ...] = (
        "Structural_Score_0_25",
        "Semantic_Score_0_25",
        "Domain_Score_0_25",
        "Functional_Score_0_25",
    )
    functional_cols: tuple[str, ...] = (
        "Functional_BM25_readiness",
        "Functional_embedding_readiness",
        "Functional_RAG_readiness",
        "Functional_keyword_indexability",
    )


COLS = Cols()


def set_theme() -> None:
    """Set a consistent Seaborn/Matplotlib theme for all plots."""
    sns.set_theme(style="whitegrid", context="talk")


def load_quality_table(path: str | Path, sheet_name: int | str = 0, sep: str = "\t") -> pd.DataFrame:
    """
    Load KO quality data from:
      - .xlsx (openpyxl)
      - .xls  (xlrd)
      - .tsv  (tab-separated)
      - .csv  (optional convenience)

    Also:
      - strips whitespace from column names
      - coerces key numeric columns
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".xlsx":
        df = pd.read_excel(p, sheet_name=sheet_name, engine="openpyxl")
    elif suffix == ".xls":
        # Requires: pip install xlrd
        df = pd.read_excel(p, sheet_name=sheet_name, engine="xlrd")
    elif suffix == ".tsv":
        df = pd.read_csv(p, sep=sep)
    elif suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .xls/.xlsx/.tsv (or .csv).")

    df.columns = df.columns.str.strip()

    # Coerce key numeric columns (support both old and new total columns)
    numeric_cols = (
            list(COLS.pillar_cols)
            + list(COLS.functional_cols)
            + ["Total_Quality_unweighted_0_100", "Total_Quality_weighted_0_100"]
    )
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def ensure_out_dir(out_dir: str | Path) -> Path:
    """Create output directory if missing."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(name: str, out_dir: str | Path = "out", dpi: int = 200) -> Path:
    """
    Save current figure to out_dir/name.png and return the path.
    Call after plt.tight_layout().
    """
    out_path = ensure_out_dir(out_dir) / f"{name}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def melt_pillars(df: pd.DataFrame) -> pd.DataFrame:
    """Wide -> long for pillar scores."""
    id_vars = [c for c in COLS.id_cols if c in df.columns]
    value_vars = [c for c in COLS.pillar_cols if c in df.columns]

    long_df = df[id_vars + value_vars].melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="Pillar",
        value_name="Score_0_25",
    )
    long_df["Pillar"] = long_df["Pillar"].str.replace("_Score_0_25", "", regex=False)
    return long_df


def melt_functional(df: pd.DataFrame) -> pd.DataFrame:
    """Wide -> long for functional readiness metrics."""
    id_vars = [c for c in COLS.id_cols if c in df.columns]
    value_vars = [c for c in COLS.functional_cols if c in df.columns]

    long_df = df[id_vars + value_vars].melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="Functional_metric",
        value_name="Score",
    )
    return long_df



def pick_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Return numeric-only subset for correlation etc."""
    sub = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    return sub


def discover_two_tsvs(data_dir: str | Path = "data") -> Tuple[Path, Path]:
    """
    Find exactly two TSV files in data_dir.
    Sorts by filename to produce deterministic (before, after) ordering.
    If you prefer a different ordering, pass explicit paths in scripts.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.tsv"))
    if len(files) != 2:
        raise ValueError(f"Expected exactly 2 TSV files in {data_path.resolve()}, found {len(files)}: {[f.name for f in files]}")
    return files[0], files[1]


def load_before_after(
    before_path: str | Path | None = None,
    after_path: str | Path | None = None,
    data_dir: str | Path = "data",
    sheet_name: int | str = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load two datasets (before/after). If paths are not provided, auto-discovers 2 TSVs in data_dir.
    Returns: before_df, after_df, before_label, after_label
    """
    if before_path is None or after_path is None:
        b, a = discover_two_tsvs(data_dir=data_dir)
    else:
        b, a = Path(before_path), Path(after_path)

    before_df = load_quality_table(b, sheet_name=sheet_name)
    after_df = load_quality_table(a, sheet_name=sheet_name)

    return before_df, after_df, b.name, a.name


def iqr(series: pd.Series) -> float:
    """Interquartile range (75th - 25th)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(np.nanpercentile(s, 75) - np.nanpercentile(s, 25))


def pct_below(series: pd.Series, threshold: float) -> float:
    """Percentage of non-null values strictly below threshold."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float((s < threshold).mean() * 100.0)


def add_file_footer(fig, before_label: str, after_label: str) -> None:
    """Adds a small footer line with filenames to the figure."""
    fig.text(0.01, 0.01, f"Before: {before_label}   |   After: {after_label}", fontsize=10, alpha=0.8)


def parse_keyword_list(val: object) -> list[str]:
    """
    Parses a keyword cell into a list.
    Handles semicolon separated (your example), comma separated, or already-list-like values.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return []
    # Prefer semicolon if present, else commas
    if ";" in s:
        parts = [p.strip() for p in s.split(";")]
    elif "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [s]
    return [p for p in parts if p]
