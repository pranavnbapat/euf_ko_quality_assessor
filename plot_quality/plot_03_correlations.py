# plot_quality/plot_03_correlations.py

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from utils import COLS, latest_tsv, load_quality_table, pick_numeric, savefig, set_theme, TOTAL_COL


# None -> newest TSV in data/. Set a path to pin a specific run.
DATA_PATH = None
SHEET_NAME = 0


def add_raw_fractions(df: pd.DataFrame) -> list[str]:
    """
    Add `<Pillar>_Total_frac`: each pillar's raw total as a fraction of its own max.

    Semantic_Total_Raw is scored out of 20 or 25 depending on whether the MNLI
    component ran for that KO, so correlating the raw column mixes two scales.
    Dividing by Semantic_Total_Max puts every row back on one scale. The other three
    pillars have a fixed max of 20, so this is a pure linear rescale for them and
    leaves their correlations numerically identical.
    """
    added: list[str] = []
    for pillar in ("Structural", "Semantic", "Domain", "Functional"):
        raw_col = f"{pillar}_Total_Raw"
        if raw_col not in df.columns:
            continue
        max_col = f"{pillar}_Total_Max"
        # Older TSVs predate the *_Total_Max columns; every pillar was out of 20 then.
        denom = pd.to_numeric(df[max_col], errors="coerce") if max_col in df.columns else 20.0
        frac_col = f"{pillar}_Total_frac"
        df[frac_col] = pd.to_numeric(df[raw_col], errors="coerce") / denom
        added.append(frac_col)
    return added


def main() -> None:
    set_theme()
    data_path = DATA_PATH if DATA_PATH is not None else latest_tsv()
    df = load_quality_table(data_path, sheet_name=SHEET_NAME)

    frac_cols = add_raw_fractions(df)

    interesting_cols = list(COLS.pillar_cols) + frac_cols + [
        "Semantic_consistency_mnli",
        "Domain_term_density",
        "Domain_similarity_title",
        "Domain_similarity_desc",
        "Domain_similarity_keywords",
        "Domain_similarity_content",
        "Functional_BM25_readiness",
        "Functional_embedding_readiness",
        "Functional_RAG_readiness",
        "Functional_keyword_indexability",
        "Total_Quality_unweighted_0_100",
        TOTAL_COL,
    ]

    # Keep only columns that actually exist (avoids crashes if your sheet differs)
    interesting_cols = [c for c in interesting_cols if c in df.columns]

    sub = pick_numeric(df, interesting_cols)
    corr = sub.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation heatmap (selected metrics)")
    plt.tight_layout()
    savefig("03_correlations_heatmap")
    plt.show()


if __name__ == "__main__":
    main()
