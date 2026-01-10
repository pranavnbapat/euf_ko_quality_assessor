# plot_quality/plot_03_correlations.py

import matplotlib.pyplot as plt
import seaborn as sns

from utils import COLS, load_quality_table, pick_numeric, savefig, set_theme, TOTAL_COL


DATA_PATH = "../assess_ko_quality/output/quality_check_20260109_202033.tsv"
SHEET_NAME = 0


def main() -> None:
    set_theme()
    df = load_quality_table(DATA_PATH, sheet_name=SHEET_NAME)

    interesting_cols = list(COLS.pillar_cols) + [
        "Structural_Total_Raw",
        "Semantic_Total_Raw",
        "Domain_Total_Raw",
        "Functional_Total_Raw",
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
