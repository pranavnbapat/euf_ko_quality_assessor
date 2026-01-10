# plot_quality/plot_04_functional_readiness.py

import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_quality_table, melt_functional, savefig, set_theme


DATA_PATH = "../assess_ko_quality/output/quality_check_20260109_201910.tsv"
SHEET_NAME = 0

order = [
    "Functional_BM25_readiness",
    "Functional_embedding_readiness",
    "Functional_RAG_readiness",
    "Functional_keyword_indexability",
]


def main() -> None:
    set_theme()
    df = load_quality_table(DATA_PATH, sheet_name=SHEET_NAME)
    func_long = melt_functional(df)

    plt.figure(figsize=(14, 6))
    ax = sns.boxplot(data=func_long, x="Functional_metric", y="Score", order=order)
    sns.stripplot(data=func_long, x="Functional_metric", y="Score", size=3, alpha=0.4, jitter=0.25, order=order, ax=ax)
    ax.set_title("Functional readiness metrics distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    savefig("04_functional_readiness")
    plt.show()


if __name__ == "__main__":
    main()
