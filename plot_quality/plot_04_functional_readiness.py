# plot_quality/plot_04_functional_readiness.py

import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_quality_table, melt_functional, savefig, set_theme


DATA_PATH = "data/temp_improved_quality_check_20251219_122545.tsv"
SHEET_NAME = 0


def main() -> None:
    set_theme()
    df = load_quality_table(DATA_PATH, sheet_name=SHEET_NAME)
    func_long = melt_functional(df)

    plt.figure(figsize=(14, 6))
    ax = sns.boxplot(data=func_long, x="Functional_metric", y="Score")
    sns.stripplot(data=func_long, x="Functional_metric", y="Score", size=3, alpha=0.4, jitter=0.25, ax=ax)
    ax.set_title("Functional readiness metrics distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    savefig("04_functional_readiness")
    plt.show()


if __name__ == "__main__":
    main()
