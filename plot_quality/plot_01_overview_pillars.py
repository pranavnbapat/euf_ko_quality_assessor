# plot_quality/plot_01_overview_pillars.py

import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_quality_table, melt_pillars, savefig, set_theme


DATA_PATH = "data/temp_improved_quality_check_20251219_122545.tsv"
SHEET_NAME = 0


def main() -> None:
    set_theme()
    df = load_quality_table(DATA_PATH, sheet_name=SHEET_NAME)
    pillars_long = melt_pillars(df)

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=pillars_long, x="Pillar", y="Score_0_25")
    sns.stripplot(
        data=pillars_long,
        x="Pillar",
        y="Score_0_25",
        size=4,
        alpha=0.5,
        jitter=0.25,
        ax=ax,
    )
    ax.set_title("Pillar score distributions (0–25)")
    ax.set_xlabel("")
    ax.set_ylabel("Score (0–25)")
    plt.tight_layout()
    savefig("01_pillars_distribution")
    plt.show()


if __name__ == "__main__":
    main()
