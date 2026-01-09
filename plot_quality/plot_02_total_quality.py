# plot_quality/plot_02_total_quality.py

import matplotlib.pyplot as plt
import seaborn as sns

from utils import COLS, load_quality_table, savefig, set_theme


DATA_PATH = "data/temp_improved_quality_check_20251219_122545.tsv"
SHEET_NAME = 0


def main() -> None:
    set_theme()
    df = load_quality_table(DATA_PATH, sheet_name=SHEET_NAME)

    # 1) Total quality distribution
    plt.figure(figsize=(10, 5))
    ax = sns.histplot(df, x="Total_Quality_0_100", bins=25, kde=True)
    ax.set_title("Total Quality distribution (0–100)")
    ax.set_xlabel("Total Quality (0–100)")
    plt.tight_layout()
    out = savefig("02_total_quality_hist")
    plt.show()

    # 2) Pillars vs total (quick relationship scan)
    for col in COLS.pillar_cols:
        if col not in df.columns:
            continue
        g = sns.lmplot(
            data=df,
            x=col,
            y="Total_Quality_0_100",
            height=5,
            aspect=1.2,
            scatter_kws={"alpha": 0.4, "s": 35},
            line_kws={"linewidth": 2},
        )
        g.set_axis_labels(col.replace("_Score_0_25", "") + " (0–25)", "Total Quality (0–100)")
        plt.title(f"{col.replace('_Score_0_25','')} vs Total Quality")
        plt.tight_layout()
        savefig(f"02_total_vs_{col.replace('_Score_0_25','').lower()}")
        plt.show()


if __name__ == "__main__":
    main()
