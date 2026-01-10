# plot_quality/plot_02_total_quality.py

import matplotlib.pyplot as plt
import seaborn as sns

from utils import COLS, load_quality_table, savefig, set_theme, WEIGHTS, TOTAL_COL


DATA_PATH = "../assess_ko_quality/output/quality_check_20260109_202033.tsv"
SHEET_NAME = 0


def main() -> None:
    set_theme()
    df = load_quality_table(DATA_PATH, sheet_name=SHEET_NAME)

    # Convert raw pillar scores (0-25) into contribution in weighted-total space.
    # This makes plots consistent with the calibrated scheme: 30/35/25/10.
    for col, w in WEIGHTS.items():
        if col in df.columns:
            df[col.replace("_Score_0_25", "_Contrib")] = (df[col] / 25.0) * w

    # 1) Total quality distribution
    plt.figure(figsize=(10, 5))

    if TOTAL_COL not in df.columns:
        raise KeyError(f"Expected column '{TOTAL_COL}' not found in input table.")
    ax = sns.histplot(df, x=TOTAL_COL, bins=25, kde=True)

    ax.set_title("Total Quality distribution (0–100)")
    ax.set_xlabel("Total Quality (0–100)")
    plt.tight_layout()
    out = savefig("02_total_quality_hist")
    plt.show()

    # 2) Pillar contributions vs total (relationship scan, consistent with weights)
    for raw_col in COLS.pillar_cols:
        if raw_col not in df.columns:
            continue

        contrib_col = raw_col.replace("_Score_0_25", "_Contrib")
        if contrib_col not in df.columns:
            # Shouldn't happen because we compute contribs above, but safe guard anyway
            continue

        # Friendly pillar name and range for labels
        pillar_name = raw_col.replace("_Score_0_25", "")
        pillar_max = WEIGHTS.get(raw_col, 25)

        g = sns.lmplot(
            data=df,
            x=contrib_col,
            y=TOTAL_COL,
            height=5,
            aspect=1.2,
            scatter_kws={"alpha": 0.4, "s": 35},
            line_kws={"linewidth": 2},
        )

        g.set_axis_labels(f"{pillar_name} contribution (0–{pillar_max})", "Total Quality (0–100)")

        # lmplot owns its own figure; use suptitle instead of plt.title
        g.fig.suptitle(f"{pillar_name} vs Total Quality", y=1.02)

        g.fig.tight_layout()
        savefig(f"02_total_vs_{pillar_name.lower()}")
        plt.show()

if __name__ == "__main__":
    main()
