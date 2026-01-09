# plot_quality/plot_06_before_after_effect_sizes.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import COLS, add_file_footer, iqr, load_before_after, pct_below, savefig, set_theme


BEFORE_PATH = None
AFTER_PATH = None
DATA_DIR = "data"
THRESHOLD = 10  # below 10/25 is "failure zone" for pillar scores


def main() -> None:
    set_theme()
    before_df, after_df, before_label, after_label = load_before_after(BEFORE_PATH, AFTER_PATH, data_dir=DATA_DIR)

    rows = []
    for col in COLS.pillar_cols:
        if col not in before_df.columns or col not in after_df.columns:
            continue

        b = before_df[col]
        a = after_df[col]

        rows.append({
            "Pillar": col.replace("_Score_0_25", ""),
            "Before_median": float(pd.to_numeric(b, errors="coerce").median()),
            "After_median": float(pd.to_numeric(a, errors="coerce").median()),
            "Δ_median": float(pd.to_numeric(a, errors="coerce").median() - pd.to_numeric(b, errors="coerce").median()),
            "Before_IQR": iqr(b),
            "After_IQR": iqr(a),
            "Δ_IQR": iqr(a) - iqr(b),
            f"Before_%<{THRESHOLD}": pct_below(b, THRESHOLD),
            f"After_%<{THRESHOLD}": pct_below(a, THRESHOLD),
            f"Δ_%<{THRESHOLD}": pct_below(a, THRESHOLD) - pct_below(b, THRESHOLD),
        })

    effects = pd.DataFrame(rows)
    if effects.empty:
        print("No pillar columns found to compare.")
        return

    # Plot: median changes + tail changes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=effects, x="Pillar", y="Δ_median", ax=axes[0])
    axes[0].axhline(0, linewidth=1)
    axes[0].set_title("Median change (After − Before)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Δ median (0–25)")

    delta_tail_col = f"Δ_%<{THRESHOLD}"
    sns.barplot(data=effects, x="Pillar", y=delta_tail_col, ax=axes[1])
    axes[1].axhline(0, linewidth=1)
    axes[1].set_title(f"Tail change: % below {THRESHOLD}/25 (After − Before)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Δ percentage points")

    fig.suptitle("Before/After effect sizes (executive summary)", y=0.98)
    add_file_footer(fig, before_label, after_label)
    plt.tight_layout()
    savefig("06_before_after_effect_sizes")
    plt.show()

    # Also save the table as CSV for slides/reports
    effects.to_csv("out/06_effect_sizes_table.csv", index=False)


if __name__ == "__main__":
    main()
