# plot_quality/plot_05_language_splits.py

import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_quality_table, melt_pillars, savefig, set_theme


DATA_PATH = "data/temp_improved_quality_check_20251219_122545.tsv"
SHEET_NAME = 0


def main() -> None:
    set_theme()
    df = load_quality_table(DATA_PATH, sheet_name=SHEET_NAME)

    # If language column missing, fail gracefully
    if "lang_meta_detected" not in df.columns:
        print("Column 'lang_meta_detected' not found; skipping language plots.")
        return

    pillars_long = melt_pillars(df)

    # Keep only the most frequent languages (avoids unreadable legends)
    top_langs = (
        df["lang_meta_detected"]
        .value_counts(dropna=False)
        .head(8)
        .index
        .tolist()
    )
    pillars_long = pillars_long[pillars_long["lang_meta_detected"].isin(top_langs)]

    plt.figure(figsize=(14, 7))
    ax = sns.violinplot(
        data=pillars_long,
        x="Pillar",
        y="Score_0_25",
        hue="lang_meta_detected",
        cut=0,
    )
    ax.set_title("Pillar scores by detected language (top 8 languages)")
    ax.set_xlabel("")
    ax.set_ylabel("Score (0–25)")
    plt.legend(title="lang", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    savefig("05_pillars_by_language_violin")
    plt.show()


if __name__ == "__main__":
    main()
