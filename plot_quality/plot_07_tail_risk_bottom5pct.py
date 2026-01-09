# plot_quality/plot_07_tail_risk_bottom5pct.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import COLS, add_file_footer, load_before_after, savefig, set_theme


BEFORE_PATH = None
AFTER_PATH = None
DATA_DIR = "data"
BOTTOM_Q = 0.05  # bottom 5%


def main() -> None:
    set_theme()
    before_df, after_df, before_label, after_label = load_before_after(BEFORE_PATH, AFTER_PATH, data_dir=DATA_DIR)

    frames = []
    for col in COLS.pillar_cols:
        if col not in before_df.columns or col not in after_df.columns:
            continue

        b = pd.to_numeric(before_df[col], errors="coerce").dropna()
        a = pd.to_numeric(after_df[col], errors="coerce").dropna()

        if b.empty or a.empty:
            continue

        b_cut = b.quantile(BOTTOM_Q)
        a_cut = a.quantile(BOTTOM_Q)

        frames.append(pd.DataFrame({
            "Pillar": col.replace("_Score_0_25", ""),
            "Score_0_25": b[b <= b_cut],
            "Dataset": f"Before ({before_label})"
        }))
        frames.append(pd.DataFrame({
            "Pillar": col.replace("_Score_0_25", ""),
            "Score_0_25": a[a <= a_cut],
            "Dataset": f"After ({after_label})"
        }))

    tail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if tail.empty:
        print("No data available for tail plots.")
        return

    plt.figure(figsize=(14, 6))
    ax = sns.violinplot(data=tail, x="Pillar", y="Score_0_25", hue="Dataset", cut=0)
    ax.set_title("Bottom 5% (tail risk) comparison by pillar")
    ax.set_xlabel("")
    ax.set_ylabel("Score (0–25)")
    plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig = plt.gcf()
    add_file_footer(fig, before_label, after_label)
    plt.tight_layout()
    savefig("07_tail_risk_bottom5pct")
    plt.show()


if __name__ == "__main__":
    main()
