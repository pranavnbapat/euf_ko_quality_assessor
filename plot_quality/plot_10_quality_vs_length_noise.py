# plot_quality/plot_10_quality_vs_length_noise.py

import matplotlib.pyplot as plt
import seaborn as sns

from utils import add_file_footer, load_quality_table, savefig, set_theme


DATA_PATH = "data/temp_improved_quality_check_20251219_122545.tsv"
SHEET_NAME = 0


def main() -> None:
    set_theme()
    df = load_quality_table(DATA_PATH, sheet_name=SHEET_NAME)

    plots = [
        ("Structural_length", "10_total_vs_struct_length", "Total Quality vs Structural length"),
        ("Structural_noise", "10_total_vs_struct_noise", "Total Quality vs Structural noise"),
    ]

    for xcol, out_name, title in plots:
        if xcol not in df.columns or "Total_Quality_0_100" not in df.columns:
            print(f"Missing columns for {title}. Need {xcol} and Total_Quality_0_100")
            continue

        g = sns.lmplot(
            data=df,
            x=xcol,
            y="Total_Quality_0_100",
            height=5,
            aspect=1.2,
            scatter_kws={"alpha": 0.35, "s": 25},
            line_kws={"linewidth": 2},
        )
        plt.title(title)
        plt.tight_layout()
        savefig(out_name)
        plt.show()


if __name__ == "__main__":
    main()
