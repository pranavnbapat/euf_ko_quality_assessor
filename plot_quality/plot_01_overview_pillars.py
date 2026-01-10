# plot_quality/plot_01_overview_pillars.py

import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_quality_table, melt_pillars, savefig, set_theme, WEIGHTS, TOTAL_COL


DATA_PATH = "../assess_ko_quality/output/quality_check_20260109_202033.tsv"
SHEET_NAME = 0

PILLAR_TO_RAW = {
    "Structural": "Structural_Score_0_25",
    "Semantic": "Semantic_Score_0_25",
    "Functional": "Functional_Score_0_25",
    "Domain": "Domain_Score_0_25",
}


def main() -> None:
    set_theme()
    df = load_quality_table(DATA_PATH, sheet_name=SHEET_NAME)
    pillars_long = melt_pillars(df)

    # Convert raw 0–25 pillar scores into contribution space consistent with weights.
    # Contribution = (raw_score / 25) * pillar_weight
    pillars_long["Weight"] = pillars_long["Pillar"].map(PILLAR_TO_RAW).map(WEIGHTS)
    if pillars_long["Weight"].isna().any():
        missing = pillars_long.loc[pillars_long["Weight"].isna(), "Pillar"].unique().tolist()
        raise KeyError(
            f"Missing weights for pillars: {missing}. "
            f"Check PILLAR_TO_RAW mapping and WEIGHTS keys."
        )

    pillars_long["Contribution_0_100"] = (pillars_long["Score_0_25"] / 25.0) * pillars_long["Weight"]

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=pillars_long, x="Pillar", y="Contribution_0_100")
    sns.stripplot(
        data=pillars_long,
        x="Pillar",
        y="Contribution_0_100",
        size=4,
        alpha=0.5,
        jitter=0.25,
        ax=ax,
    )
    ax.set_title("Pillar contribution distributions (weighted)")
    ax.set_ylabel("Contribution to Total Quality (0-100)")
    ax.set_xlabel("")
    plt.tight_layout()
    savefig("01_pillars_distribution")
    plt.show()


if __name__ == "__main__":
    main()
