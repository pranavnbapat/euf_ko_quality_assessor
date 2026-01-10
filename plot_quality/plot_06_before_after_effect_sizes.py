# plot_quality/plot_06_before_after_effect_sizes.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import COLS, add_file_footer, iqr, load_before_after, pct_below, savefig, set_theme, WEIGHTS


BEFORE_PATH = "data/quality_check_20260109_201910.tsv"
AFTER_PATH  = "data/quality_check_20260109_202033.tsv"
DATA_DIR = "data"

# Pillar failure zone threshold on the raw 0–25 pillar scale
THRESHOLD = 10


def _compute_effect_table(before_df: pd.DataFrame, after_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Compute effect sizes for each pillar on the raw 0–25 scale:
      - median (typical KO)
      - IQR (spread/consistency)
      - % below threshold (tail risk / failure zone)
    """
    rows: list[dict] = []

    for col in COLS.pillar_cols:
        if col not in before_df.columns or col not in after_df.columns:
            continue

        b = pd.to_numeric(before_df[col], errors="coerce")
        a = pd.to_numeric(after_df[col], errors="coerce")

        rows.append(
            {
                "Pillar": col.replace("_Score_0_25", ""),
                "Before_median": float(b.median()),
                "After_median": float(a.median()),
                "Δ_median": float(a.median() - b.median()),
                "Before_IQR": iqr(b),
                "After_IQR": iqr(a),
                "Δ_IQR": float(iqr(a) - iqr(b)),
                f"Before_%<{threshold}": pct_below(b, threshold),
                f"After_%<{threshold}": pct_below(a, threshold),
                f"Δ_%<{threshold}": float(pct_below(a, threshold) - pct_below(b, threshold)),
            }
        )

    return pd.DataFrame(rows)


def _compute_weighted_effect_table(
    before_df: pd.DataFrame, after_df: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """
    Compute effect sizes in "contribution to Total Quality (0–100)" space.

    Contribution per pillar:
        contribution = (pillar_score_0_25 / 25) * WEIGHTS[pillar_col]

    Threshold is still defined on the raw 0–25 scale (threshold/25 of max contribution).
    This keeps the meaning of "failure zone" consistent.
    """
    rows: list[dict] = []

    for col in COLS.pillar_cols:
        if col not in before_df.columns or col not in after_df.columns:
            continue
        if col not in WEIGHTS:
            continue

        w = float(WEIGHTS[col])  # e.g. Structural 30, Semantic 35, etc.
        fail_contrib_threshold = (threshold / 25.0) * w

        b_raw = pd.to_numeric(before_df[col], errors="coerce")
        a_raw = pd.to_numeric(after_df[col], errors="coerce")

        b = (b_raw / 25.0) * w
        a = (a_raw / 25.0) * w

        rows.append(
            {
                "Pillar": col.replace("_Score_0_25", ""),
                "Weight": w,
                "Failure_zone_threshold_contribution": fail_contrib_threshold,
                "Before_median": float(b.median()),
                "After_median": float(a.median()),
                "Δ_median": float(a.median() - b.median()),
                "Before_IQR": iqr(b),
                "After_IQR": iqr(a),
                "Δ_IQR": float(iqr(a) - iqr(b)),
                f"Before_%<fail": pct_below(b, fail_contrib_threshold),
                f"After_%<fail": pct_below(a, fail_contrib_threshold),
                f"Δ_%<fail": float(pct_below(a, fail_contrib_threshold) - pct_below(b, fail_contrib_threshold)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    set_theme()
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    before_df, after_df, before_label, after_label = load_before_after(
        BEFORE_PATH, AFTER_PATH, data_dir=DATA_DIR
    )

    # ---------- RAW (0–25) EFFECTS ----------
    effects = _compute_effect_table(before_df, after_df, THRESHOLD)
    if effects.empty:
        print("No pillar columns found to compare.")
        return

    # Keep a stable pillar order for plots/tables
    effects["Pillar"] = pd.Categorical(
        effects["Pillar"],
        categories=[c.replace("_Score_0_25", "") for c in COLS.pillar_cols if c in before_df.columns],
        ordered=True,
    )
    effects = effects.sort_values("Pillar")

    # ---------- WEIGHTED CONTRIBUTION EFFECTS ----------
    effects_w = _compute_weighted_effect_table(before_df, after_df, THRESHOLD)
    if not effects_w.empty:
        effects_w["Pillar"] = pd.Categorical(
            effects_w["Pillar"],
            categories=[c.replace("_Score_0_25", "") for c in COLS.pillar_cols if c in before_df.columns],
            ordered=True,
        )
        effects_w = effects_w.sort_values("Pillar")

    # ---------- PLOTS ----------
    # Row 1: raw scale (0–25)
    # Row 2: weighted contribution scale (0–100 contributions), if available
    n_rows = 2 if not effects_w.empty else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5.5 * n_rows))

    # Normalise axes indexing when n_rows == 1
    if n_rows == 1:
        axes = [axes]  # type: ignore

    # ---- RAW plots ----
    ax0, ax1, ax2 = axes[0]

    sns.barplot(data=effects, x="Pillar", y="Δ_median", ax=ax0)
    ax0.axhline(0, linewidth=1)
    ax0.set_title("Median change (After − Before) - raw (0–25)")
    ax0.set_xlabel("")
    ax0.set_ylabel("Δ median (0–25)")

    sns.barplot(data=effects, x="Pillar", y="Δ_IQR", ax=ax1)
    ax1.axhline(0, linewidth=1)
    ax1.set_title("IQR change (After − Before) - raw (0–25)")
    ax1.set_xlabel("")
    ax1.set_ylabel("Δ IQR (0–25)")

    delta_tail_col = f"Δ_%<{THRESHOLD}"
    sns.barplot(data=effects, x="Pillar", y=delta_tail_col, ax=ax2)
    ax2.axhline(0, linewidth=1)
    ax2.set_title(f"Tail change: % below {THRESHOLD}/25 (After − Before) - raw")
    ax2.set_xlabel("")
    ax2.set_ylabel("Δ percentage points")

    # ---- WEIGHTED plots ----
    if not effects_w.empty:
        ax3, ax4, ax5 = axes[1]

        sns.barplot(data=effects_w, x="Pillar", y="Δ_median", ax=ax3)
        ax3.axhline(0, linewidth=1)
        ax3.set_title("Median change (After − Before) - weighted contribution")
        ax3.set_xlabel("")
        ax3.set_ylabel("Δ median contribution (0–100)")

        sns.barplot(data=effects_w, x="Pillar", y="Δ_IQR", ax=ax4)
        ax4.axhline(0, linewidth=1)
        ax4.set_title("IQR change (After − Before) - weighted contribution")
        ax4.set_xlabel("")
        ax4.set_ylabel("Δ IQR contribution (0–100)")

        sns.barplot(data=effects_w, x="Pillar", y="Δ_%<fail", ax=ax5)
        ax5.axhline(0, linewidth=1)
        ax5.set_title(f"Tail change: % in failure zone (After − Before) - weighted")
        ax5.set_xlabel("")
        ax5.set_ylabel("Δ percentage points")

    fig.suptitle("Before/After effect sizes (executive summary)", fontsize=14, y=0.995)
    add_file_footer(fig, before_label, after_label)
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
    savefig("06_before_after_effect_sizes")
    plt.show()

    # ---------- SAVE TABLES ----------
    effects.to_csv("out/06_effect_sizes_table_raw_0_25.csv", index=False)
    if not effects_w.empty:
        effects_w.to_csv("out/06_effect_sizes_table_weighted_contrib.csv", index=False)


if __name__ == "__main__":
    main()
