import json
import pandas as pd
import matplotlib.pyplot as plt


# ---------- CONFIG ----------
RESULTS_JSON_PATH = "03_evaluate_chunks.json"


def load_results(path: str) -> pd.DataFrame:
    """Load the JSON list of records into a DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert list[dict] -> DataFrame for easy aggregation and plotting
    df = pd.DataFrame(data)
    return df


def plot_entail_gap_by_model(df: pd.DataFrame) -> None:
    """
    Bar chart: mean entail_gap per candidate_key (i.e. per model).
    This shows which model tends to produce better compressed summaries.
    """
    # group by model key and take mean over all records
    agg = (
        df.groupby("candidate_key")["entail_gap_LS_minus_SL"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure()
    agg.plot(kind="bar")  # default colours are fine
    plt.ylabel("Mean entail_gap (L→S - S→L)")
    plt.xlabel("candidate_key (model)")
    plt.title("Average NLI entailment gap per model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_entail_scatter(df: pd.DataFrame) -> None:
    """
    Scatter plot: L→S_entail vs S→L_entail, coloured by model (legend).
    This shows coverage vs overclaiming behaviour for each model.
    """
    plt.figure()

    # plot each model separately so we can have a legend
    for key, group in df.groupby("candidate_key"):
        plt.scatter(
            group["L_to_S_entail"],
            group["S_to_L_entail"],
            label=key,
            alpha=0.6,  # slight transparency so overlaps are visible
        )

    plt.xlabel("L→S entail (coverage)")
    plt.ylabel("S→L entail (overclaiming)")
    plt.title("Coverage vs Overclaiming by candidate model")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_contradiction_by_model(df: pd.DataFrame) -> None:
    """
    Bar chart: mean contradiction (both directions) per model.
    Helps see which model is most 'factually safe'.
    """
    # compute mean contradictions in each direction per model
    agg = df.groupby("candidate_key").agg(
        mean_L_to_S_contra=("L_to_S_contradiction", "mean"),
        mean_S_to_L_contra=("S_to_L_contradiction", "mean"),
    )

    # make a simple side-by-side bar chart
    plt.figure()
    # we use pandas' plot; it will create bars for each column
    agg.plot(kind="bar")
    plt.ylabel("Mean contradiction probability")
    plt.xlabel("candidate_key (model)")
    plt.title("Mean NLI contradiction by model (L→S and S→L)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    df = load_results(RESULTS_JSON_PATH)

    # quick sanity check: show basic stats per model in the terminal
    print("\n=== Mean metrics per model ===")
    print(
        df.groupby("candidate_key")[
            [
                "L_to_S_entail",
                "S_to_L_entail",
                "L_to_S_contradiction",
                "S_to_L_contradiction",
                "entail_gap_LS_minus_SL",
            ]
        ].mean()
    )

    # 1) Which model has the best entailment gap, on average?
    plot_entail_gap_by_model(df)

    # 2) How do coverage vs overclaiming look?
    plot_entail_scatter(df)

    # 3) Which model contradicts least?
    plot_contradiction_by_model(df)


if __name__ == "__main__":
    main()
