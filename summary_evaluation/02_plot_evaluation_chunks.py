import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------- CONFIG ----------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_JSON_PATH = SCRIPT_DIR / "output" / "02_evaluate_chunks.json"


def load_results(path: Path) -> pd.DataFrame:
    """Load JSON list of records into a DataFrame."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Normalise fields / derived columns
    df["candidate_key"] = df["candidate"].str.split(".", n=1).str[-1]

    # Compression ratio: how small S is compared to L
    df["compression_ratio"] = df["len_S"] / df["len_L"]

    # Treat moverscore = -1 as 'missing'
    df.loc[df["moverscore"] < 0, "moverscore"] = pd.NA

    return df


def plot_mean_scores_by_model(df: pd.DataFrame) -> None:
    """
    Bar chart of mean cosine_sim, BERTScore F1, and Moverscore per model.
    Helps compare overall quality across models.
    """
    metrics = ["cosine_sim", "bertscore_f1", "moverscore"]
    agg = df.groupby("candidate_key")[metrics].mean().sort_values(
        by="bertscore_f1", ascending=False
    )

    plt.figure()
    agg.plot(kind="bar")
    plt.ylabel("Mean score")
    plt.xlabel("Model (candidate_key)")
    plt.title("Mean similarity metrics per model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_length_vs_scores(df: pd.DataFrame) -> None:
    """
    Scatter plots:
    - len_S vs cosine_sim
    - len_S vs BERTScore F1
    Encodes how summary length relates to quality.
    """
    plt.figure()
    for key, group in df.groupby("candidate_key"):
        plt.scatter(
            group["len_S"],
            group["cosine_sim"],
            label=f"{key} (cosine)",
            alpha=0.6,
            marker="o",
        )
    plt.xlabel("len_S (summary length in tokens/chars)")
    plt.ylabel("cosine_sim")
    plt.title("Summary length vs cosine similarity")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    for key, group in df.groupby("candidate_key"):
        plt.scatter(
            group["len_S"],
            group["bertscore_f1"],
            label=f"{key} (BERT F1)",
            alpha=0.6,
            marker="o",
        )
    plt.xlabel("len_S (summary length in tokens/chars)")
    plt.ylabel("BERTScore F1")
    plt.title("Summary length vs BERTScore F1")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_compression_vs_scores(df: pd.DataFrame) -> None:
    """
    Scatter: compression_ratio vs BERTScore F1, coloured by model.
    Shows whether being more aggressive in compression hurts quality.
    """
    plt.figure()
    for key, group in df.groupby("candidate_key"):
        plt.scatter(
            group["compression_ratio"],
            group["bertscore_f1"],
            label=key,
            alpha=0.6,
        )
    plt.xlabel("compression_ratio = len_S / len_L")
    plt.ylabel("BERTScore F1")
    plt.title("Compression vs semantic quality")
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Plot semantic similarity evaluation outputs.")
    p.add_argument("--input", type=Path, default=DEFAULT_RESULTS_JSON_PATH, help="Path to 02 evaluation JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_results(args.input)

    # ---- Quick numeric insights in the terminal ----
    print("\n=== Mean metrics per model ===")
    print(
        df.groupby("candidate_key")[
            ["cosine_sim", "bertscore_p", "bertscore_r", "bertscore_f1", "moverscore"]
        ].mean()
    )

    print("\n=== Mean lengths & compression per model ===")
    print(
        df.groupby("candidate_key")[["len_L", "len_S", "compression_ratio"]].mean()
    )

    # ---- Plots ----
    plot_mean_scores_by_model(df)
    plot_length_vs_scores(df)
    plot_compression_vs_scores(df)


if __name__ == "__main__":
    main()
