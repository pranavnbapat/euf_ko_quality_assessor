import json
import pandas as pd
import matplotlib.pyplot as plt


# ---------- CONFIG ----------
RESULTS_JSON_PATH = "01_evaluate_chunks.json"


def load_results(path: str) -> pd.DataFrame:
    """Load the JSON list of records into a DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Ensure record_index is a string for nicer x-axis labels
    if "record_index" in df.columns:
        df["record_index"] = df["record_index"].astype(str)
    else:
        # fallback if record_index is absent
        df["record_index"] = df.index.astype(str)

    return df


def plot_style_pairs(df: pd.DataFrame) -> None:
    """
    For each record, plot Short (S) vs Long (L) for:
    - Type–Token Ratio (TTR)
    - Stopword ratio
    - Punctuation ratio
    - Flesch–Kincaid readability score
    """
    x = range(len(df))
    x_labels = df["record_index"].tolist()

    metrics = [
        ("ttr_s", "ttr_l", "Type–Token Ratio (TTR)"),
        ("stopword_ratio_s", "stopword_ratio_l", "Stopword ratio"),
        ("punct_ratio_s", "punct_ratio_l", "Punctuation ratio"),
        ("readability_fk_s", "readability_fk_l", "Flesch–Kincaid score"),
    ]

    for short_col, long_col, title in metrics:
        # skip metric if missing in this JSON
        if short_col not in df.columns or long_col not in df.columns:
            continue

        plt.figure()
        plt.plot(x, df[short_col], marker="o", label="Short (S)")
        plt.plot(x, df[long_col], marker="o", label="Long (L)")

        plt.xticks(x, x_labels, rotation=45, ha="right")
        plt.xlabel("record_index")
        plt.ylabel(title)
        plt.title(f"{title}: Short vs Long")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_compression_vs_rouge_and_heuristic(df: pd.DataFrame) -> None:
    """
    Scatter plots:
    - compression_ratio_tokens vs ROUGE-1 recall
    - compression_ratio_tokens vs ROUGE-2 recall
    - compression_ratio_tokens vs heuristic_score
    """

    # ---- Compression vs ROUGE-1 and ROUGE-2 ----
    if (
        "compression_ratio_tokens" in df.columns
        and "rouge1_recall_s_vs_l" in df.columns
        and "rouge2_recall_s_vs_l" in df.columns
    ):
        plt.figure()
        plt.scatter(
            df["compression_ratio_tokens"],
            df["rouge1_recall_s_vs_l"],
            marker="o",
            alpha=0.7,
            label="ROUGE-1 recall",
        )
        plt.scatter(
            df["compression_ratio_tokens"],
            df["rouge2_recall_s_vs_l"],
            marker="x",
            alpha=0.7,
            label="ROUGE-2 recall",
        )
        plt.xlabel("compression_ratio_tokens (len_S / len_L)")
        plt.ylabel("ROUGE recall")
        plt.title("Compression vs ROUGE overlap (S vs L)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---- Compression vs heuristic_score ----
    if "compression_ratio_tokens" in df.columns and "heuristic_score" in df.columns:
        plt.figure()
        plt.scatter(
            df["compression_ratio_tokens"],
            df["heuristic_score"],
            marker="o",
            alpha=0.7,
        )
        plt.xlabel("compression_ratio_tokens (len_S / len_L)")
        plt.ylabel("heuristic_score")
        plt.title("Compression vs heuristic_score")
        plt.tight_layout()
        plt.show()


def plot_length_comparison(df: pd.DataFrame) -> None:
    """
    Simple length comparison:
    - len_tokens_s vs len_tokens_l
    - len_chars_s vs len_chars_l
    """

    x = range(len(df))
    x_labels = df["record_index"].tolist()

    # Tokens
    if "len_tokens_s" in df.columns and "len_tokens_l" in df.columns:
        plt.figure()
        plt.plot(x, df["len_tokens_s"], marker="o", label="len_tokens_s (S)")
        plt.plot(x, df["len_tokens_l"], marker="o", label="len_tokens_l (L)")
        plt.xticks(x, x_labels, rotation=45, ha="right")
        plt.xlabel("record_index")
        plt.ylabel("Number of tokens")
        plt.title("Token length: Short vs Long")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Characters
    if "len_chars_s" in df.columns and "len_chars_l" in df.columns:
        plt.figure()
        plt.plot(x, df["len_chars_s"], marker="o", label="len_chars_s (S)")
        plt.plot(x, df["len_chars_l"], marker="o", label="len_chars_l (L)")
        plt.xticks(x, x_labels, rotation=45, ha="right")
        plt.xlabel("record_index")
        plt.ylabel("Number of characters")
        plt.title("Character length: Short vs Long")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    df = load_results(RESULTS_JSON_PATH)

    # quick sanity check: show basic stats in the terminal
    print("\n=== Basic style metrics per record ===")
    cols_to_show = [
        "record_index",
        "candidate_name",
        "compression_ratio_tokens",
        "ttr_s",
        "ttr_l",
        "stopword_ratio_s",
        "stopword_ratio_l",
        "punct_ratio_s",
        "punct_ratio_l",
        "readability_fk_s",
        "readability_fk_l",
        "rouge1_recall_s_vs_l",
        "rouge2_recall_s_vs_l",
        "heuristic_score",
        "len_tokens_s",
        "len_tokens_l",
        "len_chars_s",
        "len_chars_l",
    ]
    existing_cols = [c for c in cols_to_show if c in df.columns]
    print(df[existing_cols])

    # 1) Direct S vs L comparison for style metrics
    plot_style_pairs(df)

    # 2) How compression relates to ROUGE/heuristic
    plot_compression_vs_rouge_and_heuristic(df)

    # 3) Simple length comparison
    plot_length_comparison(df)


if __name__ == "__main__":
    main()
