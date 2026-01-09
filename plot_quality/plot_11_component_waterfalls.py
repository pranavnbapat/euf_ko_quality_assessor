# plot_quality/plot_11_component_waterfalls.py

import matplotlib.pyplot as plt
import numpy as np

from utils import add_file_footer, load_quality_table, savefig, set_theme


DATA_PATH = "data/temp_improved_quality_check_20251219_122545.tsv"
SHEET_NAME = 0

PILLARS = {
    "Structural": ["Structural_length", "Structural_completeness", "Structural_noise", "Structural_formatting"],
    "Semantic": ["Semantic_clarity", "Semantic_usefulness", "Semantic_information_density", "Semantic_consistency"],
    "Domain": ["Domain_term_density", "Domain_in_title", "Domain_in_keywords", "Domain_consistency"],
    "Functional": ["Functional_BM25_readiness", "Functional_embedding_readiness", "Functional_RAG_readiness", "Functional_keyword_indexability"],
}


def plot_waterfall(name: str, means: list[float], labels: list[str], out_name: str) -> None:
    cum = np.cumsum([0] + means[:-1])
    plt.figure(figsize=(12, 5))
    plt.bar(labels, means, bottom=cum)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean contribution (raw units)")
    plt.title(f"{name} – mean sub-metric contributions (waterfall-style)")
    plt.tight_layout()
    savefig(out_name)
    plt.show()


def main() -> None:
    set_theme()
    df = load_quality_table(DATA_PATH, sheet_name=SHEET_NAME)

    for pillar, cols in PILLARS.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            print(f"Skipping {pillar}: no columns found.")
            continue

        means = [float(df[c].mean(skipna=True)) for c in cols]
        labels = [c.replace(pillar + "_", "") for c in cols]
        plot_waterfall(pillar, means, labels, f"11_waterfall_{pillar.lower()}")


if __name__ == "__main__":
    main()
