# plot_quality/plot_11_component_waterfalls.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import add_file_footer, latest_tsv, load_quality_table, savefig, set_theme


# None -> newest TSV in data/. Set a path to pin a specific run.
DATA_PATH = None
SHEET_NAME = 0

PILLARS = {
    "Structural": ["Structural_length", "Structural_completeness", "Structural_noise", "Structural_formatting"],
    # Semantic_consistency_mnli_used is the MNLI component as it actually entered the
    # pillar: a 0-5 score where MNLI ran, blank where it was skipped (non-English, or
    # no content). Blank means "not part of this KO's pillar", not "scored zero", so
    # its mean is taken over the KOs it covers and the label reports that coverage.
    "Semantic": ["Semantic_clarity", "Semantic_usefulness", "Semantic_information_density",
                 "Semantic_consistency", "Semantic_consistency_mnli_used"],
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
    data_path = DATA_PATH if DATA_PATH is not None else latest_tsv()
    df = load_quality_table(data_path, sheet_name=SHEET_NAME)

    for pillar, cols in PILLARS.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            print(f"Skipping {pillar}: no columns found.")
            continue

        means, labels = [], []
        for c in cols:
            series = pd.to_numeric(df[c], errors="coerce")
            means.append(float(series.mean(skipna=True)))
            label = c.replace(pillar + "_", "")
            coverage = float(series.notna().mean())
            if coverage < 1.0:
                label += f"\n({coverage:.0%} of KOs)"
            labels.append(label)
        plot_waterfall(pillar, means, labels, f"11_waterfall_{pillar.lower()}")


if __name__ == "__main__":
    main()
