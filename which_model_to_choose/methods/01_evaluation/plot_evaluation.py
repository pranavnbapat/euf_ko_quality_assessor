# which_model_to_choose/methods/01_evaluation/plot_evaluation.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ----------------------------
# Config / metrics
# ----------------------------

FIELDS = ["content", "title", "subtitle", "description", "keywords"]

# Metrics that exist per field (prefix_metric)
METRICS = {
    "len_tokens": "Length (tokens)",
    "len_chars": "Length (chars)",
    "ttr": "Type–Token Ratio (lexical diversity) ↑ is richer",
    "stopword_ratio": "English stopword ratio (EN-only proxy) ↓ often means non-EN / keywordy",
    "punct_ratio": "Punctuation ratio ↑ can mean noisy extraction",
    "readability_fk_like": "FK-like readability (rough) ↑ is easier (not reliable for short strings)",
    "top5_repetition_ratio": "Top-5 repetition ratio ↑ is more repetitive / boilerplate-y",
}

# “Directionality” notes you can use in plot subtitles/annotations
INTERPRETATION = {
    "len_tokens": "Lower is better for indexing (until too small); very high = needs chunking/summarising.",
    "ttr": "Higher often means richer wording; extremely high can also mean very short text.",
    "stopword_ratio": "Works only for English; very low can indicate non-English or keyword lists.",
    "punct_ratio": "Higher can indicate tables, OCR artefacts, or UI boilerplate.",
    "readability_fk_like": "Higher = easier; ignore for short fields (title/keywords).",
    "top5_repetition_ratio": "Higher = more repetition (boilerplate/junk risk).",
}

sns.set_theme(style="whitegrid", context="talk")


# ----------------------------
# Utilities
# ----------------------------

def has_both_groups(long: pd.DataFrame) -> bool:
    """True if both is_llm True and False exist in this run."""
    vals = long["is_llm"].dropna().unique().tolist()
    return (True in vals) and (False in vals)

def load_json_rows(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)

def infer_is_llm_for_field(df: pd.DataFrame, field: str) -> pd.Series:
    """
    Decide if a field is from LLM based on the *_source column.
    e.g. title_source = title_llm, content_source = ko_content_flat_summarised
    """
    src_col = f"{field}_source"
    if src_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    src = df[src_col].fillna("").astype(str)
    # content uses ko_content_flat_summarised; other fields use *_llm
    if field == "content":
        return src.str.contains("summarised", case=False, na=False)
    return src.str.endswith("_llm")


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide -> long:
    record_index, field, metric, value, is_llm, source_key
    """
    rows = []
    for field in FIELDS:
        is_llm = infer_is_llm_for_field(df, field)
        src_col = f"{field}_source" if f"{field}_source" in df.columns else None
        source_key = df[src_col].fillna("unknown").astype(str) if src_col else "unknown"

        for metric in METRICS.keys():
            col = f"{field}_{metric}"
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")

            rows.append(pd.DataFrame({
                "record_index": df.get("record_index", pd.Series(range(len(df)))).astype(str),
                "field": field,
                "metric": metric,
                "value": vals,
                "is_llm": is_llm,
                "source_key": source_key,
            }))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    # Clean
    out = out.dropna(subset=["value"])
    return out


def savefig(output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def annotate_interpretation(ax, metric: str) -> None:
    """
    Add a small interpretation box to the top-left of the axes.
    """
    txt = INTERPRETATION.get(metric, "")
    if not txt:
        return
    ax.text(
        0.01, 0.99, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9)
    )


def robust_log1p(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.clip(lower=0)
    return np.log1p(x)


# ----------------------------
# Chart pack
# ----------------------------

def chart_manager_boxplots(long: pd.DataFrame, output: Path) -> None:
    """
    Manager-friendly: before/after distribution (LLM vs non-LLM).
    Use boxplots; token length on log scale for content.
    """
    for metric in ["len_tokens", "top5_repetition_ratio", "ttr"]:
        plt.figure(figsize=(14, 6))
        dfp = long[long["metric"] == metric].copy()
        if dfp.empty:
            continue

        # For token lengths, use log scale to avoid one giant tail dominating
        if metric == "len_tokens":
            dfp["value_plot"] = robust_log1p(dfp["value"])
            ylab = "log(1 + tokens)"
        else:
            dfp["value_plot"] = dfp["value"]
            ylab = METRICS[metric]

        use_hue = has_both_groups(dfp)

        ax = sns.boxplot(
            data=dfp,
            x="field",
            y="value_plot",
            hue=("is_llm" if use_hue else None),
            showfliers=False,
        )
        title = f"{METRICS[metric]} by field"
        if use_hue:
            title += " (LLM-selected vs original)"
        ax.set_title(title)

        ax.set_xlabel("Field")
        ax.set_ylabel(ylab)
        annotate_interpretation(ax, metric)

        if use_hue:
            ax.legend(title="LLM-selected", loc="best")
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        savefig(output, f"mgr_box_{metric}")


def chart_manager_completeness(df: pd.DataFrame, output: Path) -> None:
    """
    Bar charts for metadata completeness:
    % empty per field (subtitle missing etc.)
    """
    # Identify empties from *_len_tokens == 0 OR *_error == 'empty'
    stats = []
    for field in FIELDS:
        lt = f"{field}_len_tokens"
        err = f"{field}_error"
        if lt not in df.columns:
            continue
        empty = (pd.to_numeric(df[lt], errors="coerce").fillna(0) == 0)
        if err in df.columns:
            empty = empty | (df[err].fillna("") == "empty")
        stats.append({"field": field, "empty_rate": empty.mean() * 100.0})

    if not stats:
        return

    s = pd.DataFrame(stats).sort_values("empty_rate", ascending=False)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=s, x="field", y="empty_rate")
    ax.set_title("Metadata completeness: % empty by field (lower is better)")
    ax.set_xlabel("Field")
    ax.set_ylabel("% empty")
    for i, r in s.reset_index(drop=True).iterrows():
        ax.text(i, r["empty_rate"] + 0.5, f"{r['empty_rate']:.1f}%", ha="center", va="bottom", fontsize=10)
    savefig(output, "mgr_metadata_completeness")


def chart_engineer_scatter_failure_modes(long: pd.DataFrame, output: Path) -> None:
    """
    Engineer-friendly scatter plots (failure-mode clusters).
    """
    # 1) length vs TTR
    df_len = long[long["metric"] == "len_tokens"].rename(columns={"value": "len_tokens"}).copy()
    df_ttr = long[long["metric"] == "ttr"].rename(columns={"value": "ttr"}).copy()
    df_rep = long[long["metric"] == "top5_repetition_ratio"].rename(columns={"value": "rep"}).copy()

    # Merge per record+field
    base = df_len.merge(df_ttr[["record_index","field","ttr"]], on=["record_index","field"], how="inner")
    base = base.merge(df_rep[["record_index","field","rep"]], on=["record_index","field"], how="inner")
    base["len_log"] = robust_log1p(base["len_tokens"])

    # length vs TTR
    plt.figure(figsize=(14, 8))
    ax = sns.scatterplot(
        data=base,
        x="len_log",
        y="ttr",
        hue="field",
        alpha=0.6,
    )
    ax.set_title("Failure modes: lexical diversity vs length")
    ax.set_xlabel("log(1 + tokens)  (→ longer)")
    ax.set_ylabel("TTR  (↑ richer wording; very high can also mean very short)")
    ax.legend(loc="best", title="Field")
    savefig(output, "eng_scatter_len_vs_ttr")

    # length vs repetition
    plt.figure(figsize=(14, 8))
    ax = sns.scatterplot(
        data=base,
        x="len_log",
        y="rep",
        hue="field",
        alpha=0.6,
    )
    ax.set_title("Failure modes: repetition vs length")
    ax.set_xlabel("log(1 + tokens)  (→ longer)")
    ax.set_ylabel("Top-5 repetition ratio  (↑ more boilerplate/junk risk)")
    savefig(output, "eng_scatter_len_vs_repetition")


def chart_engineer_density(long: pd.DataFrame, output: Path) -> None:
    """
    Density plots: show how LLM normalises distributions.
    """
    # Token length density for content/description
    dfp = long[(long["metric"] == "len_tokens") & (long["field"].isin(["content","description","keywords"]))].copy()
    if dfp.empty:
        return
    dfp["len_log"] = robust_log1p(dfp["value"])

    g = sns.displot(
        data=dfp,
        x="len_log",
        hue="is_llm",
        col="field",
        kind="kde",
        fill=True,
        common_norm=False,
        height=4,
        aspect=1.2,
    )
    g.fig.suptitle("Distribution normalisation: token length density (log scale)", y=1.02)
    savefig(output, "eng_density_len_tokens_log")


def chart_engineer_correlation(long: pd.DataFrame, output: Path) -> None:
    """
    Correlation heatmaps per field (what metrics are redundant?).
    """
    # Pivot to wide metrics per record_index+field
    piv = long.pivot_table(index=["record_index","field"], columns="metric", values="value", aggfunc="mean")
    piv = piv.reset_index()

    for field in FIELDS:
        sub = piv[piv["field"] == field].copy()
        if sub.empty:
            continue
        mcols = [c for c in METRICS.keys() if c in sub.columns]
        if len(mcols) < 2:
            continue

        corr = sub[mcols].corr(numeric_only=True)

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(corr, annot=True, fmt=".2f", center=0, cmap="vlag")
        ax.set_title(f"Metric correlation heatmap (field={field})\nHelps simplify metrics: highly correlated pairs are redundant")
        ax.set_xlabel("")
        ax.set_ylabel("")
        savefig(output, f"eng_corr_{field}")


def chart_engineer_pairplot(long: pd.DataFrame, output: Path) -> None:
    """
    Pairplot for a quick multi-metric view (heavy but useful).
    Limit to content + description to keep it readable.
    """
    piv = long.pivot_table(index=["record_index","field"], columns="metric", values="value", aggfunc="mean").reset_index()
    sub = piv[piv["field"].isin(["content","description"])].copy()
    if sub.empty:
        return

    # If you have is_llm in original wide df only, we reconstruct from source_key isn’t available here.
    # So we just colour by field.
    use = [c for c in ["len_tokens","ttr","stopword_ratio","punct_ratio","top5_repetition_ratio"] if c in sub.columns]
    if len(use) < 3:
        return

    # log scale length to stop it dominating
    sub["len_tokens_log"] = robust_log1p(sub["len_tokens"])
    use_plot = ["len_tokens_log"] + [c for c in use if c != "len_tokens"]

    g = sns.pairplot(sub, vars=use_plot, hue="field", corner=True, plot_kws={"alpha": 0.4, "s": 15})
    g.fig.suptitle("Engineer overview: pairwise metric relationships (content vs description)", y=1.02)
    savefig(output, "eng_pairplot_content_description")


def chart_ops_outliers(df: pd.DataFrame, output: Path, top_n: int = 20) -> None:
    """
    Operational: action lists for QA.
    """
    # Longest content
    if "content_len_tokens" in df.columns:
        sub = df[["record_index","content_len_tokens","content_source","title_preview"]].copy()
        sub["content_len_tokens"] = pd.to_numeric(sub["content_len_tokens"], errors="coerce")
        sub = sub.sort_values("content_len_tokens", ascending=False).head(top_n)

        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=sub, y="record_index", x="content_len_tokens", hue="content_source", dodge=False)
        ax.set_title(f"Top {top_n} longest content fields (higher = chunk/summarise)")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("record_index")
        savefig(output, "ops_top_longest_content")

    # Most repetitive content
    if "content_top5_repetition_ratio" in df.columns:
        sub = df[["record_index","content_top5_repetition_ratio","content_source","title_preview"]].copy()
        sub["content_top5_repetition_ratio"] = pd.to_numeric(sub["content_top5_repetition_ratio"], errors="coerce")
        sub = sub.sort_values("content_top5_repetition_ratio", ascending=False).head(top_n)

        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=sub, y="record_index", x="content_top5_repetition_ratio", hue="content_source", dodge=False)
        ax.set_title(f"Top {top_n} most repetitive content fields (higher = boilerplate/junk risk)")
        ax.set_xlabel("Top-5 repetition ratio")
        ax.set_ylabel("record_index")
        savefig(output, "ops_top_repetition_content")


def chart_manager_dumbbell(before_df: pd.DataFrame, after_df: pd.DataFrame, output: Path) -> None:
    """
    Dramatic before/after dumbbell chart for managers.
    Requires a stable id to join. If you only have record_index, this won’t work reliably across runs.

    Best practice: include _orig_id (or @id) in your scorer output.
    Here we try '_orig_id' then '@id' then fallback to record_index.
    """
    join_key = None
    for k in ["_orig_id", "@id", "record_index"]:
        if k in before_df.columns and k in after_df.columns:
            join_key = k
            break
    if join_key is None:
        return

    b = before_df[[join_key, "content_len_tokens"]].copy()
    a = after_df[[join_key, "content_len_tokens"]].copy()

    b["content_len_tokens"] = pd.to_numeric(b["content_len_tokens"], errors="coerce")
    a["content_len_tokens"] = pd.to_numeric(a["content_len_tokens"], errors="coerce")

    m = b.merge(a, on=join_key, how="inner", suffixes=("_before","_after"))
    m = m.dropna(subset=["content_len_tokens_before","content_len_tokens_after"])
    if m.empty:
        return

    # Pick top N biggest reductions for a dramatic plot
    m["delta"] = m["content_len_tokens_before"] - m["content_len_tokens_after"]
    m = m.sort_values("delta", ascending=False).head(30)

    # Plot on log scale
    m["before_log"] = robust_log1p(m["content_len_tokens_before"])
    m["after_log"] = robust_log1p(m["content_len_tokens_after"])

    plt.figure(figsize=(14, 10))
    y = np.arange(len(m))

    plt.hlines(y=y, xmin=m["after_log"], xmax=m["before_log"], alpha=0.6)
    plt.scatter(m["before_log"], y, label="Before", s=60)
    plt.scatter(m["after_log"], y, label="After", s=60)

    plt.yticks(y, m[join_key].astype(str))
    plt.xlabel("log(1 + content tokens)")
    plt.title("Dramatic before/after: content length reduction (top 30 improvements)\nLower is better for indexing; huge drops = summarisation success")
    plt.legend(loc="lower right")

    savefig(output, "mgr_dumbbell_content_len_before_after")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", dest="in_path", required=True, help="Path to JSON output (01_evaluate_selected.json)")
    p.add_argument("--output", required=True, help="Output directory for PNG charts")
    p.add_argument("--in2", default=None, help="Optional: second JSON for before/after dumbbell charts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path).resolve()
    output = Path(args.output).resolve()

    df = load_json_rows(in_path)
    long = to_long(df)

    # --- Manager pack ---
    chart_manager_boxplots(long, output / "managers")
    chart_manager_completeness(df, output / "managers")

    # Optional dramatic dumbbell if you provide a second file
    if args.in2:
        df2 = load_json_rows(Path(args.in2).resolve())
        chart_manager_dumbbell(df, df2, output / "managers")

    # --- Engineer pack ---
    chart_engineer_scatter_failure_modes(long, output / "engineers")
    chart_engineer_density(long, output / "engineers")
    chart_engineer_correlation(long, output / "engineers")
    chart_engineer_pairplot(long, output / "engineers")

    # --- Ops/QA pack ---
    chart_ops_outliers(df, output / "ops")

    print(f"[OK] Wrote chart pack to: {output}")


if __name__ == "__main__":
    main()
