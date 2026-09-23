# assess_ko_quality/validate_metrics.py
"""
Validate the quality metrics against the 2024 human review, and calibrate thresholds.

Three questions this answers:

  1. Does each metric track reviewer judgement, and does it add anything beyond
     document length? (Length alone is a strong predictor on this corpus, so a
     metric that merely correlates with length is not carrying quality signal.)
  2. Which feature set predicts reviewer judgement best out of sample?
  3. What is the ceiling? Reviewers disagree with each other, so no scorer can
     correlate with them perfectly; the ceiling is sqrt(criterion reliability).

Usage:
    python validate_metrics.py                          # newest TSV in ./output
    python validate_metrics.py --auto output/run.tsv
    python validate_metrics.py --calibrate --kos ../input/export.json

Run from inside assess_ko_quality/ (these modules import each other flat).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from validation.human_review import (
    HUMAN_XLSX,
    inter_rater_agreement,
    latest_auto_tsv,
    load_all_human_reviews,
)

# The human dimension worth validating against. The other four have such low
# inter-rater agreement (rho 0.14-0.34) that they cannot support a conclusion.
TARGET = "human_recommend_0_1"

LENGTH_COL = "Structural_metrics_content_words"


# ---------------------------------------------------------------- statistics
def partial_spearman(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> float:
    """Spearman correlation between x and y after removing what both share with z."""
    rxy = stats.spearmanr(x, y)[0]
    rxz = stats.spearmanr(x, z)[0]
    ryz = stats.spearmanr(y, z)[0]
    denom = np.sqrt((1.0 - rxz ** 2) * (1.0 - ryz ** 2))
    return float("nan") if denom == 0 else (rxy - rxz * ryz) / denom


def spearman_brown(r_single: float, k: int) -> float:
    """Reliability of a mean of k raters, given the agreement between single raters."""
    if not np.isfinite(r_single) or r_single <= 0:
        return float("nan")
    return k * r_single / (1.0 + (k - 1) * r_single)


def cv_spearman(X: np.ndarray, y: np.ndarray, repeats: int = 20) -> Tuple[float, float]:
    """Out-of-fold Spearman from repeated 5-fold ridge regression on ranked features."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    scores = []
    for seed in range(repeats):
        oof = np.zeros(len(y))
        for train, test in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
            model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[train], y[train])
            oof[test] = model.predict(X[test])
        scores.append(stats.spearmanr(y, oof)[0])
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------- reporting
def _rank(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").rank(pct=True).fillna(0.5).to_numpy()


def report_metrics(df: pd.DataFrame, y: pd.Series, log_len: pd.Series) -> pd.DataFrame:
    """Each metric against the human target, raw and controlling for length."""
    candidates: List[Tuple[str, str, bool]] = [
        ("log(content length)                 [reference]", LENGTH_COL, False),
        ("Total_Quality_weighted_0_100        [the score]", "Total_Quality_weighted_0_100", False),
        ("info_density_mtld                   [current]", "info_density_mtld", False),
        ("info_density_unique_ratio (TTR)     [replaced]", "info_density_unique_ratio", False),
        ("Semantic_information_density 0-5", "Semantic_information_density", False),
        ("Structural_noise 0-5", "Structural_noise", False),
        ("noise_non_ascii_ratio (inverted)", "noise_non_ascii_ratio", True),
        ("noise_url_density (inverted)", "noise_url_density", True),
        ("Semantic_clarity 0-5", "Semantic_clarity", False),
        ("Semantic_consistency 0-5", "Semantic_consistency", False),
        ("Structural_Score_0_25", "Structural_Score_0_25", False),
        ("Semantic_Score_0_25", "Semantic_Score_0_25", False),
        ("Functional_Score_0_25", "Functional_Score_0_25", False),
        ("Domain_Score_0_25", "Domain_Score_0_25", False),
    ]

    rows = []
    for label, col, invert in candidates:
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        if col == LENGTH_COL:
            v = np.log1p(v)
        if invert:
            v = -v
        ok = y.notna() & v.notna()
        if ok.sum() < 3 or v[ok].std() == 0:
            continue
        rho, p = stats.spearmanr(y[ok], v[ok])
        rows.append({
            "metric": label,
            "n": int(ok.sum()),
            "spearman": rho,
            "p": p,
            "partial_on_length": partial_spearman(y[ok], v[ok], log_len[ok]),
        })
    return pd.DataFrame(rows)


def report_models(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cross-validated comparison of feature sets."""
    ok = y.notna()
    d = df[ok].reset_index(drop=True)
    yy = y[ok].to_numpy()

    length = np.log1p(pd.to_numeric(d[LENGTH_COL], errors="coerce")).rank(pct=True).fillna(0.5).to_numpy()
    pillars = [c for c in ["Structural_Score_0_25", "Semantic_Score_0_25",
                           "Functional_Score_0_25", "Domain_Score_0_25"] if c in d.columns]
    subs = [c for c in [
        "Structural_length", "Structural_completeness", "Structural_noise", "Structural_formatting",
        "Semantic_clarity", "Semantic_usefulness", "Semantic_information_density", "Semantic_consistency",
        "Functional_BM25_readiness", "Functional_embedding_readiness", "Functional_RAG_readiness",
        "Functional_keyword_indexability", "Domain_term_density", "Domain_in_title",
        "Domain_in_keywords", "Domain_consistency"] if c in d.columns]

    sets = {"length only": [length]}
    if "info_density_mtld" in d.columns:
        mtld_r = _rank(d, "info_density_mtld")
        sets["MTLD only"] = [mtld_r]
        sets["length + MTLD"] = [length, mtld_r]
    sets["4 pillars (the score's inputs)"] = [_rank(d, c) for c in pillars]
    sets["all sub-scores"] = [_rank(d, c) for c in subs]
    sets["all sub-scores + length"] = [_rank(d, c) for c in subs] + [length]

    rows = []
    for name, feats in sets.items():
        X = np.column_stack(feats)
        mean, sd = cv_spearman(X, yy)
        rows.append({"feature_set": name, "n_features": X.shape[1], "cv_spearman": mean, "sd": sd})

    rng = np.random.RandomState(0)
    perm = [stats.spearmanr(yy, rng.permutation(yy))[0] for _ in range(2000)]
    rows.append({"feature_set": "-- chance (95th pct |rho|) --", "n_features": 0,
                 "cv_spearman": float(np.percentile(np.abs(perm), 95)), "sd": float(np.std(perm))})
    return pd.DataFrame(rows)


def report_robustness(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Repeat the headline comparison on progressively cleaner subsets.

    Short content usually means extraction failed rather than that the KO is poor, so
    a predictor that only works with those rows in is detecting extraction, not quality.
    """
    rows = []
    for label, floor in [("all KOs", 0), ("excl. <80 tokens", 80),
                         ("excl. <200 tokens", 200), ("excl. <500 tokens", 500)]:
        length_raw = pd.to_numeric(df[LENGTH_COL], errors="coerce")
        keep = y.notna() & (length_raw >= floor)
        if keep.sum() < 30:
            rows.append({"subset": label, "n": int(keep.sum()), "length_only": np.nan,
                         "length_plus_mtld": np.nan, "four_pillars": np.nan})
            continue
        d = df[keep].reset_index(drop=True)
        yy = y[keep].to_numpy()
        length = np.log1p(pd.to_numeric(d[LENGTH_COL], errors="coerce")).rank(pct=True).fillna(0.5).to_numpy()
        entry = {"subset": label, "n": int(keep.sum()),
                 "length_only": cv_spearman(np.column_stack([length]), yy)[0]}
        if "info_density_mtld" in d.columns:
            entry["length_plus_mtld"] = cv_spearman(
                np.column_stack([length, _rank(d, "info_density_mtld")]), yy)[0]
        pillars = [c for c in ["Structural_Score_0_25", "Semantic_Score_0_25",
                               "Functional_Score_0_25", "Domain_Score_0_25"] if c in d.columns]
        entry["four_pillars"] = cv_spearman(np.column_stack([_rank(d, c) for c in pillars]), yy)[0]
        rows.append(entry)
    return pd.DataFrame(rows)


SUB_SCORES = [
    "Structural_length", "Structural_completeness", "Structural_noise", "Structural_formatting",
    "Semantic_clarity", "Semantic_usefulness", "Semantic_information_density", "Semantic_consistency",
    "Functional_BM25_readiness", "Functional_embedding_readiness", "Functional_RAG_readiness",
    "Functional_keyword_indexability", "Domain_term_density", "Domain_in_title",
    "Domain_in_keywords", "Domain_consistency",
]


def report_components(df: pd.DataFrame, y: pd.Series, log_len: pd.Series) -> pd.DataFrame:
    """Per sub-score: does it track reviewers, and does it add anything beyond length?"""
    rows = []
    for c in SUB_SCORES:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        ok = y.notna() & v.notna()
        if ok.sum() < 3 or v[ok].std() == 0:
            continue
        rho, p = stats.spearmanr(y[ok], v[ok])
        rows.append({
            "component": c,
            "spearman": rho,
            "p": p,
            "partial_on_length": partial_spearman(y[ok], v[ok], log_len[ok]),
            "verdict": "KEEP" if p < 0.05 and rho > 0 else ("WRONG SIGN" if p < 0.05 else "null"),
        })
    return pd.DataFrame(rows)


def report_selection(df: pd.DataFrame, y: pd.Series, ks: Sequence[int] = (1, 2, 4, 8, 16)) -> pd.DataFrame:
    """
    Nested CV: pick the top-k components inside each training fold, never on test data.

    Selecting components on the full sample and then scoring them is circular; this is
    the honest version, and it answers whether dropping weak components actually helps.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    ok = y.notna()
    d = df[ok].reset_index(drop=True)
    yy = y[ok].to_numpy()
    cols = [c for c in SUB_SCORES if c in d.columns]
    X = np.column_stack([_rank(d, c) for c in cols])

    rows = []
    for k in ks:
        if k > X.shape[1]:
            continue
        scores = []
        for seed in range(20):
            oof = np.zeros(len(yy))
            for train, test in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
                ranked = [abs(stats.spearmanr(yy[train], X[train, i])[0]) for i in range(X.shape[1])]
                keep = np.argsort(ranked)[::-1][:k]
                model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[train][:, keep], yy[train])
                oof[test] = model.predict(X[test][:, keep])
            scores.append(stats.spearmanr(yy, oof)[0])
        rows.append({"components_kept": k, "cv_spearman": float(np.mean(scores)),
                     "sd": float(np.std(scores))})
    return pd.DataFrame(rows)


def calibrate_mtld(kos_path: Path, sample: int = 3000, seed: int = 42) -> None:
    """Re-derive the MTLD_THRESHOLDS quintiles from a KO export."""
    import random

    import orjson

    from core.text_utils import mtld, norm_text, strip_stops, tokens

    raw = orjson.loads(kos_path.read_bytes())
    docs = raw["docs"] if isinstance(raw, dict) and "docs" in raw else raw
    random.seed(seed)
    if len(docs) > sample:
        docs = random.sample(docs, sample)

    values = []
    for d in docs:
        nostop = strip_stops(tokens(norm_text(d.get("ko_content_flat"))))
        if nostop:
            values.append(mtld(nostop))
    v = np.array(values)

    print(f"\n=== MTLD calibration over {len(v)} KOs from {kos_path.name} (seed {seed}) ===")
    print(f"  min {v.min():.1f}  p20 {np.percentile(v, 20):.1f}  median {np.median(v):.1f}  max {v.max():.1f}")
    print(f"  share below the extraction-failure floor (MTLD 20): {100 * (v < 20).mean():.0f}%")
    print("\n  suggested defaults:")
    for pct, score in [(80, 5), (60, 4), (40, 3)]:
        print(f"    SEM_MTLD_T{score} = {np.percentile(v, pct):.0f}   (p{pct})")
    print(f"    SEM_MTLD_T2 = 20   (p{100 * (v < 20).mean():.0f}, extraction-failure floor)")


# ---------------------------------------------------------------- driver
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auto", type=Path, default=None, help="Assessor TSV (default: newest in ./output)")
    ap.add_argument("--human", type=Path, default=HUMAN_XLSX, help="Human review workbook")
    ap.add_argument("--kos", type=Path, default=None, help="KO export JSON, for --calibrate")
    ap.add_argument("--calibrate", action="store_true", help="Re-derive MTLD thresholds and exit")
    ap.add_argument("--include-collating", action="store_true",
                    help="Treat the consolidated sheet as a third reviewer")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.calibrate:
        if args.kos is None:
            sys.exit("--calibrate needs --kos pointing at a KO export JSON")
        calibrate_mtld(args.kos)
        return

    auto_path = args.auto if args.auto is not None else latest_auto_tsv()
    print(f"[auto ] {auto_path}")
    print(f"[human] {args.human}")

    long_human, human = load_all_human_reviews(args.human, include_collating=args.include_collating)
    auto = pd.read_csv(auto_path, sep="\t", dtype=str)
    df = human.merge(auto, left_on="ko_id", right_on="_orig_id", how="inner")

    print(f"\nreviewed {len(human)} KOs | scored {len(auto)} | analysed {len(df)}")
    if LENGTH_COL not in df.columns:
        sys.exit(f"Assessor TSV is missing {LENGTH_COL!r}; re-run ko_quality_assessor_kc.py.")

    y = pd.to_numeric(df[TARGET], errors="coerce")
    log_len = np.log1p(pd.to_numeric(df[LENGTH_COL], errors="coerce"))

    ceiling = inter_rater_agreement(long_human)
    target_row = ceiling[ceiling["dimension"] == TARGET]
    r_single = float(target_row["spearman"].iloc[0]) if len(target_row) else float("nan")
    k = int(round(df["n_reviewers"].mean())) if "n_reviewers" in df.columns else 2
    reliability = spearman_brown(r_single, k)
    max_r = np.sqrt(reliability) if np.isfinite(reliability) else float("nan")

    print(f"\n=== Ceiling ===")
    print(f"  single-rater agreement on {TARGET}: rho = {r_single:.3f}")
    print(f"  reliability of the {k}-rater mean (Spearman-Brown): {reliability:.3f}")
    print(f"  max |rho| any scorer can reach against it: {max_r:.3f}")

    print(f"\n=== Individual metrics vs {TARGET} ===")
    metrics = report_metrics(df, y, log_len)
    print(metrics.round(4).to_string(index=False))
    print("  (partial_on_length near zero => the metric carries nothing beyond length)")

    print(f"\n=== Per-component signal ===")
    print(report_components(df, y, log_len).round(4).to_string(index=False))

    print(f"\n=== Nested CV: does keeping only the top-k components help? ===")
    print(report_selection(df, y).round(3).to_string(index=False))
    print("  (if the curve is flat or peaks at k=all, component selection is not the lever)")

    print(f"\n=== Cross-validated feature sets ===")
    models = report_models(df, y)
    print(models.round(3).to_string(index=False))

    print(f"\n=== Robustness across content-length floors ===")
    print(report_robustness(df, y).round(3).to_string(index=False))

    total = metrics[metrics["metric"].str.contains("Total_Quality")]
    if len(total) and np.isfinite(max_r):
        rho = float(total["spearman"].iloc[0])
        print(f"\n=== Headline ===")
        print(f"  the score reaches {100 * rho / max_r:.0f}% of the reachable ceiling "
              f"(rho {rho:+.3f} against a max of {max_r:.3f})")


if __name__ == "__main__":
    main()
