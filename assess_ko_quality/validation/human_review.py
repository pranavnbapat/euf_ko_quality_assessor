# assess_ko_quality/manual_vs_automatic_review.py
"""
Validate the automatic KO quality scores against the 2024 human review.

Reports, for the KOs that appear in both:
  - inter-rater agreement between the human reviewers (the ceiling any automatic
    scorer can be expected to reach)
  - correlation of each automatic pillar, and the total, against the human scores

Usage:
    python manual_vs_automatic_review.py                       # newest TSV in ./output
    python manual_vs_automatic_review.py --auto path/to.tsv
    python manual_vs_automatic_review.py --include-collating   # add the consolidated sheet
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


HUMAN_XLSX = Path("2024_KOs for assessment all collating.xlsx")
OUT_XLSX = Path("human_vs_auto_comparison.xlsx")

# 'collating' is a separate consolidated pass over the same KOs, not a per-reviewer
# sheet. It correlates with the reviewer mean at about r=0.7 but is not identical, so
# folding it in silently would change the ground truth. Opt in with --include-collating.
CONSOLIDATED_SHEETS = {"collating"}


# Emitted by ko_quality_assessor_kc.py and by nothing else, so it distinguishes a
# four-pillar TSV from the content-diagnostics TSVs that share the same output folder.
REQUIRED_AUTO_COL = "Total_Quality_weighted_0_100"


def latest_auto_tsv(output_dir: Path = Path("output")) -> Path:
    """Newest four-pillar assessor TSV in output_dir."""
    candidates = sorted(output_dir.glob("*.tsv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in candidates:
        try:
            header = pd.read_csv(f, sep="\t", nrows=0).columns
        except Exception:
            continue
        if REQUIRED_AUTO_COL in header:
            return f
    raise FileNotFoundError(
        f"No four-pillar assessor TSV (one with a {REQUIRED_AUTO_COL!r} column) in "
        f"{output_dir.resolve()}; found {len(candidates)} other TSV(s). "
        "Run ko_quality_assessor_kc.py first, or pass --auto."
    )


# -----------------------------
# Helpers
# -----------------------------

_OID_RE = re.compile(r"'\\$oid':\\s*'([0-9a-f]{24})'", re.IGNORECASE)

# Accept:
#  - plain 24-hex IDs
#  - strings like "{'$oid': '...'}" or '{"$oid": "..."}'
#  - actual dict objects {"$oid": "..."} (can happen depending on how pandas reads)
_OID_HEX_RE = re.compile(r"([0-9a-f]{24})", re.IGNORECASE)

def print_diagnostics(merged: pd.DataFrame, corr_df: pd.DataFrame, agree_df: pd.DataFrame) -> None:
    print("\n=== Diagnostics: merged coverage ===")
    print(f"KOs matched (inner join): {len(merged)}")
    print("Human dimension non-null counts:")
    for c in ["human_findability_0_1", "human_clarity_0_1", "human_comprehensibility_0_1", "human_usability_0_1", "human_recommend_0_1"]:
        print(f"  {c}: {int(merged[c].notna().sum())}")

    print("\nAuto proxy non-null counts:")
    for c in ["auto_findability_0_1", "auto_clarity_0_1", "auto_comprehensibility_0_1", "auto_usability_0_1", "auto_total_weighted_0_1"]:
        if c in merged.columns:
            print(f"  {c}: {int(merged[c].notna().sum())}")

    def describe01(s: pd.Series) -> str:
        s = s.dropna().astype(float)
        if s.empty:
            return "n=0"
        return (
            f"n={len(s)} min={s.min():.3f} p10={s.quantile(0.10):.3f} "
            f"median={s.median():.3f} p90={s.quantile(0.90):.3f} max={s.max():.3f}"
        )

    print("\n=== Diagnostics: score distributions (0–1) ===")
    pairs = [
        ("findability", "human_findability_0_1", "auto_findability_0_1"),
        ("clarity", "human_clarity_0_1", "auto_clarity_0_1"),
        ("comprehensibility", "human_comprehensibility_0_1", "auto_comprehensibility_0_1"),
        ("usability", "human_usability_0_1", "auto_usability_0_1"),
        ("recommendation vs total", "human_recommend_0_1", "auto_total_weighted_0_1"),
    ]
    for label, h, a in pairs:
        print(f"\n[{label}]")
        if h in merged.columns:
            print(f"  human: {describe01(merged[h])}")
        if a in merged.columns:
            print(f"  auto : {describe01(merged[a])}")

    print("\n=== Correlations (how well rankings align) ===")
    print(corr_df.to_string(index=False))

    print("\n=== Threshold agreement (binary-style) ===")
    print(agree_df.to_string(index=False))

    print("\n=== How to interpret this ===")
    print(
        "- Pearson: linear relationship; Spearman: whether humans and auto rank KOs similarly.\n"
        "- As a rough mental model: |r| < 0.2 weak, 0.2–0.4 mild, 0.4–0.6 moderate, >0.6 strong.\n"
        "- Agreement% depends heavily on the chosen threshold; good for 'screening' use-cases, not proof of equivalence.\n"
        "- If correlations are low but agreement is high, the threshold may be masking a weak continuous relationship.\n"
        "- Credibility and reusability are currently not truly mapped (missing dedicated detectors), so treat those as out-of-scope for validation."
    )

def extract_oid(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None

    # If it's already a dict-like
    if isinstance(v, dict):
        oid = v.get("$oid") or v.get("'$oid'")  # defensive
        if isinstance(oid, str) and _OID_HEX_RE.fullmatch(oid.strip()):
            return oid.strip()

    s = str(v).strip()

    # Plain 24-hex string
    if _OID_HEX_RE.fullmatch(s):
        return s

    # Embedded 24-hex anywhere inside the string
    m = _OID_HEX_RE.search(s)
    return m.group(1) if m else None

def yn_to_num(v) -> float:
    """Map Yes/No-ish values to 1/0. Return NaN if unknown."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return 1.0
    if s in {"no", "n", "false", "0"}:
        return 0.0
    return np.nan

def safe_float(v) -> float:
    if pd.isna(v):
        return np.nan
    try:
        return float(str(v).strip())
    except Exception:
        return np.nan

def normalise_0_25_to_0_1(v) -> float:
    x = safe_float(v)
    if np.isnan(x):
        return np.nan
    return max(0.0, min(1.0, x / 25.0))

def normalise_0_100_to_0_1(v) -> float:
    x = safe_float(v)
    if np.isnan(x):
        return np.nan
    return max(0.0, min(1.0, x / 100.0))

def normalise_0_5_to_0_1(v) -> float:
    x = safe_float(v)
    if np.isnan(x):
        return np.nan
    return max(0.0, min(1.0, x / 5.0))


# -----------------------------
# Human parsing
# -----------------------------
def is_reviewer_sheet(df: pd.DataFrame) -> bool:
    """
    Heuristic: reviewer sheets have the question text in row 0 around the 'Findability' block.
    """
    if df.shape[0] < 3 or df.shape[1] < 50:
        return False
    # These sheets typically have "Findability" in the column header (as a column name),
    # and row 0 contains question text like "Title Clear and complete..."
    col_candidates = [c for c in df.columns if isinstance(c, str) and "Findability" in c]
    if not col_candidates:
        return False
    c = col_candidates[0]
    v = df.loc[0, c]
    return isinstance(v, str) and "Title" in v and "reflects" in v

def parse_reviewer_sheet(sheet_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns tidy rows: one row per KO reviewed in this sheet, with normalised numeric fields.
    Assumes:
      - Row 0 contains question texts
      - Row 2+ contains data
      - Column with _id is the one where row0 == "_id"
    """
    # Find the column that contains the _id values (robust to shifted header rows / merged cells)
    id_col = None
    id_row = None

    # Search within the first few rows, because some sheets shift the "question row"
    SEARCH_ROWS = min(15, df.shape[0])

    # These workbooks carry "_id" as a real column NAME with the ids in the rows
    # beneath it, so check the header before scanning cell values.
    for c in df.columns:
        if isinstance(c, str) and c.strip().lower() in {"_id", "id"}:
            id_col = c
            id_row = None  # ids start in the data rows, not below a label cell
            break

    for r in range(SEARCH_ROWS) if id_col is None else []:
        for c in df.columns:
            v = df.loc[r, c]
            if isinstance(v, str) and v.strip().lower() in {"_id", "id"}:
                id_col = c
                id_row = r
                break
        if id_col is not None:
            break

    if id_col is None:
        # Fallback: pick a column that *looks* like it contains many Mongo ObjectIds
        best_c, best_hits = None, 0
        for c in df.columns:
            hits = 0
            # check a sample of rows for 24-hex occurrences
            for r in range(SEARCH_ROWS, min(SEARCH_ROWS + 60, df.shape[0])):
                oid = extract_oid(df.loc[r, c])
                if oid:
                    hits += 1
            if hits > best_hits:
                best_hits, best_c = hits, c

        if best_hits >= 3:  # heuristic: at least a few OIDs in the sample
            id_col = best_c
            id_row = None  # unknown header row
        else:
            raise ValueError(
                f"Could not find _id column in sheet {sheet_name}. "
                f"Tried explicit '_id' search and OID-pattern fallback."
            )

    # Map “question columns” by their row-0 text (stable across reviewer sheets)
    # We key by a short name that we’ll use downstream.
    question_map: Dict[str, str] = {}

    def find_col_containing(snippet: str) -> str | None:
        snippet_l = snippet.lower()
        for c in df.columns:
            v = df.loc[0, c]
            if isinstance(v, str) and snippet_l in v.lower():
                return c
        return None

    question_map["find_title"] = find_col_containing("Title Clear and complete")
    question_map["find_desc"] = find_col_containing("The description is clear and complete")
    question_map["find_keywords"] = find_col_containing("Keywords : do the chosen key-words")

    question_map["clarity_structured"] = find_col_containing("Visually structured well")

    question_map["comp_target_defined"] = find_col_containing("Is the target audience defined")
    question_map["comp_audience_matches"] = find_col_containing("Does this audience match")
    question_map["comp_jargon_explained"] = find_col_containing("Is jargon in the KO clearly explained")
    question_map["comp_standalone"] = find_col_containing("Is the KO sufficiently standalone")

    question_map["use_context"] = find_col_containing("context of application sufficiently")
    question_map["cred_sources"] = find_col_containing("mention the sources of the knowledge")
    question_map["reuse_licence"] = find_col_containing("licence clearly added and explained")

    # Recommendation question (free numeric 1–5)
    question_map["recommend_1_5"] = find_col_containing("would you recommend this KO")

    missing = [k for k, v in question_map.items() if v is None]
    # Missing fields can happen if sheet formatting changes; we won’t hard-fail.
    # We’ll just produce NaNs for those.
    data = []

    start_row = 2
    if id_row is not None:
        start_row = id_row + 2  # one row for headers/questions, one blank-ish, then data

    for r in range(start_row, df.shape[0]):
        oid = extract_oid(df.loc[r, id_col])
        if not oid:
            continue

        row = {
            "ko_id": oid,
            "reviewer": sheet_name,
        }

        # Binary questions
        for key in [
            "find_title", "find_desc", "find_keywords",
            "clarity_structured",
            "comp_target_defined", "comp_audience_matches", "comp_jargon_explained", "comp_standalone",
            "use_context", "cred_sources", "reuse_licence",
        ]:
            c = question_map.get(key)
            row[key] = yn_to_num(df.loc[r, c]) if c is not None else np.nan

        # Recommendation (1–5)
        c = question_map.get("recommend_1_5")
        row["recommend_1_5"] = safe_float(df.loc[r, c]) if c is not None else np.nan

        data.append(row)

    out = pd.DataFrame(data)

    # If we failed to extract any KO ids from this sheet, return an empty frame
    # with the expected columns so downstream code does not crash.
    expected_cols = [
        "ko_id", "reviewer",
        "find_title", "find_desc", "find_keywords",
        "clarity_structured",
        "comp_target_defined", "comp_audience_matches", "comp_jargon_explained", "comp_standalone",
        "use_context", "cred_sources", "reuse_licence",
        "recommend_1_5",
    ]
    if out.empty:
        return pd.DataFrame(columns=expected_cols)


    # Dimension scores (0–1): mean of available items
    out["human_findability_0_1"] = out.reindex(columns=["find_title", "find_desc", "find_keywords"]).mean(axis=1, skipna=True)
    out["human_clarity_0_1"] = out[["clarity_structured"]].mean(axis=1, skipna=True)
    out["human_comprehensibility_0_1"] = out[[
        "comp_target_defined", "comp_audience_matches", "comp_jargon_explained", "comp_standalone"
    ]].mean(axis=1, skipna=True)
    out["human_usability_0_1"] = out[["use_context"]].mean(axis=1, skipna=True)
    out["human_credibility_0_1"] = out[["cred_sources"]].mean(axis=1, skipna=True)
    out["human_reusability_0_1"] = out[["reuse_licence"]].mean(axis=1, skipna=True)
    out["human_recommend_0_1"] = out["recommend_1_5"].apply(lambda v: np.nan if np.isnan(v) else max(0.0, min(1.0, (v - 1.0) / 4.0)))

    return out

def load_all_human_reviews(
    xlsx_path: Path,
    include_collating: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (per_review_rows, per_ko_consensus).

    The per-review frame keeps one row per (KO, reviewer) so inter-rater agreement can
    be measured; the consensus frame averages those rows per KO.
    """
    xls = pd.ExcelFile(xlsx_path)
    all_rows = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if not is_reviewer_sheet(df):
            continue
        if sheet in CONSOLIDATED_SHEETS and not include_collating:
            print(f"[human] sheet={sheet} SKIPPED (consolidated; use --include-collating)")
            continue
        parsed = parse_reviewer_sheet(sheet, df)
        print(f"[human] sheet={sheet} rows={len(parsed)}")
        all_rows.append(parsed)

    if not all_rows:
        raise RuntimeError("No reviewer sheets detected. Check sheet formatting or heuristics.")

    long_df = pd.concat(all_rows, ignore_index=True)

    agg_cols = [c for c in long_df.columns if c not in {"reviewer"}]
    human_agg = long_df.groupby("ko_id", as_index=False)[agg_cols].mean(numeric_only=True)
    human_agg["n_reviewers"] = long_df.groupby("ko_id").size().reindex(human_agg["ko_id"]).to_numpy()

    return long_df, human_agg


# -----------------------------
# Inter-rater agreement (the ceiling for any automatic scorer)
# -----------------------------
HUMAN_DIMS = [
    "human_findability_0_1",
    "human_clarity_0_1",
    "human_comprehensibility_0_1",
    "human_usability_0_1",
    "human_recommend_0_1",
]


def inter_rater_agreement(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    For KOs reviewed by 2+ people, correlate reviewer 1 against reviewer 2.

    This is the headline context for everything below: an automatic scorer cannot be
    expected to track "human judgment" more closely than two humans track each other.
    Reviewer order is the sheet order, which is arbitrary but consistent.
    """
    rows = []
    two_plus = long_df.groupby("ko_id").filter(lambda g: len(g) >= 2)
    for dim in HUMAN_DIMS:
        if dim not in two_plus.columns:
            continue
        first, second = [], []
        for _, g in two_plus.groupby("ko_id"):
            vals = g[dim].dropna().to_numpy()
            if len(vals) >= 2:
                first.append(vals[0])
                second.append(vals[1])
        n = len(first)
        if n < 3:
            rows.append({"dimension": dim, "n_KOs": n, "pearson": np.nan,
                         "spearman": np.nan, "exact_agreement_pct": np.nan})
            continue
        a, b = np.array(first), np.array(second)
        with np.errstate(invalid="ignore"):
            pear = stats.pearsonr(a, b)[0] if a.std() and b.std() else np.nan
            spear = stats.spearmanr(a, b)[0] if a.std() and b.std() else np.nan
        rows.append({
            "dimension": dim,
            "n_KOs": n,
            "pearson": pear,
            "spearman": spear,
            "exact_agreement_pct": float((a == b).mean()) * 100.0,
        })
    return pd.DataFrame(rows)


def pillar_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Correlate every automatic pillar and the overall total against every human
    dimension. This is deliberately broader than the hand-built proxy mapping: if a
    pillar tracks human judgment at all, it shows up here without being told where.
    """
    auto_cols = [c for c in [
        "auto_structural_0_1", "auto_semantic_0_1", "auto_functional_0_1",
        "auto_domain_0_1", "auto_total_weighted_0_1",
    ] if c in merged.columns]

    rows = []
    for h in HUMAN_DIMS:
        if h not in merged.columns:
            continue
        for a in auto_cols:
            x = pd.to_numeric(merged[h], errors="coerce")
            y = pd.to_numeric(merged[a], errors="coerce")
            ok = x.notna() & y.notna()
            n = int(ok.sum())
            if n < 3 or x[ok].std() == 0 or y[ok].std() == 0:
                rows.append({"human_metric": h, "auto_metric": a, "n": n,
                             "pearson": np.nan, "spearman": np.nan, "p_spearman": np.nan})
                continue
            pear = stats.pearsonr(x[ok], y[ok])
            spear = stats.spearmanr(x[ok], y[ok])
            rows.append({
                "human_metric": h, "auto_metric": a, "n": n,
                "pearson": pear[0], "spearman": spear[0], "p_spearman": spear[1],
            })
    return pd.DataFrame(rows)


# -----------------------------
# Automated mapping
# -----------------------------
def compute_auto_dimension_scores(auto_df: pd.DataFrame) -> pd.DataFrame:
    df = auto_df.copy()

    # Normalise “known scales”
    df["auto_structural_0_1"] = df["Structural_Score_0_25"].apply(normalise_0_25_to_0_1)
    df["auto_semantic_0_1"] = df["Semantic_Score_0_25"].apply(normalise_0_25_to_0_1)
    df["auto_domain_0_1"] = df["Domain_Score_0_25"].apply(normalise_0_25_to_0_1)
    df["auto_functional_0_1"] = df["Functional_Score_0_25"].apply(normalise_0_25_to_0_1)

    df["auto_total_weighted_0_1"] = df["Total_Quality_weighted_0_100"].apply(normalise_0_100_to_0_1)

    # Some sub-metrics look like 0–5 (e.g., Domain_in_title, Functional_keyword_indexability)
    df["Domain_in_title_0_1"] = df["Domain_in_title"].apply(normalise_0_5_to_0_1)
    df["Domain_in_keywords_0_1"] = df["Domain_in_keywords"].apply(normalise_0_5_to_0_1)
    df["Functional_keyword_indexability_0_1"] = df["Functional_keyword_indexability"].apply(normalise_0_5_to_0_1)

    # Similarity fields appear already 0–1; coerce to float safely.
    for c in ["Domain_similarity_title", "Domain_similarity_desc", "Domain_similarity_keywords", "Domain_similarity_content"]:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)

    # Dimension proxies (0–1)
    # NOTE: These are working proxies, not “ground truth”.
    df["auto_findability_0_1"] = np.nanmean(
        np.vstack([
            df["auto_functional_0_1"].to_numpy(),
            df["Functional_keyword_indexability_0_1"].to_numpy(),
            df["Domain_in_title_0_1"].to_numpy(),
            df["Domain_in_keywords_0_1"].to_numpy(),
            df["Domain_similarity_title"].to_numpy(),
            df["Domain_similarity_desc"].to_numpy(),
            df["Domain_similarity_keywords"].to_numpy(),
        ]),
        axis=0
    )

    df["auto_clarity_0_1"] = np.nanmean(
        np.vstack([
            df["auto_structural_0_1"].to_numpy(),
            df["auto_semantic_0_1"].to_numpy(),
        ]),
        axis=0
    )

    df["Semantic_usefulness_0_1"] = df["Semantic_usefulness"].apply(normalise_0_5_to_0_1)

    df["auto_comprehensibility_0_1"] = np.nanmean(
        np.vstack([
            df["auto_semantic_0_1"].to_numpy(),
            df["auto_structural_0_1"].to_numpy(),
            df["Semantic_usefulness_0_1"].to_numpy(),
        ]),
        axis=0
    )

    df["Functional_RAG_readiness_0_1"] = df["Functional_RAG_readiness"].apply(normalise_0_5_to_0_1)

    df["auto_usability_0_1"] = np.nanmean(
        np.vstack([
            df["Functional_RAG_readiness_0_1"].to_numpy(),
            df["Semantic_usefulness_0_1"].to_numpy(),
        ]),
        axis=0
    )

    # Weak proxies / placeholders (flag these clearly in outputs)
    # Domain_term_density and Domain_consistency are 0-5 sub-scores, so they have to be
    # normalised before they can sit in a column named *_0_1.
    df["auto_credibility_proxy_0_1"] = np.nanmean(
        np.vstack([
            df["Domain_term_density"].apply(normalise_0_5_to_0_1).to_numpy(),
            df["Domain_consistency"].apply(normalise_0_5_to_0_1).to_numpy(),
        ]),
        axis=0
    )

    # No real licence detector in this TSV
    df["auto_reusability_proxy_0_1"] = np.nan

    return df


# -----------------------------
# Comparison / output
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", type=Path, default=HUMAN_XLSX, help="Human review workbook (.xlsx)")
    ap.add_argument("--auto", type=Path, default=None,
                    help="Assessor TSV. Default: newest in ./output")
    ap.add_argument("--out", type=Path, default=OUT_XLSX, help="Output .xlsx")
    ap.add_argument("--include-collating", action="store_true",
                    help="Also treat the consolidated 'collating' sheet as a reviewer")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    auto_path = args.auto if args.auto is not None else latest_auto_tsv()
    print(f"[human] workbook: {args.human}")
    print(f"[auto ] scores  : {auto_path}")

    long_human, human = load_all_human_reviews(args.human, include_collating=args.include_collating)

    auto = pd.read_csv(auto_path, sep="\t", dtype=str)
    auto = compute_auto_dimension_scores(auto)

    merged = human.merge(auto, left_on="ko_id", right_on="_orig_id", how="inner")

    # Coverage has to be stated, not assumed: the review is from 2024 and KOs drop out
    # of later exports, so the join is always a subset of what was reviewed.
    print("\n=== Coverage ===")
    print(f"  KOs human-reviewed      : {len(human)}")
    print(f"  KOs in the assessor run : {len(auto)}")
    print(f"  KOs in both (analysed)  : {len(merged)}")
    missing = sorted(set(human['ko_id']) - set(auto['_orig_id'].astype(str)))
    if missing:
        print(f"  reviewed but not scored : {len(missing)}  e.g. {missing[:3]}")
    if "n_reviewers" in merged.columns:
        print(f"  reviewers per KO        : {merged['n_reviewers'].value_counts().sort_index().to_dict()}")

    ceiling = inter_rater_agreement(long_human)
    pillars = pillar_correlations(merged)

    # Correlations for dimension scores
    dim_pairs = [
        ("human_findability_0_1", "auto_findability_0_1"),
        ("human_clarity_0_1", "auto_clarity_0_1"),
        ("human_comprehensibility_0_1", "auto_comprehensibility_0_1"),
        ("human_usability_0_1", "auto_usability_0_1"),
        ("human_recommend_0_1", "auto_total_weighted_0_1"),
    ]

    corr_rows = []
    for h, a in dim_pairs:
        x = merged[h].astype(float)
        y = merged[a].astype(float)
        corr_rows.append({
            "human_metric": h,
            "auto_metric": a,
            "pearson": x.corr(y, method="pearson"),
            "spearman": x.corr(y, method="spearman"),
            "n": int((~x.isna() & ~y.isna()).sum()),
        })
    corr_df = pd.DataFrame(corr_rows)

    # Thresholded agreement for binary dimensions (human mean >= 0.5 treated as “overall yes”)
    # You can tune thresholds per dimension if you want.
    threshold = 0.60
    agree_rows = []
    for h, a in dim_pairs[:-1]:
        human_yes = merged[h].astype(float) >= 0.5
        auto_yes = merged[a].astype(float) >= threshold
        valid = (~merged[h].isna()) & (~merged[a].isna())
        if valid.sum() == 0:
            continue
        agree_rows.append({
            "dimension": h.replace("human_", "").replace("_0_1", ""),
            "threshold_auto": threshold,
            "agreement_pct": float((human_yes[valid] == auto_yes[valid]).mean()) * 100.0,
            "n": int(valid.sum()),
        })
    agree_df = pd.DataFrame(agree_rows)

    print_diagnostics(merged, corr_df, agree_df)

    print("\n=== Inter-rater agreement (the ceiling for any automatic scorer) ===")
    print(ceiling.round(3).to_string(index=False))

    print("\n=== Every automatic pillar vs every human dimension ===")
    piv = pillars.pivot(index="auto_metric", columns="human_metric", values="spearman")
    print("Spearman:")
    print(piv.round(3).to_string())
    best = pillars.dropna(subset=["spearman"]).reindex(
        pillars.dropna(subset=["spearman"])["spearman"].abs().sort_values(ascending=False).index
    ).head(5)
    print("\nStrongest associations:")
    print(best.round(4).to_string(index=False))

    with pd.ExcelWriter(args.out, engine="openpyxl") as w:
        merged.to_excel(w, index=False, sheet_name="merged_human_auto")
        corr_df.to_excel(w, index=False, sheet_name="dimension_correlations")
        agree_df.to_excel(w, index=False, sheet_name="threshold_agreement")
        ceiling.to_excel(w, index=False, sheet_name="inter_rater_ceiling")
        pillars.to_excel(w, index=False, sheet_name="pillar_correlations")
        long_human.to_excel(w, index=False, sheet_name="human_per_review")

    print(f"\nWrote: {args.out.resolve()}")
    print(f"Merged rows: {len(merged)} / Human KOs: {len(human)} / Auto KOs: {len(auto)}")


if __name__ == "__main__":
    main()
