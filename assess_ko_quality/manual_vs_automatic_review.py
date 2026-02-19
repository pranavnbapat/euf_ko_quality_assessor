# assess_ko_quality/compare_human_vs_auto.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


HUMAN_XLSX = Path("2024_KOs for assessment all collating.xlsx")
AUTO_TSV = Path("output/quality_check_20260109_201910.tsv")
OUT_XLSX = Path("human_vs_auto_comparison.xlsx")


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

    for r in range(SEARCH_ROWS):
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

def load_all_human_reviews(xlsx_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    all_rows = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if is_reviewer_sheet(df):
            parsed = parse_reviewer_sheet(sheet, df)
            print(f"[human] sheet={sheet} rows={len(parsed)}")
            all_rows.append(parsed)

    if not all_rows:
        raise RuntimeError("No reviewer sheets detected. Check sheet formatting or heuristics.")

    long_df = pd.concat(all_rows, ignore_index=True)

    # Aggregate across reviewers per KO
    agg_cols = [c for c in long_df.columns if c not in {"reviewer"}]
    # For binaries/dimension scores: mean; for recommend: mean (you can change to median if you prefer)
    human_agg = long_df.groupby("ko_id", as_index=False)[agg_cols].mean(numeric_only=True)

    return human_agg


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
    df["auto_credibility_proxy_0_1"] = np.nanmean(
        np.vstack([
            df["Domain_term_density"].apply(safe_float).to_numpy(),
            df["Domain_consistency"].apply(safe_float).to_numpy(),
        ]),
        axis=0
    )

    # No real licence detector in this TSV
    df["auto_reusability_proxy_0_1"] = np.nan

    return df


# -----------------------------
# Comparison / output
# -----------------------------
def main() -> None:
    human = load_all_human_reviews(HUMAN_XLSX)

    auto = pd.read_csv(AUTO_TSV, sep="\t", dtype=str)
    auto = compute_auto_dimension_scores(auto)

    merged = human.merge(auto, left_on="ko_id", right_on="_orig_id", how="inner")

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

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        merged.to_excel(w, index=False, sheet_name="merged_human_auto")
        corr_df.to_excel(w, index=False, sheet_name="dimension_correlations")
        agree_df.to_excel(w, index=False, sheet_name="threshold_agreement")

    print(f"Wrote: {OUT_XLSX.resolve()}")
    print(f"Merged rows: {len(merged)} / Human KOs: {len(human)} / Auto KOs: {len(auto)}")


if __name__ == "__main__":
    main()
