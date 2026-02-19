# assess_ko_quality/ko_quality_assessor_kc.py
"""
KO Quality Assessor using improved quality modules.

This version uses the improved quality modules:
- quality_structural_new (cached metrics, configurable thresholds, comprehensive diagnostics)
- quality_semantic_new (cached spaCy, token-aware truncation, batched inference)
- quality_domain_new (batched embeddings, token-aware truncation, short-circuit optimization)
- quality_functional_new (cached metrics, internal 0-10 scaling, case-insensitive matching)

Usage:
    python ko_quality_assessor_new.py --input-dir ./input --output-dir ./output
    
Environment variables:
    KO_INPUT_DIR: Input directory (default: ./input)
    KO_OUTPUT_DIR: Output directory (default: ./output)
    AGRI_DOMAIN_CENTROID: Path to domain centroid .npy file
    W_STRUCT, W_SEM, W_FUNC, W_DOM: Scoring weights (must sum to 100)
"""

# Quality = Structural + Semantic + Domain + Functional
# Structural = length + completeness + noise + formatting
# Semantic   = clarity + usefulness + information density + consistency
# Domain     = correct agricultural terminology and context (embedding-based)
# Functional = works well for BM25, embeddings, hybrid search, RAG (proxies)

import argparse
import os
import csv

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from utils import _latest_json_file, ensure_directory, assert_readable_dir, _read_json_any, _unique_outfile

from quality_text_utils import norm_text, detect_lang_safe, _ensure_str_list
from quality_structural_kc import structural_scores
from quality_semantic_kc import semantic_scores, semantic_mnli_consistency
from quality_domain_kc import domain_scores, load_domain_centroid
from quality_functional_kc import functional_scores


# ---------- Config: I/O ----------
INPUT_FOLDER = Path(os.environ.get("KO_INPUT_DIR", "./input")).resolve()
OUTPUT_FOLDER = Path(os.environ.get("KO_OUTPUT_DIR", "./output")).resolve()

# ---------- Scoring weights ----------
# All pillar scores are in 0–25. We convert to 0–100 and then apply weights.
# Weights must sum to 100.
WEIGHTS = {
    "structural": int(os.environ.get("W_STRUCT", "30")),
    "semantic": int(os.environ.get("W_SEM", "35")),
    "functional": int(os.environ.get("W_FUNC", "25")),
    "domain": int(os.environ.get("W_DOM", "10")),
}

if sum(WEIGHTS.values()) != 100:
    raise ValueError(f"Invalid weights (must sum to 100): {WEIGHTS}")


# ---------- Per-KO assessor ----------
def assess_ko(ko: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess a single KO using improved quality modules.
    
    Args:
        ko: Knowledge Object dictionary with keys:
            - _orig_id or @id: identifier
            - title: KO title
            - subtitle: KO subtitle (optional)
            - description: KO description
            - keywords: list of keywords
            - ko_content_flat: body content
    
    Returns:
        Dictionary with all quality scores, diagnostics, and aggregates
    """
    _id = ko.get("_orig_id") or ko.get("@id") or ""
    title = norm_text(ko.get("title"))
    subtitle = norm_text(ko.get("subtitle"))
    desc = norm_text(ko.get("description"))
    content = norm_text(ko.get("ko_content_flat"))
    keywords = [norm_text(x) for x in _ensure_str_list(ko.get("keywords")) if norm_text(x)]

    meta_text = " ".join([title, subtitle, desc])
    lang_probe = (meta_text + " " + content[:500]).strip()
    lang_meta = detect_lang_safe(lang_probe) if len(lang_probe) >= 50 else "unknown"

    # --- Compute all quality scores ---
    # Each returns comprehensive diagnostics beyond just the main scores
    struct = structural_scores(title, subtitle, desc, content, keywords)
    sem = semantic_scores(title, subtitle, desc, content, keywords)
    dom = domain_scores(title, desc, content, keywords)
    func = functional_scores(title, desc, content, keywords)

    # MNLI-based semantic consistency (content → title/desc/subtitle)
    mnli_sem = semantic_mnli_consistency(
        title=title,
        subtitle=subtitle,
        desc=desc,
        content=content,
        lang_meta=lang_meta,
    )

    # --- Aggregate scores ---
    # Unweighted total (legacy): 0–100 because 4 pillars × 25
    total_unweighted_0_100 = (
        struct["Structural_Score_0_25"]
        + sem["Semantic_Score_0_25"]
        + dom["Domain_Score_0_25"]
        + func["Functional_Score_0_25"]
    )

    # Weighted total: each pillar 0–25 -> 0–100, then weighted sum -> 0–100
    s_struct = struct["Structural_Score_0_25"] * 4.0
    s_sem = sem["Semantic_Score_0_25"] * 4.0
    s_func = func["Functional_Score_0_25"] * 4.0
    s_dom = dom["Domain_Score_0_25"] * 4.0

    total_weighted_0_100 = round(
        (WEIGHTS["structural"] * s_struct
         + WEIGHTS["semantic"] * s_sem
         + WEIGHTS["functional"] * s_func
         + WEIGHTS["domain"] * s_dom) / 100.0,
        2
    )

    # --- Build notes ---
    notes: List[str] = []
    if not title:
        notes.append("Missing title")
    if not desc:
        notes.append("Missing description")
    if not content:
        notes.append("Missing content")
    if len(keywords) < 2:
        notes.append("Few keywords (<2)")
    if lang_meta != "en":
        notes.append(f"Detected non-EN metadata language: {lang_meta}")
    
    # Add any diagnostics from domain scoring
    if dom.get("Domain_content_truncated"):
        notes.append("Content was truncated for domain scoring")

    return {
        "_orig_id": _id,
        "title": title[:300],
        "lang_meta_detected": lang_meta,

        # Structural (with new diagnostics)
        **struct,
        # Semantic (lexical-based, with new diagnostics)
        **sem,
        # MNLI-based semantic consistency diagnostics
        **mnli_sem,
        # Domain (with new diagnostics)
        **dom,
        # Functional (with new diagnostics)
        **func,

        # Overall
        "Total_Quality_unweighted_0_100": total_unweighted_0_100,
        "Total_Quality_weighted_0_100": total_weighted_0_100,
        "Weights_used": f"S{WEIGHTS['structural']}/Se{WEIGHTS['semantic']}/F{WEIGHTS['functional']}/D{WEIGHTS['domain']}",
        "Notes": "; ".join(notes),
    }


def parse_args() -> argparse.Namespace:
    """
    CLI args:
      --input: explicit JSON/JSONL/NDJSON(.gz) file path to score
      --input-dir: override KO_INPUT_DIR directory scanning (default: env or ./input)
      --output-dir: override KO_OUTPUT_DIR (default: env or ./output)
    """
    p = argparse.ArgumentParser(
        description="Assess KO quality using improved modules and write TSV report."
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a specific input JSON/JSONL/NDJSON file (optionally .gz). "
             "If omitted, the latest file from --input-dir is used.",
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default=str(INPUT_FOLDER),
        help="Directory to scan for latest input file when --input is not provided.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_FOLDER),
        help="Directory to write the output TSV file.",
    )
    return p.parse_args()


# ---------- Driver ----------
def main() -> None:
    """
    Read the latest JSON/NDJSON from INPUT_FOLDER and write quality TSV to OUTPUT_FOLDER.
    Uses improved quality modules with better performance and diagnostics.
    """
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    assert_readable_dir(input_dir)
    ensure_directory(output_dir)

    if args.input:
        latest = Path(args.input).resolve()
        if not latest.exists():
            raise FileNotFoundError(f"Input file not found: {latest}")
        print(f"[INFO] Using explicit input file: {latest}")
    else:
        latest = _latest_json_file(str(input_dir))
        print(f"[INFO] Using latest file: {latest}")

    # --- First pass: load all KOs into memory ---
    all_kos: List[Dict[str, Any]] = list(_read_json_any(latest))
    if not all_kos:
        raise RuntimeError(f"No valid JSON objects found in input file: {latest}")
    print(f"[INFO] Loaded {len(all_kos)} KOs")

    # --- Load precomputed agriculture domain centroid ---
    centroid_path = os.environ.get(
        "AGRI_DOMAIN_CENTROID",
        str((Path(__file__).resolve().parent / "anchors" / "centroids" / "agri_anchor_centroid.npy"))
    )
    load_domain_centroid(centroid_path)

    # --- Second pass: actually score each KO ---
    rows: List[Dict[str, Any]] = []
    count = 0
    errors = 0

    print(f"[INFO] Starting quality assessment with weights: S{WEIGHTS['structural']}/Se{WEIGHTS['semantic']}/F{WEIGHTS['functional']}/D{WEIGHTS['domain']}")

    for ko in all_kos:
        try:
            rows.append(assess_ko(ko))
        except Exception as e:
            errors += 1
            rid = ko.get("_orig_id") or ko.get("@id") or f"row_{count}"
            rows.append({
                "_orig_id": rid,
                "title": ko.get("title", ""),
                "lang_meta_detected": "unknown",
                "Total_Quality_unweighted_0_100": 0,
                "Total_Quality_weighted_0_100": 0,
                "Weights_used": f"S{WEIGHTS['structural']}/Se{WEIGHTS['semantic']}/F{WEIGHTS['functional']}/D{WEIGHTS['domain']}",
                "Notes": f"ERROR: {type(e).__name__}: {e}",
            })

        count += 1
        if count % 500 == 0:
            print(f"[INFO] Processed {count} KOs...")

    if count == 0:
        raise RuntimeError(f"No valid JSON objects found in input file: {latest}")

    df = pd.DataFrame(rows)
    out_path = _unique_outfile(output_dir, stem="quality_check_new", ext=".tsv")
    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        lineterminator="\n",
        encoding="utf-8",
    )
    print(f"[OK] Wrote {len(df)} rows -> {out_path}")
    if errors:
        print(f"[WARN] {errors} item(s) had exceptions; see 'Notes' column for details.")
    else:
        print(f"[OK] All {len(df)} KOs processed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        raise
