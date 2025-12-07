# assess_ko_quality/ko_quality_assessor.py

# Quality = Structural + Semantic + Domain + Functional
# Structural = length + completeness + noise + formatting
# Semantic   = clarity + usefulness + information density + consistency
# Domain     = correct agricultural terminology and context (heuristic)
# Functional = works well for BM25, embeddings, hybrid search, RAG (proxies)

import os
import csv

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from utils import _latest_json_file, ensure_directory, assert_readable_dir, _read_json_any, _unique_outfile

from quality_text_utils import norm_text, detect_lang_safe, _ensure_str_list
from quality_structural import structural_scores
from quality_semantic import semantic_scores, semantic_mnli_consistency
from quality_domain import domain_scores, initialise_domain_centroid
from quality_functional import functional_scores


# ---------- Config: I/O ----------
INPUT_FOLDER = Path(os.environ.get("KO_INPUT_DIR", "./input")).resolve()
OUTPUT_FOLDER = Path(os.environ.get("KO_OUTPUT_DIR", "./output")).resolve()


# ---------- Per-KO assessor ----------
def assess_ko(ko: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess a single KO based on:
      - title
      - subtitle
      - description
      - keywords
      - ko_content_flat
    """
    _id = ko.get("_orig_id") or ko.get("@id") or ""
    title = norm_text(ko.get("title"))
    subtitle = norm_text(ko.get("subtitle"))
    desc = norm_text(ko.get("description"))
    content = norm_text(ko.get("ko_content_flat"))
    keywords = [norm_text(x) for x in _ensure_str_list(ko.get("keywords")) if norm_text(x)]

    meta_text = " ".join([title, subtitle, desc])
    lang_meta = detect_lang_safe(meta_text) if meta_text else "unknown"

    # Scores
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

    total_0_100 = (
        struct["Structural_Score_0_25"]
        + sem["Semantic_Score_0_25"]
        + dom["Domain_Score_0_25"]
        + func["Functional_Score_0_25"]
    )

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

    return {
        "_orig_id": _id,
        "title": title[:300],
        "lang_meta_detected": lang_meta,

        # Structural
        **struct,
        # Semantic (lexical-based, as before)
        **sem,
        # MNLI-based semantic consistency diagnostics
        **mnli_sem,
        # Domain
        **dom,
        # Functional
        **func,

        # Overall
        "Total_Quality_0_100": total_0_100,
        "Notes": "; ".join(notes),
    }

# ---------- Driver ----------

def main() -> None:
    """
    Read the latest JSON/NDJSON from INPUT_FOLDER and write quality TSV to OUTPUT_FOLDER.
    """
    assert_readable_dir(INPUT_FOLDER)
    ensure_directory(OUTPUT_FOLDER)

    latest = _latest_json_file(str(INPUT_FOLDER))
    print(f"[INFO] Using latest file: {latest}")

    # --- First pass: load all KOs into memory ---
    all_kos: List[Dict[str, Any]] = list(_read_json_any(latest))
    if not all_kos:
        raise RuntimeError(f"No valid JSON objects found in input file: {latest}")

    # --- Build domain centroid from KO content (data-driven) ---
    domain_texts: List[str] = []
    for ko in all_kos:
        content = norm_text(ko.get("ko_content_flat"))
        if content:
            domain_texts.append(content)

    initialise_domain_centroid(domain_texts)

    # --- Second pass: actually score each KO ---
    rows: List[Dict[str, Any]] = []
    count = 0
    errors = 0

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
                "Total_Quality_0_100": 0,
                "Notes": f"ERROR: {type(e).__name__}: {e}",
            })
        count += 1
        if count % 500 == 0:
            print(f"[INFO] Processed {count} KOs...")

    if count == 0:
        raise RuntimeError(f"No valid JSON objects found in input file: {latest}")

    df = pd.DataFrame(rows)
    out_path = _unique_outfile(OUTPUT_FOLDER, stem="quality_check", ext=".tsv")
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

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        raise
