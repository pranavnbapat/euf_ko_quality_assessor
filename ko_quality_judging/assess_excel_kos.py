#!/usr/bin/env python3
"""Run KO metadata quality checks on a fixed list of KOs supplied in an Excel file.

Unlike ``run_metadata_quality_assessment.py`` (which randomly samples eligible
English PDFs), this script judges *exactly* the KOs listed in an Excel export.
For each Excel row it:

  1. matches the row to a JSON record by ``_id`` / ``_orig_id``;
  2. verifies the Excel original metadata (title/subtitle/description/keywords)
     matches the JSON original fields and flags any that differ (records are
     never dropped for mismatching -- they are included and flagged);
  3. reuses the automated + LLM-judge scoring machinery from
     ``run_metadata_quality_assessment.py`` to compare original vs improved
     (``*_llm``) metadata.

KOs listed in the Excel but absent from the JSON are reported separately and
cannot be scored.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_metadata_quality_assessment as base  # noqa: E402

PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_EXCEL = PROJECT_ROOT / "KO_2025 ready for amb.xlsx"
DEFAULT_INPUT = PROJECT_ROOT / "input" / "final_improved_mysql_export_18_06-2026_08-00-14.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_SUBSET_CACHE = SCRIPT_DIR / "matched_excel_kos.json"

ID_RE = re.compile(r'"_id"\s*:\s*"([0-9a-f]{24})"')
MATCH_FIELDS = ("title", "subtitle", "description", "keywords")


# --------------------------------------------------------------------------- #
# Excel reading
# --------------------------------------------------------------------------- #
def read_excel_originals(path: Path, sheet: str, header_row: int) -> Dict[str, Dict[str, Any]]:
    df = pd.read_excel(path, sheet_name=sheet, header=header_row)
    if "_id" not in df.columns:
        raise KeyError(
            f"Column '_id' not found on sheet {sheet!r} with header row {header_row}. "
            f"Columns seen: {list(df.columns)[:15]}"
        )
    df = df[df["_id"].notna()]
    originals: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        rid = str(row["_id"]).strip()
        if not rid:
            continue
        originals[rid] = {field: row.get(field) for field in MATCH_FIELDS}
    return originals


# --------------------------------------------------------------------------- #
# JSON subset extraction (stream the large export, keep only wanted ids)
# --------------------------------------------------------------------------- #
def stream_matched_records(json_path: Path, wanted_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    data = json_path.read_text(encoding="utf-8")
    start = data.find("[", data.find('"docs"'))
    if start == -1:
        # plain list root or single object -> fall back to base loader
        records = base.load_records(json_path)
        return {
            base.clean_text(r.get("_id")) or base.clean_text(r.get("_orig_id")): r
            for r in records
            if (base.clean_text(r.get("_id")) or base.clean_text(r.get("_orig_id"))) in wanted_ids
        }

    i = start + 1
    length = len(data)
    found: Dict[str, Dict[str, Any]] = {}
    while i < length:
        while i < length and data[i] not in "{]":
            i += 1
        if i >= length or data[i] == "]":
            break
        depth = 0
        in_str = False
        esc = False
        obj_start = i
        while i < length:
            ch = data[i]
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
            i += 1
        chunk = data[obj_start:i]
        match = ID_RE.search(chunk[:400])
        if match and match.group(1) in wanted_ids:
            found[match.group(1)] = json.loads(chunk)
    return found


def load_or_build_subset(
    json_path: Path, wanted_ids: set[str], cache: Path, rebuild: bool
) -> Dict[str, Dict[str, Any]]:
    if cache.exists() and not rebuild:
        records = json.loads(cache.read_text(encoding="utf-8"))
        by_id = {
            base.clean_text(r.get("_id")) or base.clean_text(r.get("_orig_id")): r
            for r in records
        }
        if wanted_ids <= set(by_id):
            return {rid: by_id[rid] for rid in wanted_ids if rid in by_id}
        print("[INFO] Cache does not cover all requested ids; rebuilding from JSON export.")
    found = stream_matched_records(json_path, wanted_ids)
    cache.write_text(json.dumps(list(found.values()), ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Cached {len(found)} matched records -> {cache}")
    return found


# --------------------------------------------------------------------------- #
# Excel-vs-JSON original match flagging
# --------------------------------------------------------------------------- #
def norm_text(value: Any) -> str:
    return base.clean_text(value).lower()


def parse_keywords(value: Any) -> Tuple[str, ...]:
    """Order-independent keyword set from a real list, stringified list, or CSV."""
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = base.clean_text(value)
        if text.startswith("["):
            try:
                items = list(ast.literal_eval(text))
            except Exception:
                items = re.split(r"[,;\n]", text)
        else:
            items = re.split(r"[,;\n]", text)
    cleaned = [norm_text(item).strip("[]'\" ") for item in items]
    return tuple(sorted(item for item in cleaned if item))


def field_matches(field: str, excel_value: Any, json_value: Any) -> bool:
    if field == "keywords":
        return parse_keywords(excel_value) == parse_keywords(json_value)
    return norm_text(excel_value) == norm_text(json_value)


def match_flags(excel_original: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    mismatched = [
        field
        for field in MATCH_FIELDS
        if not field_matches(field, excel_original.get(field), record.get(field))
    ]
    return {
        "excel_all_match": len(mismatched) == 0,
        "excel_mismatch_fields": "; ".join(mismatched),
        "excel_title": base.clean_text(excel_original.get("title")),
        "excel_subtitle": base.clean_text(excel_original.get("subtitle")),
        "excel_description": base.clean_text(excel_original.get("description")),
        "excel_keywords": base.clean_text(excel_original.get("keywords")),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--header-row", type=int, default=2, help="0-indexed header row in the sheet.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Large JSON export.")
    parser.add_argument("--subset-cache", type=Path, default=DEFAULT_SUBSET_CACHE)
    parser.add_argument("--rebuild-subset", action="store_true", help="Ignore cache; re-scan the JSON export.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--no-embeddings", action="store_true", help="Force TF-IDF cosine similarity.")
    parser.add_argument("--no-llm-judge", action="store_true", help="Automated scoring only; skip the LLM judge.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-content-chars", type=int, default=4000)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--judge-jsonl", type=Path, default=None)
    parser.add_argument("--no-private-fallback", action="store_true")
    return parser.parse_args()


def default_output_path(count: int) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return DEFAULT_OUTPUT_DIR / f"ko_metadata_quality_excel_{count}_{stamp}.xlsx"


def main() -> None:
    args = parse_args()

    excel_originals = read_excel_originals(args.excel, args.sheet, args.header_row)
    wanted_ids = set(excel_originals)
    print(f"Excel KOs (rows with _id): {len(wanted_ids)}")

    matched = load_or_build_subset(args.input, wanted_ids, args.subset_cache, args.rebuild_subset)
    missing_ids = sorted(wanted_ids - set(matched))
    print(f"Matched in JSON: {len(matched)}")
    print(f"Missing from JSON (cannot score): {len(missing_ids)}")
    for rid in missing_ids:
        print(f"  missing: {rid}")

    # Deterministic order: follow Excel order.
    ordered_ids = [rid for rid in excel_originals if rid in matched]
    records = [matched[rid] for rid in ordered_ids]
    if not records:
        raise RuntimeError("No Excel KOs could be matched to the JSON export.")

    output = args.output or default_output_path(len(records))

    df, method = base.automated_score_records(records, args.embedding_model, args.no_embeddings)
    df["ko_content_flat_full"] = [base.selected_content(record)[0] for record in records]

    # Attach Excel-vs-JSON original match flags, keyed by record id.
    flag_rows = [match_flags(excel_originals[rid], matched[rid]) for rid in ordered_ids]
    flags_df = pd.DataFrame(flag_rows, index=df.index)
    df = pd.concat([df, flags_df], axis=1)

    n_mismatch = int((~df["excel_all_match"]).sum())
    print(f"Similarity method: {method}")
    print(f"KOs flagged (Excel original != JSON original on >=1 field): {n_mismatch}")

    if not args.no_llm_judge:
        df = base.run_llm_judge(df, args, output)

    base.write_excel(df, output)

    if missing_ids:
        missing_path = output.with_suffix(".missing.json")
        missing_path.write_text(json.dumps(missing_ids, indent=2), encoding="utf-8")
        print(f"Missing-id list: {missing_path}")

    print(f"Output: {output}")


if __name__ == "__main__":
    main()
