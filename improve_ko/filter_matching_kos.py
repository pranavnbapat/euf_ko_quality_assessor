# improve_ko/filter_matching_kos.py

import json
import sys

from pathlib import Path


def main(path_with_summaries: str, path_only_flat: str, out_path: str) -> None:
    p_with_summaries = Path(path_with_summaries)
    p_only_flat = Path(path_only_flat)
    p_out = Path(out_path)

    # --- Load JSON files (assumed to be arrays of objects) ---
    with p_with_summaries.open("r", encoding="utf-8") as f:
        data_with_summaries = json.load(f)

    with p_only_flat.open("r", encoding="utf-8") as f:
        data_only_flat = json.load(f)

    # --- Build a set of _orig_id values that have a summary ---
    # We also check `.get("ko_content_flat_summarised")` to avoid empty / null.
    # --- Build sets of _orig_id from both files ---
    # IDs that exist in the flat-only file
    orig_ids_in_flat = {
        rec["_orig_id"]
        for rec in data_only_flat
        if rec.get("_orig_id") is not None
    }

    # IDs that have a non-empty summary in the merged file
    orig_ids_with_summary = {
        rec["_orig_id"]
        for rec in data_with_summaries
        if rec.get("_orig_id") is not None
           and rec.get("ko_content_flat_summarised")
    }

    # We only keep IDs that are in BOTH:
    ids_to_keep = orig_ids_in_flat & orig_ids_with_summary

    print(f"Found {len(orig_ids_with_summary)} records with summaries.")
    print(f"{len(ids_to_keep)} IDs are common between both files.")

    # --- NOW: take records from the MERGED file (with summaries) ---
    filtered = [
        rec
        for rec in data_with_summaries
        if rec.get("_orig_id") in ids_to_keep
    ]

    print(f"Keeping {len(filtered)} records out of {len(data_with_summaries)}.")

    # --- Write output as a JSON array ---
    with p_out.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Wrote output to {p_out}")

if __name__ == "__main__":
    # Usage: python filter_matching_kos.py with_summaries.json only_flat.json output.json
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python filter_matching_kos.py "
            "with_summaries.json only_flat.json output.json"
        )

    main(sys.argv[1], sys.argv[2], sys.argv[3])
