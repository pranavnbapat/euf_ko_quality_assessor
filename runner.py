# runner.py

import pandas as pd

from assessor import assess_ko
from config import INPUT_FOLDER, OUTPUT_FOLDER
from io_utils import ensure_directory, assert_readable_dir, latest_json_file, read_json_any, unique_outfile, write_tsv

def main() -> None:
    assert_readable_dir(INPUT_FOLDER)
    ensure_directory(OUTPUT_FOLDER)
    latest = latest_json_file(INPUT_FOLDER)
    print(f"[INFO] Using latest file: {latest}")

    rows, errors, count = [], 0, 0
    for ko in read_json_any(latest):
        try:
            rows.append(assess_ko(ko))
        except Exception as e:
            errors += 1
            rid = ko.get("_orig_id") or ko.get("@id") or f"row_{count}"
            rows.append({
                "_orig_id": rid, "title": ko.get("title",""),
                "lang_detected": "unknown",
                "Semantic_Precision": 0, "Content_Richness": 0,
                "Cross_Field_Consistency": 0, "Linguistic_Integrity": 0,
                "Total_Score": 0, "notes": f"ERROR: {type(e).__name__}: {e}"
            })
        count += 1
        if count % 1000 == 0:
            print(f"[INFO] Processed {count} KOs...")

    if count == 0:
        raise RuntimeError(f"No valid JSON objects found in input file: {latest}")

    out_path = unique_outfile(OUTPUT_FOLDER, stem="assessments", ext=".tsv")
    df = pd.DataFrame(rows)
    write_tsv(df, out_path)
    print(f"[OK] Wrote {len(df)} rows -> {out_path}")
    if errors:
        print(f"[WARN] {errors} item(s) had exceptions; details recorded in 'notes' column.")

if __name__ == "__main__":
    main()
