# validate_assessments

import json
import os
import pandas as pd
import sys

from datetime import datetime
from jsonschema import Draft7Validator
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("KO_OUTPUT_DIR", "./output")).resolve()

SCHEMA_PATH = Path(os.environ.get("KO_SCHEMA_PATH", "./schemas/ko_assessment.schema.json")).resolve()

def _latest_data_file(folder: Path) -> Path:
    """Return the most recent *.tsv or *.csv in `folder`."""
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Output folder not found or not a directory: {folder}")
    candidates = []
    for pat in ("*.tsv", "*.csv"):
        candidates.extend(folder.glob(pat))
    if not candidates:
        raise FileNotFoundError(f"No TSV/CSV files found in {folder}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

# Pick latest data file and schema
data_path = _latest_data_file(OUTPUT_DIR)
schema_path = SCHEMA_PATH
sep = "\t" if data_path.suffix.lower() == ".tsv" else ","

print(f"[INFO] Using data file: {data_path}")
print(f"[INFO] Using schema   : {schema_path}")
print(f"[INFO] Separator      : {'TAB' if sep == '\\t' else 'COMMA'}")

# ---- Read data ----
df = pd.read_csv(str(data_path), sep=sep, dtype=str, keep_default_na=False, engine="python")

# Load schema
with open(schema_path, "r", encoding="utf-8") as f:
    schema = json.load(f)
validator = Draft7Validator(schema)

# ---- Columns to coerce ----
NUM_COLS = [
    "Semantic_Precision","Content_Richness","Cross_Field_Consistency","Linguistic_Integrity","Total_Score",
    "sp_title","sp_desc","sp_keyword_anchoring","cr_depth","cr_diversity","cr_duplication",
    "cf_topics_themes","cf_project_echo","li_spell_score","li_misspell_ratio"
]
BOOL_COLS = [
    "url_valid","doi_valid","subtitle_ok","subtitle_duplicate_title",
    "subtitle_duplicate_description","description_duplicate_title",
    "lang_match","cv_category_ok","cv_license_ok"
]

# ---- Defaults required by schema but missing in TSV ----
TODAY_VER = "v" + datetime.now().strftime("%Y-%m-%d")
DEFAULTS = {
    "rubric_version": TODAY_VER,
}

def _to_bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("true","1","yes","y","t")

def _to_float(v: str):
    try:
        return float(v)
    except Exception:
        return v  # leave as-is; schema will flag if it must be number

errors = 0
for i, row in df.iterrows():
    obj = row.to_dict()

    # Inject defaults if missing/blank
    for k, v in DEFAULTS.items():
        if obj.get(k, "") == "":
            obj[k] = v

    # Cast numerics
    for k in NUM_COLS:
        if k in obj and obj[k] != "":
            obj[k] = _to_float(obj[k])

    # Cast booleans
    for k in BOOL_COLS:
        if k in obj and obj[k] != "":
            obj[k] = _to_bool(obj[k])

    # Validate
    v_errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
    if v_errors:
        errors += 1
        ko_id = obj.get("_orig_id") or obj.get("@id") or f"row_{i}"
        print(f"[ROW {i}] {ko_id}")
        for e in v_errors:
            print("  -", e.message)
        print()

if errors == 0:
    print("[OK] All rows validate ✔")
    exit(0)
else:
    print(f"[FAIL] {errors} row(s) have schema issues")
    exit(1)

