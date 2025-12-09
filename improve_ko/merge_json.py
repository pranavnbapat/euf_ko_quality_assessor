# improve_ko/merge_json.py

import json
import re
from pathlib import Path

# --- CONFIG -------------------------------------------------------------

# Folder where your shard JSONs live.
# If you run this script from improve_ko/, "." is fine.
# If you run it from repo root, point this to the folder with the JSONs.
INPUT_DIR = Path("output")

# Base prefix of the run – everything before `_summary_sh...`
# Example filename:
#   final_output_24_11-2025_03-50-22_summary_sh1-of-2_20251208_181742.json
BASE_PREFIX = "final_output_24_11-2025_03-50-22"

# Output file name for merged result
output_file = INPUT_DIR / f"{BASE_PREFIX}_summary_merged.json"

# Regex to parse shard filenames for this base.
# It captures:
#   - base  : the base prefix
#   - idx   : shard index (1, 2, ...)
#   - total : total number of shards
#   - ts    : timestamp part (ignored for merge logic)
summary_pattern = re.compile(
    rf"^(?P<base>{re.escape(BASE_PREFIX)})_summary_sh(?P<idx>\d+)-of-(?P<total>\d+)_(?P<ts>\d+_\d+)\.json$"
)

# --- DISCOVER SUMMARY SHARDS --------------------------------------------

candidates: list[tuple[int, int, Path]] = []

# Find files like:
# final_output_24_11-2025_03-50-22_summary_sh1-of-2_*.json
for path in INPUT_DIR.glob(f"{BASE_PREFIX}_summary_sh*-of-*_*.json"):
    match = summary_pattern.match(path.name)
    if not match:
        # Ignore any files that don't match the exact pattern
        continue

    shard_idx = int(match["idx"])
    total = int(match["total"])
    candidates.append((shard_idx, total, path))

if not candidates:
    print(f"No summary shard files found for base: {BASE_PREFIX}")
    raise SystemExit(1)

# Check that all shards 1..total are present
# We assume all candidates agree on "total" (they should, by naming convention)
total = candidates[0][1]
indices_found = {idx for idx, _, _ in candidates}
expected_indices = set(range(1, total + 1))

if indices_found != expected_indices:
    print("Warning: incomplete shard set detected.")
    print(f"  Expected shards: {sorted(expected_indices)}")
    print(f"  Found shards:    {sorted(indices_found)}")
    # Abort to avoid silently merging partial data
    raise SystemExit(1)

# Sort files by shard index to control merge order (1, 2, 3, ...)
files_to_merge = [p for idx, _, p in sorted(candidates, key=lambda t: t[0])]

# --- CONFIRM WITH USER --------------------------------------------------

print("The following files will be merged (in this order):")
for p in files_to_merge:
    print(f"  - {p.name}")

answer = input("Proceed with merge? [y/N]: ").strip().lower()
if answer not in {"y", "yes"}:
    print("Aborting merge.")
    raise SystemExit(0)

# --- LOAD & MERGE -------------------------------------------------------

merged: list = []

for path in files_to_merge:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # We expect each shard file to contain a JSON list (e.g. [ {...}, {...} ])
    if not isinstance(data, list):
        raise TypeError(
            f"Expected a list in {path.name}, got {type(data).__name__}. "
            "If your shards are not lists, adjust the merge logic."
        )

    # Extend the merged list with all items from this shard
    merged.extend(data)

# --- SAVE ---------------------------------------------------------------

with output_file.open("w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Merged {len(files_to_merge)} shard(s), total {len(merged)} items → {output_file}")
