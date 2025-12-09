# improve_ko/count_json_objects.py

import json
import re

from pathlib import Path


ID_KEY = "_orig_id"
TEXT_KEY = "ko_content_flat"

def load_ids(file_path: str):
    """Return a set of IDs from the JSON list file."""
    p = Path(file_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ids = set()
    for obj in data:
        if isinstance(obj, dict) and ID_KEY in obj:
            ids.add(obj[ID_KEY])
    return ids

def load_id_to_text(file_path: str):
    """
    Return a dict mapping ID_KEY -> TEXT_KEY from the JSON list file.
    Only entries that have both keys are included.
    """
    p = Path(file_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_text = {}
    for obj in data:
        if not isinstance(obj, dict):
            continue
        if ID_KEY in obj and TEXT_KEY in obj:
            id_to_text[obj[ID_KEY]] = obj[TEXT_KEY]
    return id_to_text

def analyse_file(path: str) -> None:
    """
    Load a JSON file assumed to contain a list of objects.
    Count how many objects have / don't have ko_content_flat
    and ko_content_flat_summarised.
    """
    file_path = Path(path)

    # Read and parse the JSON file
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure we are working with a list of JSON objects
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of objects in {path}, got {type(data).__name__}")

    total = len(data)

    has_both = 0
    has_only_flat = 0
    has_only_summarised = 0
    has_neither = 0

    same_length = 0
    different_length = 0


    # Go through every JSON object and check the presence of keys
    for obj in data:
        # Safely handle non-dict entries, just in case
        if not isinstance(obj, dict):
            continue

        has_flat = "ko_content_flat" in obj and obj["ko_content_flat"] not in (None, "")
        has_summarised = "ko_content_flat_summarised" in obj and obj["ko_content_flat_summarised"] not in (None, "")

        if has_flat and has_summarised:
            has_both += 1

            # Compare the character lengths of both fields
            flat_len = len(obj["ko_content_flat"])
            sum_len = len(obj["ko_content_flat_summarised"])

            if flat_len == sum_len:
                same_length += 1
            else:
                different_length += 1
        elif has_flat and not has_summarised:
            has_only_flat += 1
        elif not has_flat and has_summarised:
            has_only_summarised += 1
        else:
            has_neither += 1

    print(f"\nFile: {file_path}")
    print(f"  Total JSON objects:                  {total}")
    print(f"  With BOTH fields present:            {has_both}")
    print(f"  Only ko_content_flat:                {has_only_flat}")
    print(f"  Only ko_content_flat_summarised:     {has_only_summarised}")
    print(f"  With NEITHER field:                  {has_neither}")

    print(f"  Of those with both fields:")
    print(f"    Same length (chars):               {same_length}")
    print(f"    Different length (chars):          {different_length}")


if __name__ == "__main__":
    analyse_file("output/final_output_24_11-2025_03-50-22_summary_sh1-of-2_20251208_181742.json")
    analyse_file("output/final_output_24_11-2025_03-50-22_summary_sh2-of-2_20251208_181751.json")
    analyse_file("input/final_output_24_11-2025_03-50-22.json")

    orig_file = "input/final_output_24_11-2025_03-50-22.json"
    shard1 = "output/final_output_24_11-2025_03-50-22_summary_sh1-of-2_20251208_181742.json"
    shard2 = "output/final_output_24_11-2025_03-50-22_summary_sh2-of-2_20251208_181751.json"

    orig_ids = load_ids(orig_file)
    sh1_ids = load_ids(shard1)
    sh2_ids = load_ids(shard2)

    shard_union = sh1_ids | sh2_ids

    missing = sorted(list(orig_ids - shard_union))

    print(f"Total original IDs: {len(orig_ids)}")
    print(f"Shard combined IDs: {len(shard_union)}")
    print(f"Missing IDs:        {len(missing)}")

    print("\nFirst 20 missing IDs:")
    for mid in missing[:20]:
        print("  ", mid)

    # Save to a file
    out_path = "missing_ids.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for mid in missing:
            f.write(str(mid) + "\n")

    print(f"\nFull list saved to: {out_path}")

    # ---- Length analysis for missing IDs ----
    id_to_text = load_id_to_text(orig_file)

    out_stats_path = "missing_ids_with_lengths.tsv"
    with open(out_stats_path, "w", encoding="utf-8") as f:
        # header
        f.write("id\tchar_len\ttoken_len\tword_len\n")

        for mid in missing:
            text = id_to_text.get(mid, "")
            char_len = len(text)
            # tokens: simple whitespace-based split (keeps punctuation)
            tokens = text.split()
            token_len = len(tokens)
            # words: regex \w+ (ignores punctuation-only tokens)
            words = re.findall(r"\w+", text)
            word_len = len(words)

            f.write(f"{mid}\t{char_len}\t{token_len}\\t{word_len}\n")

    print(f"\nLength stats for missing IDs saved to: {out_stats_path}")

    # Also show a few examples on screen
    print("\nFirst 10 missing IDs with lengths:")
    for mid in missing[:10]:
        text = id_to_text.get(mid, "")
        char_len = len(text)
        tokens = text.split()
        token_len = len(tokens)
        words = re.findall(r"\w+", text)
        word_len = len(words)
        print(f"  {mid}: chars={char_len}, tokens={token_len}, words={word_len}")
