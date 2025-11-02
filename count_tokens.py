from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer

# =======================
# CONFIG (edit as needed)
# =======================
INPUT_PATH = Path("input/final_output_14_10-2025_17-37-04.json")
OUTPUT_CSV = Path("output/token_counts.csv")
LIMIT = None            # set to 10, 20, or None for "all" records
STRICT_MISSING = False  # if True, raise on missing ko_content_flat; if False, count = 0

# Choose a tokenizer that matches GGUF lineage
TOKENIZER_MODEL = "deepseek-ai/deepseek-llm-7b-chat"

# =======================
# Tokenizer (loads once)
# =======================
tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)

def count_tokens(text: str) -> int:
    """Count tokens roughly matching what your GGUF model will see."""
    # NOTE: add_special_tokens=False so we count only content tokens
    return len(tok.encode(text, add_special_tokens=False))

# --------------------------------------------
# Helpers to support both JSON array and NDJSON
# --------------------------------------------
def load_records(path: Path):
    """
    Returns a list[dict]. Supports:
      1) A single JSON array: [ {...}, {...}, ... ]
      2) NDJSON: one JSON object per line
    """
    data = path.read_text(encoding="utf-8").strip()
    if not data:
        return []

    # Heuristic: if file starts with '[' assume a JSON array, otherwise NDJSON
    if data[0] == "[":
        return json.loads(data)

    # NDJSON fallback
    rows = []
    for ln in data.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows

def main():
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(INPUT_PATH)

    # Apply LIMIT (10, 20, or None for all)
    if isinstance(LIMIT, int):
        records = records[:LIMIT]

    out_rows = []
    for i, rec in enumerate(records, 1):
        ko_id = rec.get("@id") or rec.get("_orig_id") or f"row_{i}"
        title = rec.get("title", "")
        text = rec.get("ko_content_flat")

        if text is None:
            if STRICT_MISSING:
                raise KeyError(f"Missing 'ko_content_flat' for record index {i} (id={ko_id})")
            token_count = 0
            char_count = 0
            note = "missing ko_content_flat"
        else:
            token_count = count_tokens(text)
            char_count = len(text)
            note = ""

        out_rows.append({
            "id": ko_id,
            "title": title,
            "token_count": token_count,
            "char_count": char_count,
            "note": note
        })

    df = pd.DataFrame(out_rows)
    df.to_csv(OUTPUT_CSV.as_posix(), index=False, sep='\t')

    # Print quick stats to console for sanity check
    if not df.empty:
        print(df.head(min(5, len(df))))
        print("\nTotals:")
        print(f"- rows: {len(df)}")
        print(f"- tokens (sum): {df['token_count'].sum():,}")
        print(f"- tokens (mean): {df['token_count'].mean():.1f}")
        print(f"- tokens (p95 approx): {df['token_count'].quantile(0.95):.0f}")
        print(f"Saved → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
