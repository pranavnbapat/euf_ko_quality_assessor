from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer

# =======================
# CONFIG (edit as needed)
# =======================
INPUT_PATH = Path("input/final_output_14_10-2025_17-37-04_for_qa_llmed_runpod.jsonb")
OUTPUT_CSV = Path("output/token_counts.csv")
LIMIT = None            # set to 10, 20, or None for "all" records
STRICT_MISSING = False  # if True, raise on missing ko_content_flat; if False, count = 0
FIELDS = {
    "native": "ko_content_flat",
    "gpt_oss": "ko_content_flat_gpt_oss_20b",
    "qwen": "ko_content_flat_qwen3_30b_a3b_2507_q8_0",
}

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

def _counts_or_zero(text: str):
    """
    Return (token_count, char_count) for the given text.
    If text is None or empty, return (0, 0).
    """
    if not text:
        return 0, 0
    # Count content tokens (no special tokens), and raw characters
    return count_tokens(text), len(text)


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
        native_text = rec.get(FIELDS["native"])
        gpt_text = rec.get(FIELDS["gpt_oss"])
        qwen_text = rec.get(FIELDS["qwen"])

        # Preferred LLM content = GPT-OSS; fallback to Qwen if GPT-OSS missing
        llm_best_text = gpt_text if gpt_text else qwen_text
        llm_best_source = "gpt_oss" if gpt_text else ("qwen" if qwen_text else "")

        # Optional strictness: only raise if ALL content variants are missing
        if STRICT_MISSING and (native_text is None and gpt_text is None and qwen_text is None):
            raise KeyError(f"Missing all content fields for record index {i} (id={ko_id})")

        # Token/char counts per variant (0 if missing/empty)
        native_tok, native_chr = _counts_or_zero(native_text)
        gpt_tok, gpt_chr = _counts_or_zero(gpt_text)
        qwen_tok, qwen_chr = _counts_or_zero(qwen_text)
        best_tok, best_chr = _counts_or_zero(llm_best_text)

        # Human note about which LLM source was used (or what's missing)
        note_bits = []
        if native_text is None: note_bits.append("native:missing")
        if gpt_text is None: note_bits.append("gpt_oss:missing")
        if qwen_text is None: note_bits.append("qwen:missing")
        note = ";".join(note_bits)

        out_rows.append({
            "id": ko_id,
            "title": title,

            # Native (human/original) content
            "tokens_native": native_tok,
            "chars_native": native_chr,

            # GPT-OSS content (direct)
            "tokens_gpt_oss": gpt_tok,
            "chars_gpt_oss": gpt_chr,

            # Qwen content (direct)
            "tokens_qwen": qwen_tok,
            "chars_qwen": qwen_chr,

            # LLM best (prefers GPT-OSS; falls back to Qwen)
            "tokens_llm_best": best_tok,
            "chars_llm_best": best_chr,
            "llm_best_source": llm_best_source,

            # Notes on missing fields (for quick filtering/debug)
            "note": note,
        })

    df = pd.DataFrame(out_rows)
    df.to_csv(OUTPUT_CSV.as_posix(), index=False, sep='\t')

    # Print quick stats to console for sanity check
    if not df.empty:
        print(df.head(min(5, len(df))))
        print("\nTotals (rows):", len(df))

        def _summ(label_tokens, label_chars):
            if df[label_tokens].sum() == 0 and df[label_chars].sum() == 0:
                print(f"- {label_tokens.replace('tokens_', '')}: all missing/empty")
                return
            p95 = df[label_tokens].quantile(0.95)
            missing_frac = (df[label_tokens] == 0).mean()  # crude proxy for missing/empty
            print(f"- {label_tokens.replace('tokens_', '')}: "
                  f"sum={df[label_tokens].sum():,}, "
                  f"mean={df[label_tokens].mean():.1f}, "
                  f"p95≈{p95:.0f}, "
                  f"missing≈{missing_frac:.1%}")

        print("\nToken stats per variant:")
        _summ("tokens_native", "chars_native")
        _summ("tokens_gpt_oss", "chars_gpt_oss")
        _summ("tokens_qwen", "chars_qwen")
        _summ("tokens_llm_best", "chars_llm_best")

        print(f"\nSaved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
