# assess_ko_quality/tools/count_tokens.py
"""
Token and character counts of the content variants in an export (ko_content_flat and
two model-written versions), one TSV row per KO. Edit the paths and field names below.

    cd assess_ko_quality
    python -m tools.count_tokens
"""
from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer

INPUT_PATH = Path("input/final_output_14_10-2025_17-37-04_for_qa_llmed_runpod.jsonb")
OUTPUT_CSV = Path("output/token_counts.csv")
LIMIT = None            # None = every record
STRICT_MISSING = False  # True: fail when a record has none of the content fields
FIELDS = {
    "native": "ko_content_flat",
    "gpt_oss": "ko_content_flat_gpt_oss_20b",
    "qwen": "ko_content_flat_qwen3_30b_a3b_2507_q8_0",
}
TOKENIZER_MODEL = "deepseek-ai/deepseek-llm-7b-chat"   # close to the GGUF models' tokenizers

tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)


def count_tokens(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def _counts_or_zero(text: str):
    """(tokens, characters); (0, 0) for a missing or empty text."""
    if not text:
        return 0, 0
    return count_tokens(text), len(text)


def load_records(path: Path):
    """A JSON array or an NDJSON file, as a list of dicts."""
    data = path.read_text(encoding="utf-8").strip()
    if not data:
        return []
    if data[0] == "[":
        return json.loads(data)
    return [json.loads(ln) for ln in data.splitlines() if ln.strip()]


def main():
    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(INPUT_PATH)
    if isinstance(LIMIT, int):
        records = records[:LIMIT]

    out_rows = []
    for i, rec in enumerate(records, 1):
        ko_id = rec.get("@id") or rec.get("_orig_id") or f"row_{i}"
        title = rec.get("title", "")
        native_text = rec.get(FIELDS["native"])
        gpt_text = rec.get(FIELDS["gpt_oss"])
        qwen_text = rec.get(FIELDS["qwen"])

        # the GPT-OSS text where there is one, else the Qwen text
        llm_best_text = gpt_text if gpt_text else qwen_text
        llm_best_source = "gpt_oss" if gpt_text else ("qwen" if qwen_text else "")

        if STRICT_MISSING and (native_text is None and gpt_text is None and qwen_text is None):
            raise KeyError(f"Missing all content fields for record index {i} (id={ko_id})")

        native_tok, native_chr = _counts_or_zero(native_text)
        gpt_tok, gpt_chr = _counts_or_zero(gpt_text)
        qwen_tok, qwen_chr = _counts_or_zero(qwen_text)
        best_tok, best_chr = _counts_or_zero(llm_best_text)

        note_bits = []
        if native_text is None:
            note_bits.append("native:missing")
        if gpt_text is None:
            note_bits.append("gpt_oss:missing")
        if qwen_text is None:
            note_bits.append("qwen:missing")

        out_rows.append({
            "id": ko_id,
            "title": title,
            "tokens_native": native_tok,
            "chars_native": native_chr,
            "tokens_gpt_oss": gpt_tok,
            "chars_gpt_oss": gpt_chr,
            "tokens_qwen": qwen_tok,
            "chars_qwen": qwen_chr,
            "tokens_llm_best": best_tok,
            "chars_llm_best": best_chr,
            "llm_best_source": llm_best_source,
            "note": ";".join(note_bits),
        })

    df = pd.DataFrame(out_rows)
    df.to_csv(OUTPUT_CSV.as_posix(), index=False, sep="\t")

    if not df.empty:
        print(df.head(min(5, len(df))))
        print("\nTotals (rows):", len(df))

        def _summ(label_tokens, label_chars):
            if df[label_tokens].sum() == 0 and df[label_chars].sum() == 0:
                print(f"- {label_tokens.replace('tokens_', '')}: all missing/empty")
                return
            p95 = df[label_tokens].quantile(0.95)
            missing_frac = (df[label_tokens] == 0).mean()
            print(f"- {label_tokens.replace('tokens_', '')}: "
                  f"sum={df[label_tokens].sum():,}, "
                  f"mean={df[label_tokens].mean():.1f}, "
                  f"p95~{p95:.0f}, "
                  f"missing~{missing_frac:.1%}")

        print("\nToken stats per variant:")
        _summ("tokens_native", "chars_native")
        _summ("tokens_gpt_oss", "chars_gpt_oss")
        _summ("tokens_qwen", "chars_qwen")
        _summ("tokens_llm_best", "chars_llm_best")

        print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
