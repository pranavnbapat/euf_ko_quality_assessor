# which_model_to_choose/methods/01_evaluate_chunks.py

"""
This script compares alternative versions of `ko_content_flat` (e.g. summarised
or cleaned variants) against the original full text.

It measures how much each candidate compresses the original, how much vocabulary
and phrasing from the source is retained, and whether the result looks structurally
reasonable (length, readability, repetition).

The goal is not linguistic perfection, but to provide a lightweight, comparable
signal for deciding which content variant is safest and most useful for search
indexing and downstream QA.
"""

import json
import re
from collections import Counter
import nltk

from nltk.corpus import stopwords

# --- ensure stopwords are available only if missing ---
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))


JSON_PATH = "input/final_output_24_11-2025_03-50-22_for_qa_summary_20251205_221423_metadata_20251206_094639.json"


# ===== 2. BASIC TEXT HELPERS =====
WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str):
    """Simple whitespace/word tokenizer."""
    if not text:
        return []
    # This gives only word-like tokens (no punctuation)
    return WORD_RE.findall(text.lower())


def char_len(text: str) -> int:
    return len(text) if text else 0


def token_len(text: str) -> int:
    return len(tokenize(text))


PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


# ===== 3. METRICS IMPLEMENTATION =====

def compression_ratio(s_text: str, l_text: str):
    """len(S)/len(L) on tokens. Avoid divide by zero."""
    l_tokens = token_len(l_text)
    s_tokens = token_len(s_text)
    if l_tokens == 0:
        return 0.0
    return s_tokens / l_tokens


def type_token_ratio(text: str):
    """Unique words / total words."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def stopword_ratio(text: str):
    """% of tokens that are stopwords. High ratio in junky OCR text can be odd."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    sw_count = sum(1 for t in tokens if t in STOPWORDS)
    return sw_count / len(tokens)


def punctuation_ratio(text: str):
    """Chars that are punctuation / total chars."""
    if not text:
        return 0.0
    punct = len(PUNCT_RE.findall(text))
    return punct / len(text)


def top_token_repetition_ratio(text: str, top_k: int = 5) -> float:
    """How much the top-K tokens dominate the text. High = possible repetition."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    most_common = counts.most_common(top_k)
    top_total = sum(c for _, c in most_common)
    return top_total / len(tokens)


def sentence_split(text: str):
    """Very simple sentence splitter on .?! — good enough for readability proxy."""
    if not text:
        return []
    # Split on ., ?, ! and strip
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def flesch_kincaid_like(text: str):
    """
    Very rough readability proxy.
    Proper FK needs syllables; we’ll approximate syllables by vowels.
    This is enough to compare S vs L (relative), not to publish.
    """
    sentences = sentence_split(text)
    tokens = tokenize(text)
    if not sentences or not tokens:
        return 0.0

    # Approximate syllables by counting vowels in words
    vowels = "aeiou"
    syllables = 0
    for w in tokens:
        syllables += sum(1 for ch in w if ch in vowels)

    words_per_sentence = len(tokens) / len(sentences)
    syllables_per_word = syllables / len(tokens)

    # Classic FK formula (with approximated syllables)
    fk = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    return fk


def avg_sentence_length(text: str) -> float:
    """Average number of tokens per sentence – robust and comparable."""
    sentences = sentence_split(text)
    tokens = tokenize(text)
    if not sentences:
        return 0.0
    return len(tokens) / len(sentences)


def ngram_counts(tokens, n=1):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(candidate: str, reference: str, n=1):
    """
    ROUGE-N (recall-style): how much of reference n-grams are in candidate.
    Good to see if S still contains pieces of L.
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if len(ref_tokens) < n:
        return 0.0

    ref_ngrams = ngram_counts(ref_tokens, n)
    cand_ngrams = ngram_counts(cand_tokens, n)

    overlap = 0
    for ng, ref_count in ref_ngrams.items():
        overlap += min(ref_count, cand_ngrams.get(ng, 0))

    total_ref = sum(ref_ngrams.values())
    if total_ref == 0:
        return 0.0
    return overlap / total_ref


# ===== 4. MAIN EVALUATION LOGIC =====

def load_json(path: str):
    """Load JSON that could be a dict or a list with one dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # if isinstance(data, list) and data:
    #     return data[0]
    return data


def find_chunks(obj: dict):
    """
    We assume:
    - main large chunk is exactly 'ko_content_flat'
    - other candidates start with 'ko_content_flat' but are not equal
    """
    large = obj.get("ko_content_flat", "")
    candidates = {}
    for k, v in obj.items():
        if k.startswith("ko_content_flat") and k != "ko_content_flat":
            # skip completely empty / None candidates
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            candidates[k] = v
    return large, candidates


def evaluate_candidate(name: str, s_text: str, l_text: str):
    if not s_text or str(s_text).strip() == "":
        return {
            "candidate_name": name,
            "error": "empty_candidate"
        }

    comp = compression_ratio(s_text, l_text)

    """Run all basic/statistical metrics and return a dict."""
    metrics = {
        "candidate_name": name,
        "candidate_text": s_text,
        "compression_ratio_tokens": comp,
        "ttr_s": type_token_ratio(s_text),
        "ttr_l": type_token_ratio(l_text),
        "stopword_ratio_s": stopword_ratio(s_text),
        "stopword_ratio_l": stopword_ratio(l_text),
        "punct_ratio_s": punctuation_ratio(s_text),
        "punct_ratio_l": punctuation_ratio(l_text),
        "readability_fk_s": flesch_kincaid_like(s_text),
        "readability_fk_l": flesch_kincaid_like(l_text),
        "rouge1_recall_s_vs_l": rouge_n(s_text, l_text, n=1),
        "rouge2_recall_s_vs_l": rouge_n(s_text, l_text, n=2),
        "len_chars_s": char_len(s_text),
        "len_chars_l": char_len(l_text),
        "len_tokens_s": token_len(s_text),
        "len_tokens_l": token_len(l_text),
    }

    # heuristic
    coverage = metrics["rouge1_recall_s_vs_l"]
    heuristic_score = coverage * 1.0 + (0.15 * (1 - comp))
    metrics["heuristic_score"] = heuristic_score

    metrics["is_too_small"] = comp < 0.10
    return metrics


def main():
    data = load_json(JSON_PATH)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        print("Unexpected JSON format; expected list or single object.")
        return

    if isinstance(data, list):
        all_results = []

        for idx, obj in enumerate(data, start=1):
            l_text, candidates = find_chunks(obj)

            if not l_text:
                print(f"[record {idx}] No 'ko_content_flat' found, skipping.")
                continue

            if not candidates:
                print(f"[record {idx}] No candidate chunks starting with 'ko_content_flat'*, skipping.")
                continue

            print(f"Found {len(candidates)} candidate(s) to compare with 'ko_content_flat' in record {idx}.\n")

            for name, s_text in candidates.items():
                metrics = evaluate_candidate(name, s_text, l_text)
                metrics["record_index"] = str(idx)
                all_results.append(metrics)

        # sort across all records
        all_results.sort(key=lambda x: x.get("heuristic_score", 0), reverse=True)

        with open("01_evaluate_chunks.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        # ===== aggregated / overall view =====
        per_model = {}
        for m in all_results:
            # skip empty/error ones
            if "error" in m:
                continue

            model = m["candidate_name"]
            entry = per_model.setdefault(model, {
                "count": 0,
                "sum_compression": 0.0,
                "sum_rouge1": 0.0,
                "sum_rouge2": 0.0,
                "sum_heuristic": 0.0,
            })

            entry["count"] += 1
            entry["sum_compression"] += m["compression_ratio_tokens"]
            entry["sum_rouge1"] += m["rouge1_recall_s_vs_l"]
            entry["sum_rouge2"] += m["rouge2_recall_s_vs_l"]
            entry["sum_heuristic"] += m["heuristic_score"]

        print("\n=== Aggregated model-level summary ===")
        for model, stats in per_model.items():
            c = stats["count"]
            avg_comp = stats["sum_compression"] / c
            avg_r1 = stats["sum_rouge1"] / c
            avg_r2 = stats["sum_rouge2"] / c
            avg_h = stats["sum_heuristic"] / c
            print(
                f"- {model}: "
                f"{c} candidates | "
                f"avg compression={avg_comp:.3f} | "
                f"avg ROUGE-1={avg_r1:.3f} | "
                f"avg ROUGE-2={avg_r2:.3f} | "
                f"avg heuristic={avg_h:.3f}"
            )

            # # Pretty print
            # for m in all_results:
            #     if "error" in m:
            #         print(f"[{m['record_index']}] Candidate: {m['candidate_name']} -> {m['error']}")
            #         continue
            #
            #     s_text = m["candidate_text"]
            #     print("=" * 70)
            #     print(f"[{m['record_index']}] Candidate: {m['candidate_name']}")
            #     print(f"- Compression ratio (tokens): {m['compression_ratio_tokens']:.3f}")
            #     print(f"- ROUGE-1 recall vs L: {m['rouge1_recall_s_vs_l']:.3f}")
            #     print(f"- ROUGE-2 recall vs L: {m['rouge2_recall_s_vs_l']:.3f}")
            #     print(f"- TTR S vs L: {m['ttr_s']:.3f} vs {m['ttr_l']:.3f}")
            #     print(f"- Stopword ratio S vs L: {m['stopword_ratio_s']:.3f} vs {m['stopword_ratio_l']:.3f}")
            #     print(f"- Punctuation ratio S vs L: {m['punct_ratio_s']:.3f} vs {m['punct_ratio_l']:.3f}")
            #     print(f"- Top-5 repetition ratio S: {top_token_repetition_ratio(s_text):.3f}")
            #     print(f"- Readability (FK) S vs L: {m['readability_fk_s']:.2f} vs {m['readability_fk_l']:.2f}")
            #     print(f"- Tokens S vs L: {m['len_tokens_s']} vs {m['len_tokens_l']}")
            #     print(f"- Avg sentence length S: {avg_sentence_length(s_text):.2f}")
            #     print(f"- Heuristic score: {m['heuristic_score']:.3f}\n")
            #
            #     print()

if __name__ == "__main__":
    main()
