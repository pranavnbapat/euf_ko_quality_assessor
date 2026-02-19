# which_model_to_choose/methods/01_evaluate_chunks.py

"""
Evaluate selected text fields for Knowledge Objects (KOs) using lightweight, intrinsic heuristics.

What this script does
- Loads a JSON file containing one or more KO-like records (dicts).
- For each record, selects a single value for each field:
  - title/subtitle/description: prefer *_llm if present and non-empty, else original
  - keywords: prefer keywords_llm if present (string or list), else keywords
  - ko_content_flat: prefer ko_content_flat_summarised if present, else ko_content_flat
- Computes the same set of intrinsic “sanity” metrics for each selected text block:
  - length (chars, tokens)
  - lexical diversity (type-token ratio)
  - English stopword ratio (English-only heuristic)
  - punctuation density
  - repetition dominance (top-K non-stopword tokens)
  - rough readability proxy (FK-like; guarded for PDF/table fragments)
  - an English-suspect flag (for content that likely violates the “English content expected” rule)
- Writes per-record metrics to 01_evaluate_selected.json and prints an aggregate summary by content_source.

What it does NOT do
- It does not compare a candidate text against an original reference within the same file
  (because each input record may contain only one content variant).
- It does not assess semantic correctness or factual grounding.

Intended use
- Flag empty/junky/PDF-noisy text, unusually repetitive outputs, and likely non-English content.
- Provide comparable signals to decide which text variant is safest to index and use downstream.
"""

import json
import re
from collections import Counter
import nltk

from nltk.corpus import stopwords

# Stopwords are used for:
# - stopword_ratio (English-only heuristic)
# - filtering repetition to focus on content-bearing words
# NLTK data may be missing in fresh environments, so we download lazily.
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

JSON_PATH = "input/final_output_02_01-2026_23-13-59_summary_20260106_143343_metadata_20260107_062641.json"

# Tokenisation strategy:
# - normalise_for_tokens() first replaces high-noise patterns (URLs/emails/long numbers)
#   with stable placeholders so IDs/links do not dominate stats.
# - tokenize() then extracts "word-like" tokens in a Unicode-friendly way, keeping
#   internal apostrophes (e.g., "don't") and excluding underscores.
#
# This makes metrics like TTR and repetition more robust for derived text.
WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
LONG_NUM_RE = re.compile(r"\b\d{4,}\b")  # years/ids/long numbers
BULLET_RE = re.compile(r"^\s*[-*•]+\s+", re.MULTILINE)


def rouge_n_prf(candidate: str, reference: str, n=1):
    """
    ROUGE-N precision/recall/F1 using n-gram overlap counts.
    - recall: overlap / total_ref
    - precision: overlap / total_cand
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if len(ref_tokens) < n or len(cand_tokens) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_ngrams = ngram_counts(ref_tokens, n)
    cand_ngrams = ngram_counts(cand_tokens, n)

    overlap = 0
    for ng, ref_count in ref_ngrams.items():
        overlap += min(ref_count, cand_ngrams.get(ng, 0))

    total_ref = sum(ref_ngrams.values())
    total_cand = sum(cand_ngrams.values())

    if total_ref == 0 or total_cand == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    recall = overlap / total_ref
    precision = overlap / total_cand
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}

def normalise_for_tokens(text: str) -> str:
    """
    Reduce noise that skews token stats:
    - URLs / emails -> placeholders
    - long numbers -> placeholder (IDs, years, etc.)
    """
    if not text:
        return ""
    t = str(text)
    t = URL_RE.sub(" URLTOKEN ", t)
    t = EMAIL_RE.sub(" EMAILTOKEN ", t)
    t = LONG_NUM_RE.sub(" NUMTOKEN ", t)
    return t

def tokenize(text: str):
    """Normalised word tokenizer (keeps apostrophes inside words)."""
    if not text:
        return []
    t = normalise_for_tokens(text).lower()
    return WORD_RE.findall(t)


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


def top_token_repetition_ratio(text: str, top_k: int = 5, ignore_stopwords: bool = True) -> float:
    """
    How much the top-K tokens dominate the text.
    Default ignores stopwords to avoid flagging normal English as 'repetitive'.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0

    if ignore_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2 and t not in {"numtoken", "urltoken", "emailtoken"}]
        if not tokens:
            return 0.0

    counts = Counter(tokens)
    most_common = counts.most_common(top_k)
    top_total = sum(c for _, c in most_common)
    return top_total / len(tokens)

def sentence_split(text: str):
    """
    More robust sentence splitter for PDF-derived text:
    - handles bullets and line breaks
    - still splits on .?! primarily
    """
    if not text:
        return []

    t = str(text)

    # Fix common PDF line-break hyphenation: "exam-\nple" -> "example"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    # Turn bullets into sentence boundaries
    t = BULLET_RE.sub("\n", t)

    # Treat newlines as boundaries (common in PDFs)
    # Collapse multiple newlines first
    t = re.sub(r"\n{2,}", "\n", t)

    # Split on strong punctuation OR newline
    parts = re.split(r"[.!?]+(?:\s+|$)|\n+", t)

    # Clean up
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts


def flesch_kincaid_like(text: str):
    """
    Very rough readability proxy.
    Proper FK needs syllables; we’ll approximate syllables by vowels.
    This is enough to compare S vs L (relative), not to publish.
    """
    sentences = sentence_split(text)
    if len(sentences) < 2:
        return 0.0

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


def build_meta_fields(obj: dict) -> dict:
    """
    Build metadata using LLM fields if present, else fall back to originals.
    Also logs which source key was used for each field.
    """
    title, title_src = pick_field(obj, "title", "title_llm")
    subtitle, subtitle_src = pick_field(obj, "subtitle", "subtitle_llm")
    description, description_src = pick_field(obj, "description", "description_llm")
    keywords_text, keywords_src = pick_keywords(obj)

    meta_all = " ".join([title, subtitle, description, keywords_text]).strip()

    return {
        "meta_title": title,
        "meta_subtitle": subtitle,
        "meta_description": description,
        "meta_keywords": keywords_text,
        "meta_all": meta_all,

        # provenance / logging
        "meta_title_source": title_src,
        "meta_subtitle_source": subtitle_src,
        "meta_description_source": description_src,
        "meta_keywords_source": keywords_src,
    }


def pick_field(obj: dict, base_key: str, llm_key: str = None):
    """
    Prefer llm_key if present and non-empty; else prefer base_key if non-empty.
    Returns: (value_as_str, used_key_name)
    """
    llm_key = llm_key or f"{base_key}_llm"

    def _norm(v):
        if v is None:
            return ""
        if isinstance(v, list):
            # keywords_llm sometimes is list; normalise into one string
            v = " ".join(str(x) for x in v if x is not None)
        return str(v).strip()

    v_llm = _norm(obj.get(llm_key))
    if v_llm:
        return v_llm, llm_key

    v_base = _norm(obj.get(base_key))
    if v_base:
        return v_base, base_key

    return "", "missing"


def pick_keywords(obj: dict):
    """
    Prefer keywords_llm (string or list) if present, else keywords (list or string).
    Returns: (keywords_text, used_key_name)
    """
    # First try keywords_llm (may be list or string)
    v, used = pick_field(obj, "keywords", "keywords_llm")
    if v:
        return v, used

    # Fallback: original keywords list -> string
    kw = obj.get("keywords") or []
    if isinstance(kw, list):
        kw_text = " ".join(str(x) for x in kw if x)
    else:
        kw_text = str(kw).strip()

    return kw_text, ("keywords" if kw_text else "missing")


def pick_content(obj: dict):
    """
    Prefer ko_content_flat_summarised if present; else ko_content_flat.
    Returns: (content_text, used_key_name)
    """
    return pick_field(obj, "ko_content_flat", "ko_content_flat_summarised")


def find_selected_fields(obj: dict):
    """
    Select a single 'final' value for each field:
      - title/subtitle/description/keywords: prefer *_llm if present
      - content: prefer ko_content_flat_summarised if present
    """
    meta = build_meta_fields(obj)
    content_text, content_src = pick_content(obj)

    return content_text, meta, content_src


def score_text_block(prefix: str, text: str) -> dict:
    """
    Compute the same intrinsic metrics for any text field.
    Prefix is used to namespace the keys, e.g. 'content', 'title', 'description'.
    """
    if not text or not str(text).strip():
        return {
            f"{prefix}_error": "empty",
            f"{prefix}_len_chars": 0,
            f"{prefix}_len_tokens": 0,
        }

    t = str(text)

    tokens = tokenize(t)
    sw_ratio = stopword_ratio(t)
    # crude but effective: too-short texts are "unknown", longer texts with very low English stopwords are suspicious
    is_english_suspect = (len(tokens) >= 80 and sw_ratio < 0.06)

    return {
        f"{prefix}_preview": t[:300],  # keep short to avoid huge JSON
        f"{prefix}_len_chars": char_len(t),
        f"{prefix}_len_tokens": token_len(t),
        f"{prefix}_ttr": type_token_ratio(t),
        f"{prefix}_stopword_ratio": stopword_ratio(t),
        f"{prefix}_punct_ratio": punctuation_ratio(t),
        f"{prefix}_readability_fk_like": flesch_kincaid_like(t),
        f"{prefix}_top5_repetition_ratio": top_token_repetition_ratio(t),
        f"{prefix}_english_suspect": is_english_suspect,
    }


def evaluate_record(record_idx: int, obj: dict) -> dict:
    """
    For each KO record:
    - select best-available text per field (LLM if present else original)
    - compute a separate metrics block per field
    """
    # Select sources
    content_text, content_src = pick_content(obj)

    title_text, title_src = pick_field(obj, "title", "title_llm")
    subtitle_text, subtitle_src = pick_field(obj, "subtitle", "subtitle_llm")
    desc_text, desc_src = pick_field(obj, "description", "description_llm")
    kw_text, kw_src = pick_keywords(obj)

    # Base output row with provenance
    out = {
        "record_index": str(record_idx),
        "content_source": content_src,
        "title_source": title_src,
        "subtitle_source": subtitle_src,
        "description_source": desc_src,
        "keywords_source": kw_src,
    }

    # 5 independent score blocks (same methodology)
    out.update(score_text_block("content", content_text))
    out.update(score_text_block("title", title_text))
    out.update(score_text_block("subtitle", subtitle_text))
    out.update(score_text_block("description", desc_text))
    out.update(score_text_block("keywords", kw_text))

    return out


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
            metrics = evaluate_record(idx, obj)
            all_results.append(metrics)

        # sort across all records
        all_results.sort(key=lambda x: x.get("content_len_tokens", 0), reverse=True)

        with open("01_evaluate_selected.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        # ===== aggregated / overall view =====
        per_source = {}
        for m in all_results:
            src = m.get("content_source", "unknown")
            entry = per_source.setdefault(src, {
                "count": 0,
                "sum_content_tokens": 0.0,
                "sum_title_tokens": 0.0,
                "sum_desc_tokens": 0.0,
            })
            entry["count"] += 1
            entry["sum_content_tokens"] += m.get("content_len_tokens", 0)
            entry["sum_title_tokens"] += m.get("title_len_tokens", 0)
            entry["sum_desc_tokens"] += m.get("description_len_tokens", 0)

        print("\n=== Aggregated summary by content_source ===")
        for src, stats in per_source.items():
            c = stats["count"]
            print(
                f"- {src}: {c} items | "
                f"avg content tokens={stats['sum_content_tokens'] / c:.1f} | "
                f"avg title tokens={stats['sum_title_tokens'] / c:.1f} | "
                f"avg desc tokens={stats['sum_desc_tokens'] / c:.1f}"
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
