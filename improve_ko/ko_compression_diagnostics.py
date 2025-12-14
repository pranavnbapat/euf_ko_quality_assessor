# improve_ko/ko_compression_diagnostics.py

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
import unicodedata

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ----------------------------
# Token counting (best-effort)
# ----------------------------

def try_tiktoken_count(text: str) -> Optional[int]:
    """
    If tiktoken is installed, use it for a closer approximation to LLM tokens.
    Otherwise return None and we'll fall back to a regex-based estimate.

    NOTE: tiktoken is optional; we do not require it.
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return None


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def approx_token_count(text: str) -> int:
    """
    Simple approximation:
    - splits into "word-ish" units + punctuation
    This is NOT identical to model tokens but is stable enough for routing.
    """
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def count_tokens(text: str) -> int:
    tt = try_tiktoken_count(text)
    return tt if tt is not None else approx_token_count(text)


# ----------------------------
# spaCy setup
# ----------------------------

def _run_cmd(cmd: List[str]) -> None:
    """
    Run a command and raise a clear error if it fails.
    We use this for optional dependency bootstrap (spaCy + model).
    """
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed ({e.returncode}): {' '.join(cmd)}") from e


def load_spacy_model(model_name: str = "en_core_web_lg"):
    """
    Loads spaCy + model. If missing, attempts to install them.
    Falls back to a smaller model if the large model cannot be installed.
    """
    # 1) Ensure spaCy is installed
    try:
        import spacy  # type: ignore
    except ModuleNotFoundError:
        print("spaCy not found; installing via pip…")
        _run_cmd([sys.executable, "-m", "pip", "install", "spacy"])
        import spacy  # type: ignore

    # 2) Try to load the requested model; if missing, download it
    try:
        return spacy.load(model_name, disable=["lemmatizer"])
    except Exception:
        print(f"spaCy model '{model_name}' not available; downloading…")
        _run_cmd([sys.executable, "-m", "spacy", "download", model_name])

        # Try again after download
        try:
            return spacy.load(model_name, disable=["lemmatizer"])
        except Exception:
            # 3) Fallback: use a smaller model if large fails (often faster/less disk)
            fallback = "en_core_web_sm"
            print(f"Failed to load '{model_name}'. Falling back to '{fallback}'…")

            try:
                return spacy.load(fallback, disable=["lemmatizer"])
            except Exception:
                print(f"spaCy model '{fallback}' not available; downloading…")
                _run_cmd([sys.executable, "-m", "spacy", "download", fallback])
                return spacy.load(fallback, disable=["lemmatizer"])


# ----------------------------
# Method group 1: Lexical density
# ----------------------------

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

def lexical_density(doc) -> Dict[str, Any]:
    """
    Lexical density = content words / total tokens (excluding spaces/punct).
    Uses POS tags from spaCy.
    """
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    if not tokens:
        return {"lexical_density": None, "content_token_ratio": None, "content_tokens": 0, "total_tokens_pos": 0}

    content = [t for t in tokens if t.pos_ in CONTENT_POS]
    ld = len(content) / len(tokens)

    return {
        "lexical_density": float(ld),
        "content_token_ratio": float(ld),
        "content_tokens": int(len(content)),
        "total_tokens_pos": int(len(tokens)),
    }


# ----------------------------
# Method group 2: Entropy + repetition
# ----------------------------

def shannon_entropy_from_counts(counts: Counter) -> float:
    """
    Shannon entropy H = -sum(p_i * log2(p_i))
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return float(h)


def entropy_and_repetition(text: str) -> Dict[str, Any]:
    """
    Computes:
    - Type-Token Ratio (TTR) on word tokens
    - Unigram entropy
    - Bigram repetition ratio (how many bigrams repeat)
    """
    # Word tokens: keep simple, lower-cased
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())
    n = len(words)
    if n == 0:
        return {
            "ttr": None,
            "unigram_entropy": None,
            "bigram_repetition_ratio": None,
            "unique_word_ratio": None,
            "word_count": 0,
        }

    counts = Counter(words)
    unique = len(counts)
    ttr = unique / n

    unigram_entropy = shannon_entropy_from_counts(counts)

    # Bigrams
    if n >= 2:
        bigrams = list(zip(words[:-1], words[1:]))
        bc = Counter(bigrams)
        repeated = sum(1 for v in bc.values() if v >= 2)
        bigram_repetition_ratio = repeated / len(bc) if len(bc) else 0.0
    else:
        bigram_repetition_ratio = 0.0

    return {
        "ttr": float(ttr),
        "unique_word_ratio": float(ttr),
        "unigram_entropy": float(unigram_entropy),
        "bigram_repetition_ratio": float(bigram_repetition_ratio),
        "word_count": int(n),
        "unique_words": int(unique),
    }


# ----------------------------
# Method group 3: Semantic signals
# ----------------------------

def semantic_signals(doc, domain_terms: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Semantic signals:
    - NER density (entities per 100 tokens)
    - numeric density (numbers per 100 tokens)
    - domain term density (terms per 100 tokens) using a provided list
    """
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    token_count = len(tokens) if tokens else 0
    if token_count == 0:
        return {
            "entity_count": 0,
            "entity_density_per_100_tokens": None,
            "numeric_count": 0,
            "numeric_density_per_100_tokens": None,
            "domain_term_count": 0,
            "domain_term_density_per_100_tokens": None,
        }

    ents = list(doc.ents)
    entity_count = len(ents)

    # Numbers: include tokens like "18.5", "2022", etc.
    numeric_count = sum(1 for t in tokens if t.like_num)

    # Domain terms: simple substring matches on lower text (fast baseline).
    dt_count = None
    if domain_terms:
        dt_count = 0
        text_l = doc.text.lower()
        for term in domain_terms:
            term_l = term.strip().lower()
            if not term_l:
                continue
            # count non-overlapping occurrences
            dt_count += len(re.findall(r"\b" + re.escape(term_l) + r"\b", text_l))

    def per_100(x: int) -> float:
        return float((x / token_count) * 100.0)

    return {
        "entity_count": int(entity_count),
        "entity_density_per_100_tokens": per_100(entity_count),
        "numeric_count": int(numeric_count),
        "numeric_density_per_100_tokens": per_100(numeric_count),
        "domain_term_count": int(dt_count) if dt_count is not None else None,
        "domain_term_density_per_100_tokens": per_100(dt_count) if dt_count is not None else None,
    }


# ----------------------------
# Method group 4: Structural / syntactic complexity
# ----------------------------

VOWELS_RE = re.compile(r"[aeiouy]+", re.I)

def count_syllables(word: str) -> int:
    """
    Rough syllable estimator for English readability.
    Not perfect, but good enough for relative comparisons.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    sylls = len(VOWELS_RE.findall(w))
    # small correction: silent trailing 'e'
    if w.endswith("e") and sylls > 1:
        sylls -= 1
    return max(1, sylls)


def flesch_reading_ease(text: str) -> Optional[float]:
    """
    Flesch Reading Ease:
    206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    """
    # Very simple sentence split
    sentences = re.split(r"[.!?]+\s*", text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return None

    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
    if not words:
        return None

    syllables = sum(count_syllables(w) for w in words)
    wps = len(words) / len(sentences)
    spw = syllables / len(words)

    score = 206.835 - 1.015 * wps - 84.6 * spw
    return float(score)


def structural_complexity(doc) -> Dict[str, Any]:
    """
    Structural/syntactic indicators:
    - sentence count
    - avg sentence length (in words)
    - avg dependency depth proxy: mean token 'head distance' (simple proxy)
    - readability score (Flesch) as a proxy for complexity
    - noise ratio: non-alphanumeric chars share
    """
    sents = list(doc.sents)
    # Word tokens: excluding punct/spaces
    words = [t for t in doc if not t.is_space and not t.is_punct]
    word_count = len(words)

    sent_count = len(sents)
    avg_sent_len = (word_count / sent_count) if sent_count > 0 else None

    # Dependency head distance proxy (cheap syntactic complexity signal)
    # For each token, distance to head (absolute index difference)
    if words:
        # idx_map = {t.i: k for k, t in enumerate(doc)}
        distances = []
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            try:
                distances.append(abs(t.i - t.head.i))
            except Exception:
                pass
        dep_head_distance_mean = float(np.mean(distances)) if distances else None
    else:
        dep_head_distance_mean = None

    readability = flesch_reading_ease(doc.text)

    # Noise: proportion of non-alnum (excluding whitespace)
    raw = doc.text
    non_ws = [ch for ch in raw if not ch.isspace()]
    if non_ws:
        noise_ratio = sum(1 for ch in non_ws if not ch.isalnum()) / len(non_ws)
    else:
        noise_ratio = None

    return {
        "sentence_count": int(sent_count),
        "word_count_structural": int(word_count),
        "avg_sentence_length_words": float(avg_sent_len) if avg_sent_len is not None else None,
        "dep_head_distance_mean": dep_head_distance_mean,
        "flesch_reading_ease": readability,
        "noise_ratio_non_alnum": float(noise_ratio) if noise_ratio is not None else None,
    }


# ----------------------------
# Aggregation: compression difficulty + suggested ratio + routing
# ----------------------------

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def normalise(value: Optional[float], lo: float, hi: float) -> float:
    """
    Linear normalisation to [0,1]. If value is None, return 0.5 (neutral).
    """
    if value is None:
        return 0.5
    if hi == lo:
        return 0.5
    return clamp01((value - lo) / (hi - lo))

def compute_cds_and_policy(metrics: Dict[str, Any], token_count: int) -> Dict[str, Any]:
    """
    Compression Difficulty Score (CDS): higher means harder to compress safely.

    We combine:
    - lexical_density (higher => denser => harder)
    - unigram_entropy (higher => more novelty => harder)
    - entity_density (higher => more concepts => harder)
    - avg_sentence_length (higher => more complex => harder)
    - bigram_repetition_ratio (higher => more redundant => easier) => subtract

    The normalisation ranges are heuristic but sane.
    """
    ld = metrics.get("lexical_density")
    ent = metrics.get("unigram_entropy")
    ent_den = metrics.get("entity_density_per_100_tokens")
    sent_len = metrics.get("avg_sentence_length_words")
    bigram_rep = metrics.get("bigram_repetition_ratio")
    adj_sim = metrics.get("mean_adjacent_cosine_similarity")
    cent_sim = metrics.get("mean_centroid_cosine_similarity")

    # Normalise signals
    n_ld = normalise(ld, 0.35, 0.65)              # typical English LD range
    n_ent = normalise(ent, 5.5, 8.5)              # rough entropy range
    n_ent_den = normalise(ent_den, 2.0, 10.0)     # entities per 100 tokens
    n_sent = normalise(sent_len, 12.0, 35.0)      # words per sentence
    n_rep = normalise(bigram_rep, 0.0, 0.35)      # repetition ratio
    n_adj = normalise(adj_sim, 0.20, 0.85)
    n_cent = normalise(cent_sim, 0.25, 0.90)

    # Weighted sum (start simple; tune later)
    # Redundancy makes compression easier, hence subtract.
    cds = (
            0.22 * n_ld +
            0.22 * n_ent +
            0.22 * n_ent_den +
            0.18 * n_sent -
            0.12 * n_rep -
            0.10 * n_adj -
            0.08 * n_cent
    )

    cds = clamp01(cds)

    # Decide ratio bands
    if cds < 0.35:
        ratio_band = "low"
        summary_ratio_range = (0.10, 0.20)
        suggested_model = "small"   # e.g., 14B / GPT-OSS
    elif cds < 0.60:
        ratio_band = "medium"
        summary_ratio_range = (0.20, 0.30)
        suggested_model = "small"   # still often fine
    else:
        ratio_band = "high"
        summary_ratio_range = (0.30, 0.45)
        suggested_model = "large"   # e.g., 30B

    # Add a mild token-length override: very long chunks push to large model
    if token_count >= 900 and ratio_band != "low":
        suggested_model = "large"

    return {
        "compression_difficulty_score_0_1": float(cds),
        "difficulty_band": ratio_band,
        "suggested_summary_ratio_min": float(summary_ratio_range[0]),
        "suggested_summary_ratio_max": float(summary_ratio_range[1]),
        "suggested_summary_tokens_min": int(math.ceil(token_count * summary_ratio_range[0])),
        "suggested_summary_tokens_max": int(math.ceil(token_count * summary_ratio_range[1])),
        "suggested_model_class": suggested_model,  # "small" or "large"
    }

def clean_ko_content_chunks(chunks: list[str]) -> str:
    """
    Clean a list of text fragments extracted from PDFs/JSON for search/embedding.
    Keeps paragraphs; removes page furniture and common PDF artefacts.
    """
    # 1) Join and normalise Unicode (NFKC flattens compatibility forms)
    s = " ".join(chunks)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFKC", s)

    # 2) Whitespace: convert NBSP & friends to regular space; remove zero-widths
    #   \u00A0 NBSP; \u2000–\u200A various spaces; \u202F NNBSP; \u205F MMSP
    s = re.sub(r"[\u00A0\u2000-\u200A\u202F\u205F]", " ", s)
    #   \u200B ZWSP, \u200C ZWNJ, \u200D ZWJ, \uFEFF BOM
    s = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", s)
    #   U+00AD SOFT HYPHEN: remove entirely (PyCharm shows it as 'SHY')
    s = s.replace("\u00AD", "")
    #   HTML entity form sometimes appears in scraped text
    s = s.replace("&shy;", "")

    # 3) Remove page headers/footers like "7 / 31" at line starts
    s = re.sub(r"(?m)^\s*\d+\s*/\s*\d+\s+", "", s)

    # 4) Table-of-contents dot leaders → single space
    s = re.sub(r"\.{2,}", " ", s)

    # 5) Normalise bullets and dash spacing
    #    lines that start with a loose "-" become bullets
    s = re.sub(r"(?m)^\s*-\s+", "• ", s)
    #    collapse weird spaced hyphens/dashes to " - "
    s = re.sub(r"\s*[-–—]\s*", " - ", s)

    # Normalise special hyphen/minus to ASCII hyphen so later rules behave consistently
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2212", "-")

    # [NEW] Preserve true hyphenated compounds (collapse spaces around hyphen when both sides are word chars)
    # Examples: "EIP - AGRI" -> "EIP-AGRI", "multi - actor" -> "multi-actor"
    s = re.sub(r'(?<=\w)\s*-\s*(?=\w)', '-', s)

    # [NEW] Fix occasional split at word-start like "T hese" -> "These"
    # (Capital letter + single space + 2+ lowercase letters)
    s = re.sub(r'\b([A-Z])\s([a-z]{2,})\b', r'\1\2', s)

    # 6) Map curly quotes/ellipsis to ASCII; drop ©/®/™ clutter
    trans = {
        ord("“"): '"', ord("”"): '"', ord("„"): '"', ord("‟"): '"',
        ord("‘"): "'", ord("’"): "'", ord("‚"): "'", ord("‛"): "'",
        ord("…"): "...", ord("©"): " ", ord("®"): " ", ord("™"): " ",
    }
    s = s.translate(trans)

    # 7) Remove control characters (except \n and \t)
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)

    # 8) Light de-noising of obvious repeated headers (exact matches only, safe)
    #    Here: drop duplicate standalone "Brassica Fact Sheet" lines
    s = re.sub(r"(?m)^\s*Brassica\s+Fact\s+Sheet\s*$", "", s)

    # [NEW] Deduplicate exact lines (helps when PDFs repeat headers/URLs verbatim)
    _lines, _seen = [], set()
    for _line in s.splitlines():
        _key = _line.strip()
        if _key and _key not in _seen:
            _seen.add(_key)
            _lines.append(_line)
    s = "\n".join(_lines)

    # De-hyphenate words split across lines: "nutricio-\nnal" -> "nutricional"
    s = re.sub(r'(?<=\w)-\n(?=\w)', '', s)

    # Belt-and-braces: if a soft hyphen survived with a newline, drop both
    s = re.sub(r'\u00AD\n?', '', s)

    # [NEW] Join intra-sentence hard wraps: replace a single newline between word chars with a space
    # e.g., "Increase\nproductivity" -> "Increase productivity"
    s = re.sub(r'(?<=\w)\n(?=\w)', ' ', s)

    # ensure a space when lowercase is followed by Uppercase (productivityOptimize -> productivity Optimize)
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)

    # ensure a space after a colon (to:Developing -> to: Developing)
    s = re.sub(r':(?!\s)', ': ', s)

    # space between compact number+suffix and a 4-digit year (6,99M2018 -> 6,99M 2018)
    s = re.sub(r'(\d[\d.,]*\s*[kKmMbB])(?=\d{4}\b)', r'\1 ', s)

    # collapse duplicate "n° NN" tokens (n°19 n°19 -> n°19)
    s = re.sub(r'\b(n°\s*\d+)\s+\1\b', r'\1', s, flags=re.IGNORECASE)

    # Normalise "Nº"/"N°"/"No." variants to a single form "n°"
    s = re.sub(r'\b[Nn][oO][\.\s]?(?=\d)', 'n° ', s)  # "No 5", "No.5" -> "n° 5"
    s = re.sub(r'\b[Nn][º°]\s*(?=\d)', 'n° ', s)  # "Nº5", "N° 5" -> "n° 5"

    # Remove spaces before punctuation (e.g., "palabra :" -> "palabra:")
    s = re.sub(r'\s+([,.;:!?])', r'\1', s)

    # 9) Trim spaces around newlines; collapse excessive blank lines and spaces
    s = re.sub(r"[ \t]+\n", "\n", s)           # strip trailing spaces before NL
    s = re.sub(r"\n{3,}", "\n\n", s)           # max two newlines
    s = re.sub(r"[ \t]{2,}", " ", s)           # collapse runs of spaces/tabs
    s = re.sub(r"\s{2,}", " ", s)              # extra safety
    s = s.strip()

    return s


# ----------------------------
# Main processing
# ----------------------------

def ensure_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("Input JSON must be an object or a list of objects")


def load_sentence_transformer(model_name: str, device: str = "auto"):
    """
    Loads a SentenceTransformer model.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers not installed. Install with:\n"
            "  pip install sentence-transformers torch\n"
        ) from e

    import torch  # type: ignore

    # Auto-select GPU if available; otherwise CPU
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer(model_name, device=device)


def _sample_sentences(sent_texts: List[str], max_sents: int) -> List[str]:
    """
    Sample sentences evenly if there are too many, to keep embedding compute bounded.
    """
    if len(sent_texts) <= max_sents:
        return sent_texts
    idxs = np.linspace(0, len(sent_texts) - 1, num=max_sents, dtype=int)
    return [sent_texts[i] for i in idxs]


def sentence_embedding_similarity(
    doc,
    st_model,
    max_sentences: int = 40,
    min_sentence_chars: int = 20,
) -> Dict[str, Any]:
    """
    Computes semantic redundancy using sentence embeddings.

    Metrics returned:
    - mean_adjacent_cosine_similarity: mean cosine(sim(s_i, s_{i+1}))
      (cheap, linear time; detects local repetition)
    - mean_centroid_cosine_similarity: mean cosine(sim(s_i, centroid))
      (cheap, linear time; detects global redundancy)

    We avoid full pairwise similarity (O(n^2)) for performance.
    """
    # Extract sentence texts
    sent_texts = [s.text.strip() for s in doc.sents]
    # Filter out very short sentences that are often noise ("Figure 1", "2022", etc.)
    sent_texts = [s for s in sent_texts if len(s) >= min_sentence_chars]

    if len(sent_texts) < 2:
        return {
            "embedding_sentence_count_used": int(len(sent_texts)),
            "mean_adjacent_cosine_similarity": None,
            "mean_centroid_cosine_similarity": None,
        }

    sent_texts = _sample_sentences(sent_texts, max_sents=max_sentences)

    # Compute embeddings (normalise for cosine with dot product)
    emb = st_model.encode(
        sent_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Adjacent cosine similarities (dot product because normalised)
    adj = np.sum(emb[:-1] * emb[1:], axis=1)
    mean_adj = float(np.mean(adj)) if adj.size else None

    # Centroid similarity: average similarity to centroid vector
    centroid = np.mean(emb, axis=0, keepdims=True)
    # Normalise centroid
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    cent_sims = np.sum(emb * centroid_norm, axis=1)
    mean_cent = float(np.mean(cent_sims)) if cent_sims.size else None

    return {
        "embedding_sentence_count_used": int(len(sent_texts)),
        "mean_adjacent_cosine_similarity": mean_adj,
        "mean_centroid_cosine_similarity": mean_cent,
    }


def process_object(nlp, obj: Dict[str, Any], text_field: str, min_tokens: int,
                   domain_terms: Optional[List[str]] = None,
                   st_model=None,
                   max_embed_sentences: int = 40,
                   spacy_max_chars: int = 200_000) -> Dict[str, Any]:
    text = obj.get(text_field, "")
    if not isinstance(text, str):
        text = str(text)

    # Step 0: pre-normalisation / formatting cleanup
    clean_text = clean_ko_content_chunks([text])

    # Token count should be on the cleaned text (routing will be more stable)
    tok_count = count_tokens(clean_text)

    obj[f"{text_field}_clean"] = clean_text


    # Always store token count
    # (but only do deeper metrics if >= min_tokens)
    if tok_count < min_tokens:
        obj[f"{text_field}_metrics"] = {
            "token_count": int(tok_count),
            "skipped_reason": f"token_count < {min_tokens}",
        }
        return obj

    # spaCy parse/NER can be extremely expensive on huge texts; cap input
    spacy_text = clean_text if len(clean_text) <= spacy_max_chars else clean_text[:spacy_max_chars]
    doc = nlp(spacy_text)

    m1 = lexical_density(doc)
    m2 = entropy_and_repetition(clean_text)
    m3 = semantic_signals(doc, domain_terms=domain_terms)
    m4 = structural_complexity(doc)
    m5 = {}
    if st_model is not None:
        m5 = sentence_embedding_similarity(doc, st_model, max_sentences=max_embed_sentences)

    merged = {
        "token_count": int(tok_count),
        **m1,
        **m2,
        **m3,
        **m4,
        **m5,
    }

    policy = compute_cds_and_policy(merged, tok_count)
    merged.update(policy)

    obj[f"{text_field}_metrics"] = merged
    return obj


def build_work_items(
    items: List[Dict[str, Any]],
    text_field: str,
    min_tokens: int,
    spacy_max_chars: int,
) -> tuple[list[dict], list[Dict[str, Any]]]:
    """
    Prepares:
    - work: list of dicts for items that need spaCy processing
    - out_items: list of all dict objects (mutated in place), including skipped ones
    """
    work: list[dict] = []
    out_items: list[Dict[str, Any]] = []

    for idx, obj in enumerate(items):
        if not isinstance(obj, dict):
            continue

        out_items.append(obj)

        text = obj.get(text_field, "")
        if not isinstance(text, str):
            text = str(text)

        clean_text = clean_ko_content_chunks([text])
        tok_count = count_tokens(clean_text)

        obj[f"{text_field}_clean"] = clean_text

        if tok_count < min_tokens:
            obj[f"{text_field}_metrics"] = {
                "token_count": int(tok_count),
                "skipped_reason": f"token_count < {min_tokens}",
            }
            continue

        spacy_text = clean_text if len(clean_text) <= spacy_max_chars else clean_text[:spacy_max_chars]

        work.append({
            "idx": idx,
            "obj": obj,
            "clean_text": clean_text,
            "tok_count": tok_count,
            "spacy_text": spacy_text,
        })

    return work, out_items


def main():
    parser = argparse.ArgumentParser(description="KO density / redundancy pipeline")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--text-field", default="ko_content_flat", help="Field containing main text")
    parser.add_argument("--min-tokens", type=int, default=50, help="Skip analysis below this token count")
    parser.add_argument("--domain-terms-file", default=None,
                        help="Optional file with one domain term per line (for domain term density)")
    parser.add_argument("--st-model", default="sentence-transformers/all-mpnet-base-v2",
                        help="SentenceTransformer model name for semantic similarity")
    parser.add_argument("--max-embed-sentences", type=int, default=40,
                        help="Max sentences to embed per document (samples evenly if more)")
    parser.add_argument("--disable-embeddings", action="store_true",
                        help="Disable sentence embedding similarity computation")
    parser.add_argument("--spacy-max-chars", type=int, default=200_000,
                        help="Max characters to feed into spaCy (POS/NER/parser) per document for stability")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for sentence-transformers embeddings (auto uses CUDA if available, else CPU)")
    parser.add_argument("--spacy-batch-size", type=int, default=16, help="spaCy pipe batch size")
    parser.add_argument("--spacy-n-process", type=int, default=1,
                        help="spaCy parallel processes (1 = no multiprocessing). "
                             "Use >1 only if you disable embeddings/GPU to avoid contention.")
    parser.add_argument("--log-every", type=int, default=25, help="Progress log frequency")

    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"[{now_utc_iso()}] START ko_compression_diagnostics")

    domain_terms = None
    if args.domain_terms_file:
        with open(args.domain_terms_file, "r", encoding="utf-8") as f:
            domain_terms = [line.strip() for line in f if line.strip()]

    nlp = load_spacy_model()

    # Allow very long documents (spaCy default is 1,000,000 chars)
    nlp.max_length = max(nlp.max_length, 2_000_000)

    st_model = None
    if not args.disable_embeddings:
        st_model = load_sentence_transformer(args.st_model, device=args.device)

    print(f"[{now_utc_iso()}] SentenceTransformer device: {st_model.device if st_model else 'DISABLED'}")

    # Enable sentence segmentation
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        # en_core_web_lg normally includes parser; this is just a guard.
        nlp.add_pipe("sentencizer")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = ensure_list(data)

    # Build worklist (cleaning + token counting happens here; spaCy happens in a batch later)
    work, out_items = build_work_items(
        items=items,
        text_field=args.text_field,
        min_tokens=args.min_tokens,
        spacy_max_chars=args.spacy_max_chars,
    )

    print(
        f"[{now_utc_iso()}] Prepared {len(work)} items for spaCy; skipped {len(out_items) - len(work)} (min_tokens={args.min_tokens}).")

    # If nothing to process, just write output
    if not work:
        output_data: Union[List[Dict[str, Any]], Dict[str, Any]]
        output_data = out_items if isinstance(data, list) else (out_items[0] if out_items else {})
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"[{now_utc_iso()}] DONE (nothing to process). Wrote output to: {args.output}")
        return

    # spaCy batch processing (major speedup vs nlp(text) in a loop)
    spacy_texts = [w["spacy_text"] for w in work]

    # Warning for multiprocessing + GPU embeddings
    if args.spacy_n_process > 1 and (st_model is not None) and args.device != "cpu":
        print(f"[{now_utc_iso()}] WARNING: spacy-n-process>1 with embeddings on GPU may cause contention. "
              f"Consider --device cpu or --disable-embeddings.")

    docs_iter = nlp.pipe(
        spacy_texts,
        batch_size=args.spacy_batch_size,
        n_process=args.spacy_n_process,
    )

    # Compute metrics per doc
    for k, (w, doc) in enumerate(zip(work, docs_iter), start=1):
        t_item0 = time.perf_counter()

        obj = w["obj"]
        clean_text = w["clean_text"]
        tok_count = w["tok_count"]

        m1 = lexical_density(doc)
        m2 = entropy_and_repetition(clean_text)  # full text
        m3 = semantic_signals(doc, domain_terms=domain_terms)
        m4 = structural_complexity(doc)

        m5 = {}
        if st_model is not None:
            m5 = sentence_embedding_similarity(doc, st_model, max_sentences=args.max_embed_sentences)

        merged = {
            "token_count": int(tok_count),
            "spacy_chars_used": int(len(w["spacy_text"])),
            "spacy_chars_total": int(len(clean_text)),
            **m1,
            **m2,
            **m3,
            **m4,
            **m5,
        }

        merged.update(compute_cds_and_policy(merged, tok_count))
        obj[f"{args.text_field}_metrics"] = merged

        # Progress logs
        if k == 1 or k == len(work) or (k % args.log_every == 0):
            elapsed = time.perf_counter() - t0
            avg = elapsed / max(1, k)
            eta_s = (len(work) - k) * avg
            print(
                f"[{now_utc_iso()}] {k}/{len(work)} processed | "
                f"last={time.perf_counter() - t_item0:.2f}s avg={avg:.2f}s ETA={eta_s / 60:.1f}m"
            )

    # Write in same shape as input
    # Write in same shape as input
    output_data: Union[List[Dict[str, Any]], Dict[str, Any]]
    if isinstance(data, list):
        output_data = out_items
    else:
        output_data = out_items[0] if out_items else {}

    print(f"[{now_utc_iso()}] Writing output JSON…")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    total_s = time.perf_counter() - t0
    print(f"[{now_utc_iso()}] DONE in {total_s / 60:.2f} minutes. Wrote output to: {args.output}")


if __name__ == "__main__":
    main()
