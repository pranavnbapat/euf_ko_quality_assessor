# assess_ko_quality/core/content_report.py
"""
What the file itself is like - reported, never scored.

The file's content is the original author's work, not the contributor's, so none of
this raises or lowers the metadata score. It is still worth knowing: a KO whose text
never extracted, or that is mostly repeated boilerplate, needs a different fix from one
with a poor title.

The checks are the published text-quality filters used to clean web text for language
models, with their published thresholds, so every flag here means "this text would be
removed by filter X of paper Y":

  Gopher / MassiveText quality and repetition filters (Rae et al. 2021), as
  re-implemented in datatrove and kept in FineWeb (Penedo et al., NeurIPS 2024)
  FineWeb's own three filters, chosen by ablation from over fifty candidates
  (Penedo et al., NeurIPS 2024)
  FineWeb2's adaptation to other languages: the same thresholds, with stop words taken
  per language (Penedo et al., COLM 2025) - here from wordfreq (Speer 2022)
  C4's rule-based filters (Raffel et al., JMLR 2020; audited by Dodge et al., EMNLP 2021)

plus the declared-vs-detected language check motivated by the multilingual audit of
Kreutzer et al. (TACL 2022), and descriptive measures: MTLD lexical diversity
(McCarthy & Jarvis 2010), LIX readability (Bjornsson 1968, 1983) and provenance cues.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from core.text_utils import mtld

# ---------------------------------------------------------------- published thresholds
# Gopher quality filter (Rae et al. 2021, Table A1; datatrove GopherQualityFilter).
GOPHER_QUALITY = {
    "min_words": 50, "max_words": 100_000,
    "min_mean_word_len": 3, "max_mean_word_len": 10,
    "max_symbol_word_ratio": 0.1,
    "max_bullet_line_ratio": 0.9,
    "max_ellipsis_line_ratio": 0.3,
    "min_alpha_word_ratio": 0.8,
    "min_stop_words": 2,
}
# Gopher repetition filter (same sources).
GOPHER_REPETITION = {
    "max_dup_line_frac": 0.3, "max_dup_para_frac": 0.3,
    "max_dup_line_char_frac": 0.2, "max_dup_para_char_frac": 0.2,
    "top_ngram_char_frac": {2: 0.20, 3: 0.18, 4: 0.16},
    "dup_ngram_char_frac": {5: 0.15, 6: 0.14, 7: 0.13, 8: 0.12, 9: 0.11, 10: 0.10},
}
# FineWeb custom filters (Penedo et al. 2024, Section 3.6): a document is removed when
#   fraction of lines ending in punctuation <= 0.12
#   fraction of characters in duplicated lines >= 0.1
#   fraction of lines shorter than 30 characters >= 0.67
FINEWEB = {"max_line_punct_ratio": 0.12, "min_dup_char_ratio": 0.1,
           "short_line_len": 30, "min_short_line_ratio": 0.67}

TERMINAL = (".", "!", "?", '"', "'", "”", "’", "»", ")", "…", ":", ";")
_BULLET = re.compile(r"^\s*[•\-\*–·▪►◦‣]\s")
_WORD = re.compile(r"\w+", re.UNICODE)
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.I)
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_SYMBOLS = ("#", "...", "…")


def _lines(text: str) -> List[str]:
    return [l for l in text.splitlines() if l.strip()]


def _paragraphs(text: str) -> List[str]:
    return [p for p in re.split(r"\n{2,}", text) if p.strip()]


def _dup_stats(items: Sequence[str]) -> Dict[str, float]:
    """Fraction of items, and of their characters, that repeat an earlier item."""
    seen = set()
    dup_n = dup_chars = total_chars = 0
    for it in items:
        total_chars += len(it)
        if it in seen:
            dup_n += 1
            dup_chars += len(it)
        seen.add(it)
    n = len(items) or 1
    return {"frac": dup_n / n, "char_frac": dup_chars / (total_chars or 1)}


def _top_ngram_char_frac(words: Sequence[str], n: int, text_len: int) -> float:
    grams = Counter(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    if not grams:
        return 0.0
    gram, count = grams.most_common(1)[0]
    return (len(gram) * count) / (text_len or 1) if count > 1 else 0.0


def _dup_ngram_char_frac(words: Sequence[str], n: int, text_len: int) -> float:
    """Characters inside n-grams that occur more than once, each position counted once."""
    if len(words) < n:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(words)
    for i, g in enumerate(grams):
        if counts[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    dup_chars = sum(len(w) for w, c in zip(words, covered) if c)
    return dup_chars / (text_len or 1)


def _stop_words(lang: str, n: int = 8) -> List[str]:
    """
    The language's n commonest words. Gopher's list is eight English words; FineWeb2
    replaces it per language with frequent words from reference text. wordfreq's top
    list plays that role here.
    """
    try:
        from wordfreq import available_languages, top_n_list

        code = {"hr": "sh", "sr": "sh", "bs": "sh"}.get(lang, lang)
        if code in available_languages():
            return [w for w in top_n_list(code, n * 3) if w.isalpha()][:n]
    except Exception:
        pass
    return []


def _lix(text: str) -> Optional[float]:
    words = re.findall(r"[^\W\d_]+", text)
    if len(words) < 100:
        return None
    sents = max(1, len(re.findall(r"[.!?:;]+(?:\s|$)", text)))
    return len(words) / sents + 100.0 * sum(1 for w in words if len(w) > 6) / len(words)


def report(raw_text: Any, declared_languages: Any, detected_language: str) -> Dict[str, Any]:
    """Measure one file's extracted text. Every value is descriptive; nothing is scored."""
    text = raw_text if isinstance(raw_text, str) else ""
    words = _WORD.findall(text)
    lower = [w.lower() for w in words]
    n_words = len(words)
    lines = _lines(text)
    out: Dict[str, Any] = {"content_chars": len(text), "content_words": n_words}

    # --- Gopher quality
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    symbols = sum(text.count(s) for s in _SYMBOLS)
    mean_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    bullet = sum(1 for l in lines if _BULLET.match(l)) / (len(lines) or 1)
    ellipsis = sum(1 for l in lines if l.rstrip().endswith(("...", "…"))) / (len(lines) or 1)
    stops = _stop_words(detected_language)
    stop_hits = len(set(lower) & set(stops)) if stops else None
    g = GOPHER_QUALITY
    gq = {
        "too_few_words": n_words < g["min_words"],
        "too_many_words": n_words > g["max_words"],
        "mean_word_length_out_of_range": n_words > 0 and not (g["min_mean_word_len"] <= mean_len <= g["max_mean_word_len"]),
        "symbol_ratio_high": n_words > 0 and symbols / n_words > g["max_symbol_word_ratio"],
        "bullet_lines_high": bullet > g["max_bullet_line_ratio"],
        "ellipsis_lines_high": ellipsis > g["max_ellipsis_line_ratio"],
        "alpha_words_low": n_words > 0 and alpha_words / n_words < g["min_alpha_word_ratio"],
        "stop_words_missing": stop_hits is not None and stop_hits < g["min_stop_words"],
    }

    # --- Gopher repetition
    r = GOPHER_REPETITION
    line_dup = _dup_stats(lines)
    para_dup = _dup_stats(_paragraphs(text))
    gr = {
        "dup_lines_high": line_dup["frac"] > r["max_dup_line_frac"],
        "dup_paragraphs_high": para_dup["frac"] > r["max_dup_para_frac"],
        "dup_line_chars_high": line_dup["char_frac"] > r["max_dup_line_char_frac"],
        "dup_paragraph_chars_high": para_dup["char_frac"] > r["max_dup_para_char_frac"],
    }
    top_ng = {n: _top_ngram_char_frac(lower, n, len(text)) for n in r["top_ngram_char_frac"]}
    dup_ng = {n: _dup_ngram_char_frac(lower, n, len(text)) for n in r["dup_ngram_char_frac"]}
    gr["top_ngram_repetition_high"] = any(top_ng[n] > t for n, t in r["top_ngram_char_frac"].items())
    gr["dup_ngram_repetition_high"] = any(dup_ng[n] > t for n, t in r["dup_ngram_char_frac"].items())

    # --- FineWeb custom filters
    f = FINEWEB
    punct_ratio = sum(1 for l in lines if l.rstrip().endswith(TERMINAL)) / (len(lines) or 1)
    short_ratio = sum(1 for l in lines if len(l.strip()) <= f["short_line_len"]) / (len(lines) or 1)
    fw = {
        # With no line breaks at all the line ratios describe one line, which says
        # nothing about the document, so they are only applied when there are lines.
        "few_lines_end_in_punctuation": len(lines) > 1 and punct_ratio <= f["max_line_punct_ratio"],
        "dup_line_chars_high": line_dup["char_frac"] >= f["min_dup_char_ratio"],
        "short_lines_dominant": len(lines) > 1 and short_ratio >= f["min_short_line_ratio"],
    }

    # --- C4 rules kept by FineWeb (all but the terminal-punctuation rule)
    low_text = text.lower()
    c4 = {
        "lorem_ipsum": "lorem ipsum" in low_text,
        "curly_bracket": "{" in text,
        "javascript_lines": any("javascript" in l.lower() for l in lines),
    }

    flags = {**{f"gopher_quality.{k}": v for k, v in gq.items()},
             **{f"gopher_repetition.{k}": v for k, v in gr.items()},
             **{f"fineweb.{k}": v for k, v in fw.items()},
             **{f"c4.{k}": v for k, v in c4.items()}}
    raised = sorted(k for k, v in flags.items() if v)

    # --- language: declared vs detected (Kreutzer et al. 2022)
    declared = declared_languages if isinstance(declared_languages, list) else \
        ([declared_languages] if declared_languages else [])
    declared_codes = {_lang_code(d) for d in declared if d}
    mismatch = None
    if detected_language not in ("unknown", "") and declared_codes - {None}:
        mismatch = detected_language not in declared_codes

    out.update({
        "content_extraction_failed": n_words < g["min_words"],
        "content_filters_raised": raised,
        "content_filters_raised_count": len(raised),
        "content_would_pass_web_filters": not raised,
        "content_mean_word_length": round(mean_len, 2),
        "content_line_punct_ratio": round(punct_ratio, 3),
        "content_short_line_ratio": round(short_ratio, 3),
        "content_dup_line_char_frac": round(line_dup["char_frac"], 3),
        "content_mtld": round(mtld(lower), 1) if n_words >= 50 else None,
        "content_lix": None if (l := _lix(text)) is None else round(l, 1),
        "content_language_detected": detected_language,
        "content_language_declared": sorted(d for d in declared_codes if d),
        "content_language_mismatch": mismatch,
        "content_doi_count": len(_DOI.findall(text)),
        "content_url_count": len(_URL.findall(text)),
        "content_years_mentioned": len(set(m.group(0) for m in _YEAR.finditer(text))),
    })
    return out


def _lang_code(name: Any) -> Optional[str]:
    """Platform language name ("Spanish") -> ISO 639-1 code."""
    s = str(name).strip()
    if len(s) == 2 and s.isalpha():
        return s.lower()
    return _NAME_TO_CODE.get(s.lower())


# Platform language names -> ISO 639-1 codes (a standard table, not a judgement).
_NAME_TO_CODE = {
    "bulgarian": "bg", "croatian": "hr", "czech": "cs", "danish": "da", "dutch": "nl",
    "english": "en", "estonian": "et", "finnish": "fi", "french": "fr", "german": "de",
    "greek": "el", "hungarian": "hu", "irish": "ga", "italian": "it", "latvian": "lv",
    "lithuanian": "lt", "maltese": "mt", "polish": "pl", "portuguese": "pt",
    "romanian": "ro", "slovak": "sk", "slovene": "sl", "slovenian": "sl",
    "spanish": "es", "swedish": "sv", "catalan": "ca", "norwegian": "no",
}


__all__ = ["report", "GOPHER_QUALITY", "GOPHER_REPETITION", "FINEWEB"]
