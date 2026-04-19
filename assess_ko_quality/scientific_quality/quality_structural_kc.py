# assess_ko_quality/quality_structural_kc.py
"""
Improved structural quality scoring with configurable thresholds,
better performance, and comprehensive diagnostics.

Improvements over quality_structural.py:
1. Caches tokenization results - no repeated processing of same text
2. Configurable thresholds via environment variables
3. Fair scoring: all KOs evaluated on same components (missing = 0)
4. Input validation and type safety
5. Raw metrics included in output for debugging
6. Smarter noise detection (context-aware)
7. Better handling of optional fields
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from quality_text_utils import tokens, sentence_lengths, count_urls, count_non_ascii, count_all_caps_runs


# ============================================================================
# CONFIGURABLE THRESHOLDS (via environment variables)
# ============================================================================

# Content length thresholds (tokens)
CONTENT_TOKENS_OPTIMAL = tuple(map(int, os.environ.get("STRUCT_CONTENT_OPTIMAL", "150,3000").split(",")))
CONTENT_TOKENS_ACCEPTABLE = tuple(map(int, os.environ.get("STRUCT_CONTENT_ACCEPTABLE", "80,6000").split(",")))
CONTENT_CHARS_MIN = int(os.environ.get("STRUCT_CONTENT_CHARS_MIN", "600"))

# Title length thresholds
TITLE_WORDS_OPTIMAL = tuple(map(int, os.environ.get("STRUCT_TITLE_WORDS", "6,14").split(",")))
TITLE_CHARS_OPTIMAL = tuple(map(int, os.environ.get("STRUCT_TITLE_CHARS", "45,90").split(",")))

# Subtitle length thresholds  
SUBTITLE_WORDS_OPTIMAL = tuple(map(int, os.environ.get("STRUCT_SUBTITLE_WORDS", "8,20").split(",")))
SUBTITLE_CHARS_MAX = int(os.environ.get("STRUCT_SUBTITLE_CHARS_MAX", "140"))

# Description thresholds
DESC_WORDS_OPTIMAL = tuple(map(int, os.environ.get("STRUCT_DESC_WORDS", "40,80").split(",")))
DESC_CHARS_OPTIMAL = tuple(map(int, os.environ.get("STRUCT_DESC_CHARS", "300,600").split(",")))
DESC_SENTENCES_OPTIMAL = tuple(map(int, os.environ.get("STRUCT_DESC_SENTENCES", "2,4").split(",")))

# Keywords thresholds
KEYWORDS_OPTIMAL = tuple(map(int, os.environ.get("STRUCT_KEYWORDS", "4,10").split(",")))
KEYWORDS_ACCEPTABLE = tuple(map(int, os.environ.get("STRUCT_KEYWORDS_ACCEPTABLE", "2,15").split(",")))
KEYWORDS_MIN_REQUIRED = int(os.environ.get("STRUCT_KEYWORDS_MIN", "4"))

# Sentence length thresholds (tokens per sentence)
SENTENCE_LEN_OPTIMAL = tuple(map(int, os.environ.get("STRUCT_SENTENCE_OPTIMAL", "8,35").split(",")))
SENTENCE_LEN_GOOD = tuple(map(int, os.environ.get("STRUCT_SENTENCE_GOOD", "6,45").split(",")))
SENTENCE_LEN_ACCEPTABLE = tuple(map(int, os.environ.get("STRUCT_SENTENCE_ACCEPTABLE", "4,55").split(",")))

# Noise thresholds
NOISE_URLS_WARN = int(os.environ.get("STRUCT_NOISE_URLS_WARN", "5"))
NOISE_URLS_BAD = int(os.environ.get("STRUCT_NOISE_URLS_BAD", "15"))
NOISE_NON_ASCII_WARN = int(os.environ.get("STRUCT_NOISE_NON_ASCII_WARN", "20"))
NOISE_CAPS_WARN = int(os.environ.get("STRUCT_NOISE_CAPS_WARN", "5"))
NOISE_CAPS_BAD = int(os.environ.get("STRUCT_NOISE_CAPS_BAD", "20"))


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class TextMetrics:
    """Cached metrics for a text field to avoid recomputation."""
    text: str
    chars: int
    tokens: List[str]
    token_count: int
    sentences: List[int]  # token counts per sentence
    sentence_count: int
    
    @classmethod
    def from_text(cls, text: Optional[str]) -> "TextMetrics":
        """Compute all metrics for a text field in one pass."""
        safe_text = text if isinstance(text, str) else ""
        toks = tokens(safe_text)
        sent_lens = sentence_lengths(safe_text)
        return cls(
            text=safe_text,
            chars=len(safe_text),
            tokens=toks,
            token_count=len(toks),
            sentences=sent_lens,
            sentence_count=len(sent_lens),
        )


@dataclass(frozen=True)
class NoiseContext:
    """Context for noise evaluation to distinguish legitimate from problematic content."""
    url_count: int
    non_ascii_count: int
    all_caps_count: int
    total_chars: int
    total_tokens: int
    
    @property
    def url_density(self) -> float:
        """URLs per 1000 tokens - contextualizes raw count."""
        if self.total_tokens == 0:
            return 0.0
        return (self.url_count / self.total_tokens) * 1000
    
    @property
    def non_ascii_ratio(self) -> float:
        """Non-ASCII as proportion of text."""
        if self.total_chars == 0:
            return 0.0
        return self.non_ascii_count / self.total_chars
    
    @property
    def caps_ratio(self) -> float:
        """All-caps runs per 1000 tokens."""
        if self.total_tokens == 0:
            return 0.0
        return (self.all_caps_count / self.total_tokens) * 1000


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def _score_length_content(metrics: TextMetrics) -> int:
    """Score content body length (0-5)."""
    if metrics.token_count == 0:
        return 0
    
    # Token-based scoring
    if CONTENT_TOKENS_OPTIMAL[0] <= metrics.token_count <= CONTENT_TOKENS_OPTIMAL[1]:
        score = 5
    elif CONTENT_TOKENS_ACCEPTABLE[0] <= metrics.token_count < CONTENT_TOKENS_OPTIMAL[0] or \
         CONTENT_TOKENS_OPTIMAL[1] < metrics.token_count <= CONTENT_TOKENS_ACCEPTABLE[1]:
        score = 3
    else:
        score = 1
    
    # Char-based safeguard
    if metrics.chars < CONTENT_CHARS_MIN and score > 2:
        score = 2
    
    return score


def _score_length_title(metrics: TextMetrics) -> int:
    """Score title length (0-5). Missing title = 0."""
    if not metrics.text:
        return 0
    
    score = 5
    if not (TITLE_WORDS_OPTIMAL[0] <= metrics.token_count <= TITLE_WORDS_OPTIMAL[1]):
        score -= 2
    if not (TITLE_CHARS_OPTIMAL[0] <= metrics.chars <= TITLE_CHARS_OPTIMAL[1]):
        score -= 1
    return max(0, min(5, score))


def _score_length_subtitle(metrics: TextMetrics) -> int:
    """Score subtitle length (0-5). Missing subtitle = 0 (neutral, not penalized in aggregate)."""
    if not metrics.text:
        return 0
    
    score = 5
    if not (SUBTITLE_WORDS_OPTIMAL[0] <= metrics.token_count <= SUBTITLE_WORDS_OPTIMAL[1]):
        score -= 2
    if metrics.chars > SUBTITLE_CHARS_MAX:
        score -= 1
    return max(0, min(5, score))


def _score_length_description(metrics: TextMetrics) -> int:
    """Score description length (0-5). Missing description = 0."""
    if not metrics.text:
        return 0
    
    score = 5
    if not (DESC_WORDS_OPTIMAL[0] <= metrics.token_count <= DESC_WORDS_OPTIMAL[1]):
        score -= 2
    if not (DESC_CHARS_OPTIMAL[0] <= metrics.chars <= DESC_CHARS_OPTIMAL[1]):
        score -= 1
    if not (DESC_SENTENCES_OPTIMAL[0] <= metrics.sentence_count <= DESC_SENTENCES_OPTIMAL[1]):
        score -= 1
    return max(0, min(5, score))


def _score_length_keywords(kw_count: int) -> int:
    """Score keyword count (0-5)."""
    if kw_count == 0:
        return 0
    if KEYWORDS_OPTIMAL[0] <= kw_count <= KEYWORDS_OPTIMAL[1]:
        return 5
    if KEYWORDS_ACCEPTABLE[0] <= kw_count < KEYWORDS_OPTIMAL[0] or \
       KEYWORDS_OPTIMAL[1] < kw_count <= KEYWORDS_ACCEPTABLE[1]:
        return 3
    return 1


def _score_completeness(
    has_title: bool,
    has_desc: bool,
    has_content: bool,
    kw_count: int,
) -> Tuple[int, List[str]]:
    """
    Score completeness (0-5) and return list of missing fields.
    
    Critical fields: title, description, content
    Keywords: considered present if >= KEYWORDS_MIN_REQUIRED
    """
    missing = []
    if not has_title:
        missing.append("title")
    if not has_desc:
        missing.append("description")
    if not has_content:
        missing.append("content")
    if kw_count < KEYWORDS_MIN_REQUIRED:
        missing.append("keywords")
    
    if not missing:
        return 5, missing
    if len(missing) == 1:
        return 3, missing
    if len(missing) == 2:
        return 2, missing
    return 0, missing


def _score_noise(context: NoiseContext) -> Tuple[int, Dict[str, Any]]:
    """
    Score noise/cleanliness (0-5, higher = cleaner).
    
    Returns score plus diagnostics about what was penalized.
    """
    score = 5
    penalties = []
    
    # URL evaluation (contextualized by density)
    if context.url_count > NOISE_URLS_BAD:
        score -= 2
        penalties.append(f"excessive_urls({context.url_count})")
    elif context.url_count > NOISE_URLS_WARN:
        score -= 1
        penalties.append(f"many_urls({context.url_count})")
    
    # Non-ASCII evaluation (contextualized by ratio)
    # Allow higher absolute counts if text is long (non-ASCII ratio matters more)
    if context.non_ascii_ratio > 0.05:  # >5% non-ASCII is suspicious
        score -= 1
        penalties.append(f"high_non_ascii_ratio({context.non_ascii_ratio:.2%})")
    elif context.non_ascii_count > NOISE_NON_ASCII_WARN:
        score -= 1
        penalties.append(f"many_non_ascii({context.non_ascii_count})")
    
    # ALL CAPS evaluation (contextualized by density)
    if context.all_caps_count > NOISE_CAPS_BAD:
        score -= 2
        penalties.append(f"excessive_caps({context.all_caps_count})")
    elif context.all_caps_count > NOISE_CAPS_WARN:
        score -= 1
        penalties.append(f"many_caps({context.all_caps_count})")
    
    final_score = max(0, score)
    
    diagnostics = {
        "noise_url_count": context.url_count,
        "noise_url_density": round(context.url_density, 2),
        "noise_non_ascii_count": context.non_ascii_count,
        "noise_non_ascii_ratio": round(context.non_ascii_ratio, 4),
        "noise_caps_count": context.all_caps_count,
        "noise_caps_ratio": round(context.caps_ratio, 2),
        "noise_penalties": penalties,
    }
    
    return final_score, diagnostics


def _score_formatting(content_metrics: TextMetrics) -> Tuple[int, float]:
    """
    Score formatting/readability based on sentence length (0-5).
    
    Returns score and average sentence length.
    """
    if not content_metrics.sentences:
        return 1, 0.0
    
    avg_len = sum(content_metrics.sentences) / len(content_metrics.sentences)
    
    if SENTENCE_LEN_OPTIMAL[0] <= avg_len <= SENTENCE_LEN_OPTIMAL[1]:
        score = 5
    elif SENTENCE_LEN_GOOD[0] <= avg_len <= SENTENCE_LEN_GOOD[1]:
        score = 4
    elif SENTENCE_LEN_ACCEPTABLE[0] <= avg_len <= SENTENCE_LEN_ACCEPTABLE[1]:
        score = 3
    else:
        score = 1
    
    return score, avg_len


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def structural_scores(
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    desc: Optional[str] = None,
    content: Optional[str] = None,
    keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute Structural sub-scores with improved fairness and diagnostics.
    
    IMPROVEMENTS:
    - All metrics computed once and cached (no repeated tokenization)
    - Fair comparison: missing fields score 0, not omitted
    - Context-aware noise scoring (ratios, not just raw counts)
    - Configurable thresholds via environment variables
    - Raw metrics included for debugging
    
    Args:
        title: KO title (optional but recommended)
        subtitle: KO subtitle (optional)
        desc: KO description (optional but recommended)
        content: KO body content (optional but recommended)
        keywords: List of keyword strings (optional but recommended)
    
    Returns:
        Dictionary with scores (0-5), aggregates, and diagnostic metrics
    """
    # Normalize inputs
    keywords = keywords if isinstance(keywords, list) else []
    kw_count = len([k for k in keywords if k and str(k).strip()])
    
    # Compute all text metrics in one pass (cached)
    title_m = TextMetrics.from_text(title)
    subtitle_m = TextMetrics.from_text(subtitle)
    desc_m = TextMetrics.from_text(desc)
    content_m = TextMetrics.from_text(content)
    
    # Build combined text for noise analysis
    all_text = " ".join(filter(None, [title_m.text, subtitle_m.text, desc_m.text, content_m.text]))
    total_tokens = title_m.token_count + subtitle_m.token_count + desc_m.token_count + content_m.token_count
    
    # --- Length Component (0-5) ---
    # Core components always evaluated (missing = 0, penalized)
    length_components = [
        _score_length_content(content_m),
        _score_length_title(title_m),
        _score_length_description(desc_m),
        _score_length_keywords(kw_count),
    ]
    
    # Subtitle is OPTIONAL: only included in average if present
    # When missing, score is 0 but doesn't drag down the average
    subtitle_score = _score_length_subtitle(subtitle_m)
    if subtitle_m.text:
        length_components.append(subtitle_score)
    
    length_score = round(sum(length_components) / len(length_components))
    
    # --- Completeness Component (0-5) ---
    completeness_score, missing_fields = _score_completeness(
        has_title=bool(title_m.text),
        has_desc=bool(desc_m.text),
        has_content=bool(content_m.text),
        kw_count=kw_count,
    )
    
    # --- Noise Component (0-5) ---
    noise_context = NoiseContext(
        url_count=count_urls(all_text),
        non_ascii_count=count_non_ascii(all_text),
        all_caps_count=count_all_caps_runs(all_text),
        total_chars=len(all_text),
        total_tokens=total_tokens,
    )
    noise_score, noise_diagnostics = _score_noise(noise_context)
    
    # --- Formatting Component (0-5) ---
    formatting_score, avg_sentence_len = _score_formatting(content_m)
    
    # --- Aggregation ---
    struct_raw = length_score + completeness_score + noise_score + formatting_score
    struct_scaled = min(25, round(struct_raw * 25.0 / 20.0))  # map 0–20 → 0–25, capped
    
    return {
        # Main scores (0-5 each)
        "Structural_length": length_score,
        "Structural_completeness": completeness_score,
        "Structural_noise": noise_score,
        "Structural_formatting": formatting_score,
        
        # Aggregated scores
        "Structural_Total_Raw": struct_raw,
        "Structural_Score_0_25": struct_scaled,
        
        # Component breakdown for length (diagnostics)
        "Structural_length_content": length_components[0],
        "Structural_length_title": length_components[1],
        "Structural_length_desc": length_components[2],
        "Structural_length_keywords": length_components[3],
        "Structural_length_subtitle": subtitle_score if subtitle_m.text else 0,
        
        # Completeness diagnostics
        "Structural_missing_fields": missing_fields,
        "Structural_missing_count": len(missing_fields),
        
        # Formatting diagnostics
        "Structural_avg_sentence_len": round(avg_sentence_len, 2),
        "Structural_sentence_count": content_m.sentence_count,
        
        # Noise diagnostics (detailed)
        **noise_diagnostics,
        
        # Raw metrics (for debugging)
        "Structural_metrics_title_words": title_m.token_count,
        "Structural_metrics_title_chars": title_m.chars,
        "Structural_metrics_subtitle_words": subtitle_m.token_count,
        "Structural_metrics_subtitle_chars": subtitle_m.chars,
        "Structural_metrics_desc_words": desc_m.token_count,
        "Structural_metrics_desc_chars": desc_m.chars,
        "Structural_metrics_desc_sentences": desc_m.sentence_count,
        "Structural_metrics_content_words": content_m.token_count,
        "Structural_metrics_content_chars": content_m.chars,
        "Structural_metrics_content_sentences": content_m.sentence_count,
        "Structural_metrics_keywords_count": kw_count,
    }


# Backwards compatibility alias
structural_score = structural_scores


__all__ = ["structural_scores", "structural_score", "TextMetrics", "NoiseContext"]
