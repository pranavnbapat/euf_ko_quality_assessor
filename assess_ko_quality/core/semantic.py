# assess_ko_quality/quality_semantic_kc.py
"""
Improved semantic quality scoring with cached processing, configurable thresholds,
and comprehensive diagnostics.

Improvements over quality_semantic.py:
1. Cached spaCy analysis results - no re-processing the same text (memory-efficient)
2. Token-aware truncation for MNLI - not crude character chop
3. Configurable thresholds via environment variables
4. Device selection via env var - can force CPU
5. Input validation - handles None gracefully
6. Diagnostic metrics - full transparency into scoring
7. Language-aware processing - warns if non-English
8. Efficient text concatenation - join once, reuse
9. spaCy max length protection - prevents OOM on huge texts
10. No auto-download - fails fast with clear error message
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizerBase

from quality_text_utils import tokens, strip_stops, unique_ratio, jaccard, count_non_ascii, sentence_lengths


# ============================================================================
# CONFIGURATION
# ============================================================================

# Device selection (allow override)
MNLI_DEVICE = os.environ.get("MNLI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MNLI_MODEL_NAME = os.environ.get("MNLI_MODEL_NAME", "roberta-large-mnli")

# spaCy model
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_lg")
SPACY_MAX_CHARS = int(os.environ.get("SPACY_MAX_CHARS", "500000"))  # ~100K tokens max

# Clarity thresholds (avg sentence length, non-ascii count)
CLARITY_OPTIMAL = tuple(map(int, os.environ.get("SEM_CLARITY_OPTIMAL", "8,35,10").split(",")))
CLARITY_GOOD = tuple(map(int, os.environ.get("SEM_CLARITY_GOOD", "6,45,30").split(",")))
CLARITY_ACCEPTABLE = tuple(map(int, os.environ.get("SEM_CLARITY_ACCEPTABLE", "4,55,60").split(",")))

# Usefulness thresholds (imperative/enumeration signal weights)
USEFULNESS_IMP_WEIGHT = float(os.environ.get("SEM_USEFULNESS_IMP_W", "2.0"))
USEFULNESS_ENUM_WEIGHT = float(os.environ.get("SEM_USEFULNESS_ENUM_W", "1.5"))
USEFULNESS_THRESHOLDS = [
    (float(os.environ.get("SEM_USEFULNESS_T5", "1.2")), 5),
    (float(os.environ.get("SEM_USEFULNESS_T4", "0.8")), 4),
    (float(os.environ.get("SEM_USEFULNESS_T3", "0.5")), 3),
    (float(os.environ.get("SEM_USEFULNESS_T2", "0.2")), 2),
]

# Info density thresholds (unique ratio without stopwords)
INFO_DENSITY_THRESHOLDS = [
    (float(os.environ.get("SEM_DENSITY_T5", "0.38")), 5),
    (float(os.environ.get("SEM_DENSITY_T4", "0.30")), 4),
    (float(os.environ.get("SEM_DENSITY_T3", "0.24")), 3),
    (float(os.environ.get("SEM_DENSITY_T2", "0.18")), 2),
]

# Consistency thresholds (Jaccard similarity)
CONSISTENCY_THRESHOLDS = [
    (0.50, 5),
    (0.35, 4),
    (0.20, 3),
    (0.10, 2),
]

# MNLI thresholds
MNLI_CONTRA_THRESHOLD = float(os.environ.get("MNLI_CONTRA_THRESH", "0.5"))
MNLI_ENTAIL_STRONG = float(os.environ.get("MNLI_ENTAIL_STRONG", "0.7"))
MNLI_ENTAIL_MODERATE = float(os.environ.get("MNLI_ENTAIL_MODERATE", "0.5"))
MNLI_NEUTRAL_THRESHOLD = float(os.environ.get("MNLI_NEUTRAL_THRESH", "0.5"))


# ============================================================================
# LAZY-LOADED MODELS
# ============================================================================

_NLP: Optional[spacy.Language] = None
_MNLI_TOKENIZER: Optional[PreTrainedTokenizerBase] = None
_MNLI_MODEL: Optional[Any] = None


def _get_nlp() -> spacy.Language:
    """
    Lazy-load spaCy English model once.
    Respects SPACY_MODEL env var. Does NOT auto-download (fail fast in prod).
    Disables unused pipes (NER, textcat, lemmatizer) for performance.
    Keeps parser (for sent.root, tok.dep_) and tagger (for POS).
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    try:
        # Disable heavy pipes we don't need (we only need parser, tagger for deps/POS)
        _NLP = spacy.load(SPACY_MODEL, disable=["ner", "textcat", "lemmatizer"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{SPACY_MODEL}' not found. "
            f"Install with: python -m spacy download {SPACY_MODEL}"
        ) from e

    return _NLP


def _load_mnli_model() -> None:
    """
    Lazy-load MNLI model once and cache in globals.
    Respects MNLI_DEVICE env var.
    """
    global _MNLI_MODEL, _MNLI_TOKENIZER

    if _MNLI_MODEL is not None and _MNLI_TOKENIZER is not None:
        # Defensive: ensure type is correct (catches stale cache issues)
        if not isinstance(_MNLI_TOKENIZER, PreTrainedTokenizerBase):
            print(f"[MNLI] Warning: tokenizer wrong type ({type(_MNLI_TOKENIZER)}), reloading...")
            _MNLI_TOKENIZER = None
            _MNLI_MODEL = None
        else:
            return

    print(f"[MNLI] Loading model: {MNLI_MODEL_NAME} on device={MNLI_DEVICE} ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MNLI_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MNLI_MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"Failed to load MNLI model '{MNLI_MODEL_NAME}': {e}")
    
    model.to(MNLI_DEVICE)
    model.eval()

    # Defensive: verify tokenizer type before storing
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise RuntimeError(f"Loaded tokenizer has wrong type: {type(tokenizer)}")

    _MNLI_TOKENIZER = tokenizer
    _MNLI_MODEL = model

    print(f"[MNLI] Model loaded. Tokenizer: {type(_MNLI_TOKENIZER).__name__}")


# ============================================================================
# CONFIG VALIDATION (called lazily, not at import)
# ============================================================================

_CONFIG_VALIDATED = False


def _validate_config() -> None:
    """Validate configuration. Called once on first semantic_scores() invocation."""
    global _CONFIG_VALIDATED
    if _CONFIG_VALIDATED:
        return
    
    # Validate clarity thresholds
    for name, tup in [("OPTIMAL", CLARITY_OPTIMAL), ("GOOD", CLARITY_GOOD), ("ACCEPTABLE", CLARITY_ACCEPTABLE)]:
        if len(tup) != 3:
            raise ValueError(f"SEM_CLARITY_{name} must be 'min_len,max_len,non_ascii_max' (got {tup})")
    
    # Sanity check: ranges should be nested (optimal inside good inside acceptable)
    if not (CLARITY_ACCEPTABLE[0] <= CLARITY_GOOD[0] <= CLARITY_OPTIMAL[0] and
            CLARITY_OPTIMAL[1] <= CLARITY_GOOD[1] <= CLARITY_ACCEPTABLE[1]):
        print("[SEMANTIC] Warning: clarity thresholds look misordered (acceptable should be loosest)")
    
    _CONFIG_VALIDATED = True


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class SpaCyAnalysisResult:
    """
    Immutable results from spaCy analysis.
    Does NOT store spaCy Doc/Span objects (they're huge and unpicklable).
    Only stores the counts we actually need.
    """
    total_sentences: int = 0
    imperative_count: int = 0
    enumerated_count: int = 0
    
    @property
    def imperative_ratio(self) -> float:
        if self.total_sentences == 0:
            return 0.0
        return self.imperative_count / self.total_sentences
    
    @property
    def enumerated_ratio(self) -> float:
        if self.total_sentences == 0:
            return 0.0
        return self.enumerated_count / self.total_sentences


@dataclass
class SemanticMetrics:
    """All semantic-related metrics for a KO."""
    title: str
    subtitle: str
    desc: str
    content: str
    keywords: List[str]
    meta_text: str
    full_text: str
    kw_text: str
    content_tokens: List[str]
    content_tokens_nostop: List[str]
    meta_tokens: List[str]
    sentence_lengths: List[int]
    avg_sentence_len: float
    non_ascii_count: int
    unique_ratio: float
    jaccard_sim: float


# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================

@functools.lru_cache(maxsize=128)
def _analyze_usefulness_spacy_cached(capped_text: str) -> SpaCyAnalysisResult:
    """
    Analyze text for usefulness signals using spaCy (cached by capped text).
    
    LRU cache stores at most 128 entries. With max 500k chars per text,
    max memory for cache keys is ~64MB (plus result objects).
    
    Args:
        capped_text: Pre-capped text (max SPACY_MAX_CHARS)
    
    Returns:
        SpaCyAnalysisResult with counts only (no Doc objects)
    """
    if not capped_text or not capped_text.strip():
        return SpaCyAnalysisResult()
    
    nlp = _get_nlp()
    
    try:
        doc = nlp(capped_text)
    except Exception as e:
        print(f"[SEMANTIC] spaCy processing failed: {e}")
        return SpaCyAnalysisResult()
    
    sentences = list(doc.sents)
    total = len(sentences)
    
    if not sentences:
        return SpaCyAnalysisResult()
    
    imperative = 0
    enumerated = 0
    
    for sent in sentences:
        # Imperative heuristic: root is VERB and no nominal subject
        root = sent.root
        has_subject = any(tok.dep_.startswith("nsubj") for tok in sent)
        if root.pos_ == "VERB" and not has_subject:
            imperative += 1
        
        # Enumerated: numbered lists or bullet markers
        if len(sent) > 1:
            first = sent[0]
            if first.like_num and sent[1].text == ".":
                enumerated += 1
            elif first.text in {"-", "•", "*", "→", "⇒", ">"}:
                enumerated += 1
    
    return SpaCyAnalysisResult(
        total_sentences=total,
        imperative_count=imperative,
        enumerated_count=enumerated,
    )


def _analyze_usefulness_spacy(text: str) -> SpaCyAnalysisResult:
    """
    Analyze text for usefulness signals using spaCy (with caching).
    
    Caches by capped text content. Identical texts get cache hits.
    """
    capped = text[:SPACY_MAX_CHARS] if text else ""
    return _analyze_usefulness_spacy_cached(capped)


def _compute_metrics(
    title: Optional[str],
    subtitle: Optional[str],
    desc: Optional[str],
    content: Optional[str],
    keywords: Optional[List[str]],
) -> SemanticMetrics:
    """Compute all semantic metrics in one pass to avoid recomputation."""
    # Normalize inputs
    t = title if isinstance(title, str) else ""
    st = subtitle if isinstance(subtitle, str) else ""
    d = desc if isinstance(desc, str) else ""
    c = content if isinstance(content, str) else ""
    kw = keywords if isinstance(keywords, list) else []
    kw_text = " ".join(str(k) for k in kw if k and str(k).strip())
    
    # Combined texts (computed once)
    meta = " ".join(filter(None, [t, st, d]))
    full = " ".join(filter(None, [meta, c]))
    
    # Token metrics
    c_toks = tokens(c)
    c_toks_nostop = strip_stops(c_toks)
    m_toks = tokens(" ".join([t, d, kw_text]))
    
    # Sentence metrics (on full text for clarity)
    s_lens = sentence_lengths(full)
    avg_slen = sum(s_lens) / len(s_lens) if s_lens else 0.0
    
    return SemanticMetrics(
        title=t, subtitle=st, desc=d, content=c, keywords=kw,
        meta_text=meta, full_text=full, kw_text=kw_text,
        content_tokens=c_toks, content_tokens_nostop=c_toks_nostop,
        meta_tokens=m_toks, sentence_lengths=s_lens,
        avg_sentence_len=avg_slen, non_ascii_count=count_non_ascii(full),
        unique_ratio=unique_ratio(c_toks_nostop) if c_toks_nostop else 0.0,
        jaccard_sim=jaccard(m_toks, c_toks),
    )


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def _score_clarity(metrics: SemanticMetrics) -> Tuple[int, Dict[str, Any]]:
    """Score clarity based on sentence length and non-ASCII characters."""
    avg_len = metrics.avg_sentence_len
    non_ascii = metrics.non_ascii_count
    
    opt_len_min, opt_len_max, opt_ascii = CLARITY_OPTIMAL
    good_len_min, good_len_max, good_ascii = CLARITY_GOOD
    acc_len_min, acc_len_max, acc_ascii = CLARITY_ACCEPTABLE
    
    if opt_len_min <= avg_len <= opt_len_max and non_ascii <= opt_ascii:
        score = 5
    elif good_len_min <= avg_len <= good_len_max and non_ascii <= good_ascii:
        score = 4
    elif acc_len_min <= avg_len <= acc_len_max and non_ascii <= acc_ascii:
        score = 3
    elif avg_len > 0 or non_ascii > 0:
        score = 1
    else:
        score = 0
    
    return score, {
        "clarity_avg_sentence_len": round(avg_len, 2),
        "clarity_non_ascii": non_ascii,
    }


def _score_usefulness(metrics: SemanticMetrics) -> Tuple[int, Dict[str, Any]]:
    """Score usefulness based on imperative/enumerated sentence structure."""
    if not metrics.content_tokens:
        return 0, {"usefulness_total_sentences": 0, "usefulness_imperative": 0, "usefulness_enumerated": 0}
    
    analysis = _analyze_usefulness_spacy(metrics.full_text)
    
    if analysis.total_sentences == 0:
        return 1, {"usefulness_total_sentences": 0, "usefulness_imperative": 0, "usefulness_enumerated": 0}
    
    signal = USEFULNESS_IMP_WEIGHT * analysis.imperative_ratio + USEFULNESS_ENUM_WEIGHT * analysis.enumerated_ratio
    
    score = 1
    for threshold, sc in sorted(USEFULNESS_THRESHOLDS, key=lambda x: x[0], reverse=True):
        if signal >= threshold:
            score = sc
            break
    
    return score, {
        "usefulness_total_sentences": analysis.total_sentences,
        "usefulness_imperative": analysis.imperative_count,
        "usefulness_imperative_ratio": round(analysis.imperative_ratio, 3),
        "usefulness_enumerated": analysis.enumerated_count,
        "usefulness_enumerated_ratio": round(analysis.enumerated_ratio, 3),
        "usefulness_signal": round(signal, 3),
    }


def _score_info_density(metrics: SemanticMetrics) -> Tuple[int, Dict[str, Any]]:
    """Score information density based on lexical diversity."""
    ur = metrics.unique_ratio
    
    score = 0
    for threshold, sc in INFO_DENSITY_THRESHOLDS:
        if ur >= threshold:
            score = sc
            break
    
    if ur > 0 and score == 0:
        score = 1
    
    return score, {
        "info_density_unique_ratio": round(ur, 4),
        "info_density_content_tokens": len(metrics.content_tokens),
        "info_density_content_tokens_nostop": len(metrics.content_tokens_nostop),
    }


def _score_consistency(metrics: SemanticMetrics) -> Tuple[int, Dict[str, Any]]:
    """Score consistency based on Jaccard overlap between metadata and content."""
    jc = metrics.jaccard_sim
    
    score = 0
    for threshold, sc in CONSISTENCY_THRESHOLDS:
        if jc >= threshold:
            score = sc
            break
    
    if jc > 0 and score == 0:
        score = 1
    
    return score, {
        "consistency_jaccard": round(jc, 4),
        "consistency_meta_tokens": len(metrics.meta_tokens),
        "consistency_content_tokens": len(metrics.content_tokens),
    }


# ============================================================================
# MNLI FUNCTIONS
# ============================================================================

def _token_aware_truncate(text: str, max_tokens: int = 400) -> str:
    """Truncate text to fit within token budget using actual tokenizer."""
    if not text:
        return ""
    
    if _MNLI_TOKENIZER is None:
        return text[:max_tokens * 4]
    
    try:
        toks = _MNLI_TOKENIZER.encode(text, add_special_tokens=False)
        if len(toks) <= max_tokens:
            return text
        truncated = toks[:max_tokens]
        return _MNLI_TOKENIZER.decode(truncated, skip_special_tokens=True)
    except Exception:
        return text[:max_tokens * 4]


def _run_mnli_single(premise: str, hypothesis: str) -> Tuple[str, float, float, float]:
    """Run MNLI on a single (premise, hypothesis) pair. Returns: (label, p_entail, p_neutral, p_contra)"""
    global _MNLI_MODEL, _MNLI_TOKENIZER
    
    _load_mnli_model()
    assert _MNLI_TOKENIZER is not None and _MNLI_MODEL is not None
    
    premise_truncated = _token_aware_truncate(premise, max_tokens=400)
    
    inputs = _MNLI_TOKENIZER(
        premise_truncated, hypothesis,
        truncation=True, max_length=512, return_tensors="pt",
    )

    inputs = {k: v.to(MNLI_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _MNLI_MODEL(**inputs)
        logits = outputs.logits[0]
    
    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
    id2label = _MNLI_MODEL.config.id2label
    label_map = {id2label[i].upper(): i for i in id2label}
    
    idx_ent = label_map.get("ENTAILMENT")
    idx_neu = label_map.get("NEUTRAL")
    idx_con = label_map.get("CONTRADICTION")
    
    p_ent = float(probs[idx_ent]) if idx_ent is not None else 0.0
    p_neu = float(probs[idx_neu]) if idx_neu is not None else 0.0
    p_con = float(probs[idx_con]) if idx_con is not None else 0.0
    
    best_idx = int(probs.argmax())
    label = id2label[best_idx]
    
    return label, p_ent, p_neu, p_con


def _mnli_probs_to_score(p_ent: float, p_neu: float, p_con: float) -> float:
    """Map MNLI probabilities to 0-5 score."""
    if p_con >= MNLI_CONTRA_THRESHOLD:
        return 0.0
    if p_ent >= MNLI_ENTAIL_STRONG:
        return 5.0
    if p_ent >= MNLI_ENTAIL_MODERATE:
        return 4.0
    if p_neu >= MNLI_NEUTRAL_THRESHOLD:
        return 2.0
    return 1.0


# ============================================================================
# MAIN PUBLIC FUNCTIONS
# ============================================================================

def semantic_scores(
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    desc: Optional[str] = None,
    content: Optional[str] = None,
    keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Semantic sub-scores with improved efficiency and diagnostics."""
    _validate_config()
    
    metrics = _compute_metrics(title, subtitle, desc, content, keywords)
    
    clarity_score, clarity_diag = _score_clarity(metrics)
    usefulness_score, usefulness_diag = _score_usefulness(metrics)
    density_score, density_diag = _score_info_density(metrics)
    consistency_score, consistency_diag = _score_consistency(metrics)
    
    sem_raw = clarity_score + usefulness_score + density_score + consistency_score
    sem_scaled = min(25, round(sem_raw * 25.0 / 20.0))
    
    return {
        "Semantic_clarity": clarity_score,
        "Semantic_usefulness": usefulness_score,
        "Semantic_information_density": density_score,
        "Semantic_consistency": consistency_score,
        "Semantic_Total_Raw": sem_raw,
        "Semantic_Score_0_25": sem_scaled,
        **clarity_diag,
        **usefulness_diag,
        **density_diag,
        **consistency_diag,
        "Semantic_metrics_title_chars": len(metrics.title),
        "Semantic_metrics_desc_chars": len(metrics.desc),
        "Semantic_metrics_content_chars": len(metrics.content),
        "Semantic_metrics_content_tokens": len(metrics.content_tokens),
        "Semantic_metrics_keywords_count": len(metrics.keywords),
    }


def semantic_mnli_consistency(
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    desc: Optional[str] = None,
    content: Optional[str] = None,
    lang_meta: str = "unknown",
) -> Dict[str, Any]:
    """MNLI-based semantic consistency between metadata and content."""
    c = content if isinstance(content, str) else ""
    
    if not c or not lang_meta.startswith("en"):
        return {
            "Semantic_consistency_mnli_title": 0.0,
            "Semantic_consistency_mnli_subtitle": 0.0,
            "Semantic_consistency_mnli_desc": 0.0,
            "Semantic_consistency_mnli": 0.0,
            "Semantic_mnli_title_label": "",
            "Semantic_mnli_desc_label": "",
            "Semantic_mnli_subtitle_label": "",
            "Semantic_mnli_skipped": True,
            "Semantic_mnli_skip_reason": "non_english_or_no_content" if not c else "non_english",
        }
    
    results = {}
    scores = []
    
    fields = [
        ("title", title if isinstance(title, str) else ""),
        ("desc", desc if isinstance(desc, str) else ""),
        ("subtitle", subtitle if isinstance(subtitle, str) else ""),
    ]
    
    for field_name, field_text in fields:
        if field_text:
            try:
                label, p_ent, p_neu, p_con = _run_mnli_single(c, field_text)
                score = _mnli_probs_to_score(p_ent, p_neu, p_con)
                scores.append(score)
                
                results[f"Semantic_consistency_mnli_{field_name}"] = round(score, 2)
                results[f"Semantic_mnli_{field_name}_label"] = label
                results[f"Semantic_mnli_{field_name}_p_entail"] = round(p_ent, 4)
                results[f"Semantic_mnli_{field_name}_p_neutral"] = round(p_neu, 4)
                results[f"Semantic_mnli_{field_name}_p_contra"] = round(p_con, 4)
            except Exception as e:
                print(f"[MNLI] Error processing {field_name}: {e}")
                results[f"Semantic_consistency_mnli_{field_name}"] = 0.0
                results[f"Semantic_mnli_{field_name}_label"] = "ERROR"
                results[f"Semantic_mnli_{field_name}_p_entail"] = 0.0
                results[f"Semantic_mnli_{field_name}_p_neutral"] = 0.0
                results[f"Semantic_mnli_{field_name}_p_contra"] = 0.0
        else:
            results[f"Semantic_consistency_mnli_{field_name}"] = 0.0
            results[f"Semantic_mnli_{field_name}_label"] = ""
            results[f"Semantic_mnli_{field_name}_p_entail"] = 0.0
            results[f"Semantic_mnli_{field_name}_p_neutral"] = 0.0
            results[f"Semantic_mnli_{field_name}_p_contra"] = 0.0
    
    non_zero = [s for s in scores if s > 0]
    combined = sum(non_zero) / len(non_zero) if non_zero else 0.0
    
    results["Semantic_consistency_mnli"] = round(combined, 2)
    results["Semantic_mnli_skipped"] = False
    results["Semantic_mnli_skip_reason"] = ""
    
    return results


__all__ = ["semantic_scores", "semantic_mnli_consistency", "SemanticMetrics", "SpaCyAnalysisResult"]
