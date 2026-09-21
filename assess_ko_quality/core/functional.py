# assess_ko_quality/quality_functional_kc.py
"""
Improved functional quality scoring for search/RAG friendliness.

Improvements over quality_functional.py:
1. Cached tokenization - no repeated processing
2. Configurable thresholds via environment variables
3. Case-insensitive keyword matching
4. Configurable RAG step cues
5. Consistent integer scoring (no float accumulation)
6. Input validation - handles None gracefully
7. Comprehensive diagnostics
8. More precise scoring logic
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from quality_text_utils import tokens, jaccard, count_urls, count_all_caps_runs, _STOP_EN


# ============================================================================
# CONFIGURATION
# ============================================================================

# BM25 thresholds
BM25_LEN_OPTIMAL = tuple(map(int, os.environ.get("FUNC_BM25_LEN_OPT", "150,4000").split(",")))
BM25_LEN_ACCEPTABLE = tuple(map(int, os.environ.get("FUNC_BM25_LEN_ACC", "80,6000").split(",")))
BM25_JACCARD_TITLE_STRONG = float(os.environ.get("FUNC_BM25_JT_STRONG", "0.3"))
BM25_JACCARD_TITLE_WEAK = float(os.environ.get("FUNC_BM25_JT_WEAK", "0.15"))
BM25_JACCARD_DESC_STRONG = float(os.environ.get("FUNC_BM25_JD_STRONG", "0.3"))
BM25_JACCARD_DESC_WEAK = float(os.environ.get("FUNC_BM25_JD_WEAK", "0.15"))

# Embedding readiness thresholds
EMB_COHERENCE_HIGH = float(os.environ.get("FUNC_EMB_COH_HIGH", "0.35"))
EMB_COHERENCE_MED = float(os.environ.get("FUNC_EMB_COH_MED", "0.2"))
EMB_NOISE_URLS = int(os.environ.get("FUNC_EMB_NOISE_URLS", "5"))
EMB_NOISE_CAPS = int(os.environ.get("FUNC_EMB_NOISE_CAPS", "5"))

# RAG thresholds
RAG_LEN_OPTIMAL = tuple(map(int, os.environ.get("FUNC_RAG_LEN_OPT", "200,3000").split(",")))
RAG_STEP_CUES = [
    c.strip().lower()
    for c in os.environ.get(
        "FUNC_RAG_CUES",
        "step 1,step 2,step 3,step 4,step 5,section,introduction,conclusion,summary",
    ).split(",")
    if c.strip()
]

# Keyword indexability thresholds (count -> score mapping)
KW_INDEX_THRESHOLDS = [
    (int(os.environ.get("FUNC_KW_T5", "3")), 5),
    (int(os.environ.get("FUNC_KW_T4", "2")), 4),
    (int(os.environ.get("FUNC_KW_T3", "1")), 3),
]
KW_INDEX_MIN_PRESENT = int(os.environ.get("FUNC_KW_MIN", "1"))  # Has at least 1 keyword but none in content


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FunctionalMetrics:
    """Cached metrics for functional scoring."""
    title: str
    desc: str
    content: str
    keywords: List[str]
    
    # Token metrics
    title_tokens: List[str]
    desc_tokens: List[str]
    content_tokens: List[str]
    content_tokens_lower_set: Set[str]  # For O(1) case-insensitive lookup
    kw_tokens: List[str]
    meta_tokens: List[str]
    
    # Lengths
    content_len: int
    
    # Jaccard similarities (precomputed)
    jaccard_title_content: float
    jaccard_desc_content: float
    jaccard_meta_content: float


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def _compute_metrics(
    title: str,
    desc: str,
    content: str,
    keywords: List[str],
) -> FunctionalMetrics:
    """Compute all functional metrics in one pass."""
    # Normalize inputs
    t = title if isinstance(title, str) else ""
    d = desc if isinstance(desc, str) else ""
    c = content if isinstance(content, str) else ""
    kw = keywords if isinstance(keywords, list) else []
    
    # Tokenize once
    t_toks = tokens(t)
    d_toks = tokens(d)
    c_toks = tokens(c)
    c_toks_lower_set = {tok.lower() for tok in c_toks}  # For case-insensitive lookup
    kw_toks = tokens(" ".join(str(k) for k in kw))
    m_toks = tokens(t + " " + d)
    
    return FunctionalMetrics(
        title=t, desc=d, content=c, keywords=kw,
        title_tokens=t_toks, desc_tokens=d_toks,
        content_tokens=c_toks, content_tokens_lower_set=c_toks_lower_set,
        kw_tokens=kw_toks, meta_tokens=m_toks,
        content_len=len(c_toks),
        jaccard_title_content=jaccard(t_toks, c_toks),
        jaccard_desc_content=jaccard(d_toks, c_toks),
        jaccard_meta_content=jaccard(m_toks, c_toks),
    )


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def _score_bm25(metrics: FunctionalMetrics) -> Tuple[int, Dict[str, Any]]:
    """
    Score BM25 readiness (0-5).
    Based on content length and lexical overlap between title/desc and content.
    
    NOTE: Uses internal 0-10 scale then maps to 0-5 to preserve nuance with integer output.
    This preserves "weak" overlap credit (1 point on 0-10 scale = 0.5 on 0-5 scale).
    """
    diagnostics = {
        "bm25_content_len": metrics.content_len,
        "bm25_jaccard_title": round(metrics.jaccard_title_content, 3),
        "bm25_jaccard_desc": round(metrics.jaccard_desc_content, 3),
    }
    
    # Internal 0-10 scale for nuance
    internal_score = 0
    
    # Content length component (0-4 points on 0-10 scale = 0-2 on 0-5 scale)
    opt_min, opt_max = BM25_LEN_OPTIMAL
    acc_min, acc_max = BM25_LEN_ACCEPTABLE
    
    if opt_min <= metrics.content_len <= opt_max:
        internal_score += 4
        diagnostics["bm25_len_score"] = 2
    elif acc_min <= metrics.content_len < opt_min or opt_max < metrics.content_len <= acc_max:
        internal_score += 2
        diagnostics["bm25_len_score"] = 1
    else:
        diagnostics["bm25_len_score"] = 0
    
    # Title overlap component (0-4 points on 0-10 scale = 0-2 on 0-5 scale)
    if metrics.jaccard_title_content >= BM25_JACCARD_TITLE_STRONG:
        internal_score += 4
        diagnostics["bm25_title_score"] = 2
    elif metrics.jaccard_title_content >= BM25_JACCARD_TITLE_WEAK:
        internal_score += 2
        diagnostics["bm25_title_score"] = 1
    else:
        diagnostics["bm25_title_score"] = 0
    
    # Description overlap component (0-2 points on 0-10 scale = 0-1 on 0-5 scale)
    # Strong overlap gets 2 points; weak overlap gets 1 point (preserves nuance!)
    if metrics.jaccard_desc_content >= BM25_JACCARD_DESC_STRONG:
        internal_score += 2
        diagnostics["bm25_desc_score"] = 1
    elif metrics.jaccard_desc_content >= BM25_JACCARD_DESC_WEAK:
        internal_score += 1  # Half credit on 0-5 scale (1 point = 0.5)
        diagnostics["bm25_desc_score"] = 0.5
    else:
        diagnostics["bm25_desc_score"] = 0
    
    # Map 0-10 internal to 0-5 final (round for integer output)
    final_score = min(5, round(internal_score / 2))
    diagnostics["bm25_internal_score"] = internal_score
    diagnostics["bm25_raw_score"] = internal_score / 2  # Equivalent 0-5 scale
    
    return final_score, diagnostics


def _score_embedding(metrics: FunctionalMetrics) -> Tuple[int, Dict[str, Any]]:
    """
    Score embedding readiness (0-5).
    Based on coherence (meta-content overlap) and noise level.
    """
    diagnostics = {
        "emb_coherence": round(metrics.jaccard_meta_content, 3),
    }
    
    # Noise detection
    url_count = count_urls(metrics.content)
    all_caps = count_all_caps_runs(metrics.content)
    
    noise_score = 5
    if url_count > EMB_NOISE_URLS:
        noise_score -= 1
    if all_caps > EMB_NOISE_CAPS:
        noise_score -= 1
    noise_score = max(0, noise_score)
    
    diagnostics["emb_noise_url_count"] = url_count
    diagnostics["emb_noise_caps_count"] = all_caps
    diagnostics["emb_noise_score"] = noise_score
    
    # Coherence component (0-3 points)
    score = 0
    if metrics.jaccard_meta_content >= EMB_COHERENCE_HIGH:
        score += 3
        diagnostics["emb_coherence_score"] = 3
    elif metrics.jaccard_meta_content >= EMB_COHERENCE_MED:
        score += 2
        diagnostics["emb_coherence_score"] = 2
    elif metrics.jaccard_meta_content > 0:
        score += 1
        diagnostics["emb_coherence_score"] = 1
    else:
        diagnostics["emb_coherence_score"] = 0
    
    # Noise bonus (0-2 points)
    if noise_score >= 4:
        score += 2
        diagnostics["emb_noise_bonus"] = 2
    elif noise_score >= 3:
        score += 1
        diagnostics["emb_noise_bonus"] = 1
    else:
        diagnostics["emb_noise_bonus"] = 0
    
    final_score = min(5, score)
    diagnostics["emb_raw_score"] = score
    
    return final_score, diagnostics


def _score_rag(metrics: FunctionalMetrics) -> Tuple[int, Dict[str, Any]]:
    """
    Score RAG readiness (0-5).
    Based on segmentation cues and moderate content length.
    """
    lower_content = metrics.content.lower()
    
    # Count step/section cues
    step_hits = sum(1 for cue in RAG_STEP_CUES if cue in lower_content)
    
    diagnostics = {
        "rag_step_hits": step_hits,
        "rag_step_cues_found": [c for c in RAG_STEP_CUES if c in lower_content],
        "rag_content_len": metrics.content_len,
    }
    
    score = 0
    
    # Length component (0-2 points)
    opt_min, opt_max = RAG_LEN_OPTIMAL
    if opt_min <= metrics.content_len <= opt_max:
        score += 2
        diagnostics["rag_len_score"] = 2
    else:
        diagnostics["rag_len_score"] = 0
    
    # Step cues component (0-3 points)
    if step_hits >= 3:
        score += 3
        diagnostics["rag_step_score"] = 3
    elif step_hits == 2:
        score += 2
        diagnostics["rag_step_score"] = 2
    elif step_hits == 1:
        score += 1
        diagnostics["rag_step_score"] = 1
    else:
        diagnostics["rag_step_score"] = 0
    
    final_score = min(5, score)
    diagnostics["rag_raw_score"] = score
    
    return final_score, diagnostics


def _score_keyword_indexability(metrics: FunctionalMetrics) -> Tuple[int, Dict[str, Any]]:
    """
    Score keyword indexability (0-5).
    Based on how many non-stopword keywords actually appear in content.
    Case-insensitive matching.
    """
    # Filter stopwords and empty tokens
    kw_clean = [k for k in metrics.kw_tokens if k and k not in _STOP_EN]
    
    if not kw_clean:
        return 0, {
            "kw_total": 0,
            "kw_clean": 0,
            "kw_in_content": 0,
            "kw_in_content_list": [],
        }
    
    # Case-insensitive check using pre-computed lowercase set
    kw_in_content = [k for k in kw_clean if k.lower() in metrics.content_tokens_lower_set]
    
    count = len(kw_in_content)
    
    # Score mapping using configurable thresholds (descending order)
    score = 0
    for threshold, sc in sorted(KW_INDEX_THRESHOLDS, key=lambda x: x[0], reverse=True):
        if count >= threshold:
            score = sc
            break
    
    # Has keywords but none in content
    if count == 0 and len(kw_clean) >= KW_INDEX_MIN_PRESENT:
        score = 1
    
    diagnostics = {
        "kw_total": len(metrics.kw_tokens),
        "kw_clean": len(kw_clean),
        "kw_in_content": count,
        "kw_in_content_list": sorted(set(kw_in_content)),
        "kw_score_mapping": f"{count} keywords -> score {score}",
    }
    
    return score, diagnostics


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def functional_scores(
    title: str,
    desc: str,
    content: str,
    keywords: List[str],
) -> Dict[str, Any]:
    """
    Compute Functional sub-scores with improved efficiency and diagnostics.
    
    IMPROVEMENTS:
    - All metrics computed once, cached
    - Configurable thresholds via environment variables
    - Case-insensitive keyword matching
    - Consistent integer scoring (no float accumulation)
    - Comprehensive diagnostics for each component
    
    Args:
        title: KO title
        desc: KO description
        content: KO body content
        keywords: List of keyword strings
    
    Returns:
        Dictionary with scores (0-5), aggregates, and diagnostic metrics
    """
    # Compute all metrics in one pass
    metrics = _compute_metrics(title, desc, content, keywords)
    
    # Score each component
    bm25_score, bm25_diag = _score_bm25(metrics)
    emb_score, emb_diag = _score_embedding(metrics)
    rag_score, rag_diag = _score_rag(metrics)
    kw_score, kw_diag = _score_keyword_indexability(metrics)
    
    # Aggregate
    func_raw = bm25_score + emb_score + rag_score + kw_score
    func_scaled = min(25, round(func_raw * 25.0 / 20.0))
    
    return {
        # Main scores
        "Functional_BM25_readiness": bm25_score,
        "Functional_embedding_readiness": emb_score,
        "Functional_RAG_readiness": rag_score,
        "Functional_keyword_indexability": kw_score,
        
        # Aggregates
        "Functional_Total_Raw": func_raw,
        "Functional_Score_0_25": func_scaled,
        
        # Component diagnostics
        **bm25_diag,
        **emb_diag,
        **rag_diag,
        **kw_diag,
        
        # Legacy compatibility
        "Functional_keywords_in_content": ";".join(kw_diag["kw_in_content_list"]),
    }


__all__ = ["functional_scores", "FunctionalMetrics"]
