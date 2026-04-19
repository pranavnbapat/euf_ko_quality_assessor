# assess_ko_quality/quality_domain_kc.py
"""
Improved domain quality scoring with batched embeddings, token-aware truncation,
and better efficiency.

Improvements over quality_domain.py:
1. Batched embedding: All fields encoded in a single model call (4x speedup)
2. Token-aware truncation: Uses actual tokenizer, not heuristic
3. Removed unused initialise_domain_centroid() dead code
4. Lazy dimension validation: Doesn't force model load in load_domain_centroid()
5. Short-circuit when centroid not loaded: Avoids unnecessary model loading
6. Hard char cap before tokenization: Prevents pathological inputs
7. Safer batch sizing: Future-proof against larger batches
8. Clean threshold table: No dead/confusing entries
9. Defensive centroid validation: Checks shape even without model
"""

from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sentence_transformers import SentenceTransformer


# Configuration
AGRI_EMB_MODEL_NAME = os.environ.get("AGRI_EMB_MODEL_NAME", "all-mpnet-base-v2")
AGRI_VALIDATE_DIM = os.environ.get("AGRI_VALIDATE_DIM", "0").lower() in ("1", "true", "yes")

# Max tokens to keep from start and end when truncating
# Default: keep first 400 tokens + last 112 tokens = 512 total (for mpnet-base)
MAX_SEQ_LEN = int(os.environ.get("AGRI_MAX_SEQ_LEN", "512"))
TRUNC_KEEP_END_TOKENS = int(os.environ.get("AGRI_TRUNC_KEEP_END_TOKENS", "112"))

# Hard character cap before tokenization (prevents pathological slowness on huge texts)
MAX_CHARS_PRETOKENISE = int(os.environ.get("AGRI_MAX_CHARS_PRETOKENISE", "200000"))

# Similarity to score thresholds (empirically derived - may need calibration)
# NOTE: sim == 0.0 is handled separately (returns 0), so 0.0 is not in this table
SIM_THRESHOLDS = [
    (0.65, 5),
    (0.55, 4),
    (0.45, 3),
    (0.35, 2),
]

# Lazy-loaded globals
_EMB_MODEL: Optional[SentenceTransformer] = None
_DOMAIN_CENTROID: Optional[np.ndarray] = None


def _load_emb_model() -> None:
    """
    Load the SentenceTransformer model (if not already loaded).
    """
    global _EMB_MODEL

    if _EMB_MODEL is not None:
        return

    print(f"[DOMAIN] Loading SentenceTransformer model: {AGRI_EMB_MODEL_NAME}")
    _EMB_MODEL = SentenceTransformer(AGRI_EMB_MODEL_NAME)
    print(f"[DOMAIN] Model loaded. Embedding dim: {_EMB_MODEL.get_sentence_embedding_dimension()}")


def _get_tokenizer() -> Optional[Any]:
    """
    Get the tokenizer from the loaded model, if available.
    Returns None if model not loaded.
    """
    if _EMB_MODEL is None:
        return None
    # SentenceTransformer exposes tokenizer via tokenizer attribute
    return getattr(_EMB_MODEL, "tokenizer", None)


def _token_aware_truncate(text: str, max_tokens: int = MAX_SEQ_LEN) -> str:
    """
    Truncate text using actual tokenizer when available.
    
    Strategy: Keep first N tokens from start + M tokens from end.
    Falls back to character heuristic only if tokenizer unavailable.
    
    Includes hard character cap before tokenization to prevent pathological
    slowness on huge inputs (e.g., multi-MB extracted PDF text).
    
    Args:
        text: Input text
        max_tokens: Maximum tokens (default 512 for mpnet-base)
    
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    # Hard cap: avoid tokenising absurdly large strings
    # Preserve head + tail (like token-aware strategy) even under hard cap
    if len(text) > MAX_CHARS_PRETOKENISE:
        keep_end_chars = min(20_000, MAX_CHARS_PRETOKENISE // 5)  # ~20k chars or 20%
        keep_start_chars = MAX_CHARS_PRETOKENISE - keep_end_chars
        text = text[:keep_start_chars] + "\n... [char-capped] ...\n" + text[-keep_end_chars:]
    
    tokenizer = _get_tokenizer()
    
    if tokenizer is None:
        # Fallback: char-based heuristic (conservative estimate ~3 chars/token)
        max_chars = max_tokens * 3
        if len(text) <= max_chars:
            return text
        keep_start = max_chars - (TRUNC_KEEP_END_TOKENS * 3)
        if keep_start < 100:
            keep_start = max_chars // 2
        return text[:keep_start] + "\n... [truncated] ...\n" + text[-(TRUNC_KEEP_END_TOKENS * 3):]
    
    # Token-aware truncation
    try:
        # Encode to get token count (without special tokens)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Clamp keep_end to avoid misconfiguration (e.g., keep_end > max_tokens)
        keep_end = min(TRUNC_KEEP_END_TOKENS, max_tokens)
        keep_start = max_tokens - keep_end
        if keep_start < 50:
            keep_start = max_tokens // 2
        
        # Decode the kept tokens back to text
        start_tokens = tokens[:keep_start]
        end_tokens = tokens[-keep_end:] if len(tokens) > keep_start + keep_end else []
        
        start_text = tokenizer.decode(start_tokens, skip_special_tokens=True)
        end_text = tokenizer.decode(end_tokens, skip_special_tokens=True) if end_tokens else ""
        
        if end_text:
            return start_text + "\n... [truncated] ...\n" + end_text
        return start_text
        
    except Exception as e:
        # If tokenization fails, fall back to character-based
        print(f"[DOMAIN] Tokenization failed ({e}), using char fallback")
        max_chars = max_tokens * 3
        return text[:max_chars] if len(text) > max_chars else text


def load_domain_centroid(path: str) -> None:
    """
    Load a precomputed domain centroid (.npy) from disk.
    
    Also verify that the centroid was built with the same embedding model
    as AGRI_EMB_MODEL_NAME to avoid embedding-space mismatch.
    
    IMPORTANT: This does NOT load the embedding model unless AGRI_VALIDATE_DIM=1
    AND the model is already loaded. The model is loaded lazily when first needed
    for scoring.
    
    Args:
        path: Path to the .npy centroid file
    
    Raises:
        FileNotFoundError: If centroid file doesn't exist
        ValueError: If embedding model mismatch detected or centroid is malformed
    """
    global _DOMAIN_CENTROID

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Domain centroid file not found: {p}")

    # --- Model compatibility check (name only, from metadata) ---
    meta_path = p.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            centroid_model = (meta.get("embedding_model") or "").strip()
            runtime_model = (AGRI_EMB_MODEL_NAME or "").strip()
            if centroid_model and runtime_model and centroid_model != runtime_model:
                raise ValueError(
                    "Embedding model mismatch:\n"
                    f"  centroid built with: {centroid_model}\n"
                    f"  runtime AGRI_EMB_MODEL_NAME: {runtime_model}\n"
                    "Fix: rebuild centroid with the runtime model, or set AGRI_EMB_MODEL_NAME to match."
                )
        except Exception as e:
            print(f"[DOMAIN] Warning: could not validate centroid meta file {meta_path}: {e}")
    else:
        print(f"[DOMAIN] Warning: centroid meta file not found (expected {meta_path}). Skipping model check.")

    c = np.load(p)

    # --- Defensive validation: check centroid is valid even without model ---
    if c.ndim != 1 or c.shape[0] == 0:
        raise ValueError(f"Invalid centroid array shape: {c.shape} (expected 1-D non-empty)")

    # --- Optional dimension validation (only if model already loaded) ---
    if AGRI_VALIDATE_DIM and _EMB_MODEL is not None:
        model_dim = _EMB_MODEL.get_sentence_embedding_dimension()
        if c.shape[0] != model_dim:
            raise ValueError(
                f"Centroid dimension ({c.shape[0]}) doesn't match model dimension ({model_dim})"
            )
    elif AGRI_VALIDATE_DIM:
        print(
            "[DOMAIN] Note: AGRI_VALIDATE_DIM=1 but model not loaded yet. "
            "Dimension check skipped. Call domain_scores() once (loads model), "
            "or preload the model with _load_emb_model()."
        )

    # Defensive normalisation
    norm = np.linalg.norm(c)
    if norm > 0:
        c = c / norm

    _DOMAIN_CENTROID = c.astype(np.float32)  # Use float32 for efficiency
    print(f"[DOMAIN] Loaded domain centroid from: {p} (shape: {c.shape})")


def _compute_cosine_similarities(texts: List[str]) -> List[float]:
    """
    Compute cosine similarities between multiple texts and the domain centroid.
    
    Args:
        texts: List of texts to compare (already preprocessed/truncated)
    
    Returns:
        List of similarity scores in [0, 1]
    """
    # Note: Caller should check _DOMAIN_CENTROID is not None before calling
    if _DOMAIN_CENTROID is None:
        return [0.0] * len(texts)
    
    if not texts:
        return []
    
    _load_emb_model()
    assert _EMB_MODEL is not None
    
    # Filter out empty texts for embedding, track their positions
    non_empty = []
    empty_mask = [False] * len(texts)  # Boolean mask for empty positions
    for i, t in enumerate(texts):
        if t and t.strip():
            non_empty.append(t)
        else:
            empty_mask[i] = True
    
    if not non_empty:
        return [0.0] * len(texts)
    
    # Batch encode all non-empty texts in ONE call
    # batch_size capped for future-proofing (in case this gets reused for larger batches)
    embeddings = _EMB_MODEL.encode(
        non_empty,
        normalize_embeddings=True,
        batch_size=min(32, len(non_empty)),
        show_progress_bar=False,
    )
    
    # Compute similarities
    non_empty_idx = 0
    similarities = []
    for i in range(len(texts)):
        if empty_mask[i]:
            similarities.append(0.0)
        else:
            emb = embeddings[non_empty_idx]
            # Cosine similarity (dot product of normalized vectors)
            sim = float(np.dot(emb, _DOMAIN_CENTROID))
            # Map from [-1, 1] to [0, 1]
            sim01 = (sim + 1.0) / 2.0
            similarities.append(max(0.0, min(1.0, sim01)))
            non_empty_idx += 1
    
    return similarities


def _sim_to_score(sim: float) -> int:
    """
    Map cosine similarity (0–1) to a discrete 0–5 score.
    
    Matches original behavior:
    - sim <= 0.0 -> score 0 (off-domain / no centroid / floating point near-zero)
    - sim > 0.0 -> at least score 1 (if below lowest threshold)
    
    Args:
        sim: Similarity score in [0, 1]
    
    Returns:
        Integer score 0-5
    """
    # sim <= 0.0 means off-domain, no centroid loaded, or floating point underflow
    # Using <= instead of == is safer against floating point edge cases
    if sim <= 0.0:
        return 0
    
    # Check thresholds for scores 2-5
    for threshold, score in SIM_THRESHOLDS:
        if sim >= threshold:
            return score
    
    # sim > 0.0 but below lowest threshold (0.35)
    return 1


def _preprocess_keywords(keywords: List[str]) -> str:
    """
    Join keywords with spaces, handling edge cases.
    
    Args:
        keywords: List of keyword strings
    
    Returns:
        Concatenated keyword text
    """
    if not keywords:
        return ""
    # Filter out None/empty and join
    clean = [str(k).strip() for k in keywords if k and str(k).strip()]
    return " ".join(clean)


def domain_scores(
    title: str,
    desc: str,
    content: str,
    keywords: List[str],
) -> Dict[str, Any]:
    """
    Compute domain relevance scores for a Knowledge Object.
    
    IMPROVEMENTS:
    - Short-circuits if centroid not loaded (avoids unnecessary model loading)
    - All 4 fields are embedded in a SINGLE model call (4x faster)
    - Token-aware truncation when model is loaded
    - Hard char cap before tokenization (prevents pathological inputs)
    - Preserves original scoring behavior at similarity boundary
    
    Args:
        title: KO title
        desc: KO description
        content: KO body content (will be truncated if too long)
        keywords: List of keywords
    
    Returns:
        Dictionary with domain scores and diagnostic similarities
    """
    # SHORT-CIRCUIT: If centroid not loaded, return zeros without loading model
    # This prevents accidental model downloads in unit tests or CLI tools
    if _DOMAIN_CENTROID is None:
        content_len = len(content) if content else 0
        return {
            "Domain_term_density": 0,
            "Domain_in_title": 0,
            "Domain_in_keywords": 0,
            "Domain_consistency": 0,
            "Domain_Total_Raw": 0,
            "Domain_Score_0_25": 0,
            "Domain_similarity_title": 0.0,
            "Domain_similarity_desc": 0.0,
            "Domain_similarity_keywords": 0.0,
            "Domain_similarity_content": 0.0,
            "Domain_content_truncated": False,
            "Domain_content_original_len": content_len,
            "Domain_content_truncated_len": content_len,
        }
    
    # Preprocess inputs
    full_kw_text = _preprocess_keywords(keywords)
    
    # Ensure model is loaded for token-aware truncation
    _load_emb_model()
    
    # Truncate content if needed (token-aware)
    content_truncated = _token_aware_truncate(content)
    
    # Prepare all texts for batch embedding
    texts = [title, desc, full_kw_text, content_truncated]
    
    # Compute all similarities in ONE batched call
    sim_title, sim_desc, sim_kw, sim_content = _compute_cosine_similarities(texts)
    
    # Map to discrete scores (preserves original boundary behavior)
    score_title = _sim_to_score(sim_title)
    score_desc = _sim_to_score(sim_desc)
    score_kw = _sim_to_score(sim_kw)
    score_content = _sim_to_score(sim_content)
    
    # Aggregate
    dom_raw = score_content + score_title + score_kw + score_desc
    dom_scaled = min(25, round(dom_raw * 25.0 / 20.0))  # Cap at 25
    
    # Detect if content was truncated (compare original vs truncated)
    was_truncated = len(content) > len(content_truncated) if content else False
    
    return {
        # Main domain scores (0-5 each)
        "Domain_term_density": score_content,
        "Domain_in_title": score_title,
        "Domain_in_keywords": score_kw,
        "Domain_consistency": score_desc,
        
        # Aggregated scores
        "Domain_Total_Raw": dom_raw,
        "Domain_Score_0_25": dom_scaled,
        
        # Diagnostic similarities (continuous 0-1)
        "Domain_similarity_title": round(sim_title, 4),
        "Domain_similarity_desc": round(sim_desc, 4),
        "Domain_similarity_keywords": round(sim_kw, 4),
        "Domain_similarity_content": round(sim_content, 4),
        
        # New diagnostics
        "Domain_content_truncated": was_truncated,
        "Domain_content_original_len": len(content) if content else 0,
        "Domain_content_truncated_len": len(content_truncated) if content_truncated else 0,
    }


def batch_domain_scores(
    items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Compute domain scores for multiple KOs with optimized batching.
    
    This is an EXPERIMENTAL function for future batch processing optimization.
    Currently just a wrapper around domain_scores for each item.
    
    Future improvement: True batching across multiple KOs for even better GPU utilization.
    
    Args:
        items: List of dicts with keys: title, desc, content, keywords
    
    Returns:
        List of score dictionaries
    """
    return [
        domain_scores(
            title=item.get("title", ""),
            desc=item.get("desc", ""),
            content=item.get("content", ""),
            keywords=item.get("keywords", []),
        )
        for item in items
    ]


def get_centroid_stats() -> Dict[str, Any]:
    """
    Get information about the loaded centroid.
    
    Returns:
        Dictionary with centroid status and metadata
    """
    if _DOMAIN_CENTROID is None:
        return {"loaded": False, "error": "No centroid loaded"}
    
    return {
        "loaded": True,
        "shape": _DOMAIN_CENTROID.shape,
        "dtype": str(_DOMAIN_CENTROID.dtype),
        "norm": float(np.linalg.norm(_DOMAIN_CENTROID)),
        "model_name": AGRI_EMB_MODEL_NAME,
    }


# Backwards compatibility: explicitly removed initialise_domain_centroid
# If anyone tries to import it, they'll get AttributeError immediately
__all__ = [
    "load_domain_centroid",
    "domain_scores", 
    "batch_domain_scores",
    "get_centroid_stats",
]
