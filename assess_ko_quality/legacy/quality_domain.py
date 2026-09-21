# assess_ko_quality/quality_domain.py

import json
import os

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sentence_transformers import SentenceTransformer


AGRI_EMB_MODEL_NAME = os.environ.get("AGRI_EMB_MODEL_NAME", "all-mpnet-base-v2")

# Lazy-loaded globals
_EMB_MODEL: Optional[SentenceTransformer] = None
_DOMAIN_CENTROID: Optional[np.ndarray] = None


def load_domain_centroid(path: str) -> None:
    """
    Load a precomputed domain centroid (.npy) from disk.
    Also verify that the centroid was built with the same embedding model
    as AGRI_EMB_MODEL_NAME to avoid embedding-space mismatch.
    """
    global _DOMAIN_CENTROID
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Domain centroid file not found: {p}")

    # --- Model compatibility check (strongly recommended) ---
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

    # Defensive normalisation
    norm = np.linalg.norm(c)
    if norm > 0:
        c = c / norm

    _DOMAIN_CENTROID = c
    print(f"[DOMAIN] Loaded domain centroid from: {p}")


def _load_emb_model() -> None:
    """
    Load the SentenceTransformer model (if not already loaded).
    """
    global _EMB_MODEL

    if _EMB_MODEL is not None:
        return

    print(f"[DOMAIN] Loading SentenceTransformer model: {AGRI_EMB_MODEL_NAME}")
    _EMB_MODEL = SentenceTransformer(AGRI_EMB_MODEL_NAME)


def initialise_domain_centroid(texts: List[str]) -> None:
    """
    Given a list of representative KO texts (e.g. all ko_content_flat,
    or content + title + description), compute a single domain centroid
    vector as the mean of their embeddings.

    This is *data-driven*: no hard-coded agriculture terms or descriptions.
    """
    global _DOMAIN_CENTROID

    # Filter out empty strings
    clean_texts = [t for t in texts if t and t.strip()]
    if not clean_texts:
        # If nothing provided, leave centroid as None; scoring will degrade gracefully.
        print("[DOMAIN] Warning: no texts provided to initialise domain centroid.")
        _DOMAIN_CENTROID = None
        return

    _load_emb_model()
    assert _EMB_MODEL is not None

    print(f"[DOMAIN] Computing domain centroid from {len(clean_texts)} texts...")

    embs = _EMB_MODEL.encode(clean_texts, normalize_embeddings=True)
    _DOMAIN_CENTROID = np.mean(embs, axis=0)


def _cos_sim_to_agri(text: str) -> float:
    """
    Compute cosine similarity between given text and the learned domain centroid.
    Returns a value in [0, 1] (since we use normalised embeddings).

    If the centroid has not been initialised, we return 0.0 (off-domain)
    """
    if not text or not text.strip():
        return 0.0

    _load_emb_model()
    assert _EMB_MODEL is not None

    if _DOMAIN_CENTROID is None:
        # No centroid initialised yet – treat as unknown/off-domain.
        return 0.0

    emb = _EMB_MODEL.encode([text], normalize_embeddings=True)[0]

    # keep cosine in [-1, 1]
    sim = float(np.dot(emb, _DOMAIN_CENTROID))
    # optional: map to [0,1] while preserving negatives: (sim + 1) / 2
    sim01 = (sim + 1.0) / 2.0
    return max(0.0, min(1.0, sim01))



def _sim_to_score(sim: float) -> int:
    """
    Map cosine similarity (0–1) to a discrete 0–5 score.
    Thresholds are heuristic.
    """
    if sim >= 0.65:
        return 5
    if sim >= 0.55:
        return 4
    if sim >= 0.45:
        return 3
    if sim >= 0.35:
        return 2
    if sim > 0.0:
        return 1
    return 0


def domain_scores(
    title: str,
    desc: str,
    content: str,
    keywords: List[str],
) -> Dict[str, Any]:
    """
    Domain sub-scores (agricultural context, embedding-based):

      We compute cosine similarity between each metadata field / content and
      an 'agriculture prototype' embedding:

        - title_sim     ~ how agricultural the title looks
        - desc_sim      ~ how agricultural the description looks
        - kw_sim        ~ how agricultural the keywords look
        - content_sim   ~ how agricultural the KO body looks

      Then we map each similarity to a 0–5 score. For backwards compatibility,
      we re-use the old column names:

        - Domain_term_density  -> content-based score
        - Domain_in_title      -> title-based score
        - Domain_in_keywords   -> keyword-based score
        - Domain_consistency   -> description-based score

      Plus:
        - Domain_Total_Raw (0–20)
        - Domain_Score_0_25 (0–25)
        - additional diagnostic similarity columns.
    """
    full_kw_text = " ".join(keywords) if keywords else ""

    sim_title = _cos_sim_to_agri(title)
    sim_desc = _cos_sim_to_agri(desc)
    sim_kw = _cos_sim_to_agri(full_kw_text)
    sim_content = _cos_sim_to_agri(content)

    score_title = _sim_to_score(sim_title)
    score_desc = _sim_to_score(sim_desc)
    score_kw = _sim_to_score(sim_kw)
    score_content = _sim_to_score(sim_content)

    # Aggregate – keep semantics roughly similar to old 4×(0–5) structure
    dom_raw = score_content + score_title + score_kw + score_desc
    dom_scaled = round(dom_raw * 25.0 / 20.0)

    return {
        # "Old" score names – but now embedding based
        "Domain_term_density": score_content,       # KO body looks agricultural
        "Domain_in_title": score_title,             # title looks agricultural
        "Domain_in_keywords": score_kw,             # keywords look agricultural
        "Domain_consistency": score_desc,           # description is on-domain

        "Domain_Total_Raw": dom_raw,
        "Domain_Score_0_25": dom_scaled,

        # Extra diagnostics (continuous similarities, 0–1)
        "Domain_similarity_title": round(sim_title, 3),
        "Domain_similarity_desc": round(sim_desc, 3),
        "Domain_similarity_keywords": round(sim_kw, 3),
        "Domain_similarity_content": round(sim_content, 3),
    }
