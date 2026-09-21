# assess_ko_quality/transformer_similarity/ko_similarity_ref_based.py

"""
Reference-based semantic similarity for KO metadata.

What this does (per KO):
1) Compare ORIGINAL vs IMPROVED text fields using transformer embeddings:
   - title vs title_llm
   - subtitle vs subtitle_llm
   - description vs description_llm
   - keywords vs keywords_llm

2) Compare IMPROVED metadata vs content summary (or content) as a "reference":
   - title_llm/subtitle_llm/description_llm/keywords_llm vs ko_content_flat_summarised (preferred)
   This detects drift: improved metadata not supported by extracted content.

Outputs:
- CSV with per-field similarity + weighted scores + diagnostics
- Optional JSONL with the same info

Usage:
    On runpod:
    cd workspace
    mkdir -p assess_ko_quality/transformer_similarity && cd assess_ko_quality/transformer_similarity
    python3 -m venv .venv
    source .venv/bin/activate
    python --version
    python -m pip install --upgrade pip
    python -m pip install numpy pandas tqdm sentence-transformers

  python -m assess_ko_quality.transformer_similarity.ko_similarity_ref_based \
      --input /path/to/kos.json \
      --out_csv /path/to/out.csv \
      --model intfloat/e5-base-v2
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


# ----------------------------
# Text normalisation helpers
# ----------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def strip_html(text: str) -> str:
    # Very basic HTML strip (good enough for descriptions)
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()

def normalise_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = strip_html(text)
    text = text.replace("\u00a0", " ")  # non-breaking spaces
    text = _WS_RE.sub(" ", text).strip()
    return text

def keywords_to_text(keywords: Any) -> str:
    """
    Convert keywords field (list or string) to a single text string.
    """
    if keywords is None:
        return ""
    if isinstance(keywords, list):
        parts = []
        for k in keywords:
            if k is None:
                continue
            s = str(k).strip()
            if s:
                parts.append(s)
        return "; ".join(parts)
    return str(keywords).strip()

def safe_get(d: Dict[str, Any], key: str) -> Any:
    return d.get(key, None)

def is_probably_garbage_extraction(text: str) -> bool:
    """
    Heuristic detector for UI fragments / boilerplate / corrupted pulls.
    Not perfect, but catches a lot cheaply.
    """
    if not text:
        return True

    # Very short or mostly punctuation is suspicious
    if len(text) < 80:
        return True

    # Lots of repeated UI-ish tokens (example patterns)
    lower = text.lower()
    ui_markers = ["impressum", "datenschutz", "cookies", "terms", "richtlinien", "youtube", "google llc", "sign in"]
    hits = sum(1 for m in ui_markers if m in lower)
    if hits >= 2:
        return True

    # Too many very short lines / menu-like chunks
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 20:
        short_lines = sum(1 for ln in lines if len(ln) <= 18)
        if short_lines / max(1, len(lines)) > 0.6:
            return True

    return False


# ----------------------------
# Embedding + similarity
# ----------------------------

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two 2D matrices (n x d) and (m x d).
    Returns (n x m).
    """
    # Normalise to unit vectors
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1D vectors.
    """
    # Cast norms to Python floats to keep static type checkers happy (PyCharm/mypy)
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


@dataclass
class FieldPair:
    name: str
    orig_key: str
    imp_key: str


FIELD_PAIRS: List[FieldPair] = [
    FieldPair("title", "title", "title_llm"),
    FieldPair("subtitle", "subtitle", "subtitle_llm"),
    FieldPair("description", "description", "description_llm"),
    FieldPair("keywords", "keywords", "keywords_llm"),
]


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError("Input JSON must be a list[object] or a single object.")


def build_texts_for_embedding(records: List[Dict[str, Any]]) -> Tuple[List[str], Dict[Tuple[int, str], int]]:
    """
    Create a global list of texts to embed once.
    Returns:
      - texts: list of unique texts
      - index_map: mapping from (record_idx, slot_name) -> index in texts
    Slot names include:
      - f"orig:{field}"
      - f"imp:{field}"
      - "ref:content"  (summary/content reference)
    """
    texts: List[str] = []
    index_map: Dict[Tuple[int, str], int] = {}

    def add(idx: int, slot: str, text: str) -> None:
        index_map[(idx, slot)] = len(texts)
        texts.append(text)

    for i, r in enumerate(records):
        # Original vs improved fields
        for fp in FIELD_PAIRS:
            if fp.name == "keywords":
                orig_text = keywords_to_text(safe_get(r, fp.orig_key))
                imp_text = keywords_to_text(safe_get(r, fp.imp_key))
            else:
                orig_text = normalise_text(safe_get(r, fp.orig_key))
                imp_text = normalise_text(safe_get(r, fp.imp_key))

            add(i, f"orig:{fp.name}", orig_text)
            add(i, f"imp:{fp.name}", imp_text)

        # Reference content: prefer summarised; fall back to raw content
        # ref = normalise_text(safe_get(r, "ko_content_flat_summarised"))
        # if not ref:
        #     ref = normalise_text(safe_get(r, "ko_content_flat"))
        # add(i, "ref:content", ref)

    return texts, index_map


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed all texts in batches. Returns a float32 numpy matrix (n_texts x dim).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we'll normalise ourselves where needed
    )
    return embeddings.astype(np.float32)


def similarity_pipeline(
    records: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index_map: Dict[Tuple[int, str], int],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute per-record similarity scores:
    - orig_vs_imp_{field}
    - imp_vs_ref_{field}
    - weighted aggregate scores
    - diagnostics
    """
    if weights is None:
        # You can tune these later. Title is short/noisy; description carries meaning.
        weights = {
            "title": 0.4258,
            "subtitle": 0.0454,
            "description": 0.2838,
            "keywords": 0.2450,
        }

    rows: List[Dict[str, Any]] = []

    for i, r in enumerate(records):
        row: Dict[str, Any] = {}

        ko_id = r.get("@id") or r.get("_orig_id") or r.get("id") or f"idx:{i}"
        row["ko_id"] = ko_id
        row["language"] = (r.get("languages") or [""])[0] if isinstance(r.get("languages"), list) else (r.get("languages") or "")
        row["category"] = r.get("category", "")

        # Diagnostics around content extraction quality
        # ref_text = normalise_text(r.get("ko_content_flat_summarised")) or normalise_text(r.get("ko_content_flat"))
        # row["ref_is_probably_garbage"] = bool(is_probably_garbage_extraction(ref_text))
        # row["ref_char_len"] = len(ref_text)

        # Similarities
        orig_imp_scores = {}
        imp_ref_scores = {}

        for fp in FIELD_PAIRS:
            o_idx = index_map[(i, f"orig:{fp.name}")]
            p_idx = index_map[(i, f"imp:{fp.name}")]
            # ref_idx = index_map[(i, "ref:content")]

            o_emb = embeddings[o_idx]
            p_emb = embeddings[p_idx]
            # ref_emb = embeddings[ref_idx]

            s_orig_imp = cosine_sim(o_emb, p_emb)
            # s_imp_ref = cosine_sim(p_emb, ref_emb)

            row[f"orig_vs_imp_{fp.name}_cos"] = s_orig_imp
            # row[f"imp_vs_ref_{fp.name}_cos"] = s_imp_ref

            # Track for aggregation
            orig_imp_scores[fp.name] = s_orig_imp
            # imp_ref_scores[fp.name] = s_imp_ref

        # Weighted aggregates
        def weighted_mean(scores: Dict[str, float]) -> float:
            num = 0.0
            den = 0.0
            for k, w in weights.items():
                v = scores.get(k, float("nan"))
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                num += w * float(v)
                den += w
            return (num / den) if den > 1e-12 else float("nan")

        row["score_orig_vs_imp_weighted"] = weighted_mean(orig_imp_scores)
        # row["score_imp_vs_ref_weighted"] = weighted_mean(imp_ref_scores)

        # A combined score: "changed but still grounded"
        # You can interpret it as: improvement changes + grounding.
        # row["score_combined"] = (
        #     0.6 * row["score_orig_vs_imp_weighted"] +
        #     0.4 * row["score_imp_vs_ref_weighted"]
        # )

        # Useful extra flags:
        # - Low orig_vs_imp but also low imp_vs_ref => likely drift/hallucination
        # row["flag_possible_drift"] = bool(
        #     (row["score_orig_vs_imp_weighted"] < 0.45) and (row["score_imp_vs_ref_weighted"] < 0.35)
        # )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Input JSON file (list or object).")
    ap.add_argument("--out_csv", required=True, type=Path, help="Output CSV path.")
    ap.add_argument("--out_jsonl", default=None, type=Path, help="Optional output JSONL path.")
    ap.add_argument("--model", default="intfloat/e5-base-v2", help="SentenceTransformer model name.")
    ap.add_argument("--batch_size", default=64, type=int, help="Embedding batch size.")
    ap.add_argument("--device", default=None, help="Force device: 'cpu' or 'cuda'. Default: auto.")
    args = ap.parse_args()

    records = load_json_records(args.input)

    # Build text list (embed once)
    texts, index_map = build_texts_for_embedding(records)

    # Load model
    # E5 family typically works best when you prefix queries/docs, but for
    # within-field similarity and grounding checks this is already useful.
    model = SentenceTransformer(args.model, device=args.device)

    embeddings = embed_texts(model, texts, batch_size=args.batch_size, show_progress=True)

    df = similarity_pipeline(records, embeddings, index_map)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.out_csv}")
    if args.out_jsonl:
        print(f"Wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()
