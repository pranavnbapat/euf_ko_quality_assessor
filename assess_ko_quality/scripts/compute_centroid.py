# assess_ko_quality/scripts/compute_centroid.py

"""
Compute and save a centroid vector from anchor JSONL files.

Each input JSONL line should contain either:
- "anchor_text" (preferred), or
- "text" (fallback)

Outputs:
- .npy centroid (L2-normalised)
- .json metadata (model name, counts, sources, limits)

Examples:
  # Default: use anchors/agrovoc + anchors/nalt if present (and wikibooks if present)
  python scripts/compute_centroid.py

  # Explicit inputs + custom output names
  python scripts/compute_centroid.py \
    --inputs anchors/agrovoc/agrovoc_anchor_texts.jsonl anchors/nalt/nalt_anchor_texts.jsonl \
    --out anchors/centroids/agri_agrovoc_nalt.npy \
    --meta anchors/centroids/agri_agrovoc_nalt.meta.json

  # Change model / cap / batch size
  python scripts/compute_centroid.py --model BAAI/bge-base-en-v1.5 --limit 50000 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-mpnet-base-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute an embedding centroid from anchor JSONL files.")
    p.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="One or more anchor JSONL files. If omitted, uses known defaults if they exist.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output centroid .npy path. Default: anchors/centroids/agri_anchor_centroid.npy",
    )
    p.add_argument(
        "--meta",
        default=None,
        help="Output metadata .json path. Default: anchors/centroids/agri_anchor_centroid.meta.json",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"SentenceTransformer model name (default: {DEFAULT_MODEL})")
    p.add_argument("--limit", type=int, default=0, help="Max anchor texts to embed total (0 = no limit)")
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    p.add_argument(
        "--field",
        default="anchor_text",
        help="Primary JSON field to read text from (default: anchor_text). Falls back to 'text'.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device, e.g. 'cpu' or 'cuda'. Default lets SentenceTransformers decide.",
    )
    return p.parse_args()


def iter_anchor_texts(jsonl_paths: List[Path], field: str, limit: Optional[int]) -> Iterable[str]:
    """
    Stream anchor texts from JSONL files. Uses `field` first, falls back to 'text'.
    """
    count = 0
    for p in jsonl_paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                t = (obj.get(field) or obj.get("text") or "").strip()
                if not t:
                    continue

                yield t
                count += 1
                if limit is not None and limit > 0 and count >= limit:
                    return


def resolve_default_inputs(base: Path) -> List[Path]:
    """
    If --inputs is not provided, pick from known anchor locations if they exist.
    """
    anchors_dir = base / "anchors"
    candidates = [
        anchors_dir / "agrovoc" / "agrovoc_anchor_texts.jsonl",
        anchors_dir / "nalt" / "nalt_anchor_texts.jsonl",
        anchors_dir / "wikibooks" / "wikibooks_agriculture_chunks.jsonl",
        anchors_dir / "cabi" / "cabi_anchor_texts.jsonl",
    ]
    return [p for p in candidates if p.exists()]


def main() -> None:
    args = parse_args()

    base = Path(__file__).resolve().parents[1]
    anchors_dir = base / "anchors"
    out_dir = anchors_dir / "centroids"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide inputs
    if args.inputs:
        paths = [Path(x).resolve() for x in args.inputs]
    else:
        paths = resolve_default_inputs(base)

    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(
            "No anchor JSONL files found.\n"
            "Either pass --inputs <file1.jsonl> <file2.jsonl> ... or run the anchor builder scripts first."
        )

    # Decide outputs
    centroid_path = Path(args.out).resolve() if args.out else (out_dir / "agri_anchor_centroid.npy")
    meta_path = Path(args.meta).resolve() if args.meta else (out_dir / "agri_anchor_centroid.meta.json")
    centroid_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model_name = args.model
    if args.device:
        model = SentenceTransformer(model_name, device=args.device)
    else:
        model = SentenceTransformer(model_name)

    # Stream texts into a list (encode() wants a sequence; we keep a cap via limit)
    limit = None if args.limit == 0 else args.limit
    texts = list(iter_anchor_texts(paths, field=args.field, limit=limit))
    if not texts:
        raise RuntimeError(f"No usable anchor texts found in: {[str(p) for p in paths]}")

    print(f"[CENTROID] Loaded {len(texts)} anchor texts from {len(paths)} file(s).")
    print(f"[CENTROID] Model={model_name} batch_size={args.batch_size} limit={args.limit}")

    # Encode -> normalised embeddings
    embs = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Mean and normalise centroid
    centroid = np.mean(embs, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    np.save(centroid_path, centroid)

    meta = {
        "embedding_model": model_name,
        "num_texts": len(texts),
        "sources": [str(p) for p in paths],
        "field": args.field,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "device": args.device or "auto",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Saved centroid -> {centroid_path}")
    print(f"[OK] Saved metadata -> {meta_path}")


if __name__ == "__main__":
    main()
