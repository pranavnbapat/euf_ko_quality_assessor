from __future__ import annotations

import random
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .config import CANDIDATE_RUNS_DIR, INPUT_DIR, OUTPUT_DIR


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_json(input_dir: Path | None = None) -> Path:
    input_dir = input_dir or INPUT_DIR
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")
    return max(json_files, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def candidate_run_dir(run_id: str) -> Path:
    return ensure_dir(CANDIDATE_RUNS_DIR / run_id)


def output_run_dir(run_id: str) -> Path:
    return ensure_dir(OUTPUT_DIR / run_id)


def normalize_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        docs = data.get("docs")
        if isinstance(docs, list):
            return [item for item in docs if isinstance(item, dict)]
        return [data]
    raise TypeError(f"Unsupported JSON top-level type: {type(data).__name__}")


def record_id(record: Dict[str, Any], idx: int) -> str:
    return str(record.get("_orig_id") or record.get("@id") or record.get("id") or f"row_{idx}")


def sample_records(records: Sequence[Dict[str, Any]], sample_size: int | None, seed: int = 42) -> List[Dict[str, Any]]:
    rows = list(records)
    if sample_size is None or sample_size <= 0 or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[i] for i in sampled_indices]
