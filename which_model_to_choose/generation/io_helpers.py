from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import CANDIDATE_RUNS_DIR, INPUT_DIR


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
