# which_model_to_choose/improve_ko/io_helpers.py

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Dict, List, Union

def find_latest_json(input_dir: Path) -> Path:
    """Return the path to the most recently modified *.json file in input_dir."""
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")
    return max(json_files, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    """Write JSON atomically: path.tmp -> rename. Supports dict or list top-level."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_model_result_dict_mode(augmented: Dict[str, Any], out_path: Path, key: str, value: Any) -> None:
    augmented[key] = value
    atomic_write_json(out_path, augmented)


def append_model_result_list_mode(out_items: List[Dict[str, Any]],
                                  current_snapshot: Dict[str, Any],
                                  out_path: Path,
                                  key: str,
                                  value: Any) -> None:
    current_snapshot[key] = value
    # Persist full list snapshot: already-completed items + current in-progress snapshot
    atomic_write_json(out_path, out_items + [current_snapshot])
