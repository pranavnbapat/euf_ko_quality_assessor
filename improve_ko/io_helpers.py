# which_model_to_choose/improve_ko/io_helpers.py

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Dict, List, Union


# Base directory: folder where this io_helpers.py lives
_BASE_DIR = Path(__file__).resolve().parent

# input/ and output/ folders on the same level as this file
INPUT_DIR = _BASE_DIR / "input"
OUTPUT_DIR = _BASE_DIR / "output"


def get_input_dir() -> Path:
    """
    Return the input/ folder next to this file.

    If it does not exist, raise a clear error instead of silently failing.
    """
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")
    return INPUT_DIR


def get_output_dir(create: bool = True) -> Path:
    """
    Return the output/ folder next to this file.

    If it does not exist:
      - create it when create=True
      - raise an error when create=False
    """
    if OUTPUT_DIR.is_dir():
        return OUTPUT_DIR

    if create:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR

    raise FileNotFoundError(f"Output folder not found: {OUTPUT_DIR}")


def find_latest_json(input_dir: Path | None = None) -> Path:
    """
    Return the path to the most recently modified *.json file.

    By default, it looks in the input/ folder that lives next to this file.
    You can still pass a custom input_dir if you really want to.
    """
    # If no directory is provided, use the sibling input/ folder
    if input_dir is None:
        input_dir = get_input_dir()

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

