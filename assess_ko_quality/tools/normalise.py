# assess_ko_quality/normalise_for_ko_quality_assessment.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple


def remove_fields_recursive(obj: Any, fields: set[str]) -> Tuple[Any, int]:
    """
    Recursively remove every key listed in `fields` from dicts within obj.
    Returns (new_obj, removed_count).
    """
    removed = 0

    if isinstance(obj, dict):
        # Copy once if we actually remove something, to avoid mutating original
        new_obj = obj
        for f in fields:
            if f in new_obj:
                if new_obj is obj:
                    new_obj = dict(obj)  # copy on first removal
                new_obj.pop(f, None)
                removed += 1

        # Recurse into remaining keys
        new_d: dict[str, Any] = {}
        for k, v in new_obj.items():
            new_v, r = remove_fields_recursive(v, fields)
            removed += r
            new_d[k] = new_v
        return new_d, removed

    if isinstance(obj, list):
        new_l = []
        for item in obj:
            new_item, r = remove_fields_recursive(item, fields)
            removed += r
            new_l.append(new_item)
        return new_l, removed

    return obj, removed


def process_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    removed = 0
    renamed = 0

    def normalise_one(obj: Any) -> None:
        """Normalise one top-level KO dict in-place."""
        nonlocal removed, renamed
        if not isinstance(obj, dict):
            return

        # 1) Remove old fields
        for f in ("title", "subtitle", "keywords", "description", "ko_content_flat"):
            if f in obj:
                obj.pop(f, None)
                removed += 1

        # 2) Rename *_llm → canonical names
        rename_map = {
            "title_llm": "title",
            "subtitle_llm": "subtitle",
            "keywords_llm": "keywords",
            "description_llm": "description",
            "ko_content_flat_summarised": "ko_content_flat",
        }
        for src, dst in rename_map.items():
            if src in obj:
                obj[dst] = obj.pop(src)
                renamed += 1

    # Support either a single dict or a list of dicts
    if isinstance(data, dict):
        normalise_one(data)
    elif isinstance(data, list):
        for item in data:
            normalise_one(item)
    else:
        return 0

    if removed > 0 or renamed > 0:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return removed + renamed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a single JSON file to modify in-place",
    )
    args = ap.parse_args()

    src: Path = args.input
    if not src.exists():
        print(f"[ERROR] File not found: {src}")
        return 1
    if src.is_dir():
        print(f"[ERROR] --input must be a file, not a directory: {src}")
        return 1

    try:
        removed = process_file(src)
    except Exception as e:
        print(f"[ERROR] {src}: {e}")
        return 1

    print(f"Done. File: {src}. Total fields removed: {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
