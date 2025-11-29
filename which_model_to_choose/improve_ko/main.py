# which_model_to_choose/improve_ko/main.py

from __future__ import annotations

import sys
import time

from typing import Any, Dict, List

from config import INPUT_DIR
from io_helpers import find_latest_json, load_json, atomic_write_json
from pipeline import process_one_dict_item, process_one_list_item
from utils import fmt


def main() -> None:
    t_script = time.perf_counter()

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m which_model_to_choose.improve_ko.main "
            "[clean|summary|metadata]"
        )
    mode = sys.argv[1].strip().lower()
    if mode not in {"clean", "summary", "metadata"}:
        raise SystemExit(
            f"Invalid mode '{mode}'. Expected one of: clean, summary, metadata."
        )

    latest_path = find_latest_json(INPUT_DIR)
    data = load_json(latest_path)
    out_path = latest_path.with_name(latest_path.stem + "_llmed.json")

    if isinstance(data, dict):
        augmented_one = dict(data)
        try:
            content = data.get("ko_content_flat")
            if not isinstance(content, str) or not content.strip():
                raise KeyError("'ko_content_flat' missing or empty")
            augmented_one = process_one_dict_item(augmented_one, out_path, content, mode=mode)
            atomic_write_json(out_path, augmented_one)
            print(f"[DONE] Wrote: {out_path}")
        except KeyboardInterrupt:
            atomic_write_json(out_path, augmented_one)
            print("\n[INTERRUPTED] Progress saved.", file=sys.stderr)
            raise

    elif isinstance(data, list):
        out_items: List[Dict[str, Any]] = []
        if out_path.exists():
            try:
                existing = load_json(out_path)
                if isinstance(existing, list):
                    out_items = existing
            except Exception:
                pass

        total = len(data)
        for idx, obj in enumerate(data, 1):
            if not isinstance(obj, dict):
                print(f"[WARN] Skipping non-dict item at index {idx}", file=sys.stderr)
                continue
            print(f"[INFO] Item {idx}/{total}")

            if len(out_items) >= idx:
                continue  # already persisted

            current_snapshot = dict(obj)
            try:
                content = obj.get("ko_content_flat")
                if not isinstance(content, str) or not content.strip():
                    raise KeyError("'ko_content_flat' missing or empty")
                current_snapshot = process_one_list_item(out_items, current_snapshot, out_path, content, mode=mode)
                out_items.append(current_snapshot)
                atomic_write_json(out_path, out_items)
            except KeyboardInterrupt:
                atomic_write_json(out_path, out_items + [current_snapshot])
                print("\n[INTERRUPTED] Progress saved.", file=sys.stderr)
                raise
            except Exception as e:
                atomic_write_json(out_path, out_items + [current_snapshot])
                print(f"[ERROR] Item {idx}: {e}", file=sys.stderr)

        print(f"[DONE] Wrote: {out_path}")
    else:
        raise TypeError(f"Unsupported JSON top-level type: {type(data).__name__}")

    print(f"[TIMER] Script total: {fmt(time.perf_counter() - t_script)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
