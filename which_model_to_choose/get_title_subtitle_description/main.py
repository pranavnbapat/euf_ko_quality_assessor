# which_model_to_choose/get_title_subtitle_description/main.py

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
    candidates = list(INPUT_DIR.glob("*_llmed.json"))
    if not candidates:
        raise FileNotFoundError(f"No files ending with '_llmed.json' in: {INPUT_DIR}")
    latest_path = max(candidates, key=lambda p: p.stat().st_mtime)
    data = load_json(latest_path)

    out_path = latest_path.with_name(latest_path.stem + "_tsd.json")
    print(f"[INFO] Reading: {latest_path.name}")
    print(f"[INFO] Will write: {out_path.name}")

    if isinstance(data, dict):
        augmented_one = dict(data)
        try:
            content = data.get("ko_content_flat_summarised_qwenb_30b_instruct")
            if not isinstance(content, str) or not content.strip():
                raise KeyError("No usable content: 'ko_content_flat_summarised_qwenb_30b_instruct' is empty or not found")
            augmented_one = process_one_dict_item(augmented_one, out_path, content, parallel=True)
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
            print(f"[INFO] Item {idx}/{total} (type={type(obj).__name__})")

            # --- normalise obj: if it's a one-element list containing a dict, unwrap it ---
            if isinstance(obj, list):
                if len(obj) == 1 and isinstance(obj[0], dict):
                    obj = obj[0]
                    print(f"[INFO]   unwrapped one-element list at index {idx}")
                else:
                    print(f"[WARN]   skipping: list item not a dict at index {idx}")
                    continue

            if not isinstance(obj, dict):
                print(f"[WARN]   skipping non-dict item at index {idx}")
                continue
            # If resuming by index
            if len(out_items) >= idx:
                continue
            current_snapshot = dict(obj)

            try:
                content = obj.get("ko_content_flat_summarised_qwenb_30b_instruct")
                if not isinstance(content, str) or not content.strip():
                    raise KeyError("No usable content in item: expected 'ko_content_flat_summarised_qwenb_30b_instruct'")
                current_snapshot = process_one_list_item(
                    out_items, current_snapshot, out_path, content, parallel=True
                )
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
