# improve_ko_vllm/main.py

from __future__ import annotations

import logging
import sys
import time
from tiktoken import get_encoding

from typing import Any, Dict, List

from config import COMBINE_NUM_PREDICT, NEAR_LIMIT_CTX_THRESHOLD_TOK, EXTREME_CTX_THRESHOLD_TOK
from io_helpers import find_latest_json, load_json, atomic_write_json, get_output_dir
from pipeline import process_one_dict_item, process_one_list_item
from prompts import DEFAULT_PROMPT
from utils import fmt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def estimate_tokens(text: str) -> int:
    """
    Estimate tokens using Qwen/LLaMA-compatible BPE.
    """
    enc = get_encoding("cl100k_base")   # closest to Qwen tokenizer
    return len(enc.encode(text))


def main() -> None:
    t_script = time.perf_counter()

    # ---------------------------------------------------------------
    # Parse CLI arguments
    # ---------------------------------------------------------------
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m improve_ko.main "
            "[clean|summary|metadata] [--shard-index i --num-shards n]"
        )
    mode = sys.argv[1].strip().lower()
    if mode not in {"clean", "summary", "metadata"}:
        raise SystemExit(
            f"Invalid mode '{mode}'. Expected one of: clean, summary, metadata."
        )

    shard_index = 0
    num_shards = 1
    args = sys.argv[2:]
    if "--shard-index" in args:
        i = args.index("--shard-index")
        shard_index = int(args[i + 1])
    if "--num-shards" in args:
        i = args.index("--num-shards")
        num_shards = int(args[i + 1])

    if not (0 <= shard_index < num_shards):
        raise SystemExit(f"Invalid shard config: index={shard_index}, total={num_shards}")

    # ---------------------------------------------------------------
    # Load input file
    # ---------------------------------------------------------------
    latest_path = find_latest_json()

    # Log which input file and folder we are using
    print(f"[INFO] Input folder: {latest_path.parent}")
    print(f"[INFO] Input file:   {latest_path.name}")

    data = load_json(latest_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # ---------------------------------------------------------------
    # Output file setup
    # ---------------------------------------------------------------
    if mode == "summary":
        base_suffix = "_summary"
    elif mode == "metadata":
        base_suffix = "_metadata"
    elif mode == "clean":
        base_suffix = "_clean"
    else:
        base_suffix = "_llmed"

    # Make filenames distinct per shard when using multiple shards
    if num_shards > 1:
        suffix = f"{base_suffix}_sh{shard_index + 1}-of-{num_shards}_{timestamp}"
    else:
        suffix = f"{base_suffix}_{timestamp}"

    # Ensure we have an output/ folder next to io_helpers.py
    output_dir = get_output_dir(create=True)

    # Save processed file into output/ instead of input/
    out_path = output_dir / f"{latest_path.stem}{suffix}.json"

    # Log where the output will be written
    print(f"[INFO] Output folder: {out_path.parent}")
    print(f"[INFO] Output file:   {out_path.name}")


    # ---------------------------------------------------------------
    # DICT INPUT
    # ---------------------------------------------------------------
    if isinstance(data, dict):
        augmented_one = dict(data)
        try:
            content = data.get("ko_content_flat")
            if not isinstance(content, str) or not content.strip():
                raise KeyError("'ko_content_flat' missing or empty")

            # ---------------- TOKEN CHECK (DICT CASE) ----------------
            prompt_tokens = estimate_tokens(DEFAULT_PROMPT)
            content_tokens = estimate_tokens(content)
            requested_tokens = COMBINE_NUM_PREDICT
            total_needed = prompt_tokens + content_tokens + requested_tokens

            print(f"[TOKENS] prompt={prompt_tokens}, input={content_tokens}, "
                  f"generation={requested_tokens}, total={total_needed}")

            if total_needed > NEAR_LIMIT_CTX_THRESHOLD_TOK:
                print(
                    f"⚠️ WARNING: This KO is near the vLLM context limit "
                    f"({total_needed} / {EXTREME_CTX_THRESHOLD_TOK}). Chunking may be required."
                )

            augmented_one = process_one_dict_item(augmented_one, out_path, content, mode=mode)
            atomic_write_json(out_path, augmented_one)
            print(f"[DONE] Wrote: {out_path}")
        except KeyboardInterrupt:
            atomic_write_json(out_path, augmented_one)
            print("\n[INTERRUPTED] Progress saved.", file=sys.stderr)
            raise

    elif isinstance(data, list):
        out_items: List[Dict[str, Any]] = []

        # Resume logic
        if out_path.exists():
            try:
                existing = load_json(out_path)
                if isinstance(existing, list):
                    out_items = existing
                    print(f"[INFO] Resuming from existing output file with {len(out_items)} items: {out_path}")
            except Exception:
                print(f"[WARN] Failed to load existing output file, will overwrite: {out_path}", file=sys.stderr)
                pass

        # Build a set of already-processed IDs for robust resume
        processed_ids = set()
        for item in out_items:
            if not isinstance(item, dict):
                continue

            pid = item.get("_orig_id") or item.get("id")
            if pid is None:
                continue

            # if isinstance(item, dict):
            #     pid = item.get("_orig_id") or item.get("id")
            #     if pid is not None:
            #         processed_ids.add(pid)

            # Only count as processed if the summary exists and is non-empty
            s = item.get("ko_content_flat_summarised")
            if isinstance(s, str) and s.strip():
                processed_ids.add(pid)

        total = len(data)
        for idx, obj in enumerate(data, 1):
            if not isinstance(obj, dict):
                print(f"[WARN] Skipping non-dict item at index {idx}", file=sys.stderr)
                continue

            # Sharding: only process items assigned to this shard.
            # idx is 1-based; convert to 0-based for modulo division.
            if (idx - 1) % num_shards != shard_index:
                continue

            # Decide the identifier for this KO
            orig_id = obj.get("_orig_id") or obj.get("id") or idx

            # Skip if already processed (robust resume)
            if orig_id in processed_ids:
                continue

            print(f"[INFO] Item {idx}/{total}")

            current_snapshot = dict(obj)
            try:
                content = obj.get("ko_content_flat")
                if not isinstance(content, str) or not content.strip():
                    raise KeyError("'ko_content_flat' missing or empty")

                # ---------------- TOKEN CHECK (LIST CASE) ----------------
                prompt_tokens = estimate_tokens(DEFAULT_PROMPT)
                content_tokens = estimate_tokens(content)
                requested_tokens = COMBINE_NUM_PREDICT
                total_needed = prompt_tokens + content_tokens + requested_tokens

                print(f"[TOKENS] prompt={prompt_tokens}, input={content_tokens}, "
                      f"generation={requested_tokens}, total={total_needed}")

                if total_needed > NEAR_LIMIT_CTX_THRESHOLD_TOK:
                    print(
                        f"⚠️ WARNING: This KO is near the vLLM context limit "
                        f"({total_needed} / {EXTREME_CTX_THRESHOLD_TOK}). Chunking may be required."
                    )

                current_snapshot = process_one_list_item(out_items, current_snapshot, out_path, content, mode=mode)
                out_items.append(current_snapshot)
                processed_ids.add(orig_id)
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
