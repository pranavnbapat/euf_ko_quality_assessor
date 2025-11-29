# which_model_to_choose/improve_ko/pipeline.py

from __future__ import annotations

import sys
import time

from typing import Any, Dict, List

from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (models_available, MODEL_TO_HOST, EXTREME_CTX_THRESHOLD_TOK, NEAR_LIMIT_CTX_THRESHOLD_TOK,
                    DEFAULT_NUM_PREDICT, LONG_NUM_PREDICT, COMBINE_NUM_PREDICT, SUMMARY_MAX_ATTEMPTS, PRIMARY_MODEL)
from io_helpers import append_model_result_dict_mode, append_model_result_list_mode
from ollama_client import call_ollama, warm_up_models
from prompts import DEFAULT_PROMPT, COMBINE_PROMPT, CLEAN_PROMPT, METADATA_PROMPT
from utils import (fmt, approx_token_count, split_into_tokenish_chunks, normalise_model_key, extract_summary_json,
                   extract_cleaned_json, extract_metadata_json)


def _maybe_cap_ctx(model: str, opts: dict) -> dict:
    if model.startswith("qwen3:30b") or model.startswith("gpt-oss:20b"):
        opts["num_ctx"] = min(opts.get("num_ctx", 32768), 32768)
    return opts

def _run_cleaning(model: str, content: str) -> str:
    """
    Run cleaning task: returns cleaned text in same language.
    No chunk-combine complexity: if content is huge, we can still chunk,
    but generally cleaning should be cheaper than summarisation.
    """
    tok = approx_token_count(content)
    last_err: Exception | None = None

    for attempt in range(1, SUMMARY_MAX_ATTEMPTS + 1):
        try:
            if tok > EXTREME_CTX_THRESHOLD_TOK:
                chunks = split_into_tokenish_chunks(content, 16_000, 400)
                cleaned_parts: List[str] = []
                for ch in chunks:
                    opts = _maybe_cap_ctx(
                        model,
                        {"num_ctx": 32768, "num_predict": DEFAULT_NUM_PREDICT, "temperature": 0.1},
                    )
                    raw_part = call_ollama(
                        model,
                        CLEAN_PROMPT,
                        ch,
                        options_override=opts,
                        base_url=MODEL_TO_HOST[model],
                    )

                    obj_part = extract_cleaned_json(raw_part)
                    part_clean = (obj_part.get("cleaned") or "").strip()

                    if not part_clean:
                        raise ValueError("Empty cleaned chunk from model")
                    cleaned_parts.append(part_clean)

                return "\n\n".join(cleaned_parts)

            else:
                opts = _maybe_cap_ctx(
                    model,
                    {"num_predict": DEFAULT_NUM_PREDICT, "temperature": 0.1},
                )
                raw = call_ollama(
                    model,
                    CLEAN_PROMPT,
                    content,
                    options_override=opts,
                    base_url=MODEL_TO_HOST[model],
                )
                obj = extract_cleaned_json(raw)
                cleaned = (obj.get("cleaned") or "").strip()

                if not cleaned:
                    raise ValueError("Empty cleaned output from model")
                return cleaned

        except Exception as e:
            last_err = e
            if attempt < SUMMARY_MAX_ATTEMPTS:
                print(
                    f"[WARN] _run_cleaning: model={model} attempt {attempt} failed "
                    f"with {type(e).__name__}: {e}. Retrying...",
                    file=sys.stderr,
                )
                continue
            break

    if last_err is not None:
        raise last_err
    raise RuntimeError(
        f"Model {model} returned no usable cleaned text after {SUMMARY_MAX_ATTEMPTS} attempts"
    )


def _run_single_model(model: str, content: str) -> str:
    tok = approx_token_count(content)
    last_err: Exception | None = None

    for attempt in range(1, SUMMARY_MAX_ATTEMPTS + 1):
        try:
            if tok > EXTREME_CTX_THRESHOLD_TOK:
                chunks = split_into_tokenish_chunks(content, 16_000, 400)
                partials: List[str] = []
                for ch in chunks:
                    map_opts = _maybe_cap_ctx(
                        model,
                        {"num_ctx": 32768, "num_predict": DEFAULT_NUM_PREDICT, "temperature": 0.2},
                    )
                    raw_part = call_ollama(
                        model,
                        DEFAULT_PROMPT,
                        ch,
                        options_override=map_opts,
                        base_url=MODEL_TO_HOST[model],
                    )
                    obj_part = extract_summary_json(raw_part)
                    part_summary = (obj_part.get("summary") or "").strip()
                    if not part_summary:
                        raise ValueError("Empty partial summary from model")
                    partials.append(part_summary)

                combined_input = "\n\n---- PARTIAL SUMMARY ----\n".join(partials)
                combine_opts = _maybe_cap_ctx(
                    model,
                    {"num_ctx": 32768, "num_predict": COMBINE_NUM_PREDICT, "temperature": 0.2},
                )
                raw_combined = call_ollama(
                    model,
                    COMBINE_PROMPT,
                    combined_input,
                    options_override=combine_opts,
                    base_url=MODEL_TO_HOST[model],
                )
                obj_json = extract_summary_json(raw_combined)
                final_summary = (obj_json.get("summary") or "").strip()
                if not final_summary:
                    raise ValueError("Empty combined summary from model")
                return final_summary

            elif tok > NEAR_LIMIT_CTX_THRESHOLD_TOK:
                near_opts = _maybe_cap_ctx(
                    model,
                    {"num_ctx": 131072, "num_predict": LONG_NUM_PREDICT, "temperature": 0.2},
                )
                raw = call_ollama(
                    model,
                    DEFAULT_PROMPT,
                    content,
                    options_override=near_opts,
                    base_url=MODEL_TO_HOST[model],
                )
                obj_json = extract_summary_json(raw)
                summary = (obj_json.get("summary") or "").strip()
                if not summary:
                    raise ValueError("Empty summary from model")
                return summary

            else:
                opts = _maybe_cap_ctx(
                    model,
                    {"num_predict": DEFAULT_NUM_PREDICT, "temperature": 0.2},
                )
                raw = call_ollama(
                    model,
                    DEFAULT_PROMPT,
                    content,
                    options_override=opts,
                    base_url=MODEL_TO_HOST[model],
                )
                obj_json = extract_summary_json(raw)
                summary = (obj_json.get("summary") or "").strip()
                if not summary:
                    raise ValueError("Empty summary from model")
                return summary

        except Exception as e:
            # Capture and optionally retry
            last_err = e
            if attempt < SUMMARY_MAX_ATTEMPTS:
                print(
                    f"[WARN] {_run_single_model.__name__}: model={model} attempt {attempt} failed "
                    f"with {type(e).__name__}: {e}. Retrying...",
                    file=sys.stderr,
                )
                continue
            else:
                # No more attempts left
                break

    # If we reach here, all attempts failed or produced empty summaries
    if last_err is not None:
        raise last_err
    raise RuntimeError(
        f"Model {model} returned no usable summary after {SUMMARY_MAX_ATTEMPTS} attempts"
    )


def _run_metadata(model: str,
                  summary_en: str,
                  existing: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Generate or refine title/subtitle/description/keywords in British English,
    based on an English summary and optional existing metadata.

    Returns a dict with keys:
      - title_improved
      - subtitle_improved
      - description_improved
      - keywords_improved (list of strings)
    """

    meta_context = ""
    if existing:
        meta_context = (
            "Existing metadata (may be empty):\n"
            f"TITLE: {existing.get('title', '')}\n"
            f"SUBTITLE: {existing.get('subtitle', '')}\n"
            f"DESCRIPTION: {existing.get('description', '')}\n"
            f"KEYWORDS: {', '.join(existing.get('keywords', []) or [])}\n"
        )

    content = f"{meta_context}\n\nSUMMARY:\n{summary_en}"

    last_err: Exception | None = None
    for attempt in range(1, SUMMARY_MAX_ATTEMPTS + 1):
        try:
            opts = _maybe_cap_ctx(
                model,
                {"num_predict": DEFAULT_NUM_PREDICT, "temperature": 0.3},
            )
            raw = call_ollama(
                model,
                METADATA_PROMPT,
                content,
                options_override=opts,
                base_url=MODEL_TO_HOST[model],
            )
            obj = extract_metadata_json(raw)
            return obj
        except Exception as e:
            last_err = e
            if attempt < SUMMARY_MAX_ATTEMPTS:
                print(
                    f"[WARN] _run_metadata: model={model} attempt {attempt} failed "
                    f"with {type(e).__name__}: {e}. Retrying...",
                    file=sys.stderr,
                )
                continue
            break

    if last_err is not None:
        raise last_err
    raise RuntimeError("Metadata generation failed after retries")


def process_one_dict_item(
    augmented_one: Dict[str, Any],
    out_path,
    content: str,
    mode: str,
) -> Dict[str, Any]:
    """
    mode:
      - "clean":    only write ko_content_flat_cleaned
      - "summary":  only write ko_content_flat_cleaned_summarised (requires cleaned)
      - "metadata": only write *_llm_en fields (requires summary)
    """

    t_item = time.perf_counter()
    warmed: set[str] = set()

    model = PRIMARY_MODEL  # from config

    # --- CLEAN ONLY ---
    if mode == "clean":
        cleaned_field = "ko_content_flat_cleaned"

        if isinstance(content, str) and "No content present" in content:
            if not augmented_one.get(cleaned_field):
                append_model_result_dict_mode(
                    augmented_one,
                    out_path,
                    cleaned_field,
                    "No content present",
                )
            print(f"[TIMER] Item total (clean): {fmt(time.perf_counter() - t_item)}")
            return augmented_one

        if not augmented_one.get(cleaned_field):
            if model not in warmed:
                warm_up_models([model], base_url=MODEL_TO_HOST[model])
                warmed.add(model)
            cleaned_text = _run_cleaning(model, content)
            append_model_result_dict_mode(augmented_one, out_path, cleaned_field, cleaned_text)
        # nothing else in this run
        print(f"[TIMER] Item total (clean): {fmt(time.perf_counter() - t_item)}")
        return augmented_one

    # --- SUMMARY ONLY (requires cleaned) ---
    if mode == "summary":
        cleaned_field = "ko_content_flat"
        summary_field = "ko_content_flat_summarised"

        cleaned_text = augmented_one.get(cleaned_field)
        if not isinstance(cleaned_text, str) or not cleaned_text.strip():
            raise KeyError(
                f"{cleaned_field} missing or empty; run in 'clean' mode before 'summary' mode."
            )

        if not augmented_one.get(summary_field):
            if model not in warmed:
                warm_up_models([model], base_url=MODEL_TO_HOST[model])
                warmed.add(model)
            summary_en = _run_single_model(model, cleaned_text)
            append_model_result_dict_mode(augmented_one, out_path, summary_field, summary_en)

        print(f"[TIMER] Item total (summary): {fmt(time.perf_counter() - t_item)}")
        return augmented_one

    # --- METADATA ONLY (requires English summary) ---
    if mode == "metadata":
        summary_field = "ko_content_flat_cleaned_summarised"
        meta_done_field = "metadata_llm_en_done"

        summary_en = augmented_one.get(summary_field)
        if not isinstance(summary_en, str) or not summary_en.strip():
            raise KeyError(
                f"{summary_field} missing or empty; run in 'summary' mode before 'metadata' mode."
            )

        if not augmented_one.get(meta_done_field):
            if model not in warmed:
                warm_up_models([model], base_url=MODEL_TO_HOST[model])
                warmed.add(model)

            existing_meta = {
                "title": augmented_one.get("title") or "",
                "subtitle": augmented_one.get("subtitle") or "",
                "description": augmented_one.get("description") or "",
                "keywords": augmented_one.get("keywords") or [],
            }
            meta = _run_metadata(model, summary_en, existing=existing_meta)
            append_model_result_dict_mode(augmented_one, out_path, "title_llm_en", meta["title_en"])
            append_model_result_dict_mode(augmented_one, out_path, "subtitle_llm_en", meta["subtitle_en"])
            append_model_result_dict_mode(augmented_one, out_path, "description_llm_en", meta["description_en"])
            append_model_result_dict_mode(augmented_one, out_path, "keywords_llm_en", meta["keywords_en"])
            append_model_result_dict_mode(augmented_one, out_path, meta_done_field, True)

        print(f"[TIMER] Item total (metadata): {fmt(time.perf_counter() - t_item)}")
        return augmented_one

    raise ValueError(f"Unsupported mode '{mode}'. Expected 'clean', 'summary', or 'metadata'.")

def process_one_list_item(
    out_items: List[Dict[str, Any]],
    current_snapshot: Dict[str, Any],
    out_path,
    content: str,
    mode: str,
) -> Dict[str, Any]:
    """
    Same semantics as process_one_dict_item, but for list batches.
    """
    t_item = time.perf_counter()

    # Reuse single-item logic; this function mostly exists so we can
    # keep the calling pattern in main.py the same.
    updated = process_one_dict_item(current_snapshot, out_path, content, mode=mode)

    print(f"[TIMER] Item total ({mode}): {fmt(time.perf_counter() - t_item)}")
    return updated
