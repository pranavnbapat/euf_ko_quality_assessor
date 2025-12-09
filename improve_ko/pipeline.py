# which_model_to_choose/improve_ko/pipeline.py

from __future__ import annotations

import logging
import sys
import time

from tiktoken import get_encoding
from typing import Any, Dict, List

# from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (models_available, MODEL_TO_HOST, EXTREME_CTX_THRESHOLD_TOK, NEAR_LIMIT_CTX_THRESHOLD_TOK,
                    CHUNK_TARGET_TOK, CHUNK_OVERLAP_TOK, DEFAULT_NUM_PREDICT, LONG_NUM_PREDICT, COMBINE_NUM_PREDICT,
                    SUMMARY_MAX_ATTEMPTS, PRIMARY_MODEL, LLM_BACKEND)
from io_helpers import append_model_result_dict_mode
from ollama_client import call_ollama, warm_up_models
from prompts import DEFAULT_PROMPT, COMBINE_PROMPT, CLEAN_PROMPT, METADATA_PROMPT
from utils import (fmt, approx_token_count, split_into_tokenish_chunks, extract_summary_json,
                   extract_cleaned_json, extract_metadata_text, extract_metadata_keywords, should_summarise_text)


logger = logging.getLogger(__name__)

_enc = get_encoding("cl100k_base")

def estimate_tokens(text: str) -> int:
    """Approximate token count with cl100k_base (Qwen-compatible enough)."""
    return len(_enc.encode(text))

def split_by_tokens(text: str, max_tokens: int, overlap: int = 256) -> List[str]:
    """
    Split text into token-limited chunks with optional overlapping window.
    - max_tokens: maximum tokens per chunk (input-side only).
    - overlap: number of tokens to carry over between chunks.
    """
    ids = _enc.encode(text)
    n = len(ids)
    chunks = []

    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        chunk_ids = ids[start:end]
        chunks.append(_enc.decode(chunk_ids))

        if end >= n:
            break

        # step back by overlap to avoid hard cuts
        start = max(0, end - overlap)

    return chunks

def _maybe_cap_ctx(model: str, opts: dict) -> dict:
    """
    Normalise context-related options depending on backend.

    - For vLLM (OpenAI-compatible endpoint):
        * We completely ignore `num_ctx` – context is controlled by
          `--max-model-len` on the server, not per-request.
        * We keep `num_predict` → mapped to `max_tokens` in ollama_client._call_vllm_openai_chat().
    - For Ollama:
        * Keep a conservative cap on `num_ctx` for big models.
    """
    if LLM_BACKEND == "vllm":
        # vLLM's /v1/chat/completions does not know about num_ctx.
        # Deleting it avoids giving the illusion that it is honoured.
        opts.pop("num_ctx", None)
        return opts

    # Ollama backend – keep older safety cap for qwen3 / gpt-oss models.
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
                # Token-true chunking via tiktoken; sizes controlled via config.py
                # chunks = split_by_tokens(
                #     content,
                #     max_tokens=CHUNK_TARGET_TOK,
                #     overlap=CHUNK_OVERLAP_TOK,
                # )
                chunks = split_into_tokenish_chunks(content, CHUNK_TARGET_TOK, CHUNK_OVERLAP_TOK)

                cleaned_parts: List[str] = []

                for ch in chunks:
                    opts = _maybe_cap_ctx(
                        model,
                        {
                            # For vLLM: num_ctx will be dropped; num_predict still used
                            "num_ctx": 32768,
                            "num_predict": DEFAULT_NUM_PREDICT,
                            "temperature": 0.1,
                        },
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
                logger.warning(
                    "_run_cleaning: model=%s attempt %d failed with %s: %s. Retrying...",
                    model,
                    attempt,
                    type(e).__name__,
                    e,
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
                # Very long input → map–reduce summarisation
                # 1) Map step: summarise token-limited chunks
                # chunks = split_by_tokens(
                #     content,
                #     max_tokens=CHUNK_TARGET_TOK,
                #     overlap=CHUNK_OVERLAP_TOK,
                # )
                chunks = split_into_tokenish_chunks(content, CHUNK_TARGET_TOK, CHUNK_OVERLAP_TOK)
                partials: List[str] = []

                for ch in chunks:
                    map_opts = _maybe_cap_ctx(
                        model,
                        {
                            # vLLM: num_ctx dropped, num_predict → max_tokens
                            "num_ctx": 32768,
                            "num_predict": DEFAULT_NUM_PREDICT,
                            "temperature": 0.2,
                        },
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

                # 2) Reduce step: combine partial summaries into one
                combined_input = "\n\n---- PARTIAL SUMMARY ----\n".join(partials)
                combine_opts = _maybe_cap_ctx(
                    model,
                    {
                        "num_ctx": 32768,
                        "num_predict": COMBINE_NUM_PREDICT,
                        "temperature": 0.2,
                    },
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
                    {"num_ctx": 32768, "num_predict": LONG_NUM_PREDICT, "temperature": 0.2},
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
                logger.warning(
                    "%s: model=%s attempt %d failed with %s: %s. Retrying...",
                    _run_single_model.__name__,
                    model,
                    attempt,
                    type(e).__name__,
                    e,
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


def _run_metadata_field(
    model: str,
    summary_en: str,
    field: str,
    existing_value: Any | None = None,
) -> Any:
    """
    Improve a single metadata FIELD ('TITLE', 'SUBTITLE', 'DESCRIPTION', 'KEYWORDS')
    using the METADATA_PROMPT.

    Returns:
      - For TITLE/SUBTITLE/DESCRIPTION: a string.
      - For KEYWORDS: a List[str].
    """
    field = field.upper().strip()
    if field not in {"TITLE", "SUBTITLE", "DESCRIPTION", "KEYWORDS"}:
        raise ValueError(f"Unsupported metadata field '{field}'")

    existing_str = ""
    # For keywords, existing_value may be a list; normalise to comma-separated text
    if field == "KEYWORDS":
        if isinstance(existing_value, list):
            existing_str = ", ".join([str(x) for x in existing_value if str(x).strip()])
        elif isinstance(existing_value, str):
            existing_str = existing_value
    else:
        if isinstance(existing_value, str):
            existing_str = existing_value
        elif existing_value is not None:
            existing_str = str(existing_value)

    meta_context = (
        f"FIELD: {field}\n\n"
        f"EXISTING VALUE:\n{existing_str}\n\n"
        f"SUMMARY:\n{summary_en}"
    )

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
                meta_context,
                options_override=opts,
                base_url=MODEL_TO_HOST[model],
                force_no_schema=True
            )

            if field == "KEYWORDS":
                # Expect a list[str]
                return extract_metadata_keywords(raw)
            else:
                # Expect a single string
                return extract_metadata_text(raw)

        except Exception as e:
            last_err = e
            if attempt < SUMMARY_MAX_ATTEMPTS:
                logger.warning("_run_metadata_field: field=%s model=%s attempt %d failed with %s: %s. Retrying...",
                               field, model, attempt, type(e).__name__, e,)
                continue
            break

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Metadata generation for field '{field}' failed after retries")


def process_one_dict_item(
    augmented_one: Dict[str, Any],
    out_path,
    content: str,
    mode: str,
    log_timer: bool = True,
) -> Dict[str, Any]:
    """
    mode:
      - "clean":    only write ko_content_flat_cleaned
      - "summary":  only write ko_content_flat_cleaned_summarised (requires cleaned)
      - "metadata": only write *_llm_en fields (requires summary)
    """

    t_item = time.perf_counter() if log_timer else None
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
            if log_timer and t_item is not None:
                logger.info("Item total (clean): %s", fmt(time.perf_counter() - t_item))
            return augmented_one

        if not augmented_one.get(cleaned_field):
            if model not in warmed:
                warm_up_models([model], base_url=MODEL_TO_HOST[model])
                warmed.add(model)
            cleaned_text = _run_cleaning(model, content)
            append_model_result_dict_mode(augmented_one, out_path, cleaned_field, cleaned_text)

        # nothing else in this run
        if log_timer and t_item is not None:
            logger.info("Item total (clean): %s", fmt(time.perf_counter() - t_item))
        return augmented_one

    # --- SUMMARY ONLY (requires cleaned) ---
    if mode == "summary":
        cleaned_field = "ko_content_flat"
        summary_field = "ko_content_flat_summarised"

        if isinstance(content, str) and "No content present" in content:
            if not augmented_one.get(summary_field):
                append_model_result_dict_mode(
                    augmented_one,
                    out_path,
                    summary_field,
                    "No content present",
                )
            if log_timer and t_item is not None:
                logger.info("Item total (summary): %s", fmt(time.perf_counter() - t_item))
            return augmented_one

        cleaned_text = augmented_one.get(cleaned_field)
        if not isinstance(cleaned_text, str) or not cleaned_text.strip():
            raise KeyError(
                f"{cleaned_field} missing or empty; run in 'clean' mode before 'summary' mode."
            )

        if not should_summarise_text(cleaned_text):
            if not augmented_one.get(summary_field):
                append_model_result_dict_mode(
                    augmented_one,
                    out_path,
                    summary_field,
                    cleaned_text,
                )
            if log_timer and t_item is not None:
                logger.info("Item total (summary): %s", fmt(time.perf_counter() - t_item))
            return augmented_one

        if not augmented_one.get(summary_field):
            if model not in warmed:
                warm_up_models([model], base_url=MODEL_TO_HOST[model])
                warmed.add(model)
            summary_en = _run_single_model(model, cleaned_text)
            append_model_result_dict_mode(augmented_one, out_path, summary_field, summary_en)

        if log_timer and t_item is not None:
            logger.info("Item total (summary): %s", fmt(time.perf_counter() - t_item))
        return augmented_one

    # --- METADATA ONLY ---
    if mode == "metadata":
        summary_field = "ko_content_flat_summarised"
        title_llm_field = "title_llm"
        subtitle_llm_field = "subtitle_llm"
        description_llm_field = "description_llm"
        keywords_llm_field = "keywords_llm"

        # 1) Look at the SUMMARY to decide whether to call the LLM at all.
        summary_en = augmented_one.get(summary_field)
        summary_text = summary_en if isinstance(summary_en, str) else ""

        # Normalise ko_content_flat to a string for checks
        raw_content = content if isinstance(content, str) else ""

        # If we had a "no content" sentinel earlier, just propagate a basic title and stop.
        # No point asking the model for richer metadata without content.
        if ("No content present" in summary_text or not summary_text.strip() or len(summary_text.split()) < 50):
            # Title
            if not augmented_one.get(title_llm_field):
                append_model_result_dict_mode(
                    augmented_one,
                    out_path,
                    title_llm_field,
                    augmented_one.get("title", ""),
                )

            # Subtitle
            if not augmented_one.get(subtitle_llm_field):
                append_model_result_dict_mode(
                    augmented_one,
                    out_path,
                    subtitle_llm_field,
                    augmented_one.get("subtitle", ""),
                )

            # Description
            if not augmented_one.get(description_llm_field):
                append_model_result_dict_mode(
                    augmented_one,
                    out_path,
                    description_llm_field,
                    augmented_one.get("description", ""),
                )

            # Keywords
            if not augmented_one.get(keywords_llm_field):
                append_model_result_dict_mode(
                    augmented_one,
                    out_path,
                    keywords_llm_field,
                    augmented_one.get("keywords", []),
                )

            if log_timer and t_item is not None:
                logger.info("Item total (metadata): %s", fmt(time.perf_counter() - t_item))
            return augmented_one

        # Warm the model once per item (before any metadata calls),
        if model not in warmed:
            warm_up_models([model], base_url=MODEL_TO_HOST[model])
            warmed.add(model)

        # ----- SUBTITLE -----
        if not augmented_one.get(subtitle_llm_field):
            existing_subtitle = augmented_one.get("subtitle", "")
            improved_subtitle = _run_metadata_field(
                model=model,
                summary_en=summary_en,
                field="SUBTITLE",
                existing_value=existing_subtitle,
            )
            # Fallback: if model gives empty/garbage, keep the original value.
            if not isinstance(improved_subtitle, str) or not improved_subtitle.strip():
                improved_subtitle = existing_subtitle
            append_model_result_dict_mode(
                augmented_one,
                out_path,
                subtitle_llm_field,
                improved_subtitle,
            )

        # ----- DESCRIPTION -----
        if not augmented_one.get(description_llm_field):
            existing_description = augmented_one.get("description", "")
            improved_description = _run_metadata_field(
                model=model,
                summary_en=summary_en,
                field="DESCRIPTION",
                existing_value=existing_description,
            )
            if not isinstance(improved_description, str) or not improved_description.strip():
                improved_description = existing_description
            append_model_result_dict_mode(
                augmented_one,
                out_path,
                description_llm_field,
                improved_description,
            )

        # ----- KEYWORDS -----
        if not augmented_one.get(keywords_llm_field):
            existing_keywords = augmented_one.get("keywords", [])
            improved_keywords = _run_metadata_field(
                model=model,
                summary_en=summary_en,
                field="KEYWORDS",
                existing_value=existing_keywords,
            )
            # improved_keywords should be List[str], thanks to extract_metadata_keywords().
            append_model_result_dict_mode(
                augmented_one,
                out_path,
                keywords_llm_field,
                improved_keywords,
            )

        # ----- TITLE -----
        # Only generate title_llm_en if it does not exist yet; this keeps the run idempotent.
        if not augmented_one.get(title_llm_field):
            existing_title = augmented_one.get("title", "")
            improved_title = _run_metadata_field(
                model=model,
                summary_en=summary_en,
                field="TITLE",
                existing_value=existing_title,
            )

            # Fallback: never write an empty string – keep original title instead.
            if not isinstance(improved_title, str) or not improved_title.strip():
                improved_title = existing_title

            append_model_result_dict_mode(
                augmented_one,
                out_path,
                title_llm_field,
                improved_title,
            )

        if log_timer and t_item is not None:
            logger.info("Item total (metadata): %s", fmt(time.perf_counter() - t_item))
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
    updated = process_one_dict_item(
        current_snapshot,
        out_path,
        content,
        mode=mode,
        log_timer=False,
    )

    orig_id = current_snapshot.get("_orig_id") or current_snapshot.get("id") or "UNKNOWN"

    logger.info(
        "Item total (%s) _orig_id=%s: %s",
        mode,
        orig_id,
        fmt(time.perf_counter() - t_item),
    )
    return updated
