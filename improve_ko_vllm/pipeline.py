# improve_ko_vllm/pipeline.py

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from config import (
    MODEL_TO_HOST,
    EXTREME_CTX_THRESHOLD_TOK,
    NEAR_LIMIT_CTX_THRESHOLD_TOK,
    CHUNK_TARGET_TOK,
    CHUNK_OVERLAP_TOK,
    DEFAULT_NUM_PREDICT,
    LONG_NUM_PREDICT,
    COMBINE_NUM_PREDICT,
    SUMMARY_MAX_ATTEMPTS,
    PRIMARY_MODEL,
)
from io_helpers import append_model_result_dict_mode
from llm_client import call_vllm_chat, warm_up_models
from prompts import (CLEAN_PROMPT, METADATA_PROMPT, LOW_CDS_PROMPT, DEFAULT_PROMPT, HIGH_CDS_PROMPT,
                     NOISY_SOURCE_PROMPT, CHUNK_SUMMARY_PROMPT, COMBINE_PROMPT, UNIVERSAL_SUMMARY_PROMPT)
from utils import (fmt, approx_token_count, split_into_tokenish_chunks, extract_summary_json, extract_cleaned_json,
                   extract_metadata_text, extract_metadata_keywords, should_summarise_text)

logger = logging.getLogger(__name__)

def _select_summary_prompt(augmented_one: Dict[str, Any], text_field: str = "ko_content_flat",) -> Tuple[str, str]:
    """
    Choose a summarisation prompt using diagnostics if available.
    Falls back to DEFAULT_PROMPT if no metrics exist.
    Returns:
      (prompt_name, prompt_text)
    """
    metrics = augmented_one.get(f"{text_field}_metrics", {})
    if not isinstance(metrics, dict):
        return "DEFAULT", DEFAULT_PROMPT

    tokens = metrics.get("tokens")
    noise = metrics.get("noise_ratio_non_alnum")
    flags = metrics.get("field_quality_flags") or []
    bigram_rep = metrics.get("bigram_repetition_ratio")
    ent_density = metrics.get("entity_density_per_100_tokens")
    centroid = metrics.get("mean_centroid_cosine_similarity")
    drift = metrics.get("first_last_block_cosine_similarity")

    flagset = {str(x) for x in flags} if isinstance(flags, list) else set()

    if (
            "very_hard_to_read" in flagset
            or "truncated_for_spacy" in flagset
            or (isinstance(noise, (int, float)) and noise >= 0.12)
            or (isinstance(drift, (int, float)) and drift < 0.35)
    ):
        return "NOISY_SOURCE", NOISY_SOURCE_PROMPT

    # 2) HIGH_CDS: dense/technical content worth preserving in detail
    # Use entity density as a strong proxy (works well across languages).
    if isinstance(ent_density, (int, float)) and ent_density >= 6.0:
        return "HIGH_CDS", HIGH_CDS_PROMPT

    # 3) LOW_CDS: repetitive / redundant content where we can compress harder
    if (
            (isinstance(centroid, (int, float)) and centroid > 0.85)
            or (isinstance(bigram_rep, (int, float)) and bigram_rep > 0.22)
            or (isinstance(tokens, int) and tokens >= 8000)  # optional: cost-control for long docs
    ):
        return "LOW_CDS", LOW_CDS_PROMPT

    # Default band or unknown values
    return "DEFAULT", DEFAULT_PROMPT


def _run_cleaning(model: str, content: str) -> str:
    tok = approx_token_count(content)

    for attempt in range(1, SUMMARY_MAX_ATTEMPTS + 1):
        try:
            if tok > EXTREME_CTX_THRESHOLD_TOK:
                chunks = split_into_tokenish_chunks(
                    content, CHUNK_TARGET_TOK, CHUNK_OVERLAP_TOK
                )
                cleaned_parts: List[str] = []

                for ch in chunks:
                    raw = call_vllm_chat(
                        model,
                        CLEAN_PROMPT,
                        ch,
                        options_override={
                            "max_tokens": DEFAULT_NUM_PREDICT,
                            "temperature": 0.1,
                        },
                        base_url=MODEL_TO_HOST[model],
                    )
                    obj = extract_cleaned_json(raw)
                    cleaned_parts.append(obj["cleaned"])

                return "\n\n".join(cleaned_parts)

            raw = call_vllm_chat(
                model,
                CLEAN_PROMPT,
                content,
                options_override={
                    "max_tokens": DEFAULT_NUM_PREDICT,
                    "temperature": 0.1,
                },
                base_url=MODEL_TO_HOST[model],
            )
            return extract_cleaned_json(raw)["cleaned"]

        except Exception as e:
            if attempt < SUMMARY_MAX_ATTEMPTS:
                logger.warning("clean attempt %d failed: %s", attempt, e)
            else:
                raise


def _run_single_model(model: str, content: str, prompt: str) -> str:
    tok = approx_token_count(content)

    for attempt in range(1, SUMMARY_MAX_ATTEMPTS + 1):
        try:
            if tok > EXTREME_CTX_THRESHOLD_TOK:
                chunks = split_into_tokenish_chunks(
                    content, CHUNK_TARGET_TOK, CHUNK_OVERLAP_TOK
                )
                partials = []

                for ch in chunks:
                    raw = call_vllm_chat(
                        model,
                        CHUNK_SUMMARY_PROMPT,
                        ch,
                        options_override={"max_tokens": DEFAULT_NUM_PREDICT},
                        base_url=MODEL_TO_HOST[model],
                    )
                    partials.append(extract_summary_json(raw)["summary"])

                combined = "\n\n---- PART ----\n".join(partials)
                raw = call_vllm_chat(
                    model,
                    COMBINE_PROMPT,
                    combined,
                    options_override={"max_tokens": COMBINE_NUM_PREDICT},
                    base_url=MODEL_TO_HOST[model],
                )
                return extract_summary_json(raw)["summary"]

            max_toks = (
                LONG_NUM_PREDICT
                if tok > NEAR_LIMIT_CTX_THRESHOLD_TOK
                else DEFAULT_NUM_PREDICT
            )

            raw = call_vllm_chat(
                model,
                prompt,
                content,
                options_override={"max_tokens": max_toks},
                base_url=MODEL_TO_HOST[model],
            )
            return extract_summary_json(raw)["summary"]

        except Exception as e:
            if attempt < SUMMARY_MAX_ATTEMPTS:
                logger.warning("summary attempt %d failed: %s", attempt, e)
            else:
                raise


def _run_metadata_field(
    model: str,
    summary_en: str,
    field: str,
    existing_value: Any | None = None,
) -> Any:
    existing_str = ""
    if isinstance(existing_value, list):
        existing_str = ", ".join(map(str, existing_value))
    elif isinstance(existing_value, str):
        existing_str = existing_value

    meta_context = (
        f"FIELD: {field}\n\n"
        f"EXISTING VALUE:\n{existing_str}\n\n"
        f"SUMMARY:\n{summary_en}"
    )

    raw = call_vllm_chat(
        model,
        METADATA_PROMPT,
        meta_context,
        options_override={"max_tokens": DEFAULT_NUM_PREDICT, "temperature": 0.3},
        base_url=MODEL_TO_HOST[model],
    )

    if field == "KEYWORDS":
        return extract_metadata_keywords(raw)
    return extract_metadata_text(raw)


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
        # cleaned_field = "ko_content_flat_clean"
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

        # Prefer cleaned text if it exists, but DO NOT require it.
        # Some pipelines skip the cleaning stage entirely.
        cleaned_text = augmented_one.get("ko_content_flat_cleaned")
        if not isinstance(cleaned_text, str) or not cleaned_text.strip():
            cleaned_text = augmented_one.get("ko_content_flat_clean")  # backwards compatibility

        # Fall back to raw ko_content_flat passed into this function.
        source_text = cleaned_text if isinstance(cleaned_text, str) and cleaned_text.strip() else content

        if not isinstance(source_text, str) or not source_text.strip():
            raise KeyError("'ko_content_flat' missing or empty")

        if not should_summarise_text(source_text):
            if not augmented_one.get(summary_field):
                append_model_result_dict_mode(
                    augmented_one,
                    out_path,
                    summary_field,
                    source_text,
                )
            if log_timer and t_item is not None:
                logger.info("Item total (summary): %s", fmt(time.perf_counter() - t_item))
            return augmented_one

        if not augmented_one.get(summary_field):
            if model not in warmed:
                warm_up_models([model], base_url=MODEL_TO_HOST[model])
                warmed.add(model)

            # prompt = _select_summary_prompt(augmented_one, text_field="ko_content_flat")
            # prompt_name, prompt_text = _select_summary_prompt(augmented_one, text_field="ko_content_flat")
            ko_id = augmented_one.get("_orig_id") or augmented_one.get("id") or augmented_one.get("identifier") or "unknown"
            #
            # m = augmented_one.get("ko_content_flat_metrics", {})
            # tok = m.get("tokens") if isinstance(m, dict) else None
            # noise = m.get("noise_ratio_non_alnum") if isinstance(m, dict) else None
            # drift = m.get("first_last_block_cosine_similarity") if isinstance(m, dict) else None
            # ent = m.get("entity_density_per_100_tokens") if isinstance(m, dict) else None
            # rep = m.get("bigram_repetition_ratio") if isinstance(m, dict) else None
            # flags = m.get("field_quality_flags") if isinstance(m, dict) else None
            #
            # logger.info(
            #     "Summarise: id=%s model=%s prompt=%s tokens=%s noise=%s drift=%s ent_dens=%s bigram_rep=%s flags=%s",
            #     ko_id, model, prompt_name, tok, noise, drift, ent, rep, flags
            # )
            #
            # summary_en = _run_single_model(model, source_text, prompt=prompt_text)

            prompt_name, prompt_text = "UNIVERSAL", UNIVERSAL_SUMMARY_PROMPT
            logger.info("Summarise: id=%s model=%s prompt=%s", ko_id, model, prompt_name)
            summary_en = _run_single_model(model, source_text, prompt=prompt_text)

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
