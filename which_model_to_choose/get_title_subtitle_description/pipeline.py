# which_model_to_choose/get_title_subtitle_description/pipeline.py

from __future__ import annotations

from typing import Any, Dict, List

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    models_available, MODEL_TO_HOST,
    NEAR_LIMIT_CTX_THRESHOLD_TOK,
    DEFAULT_NUM_PREDICT, LONG_NUM_PREDICT,
)

from io_helpers import append_model_result_dict_mode, append_model_result_list_mode
from ollama_client import call_ollama, warm_up_models
from prompts import DEFAULT_PROMPT
from utils import (fmt, approx_token_count, normalise_model_key, extract_metadata_json, clamp_metadata_lengths)

def _maybe_cap_ctx(model: str, opts: dict) -> dict:
    return opts


def _retry_repair_prompt(original_filled: str) -> str:
    # Append a short, explicit correction without changing the rest of the prompt.
    return original_filled + (
        "\n\nIMPORTANT\n"
        "- Do not use the key \"summary\".\n"
        "- Return ONLY this JSON object with EXACTLY these keys: "
        "{\"title\",\"subtitle\",\"description\"}.\n"
        "- No extra keys or explanations."
    )


def _run_single_model(
    model: str,
    context_chunk: str,
    *,
    given_title: str,
    given_subtitle: str,
    given_description: str
) -> Dict[str, str]:
    """
    Call the model with the filled prompt and return a dict:
      {"title": "...", "subtitle": "...", "description": "..."}
    Applies model-specific ctx caps and clamps output lengths.
    """
    def fill_prompt(chunk: str) -> str:
        return DEFAULT_PROMPT.format(
            title=(given_title or "").strip(),
            subtitle=(given_subtitle or "").strip(),
            description=(given_description or "").strip(),
            context_chunk=(chunk or "").strip(),
        )

    # IMPORTANT: use context_chunk (the parameter) — 'content' was undefined before.
    tok = approx_token_count(context_chunk)

    # Keep it simple for now: skip the "extreme" map–reduce branch.
    if tok > NEAR_LIMIT_CTX_THRESHOLD_TOK:
        near_opts = _maybe_cap_ctx(model, {
            "num_ctx": 131072,
            "num_predict": LONG_NUM_PREDICT,
            "temperature": 0.2,
            "format": "json",
        })
        raw = call_ollama(
            model,
            fill_prompt(context_chunk),  # merged system+user style prompt
            "",
            options_override=near_opts,
            base_url=MODEL_TO_HOST[model],
        )
        try:
            obj = extract_metadata_json(raw)
        except ValueError:
            # one repair attempt
            raw = call_ollama(
                model,
                _retry_repair_prompt(fill_prompt(context_chunk)),
                "",
                options_override=near_opts,  # or opts in the normal branch
                base_url=MODEL_TO_HOST[model],
            )
            obj = extract_metadata_json(raw)
        return clamp_metadata_lengths(obj)

    # Normal path
    opts = _maybe_cap_ctx(model, {
        "num_predict": DEFAULT_NUM_PREDICT,
        "temperature": 0.2,
        "format": "json",
    })
    raw = call_ollama(
        model,
        fill_prompt(context_chunk),
        "",
        options_override=opts,
        base_url=MODEL_TO_HOST[model],
    )
    try:
        obj = extract_metadata_json(raw)
    except ValueError:
        # one repair attempt
        raw = call_ollama(
            model,
            _retry_repair_prompt(fill_prompt(context_chunk)),
            "",
            options_override=opts,  # or opts in the normal branch
            base_url=MODEL_TO_HOST[model],
        )
        obj = extract_metadata_json(raw)
    return clamp_metadata_lengths(obj)

def process_one_dict_item(augmented_one: Dict[str, Any], out_path, content: str, parallel: bool = True) -> Dict[str, Any]:
    warmed: set[str] = set()
    t_item = time.perf_counter()

    # Capture the original metadata to feed into the prompt
    given_title = (augmented_one.get("title") or "").strip()
    given_subtitle = (augmented_one.get("subtitle") or "").strip()
    given_description = (augmented_one.get("description") or "").strip()

    # Optionally run the three models in parallel (benefits 2×GPU)
    if parallel:
        with ThreadPoolExecutor(max_workers=len(models_available)) as pool:
            futures = {}
            for model in models_available:
                key_suffix = normalise_model_key(model)
                base_suffix = key_suffix  # e.g. 'gpt_oss_20b' or 'qwen3_30b_a3b_2507_q8_0'

                have_all = all(
                    isinstance(augmented_one.get(f"{k}_{base_suffix}"), str)
                    and augmented_one.get(f"{k}_{base_suffix}").strip()
                    for k in ("title", "subtitle", "description")
                )
                if have_all:
                    continue

                if model not in warmed:
                    warm_up_models([model], base_url=MODEL_TO_HOST[model])
                    warmed.add(model)

                futures[pool.submit(
                    _run_single_model,
                    model,
                    content,
                    given_title=given_title,
                    given_subtitle=given_subtitle,
                    given_description=given_description
                )] = base_suffix

            for fut in as_completed(futures):
                base_suffix = futures[fut]
                try:
                    md = fut.result()
                    append_model_result_dict_mode(augmented_one, out_path, f"title_{base_suffix}", md.get("title", ""))
                    append_model_result_dict_mode(augmented_one, out_path, f"subtitle_{base_suffix}",
                                                  md.get("subtitle", ""))
                    append_model_result_dict_mode(augmented_one, out_path, f"description_{base_suffix}",
                                                  md.get("description", ""))
                except Exception as e:
                    append_model_result_dict_mode(augmented_one, out_path, f"metadata_{base_suffix}_error", str(e))

    else:
        for model in models_available:
            key_suffix = normalise_model_key(model)
            base_suffix = key_suffix

            have_all = all(
                isinstance(augmented_one.get(f"{k}_{base_suffix}"), str) and augmented_one.get(
                    f"{k}_{base_suffix}").strip()
                for k in ("title", "subtitle", "description")
            )
            if have_all:
                continue

            if model not in warmed:
                warm_up_models([model], base_url=MODEL_TO_HOST[model])
                warmed.add(model)

            try:
                md = _run_single_model(
                    model,
                    content,
                    given_title=given_title,
                    given_subtitle=given_subtitle,
                    given_description=given_description
                )
                append_model_result_dict_mode(augmented_one, out_path, f"title_{base_suffix}", md.get("title", ""))
                append_model_result_dict_mode(augmented_one, out_path, f"subtitle_{base_suffix}",
                                              md.get("subtitle", ""))
                append_model_result_dict_mode(augmented_one, out_path, f"description_{base_suffix}",
                                              md.get("description", ""))
            except Exception as e:
                append_model_result_dict_mode(augmented_one, out_path, f"metadata_{base_suffix}_error", str(e))

    print(f"[TIMER] Item total: {fmt(time.perf_counter() - t_item)}")
    return augmented_one

def process_one_list_item(out_items: List[Dict[str, Any]], current_snapshot: Dict[str, Any], out_path, content: str,
                          parallel: bool = True) -> Dict[str, Any]:
    warmed: set[str] = set()
    t_item = time.perf_counter()

    given_title = (current_snapshot.get("title") or "").strip()
    given_subtitle = (current_snapshot.get("subtitle") or "").strip()
    given_description = (current_snapshot.get("description") or "").strip()

    if parallel:
        with ThreadPoolExecutor(max_workers=len(models_available)) as pool:
            futures = {}
            for model in models_available:
                key_suffix = normalise_model_key(model)
                base_suffix = key_suffix

                have_all = all(
                    isinstance(current_snapshot.get(f"{k}_{base_suffix}"), str)
                    and current_snapshot.get(f"{k}_{base_suffix}").strip()
                    for k in ("title", "subtitle", "description")
                )
                if have_all:
                    continue

                if model not in warmed:
                    warm_up_models([model], base_url=MODEL_TO_HOST[model])
                    warmed.add(model)

                futures[pool.submit(
                    _run_single_model,
                    model,
                    content,
                    given_title=given_title,
                    given_subtitle=given_subtitle,
                    given_description=given_description
                )] = base_suffix

            for fut in as_completed(futures):
                base_suffix = futures[fut]
                try:
                    md = fut.result()
                    append_model_result_list_mode(out_items, current_snapshot, out_path, f"title_{base_suffix}",
                                                  md.get("title", ""))
                    append_model_result_list_mode(out_items, current_snapshot, out_path, f"subtitle_{base_suffix}",
                                                  md.get("subtitle", ""))
                    append_model_result_list_mode(out_items, current_snapshot, out_path, f"description_{base_suffix}",
                                                  md.get("description", ""))
                except Exception as e:
                    append_model_result_list_mode(out_items, current_snapshot, out_path,
                                                  f"metadata_{base_suffix}_error", str(e))


    else:
        for model in models_available:
            key_suffix = normalise_model_key(model)
            base_suffix = key_suffix

            have_all = all(
                isinstance(current_snapshot.get(f"{k}_{base_suffix}"), str) and current_snapshot.get(
                    f"{k}_{base_suffix}").strip()
                for k in ("title", "subtitle", "description")
            )
            if have_all:
                continue

            if model not in warmed:
                warm_up_models([model], base_url=MODEL_TO_HOST[model])
                warmed.add(model)

            try:
                md = _run_single_model(
                    model,
                    content,
                    given_title=given_title,
                    given_subtitle=given_subtitle,
                    given_description=given_description
                )
                append_model_result_list_mode(out_items, current_snapshot, out_path, f"title_{base_suffix}",
                                              md.get("title", ""))
                append_model_result_list_mode(out_items, current_snapshot, out_path, f"subtitle_{base_suffix}",
                                              md.get("subtitle", ""))
                append_model_result_list_mode(out_items, current_snapshot, out_path, f"description_{base_suffix}",
                                              md.get("description", ""))
            except Exception as e:
                append_model_result_list_mode(out_items, current_snapshot, out_path, f"metadata_{base_suffix}_error",
                                              str(e))

    print(f"[TIMER] Item total: {fmt(time.perf_counter() - t_item)}")
    return current_snapshot
