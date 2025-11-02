# pipeline.py

from __future__ import annotations

from typing import Any, Dict, List

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    models_available, MODEL_TO_HOST,
    EXTREME_CTX_THRESHOLD_TOK, NEAR_LIMIT_CTX_THRESHOLD_TOK,
    DEFAULT_NUM_PREDICT, LONG_NUM_PREDICT, COMBINE_NUM_PREDICT,
)

from io_helpers import append_model_result_dict_mode, append_model_result_list_mode
from ollama_client import call_ollama, warm_up_models
from prompts import DEFAULT_PROMPT, COMBINE_PROMPT
from utils import fmt, approx_token_count, split_into_tokenish_chunks, normalise_model_key, extract_summary_json

def _maybe_cap_ctx(model: str, opts: dict) -> dict:
    if model.startswith("qwen3:30b"):
        opts["num_ctx"] = min(opts.get("num_ctx", 32768), 32768)
    return opts

def _run_single_model(model: str, content: str) -> str:
    tok = approx_token_count(content)
    if tok > EXTREME_CTX_THRESHOLD_TOK:
        chunks = split_into_tokenish_chunks(content, 16_000, 400)
        partials: List[str] = []
        for ch in chunks:
            map_opts = _maybe_cap_ctx(model, {"num_ctx": 32768, "num_predict": DEFAULT_NUM_PREDICT, "temperature": 0.2})
            raw_part = call_ollama(model, DEFAULT_PROMPT, ch, options_override=map_opts, base_url=MODEL_TO_HOST[model])
            obj_part = extract_summary_json(raw_part)
            partials.append(obj_part["summary"])
        combined_input = "\n\n---- PARTIAL SUMMARY ----\n".join(partials)
        combine_opts = _maybe_cap_ctx(model, {"num_ctx": 32768, "num_predict": COMBINE_NUM_PREDICT, "temperature": 0.2})
        raw_combined = call_ollama(model, COMBINE_PROMPT, combined_input, options_override=combine_opts,
                                   base_url=MODEL_TO_HOST[model])
        obj_json = extract_summary_json(raw_combined)
        return obj_json["summary"]
    elif tok > NEAR_LIMIT_CTX_THRESHOLD_TOK:
        near_opts = _maybe_cap_ctx(model, {"num_ctx": 131072, "num_predict": LONG_NUM_PREDICT, "temperature": 0.2})
        raw = call_ollama(model, DEFAULT_PROMPT, content, options_override=near_opts, base_url=MODEL_TO_HOST[model])
        obj_json = extract_summary_json(raw)
        return obj_json["summary"]
    else:
        opts = _maybe_cap_ctx(model, {"num_predict": DEFAULT_NUM_PREDICT, "temperature": 0.2})
        raw = call_ollama(model, DEFAULT_PROMPT, content, options_override=opts, base_url=MODEL_TO_HOST[model])
        obj_json = extract_summary_json(raw)
        return obj_json["summary"]

def process_one_dict_item(augmented_one: Dict[str, Any], out_path, content: str, parallel: bool = True) -> Dict[str, Any]:
    warmed: set[str] = set()
    t_item = time.perf_counter()

    # Optionally run the three models in parallel (benefits 2×GPU)
    if parallel:
        with ThreadPoolExecutor(max_workers=len(models_available)) as pool:
            futures = {}
            for model in models_available:
                key_suffix = normalise_model_key(model)
                field_name = f"ko_content_flat_{key_suffix}"
                if field_name in augmented_one and isinstance(augmented_one[field_name], str) and augmented_one[field_name].strip():
                    continue
                # warm model once per run on its host
                if model not in warmed:
                    warm_up_models([model], base_url=MODEL_TO_HOST[model])
                    warmed.add(model)
                futures[pool.submit(_run_single_model, model, content)] = (model, field_name)

            for fut in as_completed(futures):
                model, field_name = futures[fut]
                try:
                    summary = fut.result()
                    append_model_result_dict_mode(augmented_one, out_path, field_name, summary)
                except Exception as e:
                    append_model_result_dict_mode(augmented_one, out_path, f"{field_name}_error", str(e))

    else:
        for model in models_available:
            key_suffix = normalise_model_key(model)
            field_name = f"ko_content_flat_{key_suffix}"
            if field_name in augmented_one and isinstance(augmented_one[field_name], str) and augmented_one[field_name].strip():
                continue
            if model not in warmed:
                warm_up_models([model], base_url=MODEL_TO_HOST[model])
                warmed.add(model)
            try:
                summary = _run_single_model(model, content)
                append_model_result_dict_mode(augmented_one, out_path, field_name, summary)
            except Exception as e:
                append_model_result_dict_mode(augmented_one, out_path, f"{field_name}_error", str(e))

    print(f"[TIMER] Item total: {fmt(time.perf_counter() - t_item)}")
    return augmented_one

def process_one_list_item(out_items: List[Dict[str, Any]], current_snapshot: Dict[str, Any], out_path, content: str,
                          parallel: bool = True) -> Dict[str, Any]:
    warmed: set[str] = set()
    t_item = time.perf_counter()

    if parallel:
        with ThreadPoolExecutor(max_workers=len(models_available)) as pool:
            futures = {}
            for model in models_available:
                key_suffix = normalise_model_key(model)
                field_name = f"ko_content_flat_{key_suffix}"
                if field_name in current_snapshot and isinstance(current_snapshot[field_name], str) and current_snapshot[field_name].strip():
                    continue
                if model not in warmed:
                    warm_up_models([model], base_url=MODEL_TO_HOST[model])
                    warmed.add(model)
                futures[pool.submit(_run_single_model, model, content)] = (model, field_name)

            for fut in as_completed(futures):
                model, field_name = futures[fut]
                try:
                    summary = fut.result()
                    append_model_result_list_mode(out_items, current_snapshot, out_path, field_name, summary)
                except Exception as e:
                    append_model_result_list_mode(out_items, current_snapshot, out_path, f"{field_name}_error", str(e))
    else:
        for model in models_available:
            key_suffix = normalise_model_key(model)
            field_name = f"ko_content_flat_{key_suffix}"
            if field_name in current_snapshot and isinstance(current_snapshot[field_name], str) and current_snapshot[field_name].strip():
                continue
            if model not in warmed:
                warm_up_models([model], base_url=MODEL_TO_HOST[model])
                warmed.add(model)
            try:
                summary = _run_single_model(model, content)
                append_model_result_list_mode(out_items, current_snapshot, out_path, field_name, summary)
            except Exception as e:
                append_model_result_list_mode(out_items, current_snapshot, out_path, f"{field_name}_error", str(e))

    print(f"[TIMER] Item total: {fmt(time.perf_counter() - t_item)}")
    return current_snapshot
