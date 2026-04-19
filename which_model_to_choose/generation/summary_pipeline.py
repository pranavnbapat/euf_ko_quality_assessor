from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

from .client import chat_completion, extract_text
from .prompts import COMBINE_SUMMARY_PROMPT, DEFAULT_SUMMARY_PROMPT
from .utils import approx_token_count, extract_summary_json, split_into_tokenish_chunks


EXTREME_CTX_THRESHOLD_TOK = 128_000
NEAR_LIMIT_CTX_THRESHOLD_TOK = 110_000
CHUNK_TARGET_TOK = 16_000
CHUNK_OVERLAP_TOK = 400
DEFAULT_NUM_PREDICT = 2048
LONG_NUM_PREDICT = 4096
COMBINE_NUM_PREDICT = 8192
SUMMARY_MAX_ATTEMPTS = 3


def iter_records(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        docs = data.get("docs")
        if isinstance(docs, list):
            for item in docs:
                if isinstance(item, dict):
                    yield item
            return
        yield data
        return
    raise TypeError(f"Unsupported JSON top-level type: {type(data).__name__}")


def _messages(prompt: str, content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]


def _run_once(model_key: str, prompt: str, content: str, max_tokens: int) -> str:
    response = chat_completion(
        model=model_key,
        messages=_messages(prompt, content),
        max_tokens=max_tokens,
        temperature=0.2,
    )
    raw = extract_text(response)
    obj = extract_summary_json(raw)
    summary = (obj.get("summary") or "").strip()
    if not summary:
        raise ValueError("Empty summary from model output")
    return summary


def generate_summary_for_text(model_key: str, content: str) -> str:
    tok = approx_token_count(content)
    last_err: Exception | None = None

    for attempt in range(1, SUMMARY_MAX_ATTEMPTS + 1):
        try:
            if tok > EXTREME_CTX_THRESHOLD_TOK:
                chunks = split_into_tokenish_chunks(content, CHUNK_TARGET_TOK, CHUNK_OVERLAP_TOK)
                partials: List[str] = []
                for ch in chunks:
                    partials.append(_run_once(model_key, DEFAULT_SUMMARY_PROMPT, ch, DEFAULT_NUM_PREDICT))
                combined_input = "\n\n---- PARTIAL SUMMARY ----\n".join(partials)
                return _run_once(model_key, COMBINE_SUMMARY_PROMPT, combined_input, COMBINE_NUM_PREDICT)

            if tok > NEAR_LIMIT_CTX_THRESHOLD_TOK:
                return _run_once(model_key, DEFAULT_SUMMARY_PROMPT, content, LONG_NUM_PREDICT)

            return _run_once(model_key, DEFAULT_SUMMARY_PROMPT, content, DEFAULT_NUM_PREDICT)
        except Exception as e:
            last_err = e
            if attempt >= SUMMARY_MAX_ATTEMPTS:
                break
            time.sleep(1.5 * attempt)

    if last_err is not None:
        raise last_err
    raise RuntimeError("Model returned no usable summary")


def summarize_dataset(data: Any, model_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(iter_records(data), 1):
        rec_id = record.get("_orig_id") or record.get("@id") or record.get("id") or f"row_{idx}"
        title = record.get("title") or record.get("title_llm") or ""
        content = record.get("ko_content_flat")

        row: Dict[str, Any] = {
            "record_index": idx,
            "id": rec_id,
            "title": title,
        }

        if not isinstance(content, str) or not content.strip():
            row["status"] = "error"
            row["error"] = "'ko_content_flat' missing or empty"
            rows.append(row)
            continue

        try:
            row["summary"] = generate_summary_for_text(model_key, content)
            row["status"] = "ok"
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows
