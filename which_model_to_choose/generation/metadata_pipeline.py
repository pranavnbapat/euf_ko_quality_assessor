from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

from .client import chat_completion, extract_text
from .prompts import DEFAULT_METADATA_PROMPT
from .utils import clamp_metadata_lengths, extract_metadata_json


METADATA_MAX_ATTEMPTS = 3
DEFAULT_NUM_PREDICT = 1024
LONG_NUM_PREDICT = 2048


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


def _build_prompt(record: Dict[str, Any], summary_text: str) -> str:
    title = str(record.get("title") or record.get("title_llm") or "").strip()
    subtitle = str(record.get("subtitle") or record.get("subtitle_llm") or "").strip()
    description = str(record.get("description") or record.get("description_llm") or "").strip()
    keywords = record.get("keywords_llm") or record.get("keywords") or []
    if isinstance(keywords, list):
        keywords = ", ".join(str(x).strip() for x in keywords if str(x).strip())
    else:
        keywords = str(keywords).strip()

    return DEFAULT_METADATA_PROMPT.format(
        title=title,
        subtitle=subtitle,
        description=description,
        keywords=keywords,
        context_chunk=summary_text.strip(),
    )


def _run_once(served_model_name: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    response = chat_completion(
        model=served_model_name,
        messages=[
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    raw = extract_text(response)
    obj = extract_metadata_json(raw)
    return clamp_metadata_lengths(obj)


def generate_metadata_for_record(served_model_name: str, record: Dict[str, Any], summary_text: str) -> Dict[str, Any]:
    prompt = _build_prompt(record, summary_text)
    last_err: Exception | None = None
    for attempt in range(1, METADATA_MAX_ATTEMPTS + 1):
        try:
            max_tokens = LONG_NUM_PREDICT if len(summary_text) > 12000 else DEFAULT_NUM_PREDICT
            return _run_once(served_model_name, prompt, max_tokens)
        except Exception as e:
            last_err = e
            if attempt >= METADATA_MAX_ATTEMPTS:
                break
            time.sleep(1.5 * attempt)
    if last_err is not None:
        raise last_err
    raise RuntimeError("Model returned no usable metadata")


def generate_metadata_dataset(
    data: Any,
    model_key: str,
    served_model_name: str,
    summary_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records = list(iter_records(data))
    summary_by_id = {
        row.get("id"): row
        for row in summary_rows
        if row.get("status") == "ok" and isinstance(row.get("summary"), str)
    }
    filtered_records: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, 1):
        rec_id = record.get("_orig_id") or record.get("@id") or record.get("id") or f"row_{idx}"
        if rec_id in summary_by_id:
            filtered_records.append(record)

    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(filtered_records, 1):
        rec_id = record.get("_orig_id") or record.get("@id") or record.get("id") or f"row_{idx}"
        row: Dict[str, Any] = {
            "record_index": idx,
            "id": rec_id,
            "title": record.get("title") or record.get("title_llm") or "",
        }

        summary_row = summary_by_id.get(rec_id)
        try:
            md = generate_metadata_for_record(served_model_name, record, summary_row["summary"])
            row.update(md)
            row["status"] = "ok"
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows
