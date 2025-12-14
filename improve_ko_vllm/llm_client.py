# improve_ko_vllm/llm_client.py

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from config import (
    BASE_VLLM_HOST,
    PER_REQUEST_TIMEOUT,
)

# ------------------------------------------------------------------
# Shared HTTP session with retries
# ------------------------------------------------------------------
_session = requests.Session()
_retries = Retry(
    total=4,
    backoff_factor=0.7,
    status_forcelist=(408, 409, 425, 429, 499, 500, 502, 503, 504, 524),
    allowed_methods=frozenset(["POST"]),
    raise_on_status=False,
)
_session.mount("https://", HTTPAdapter(max_retries=_retries))
_session.mount("http://", HTTPAdapter(max_retries=_retries))


def _sleep_with_jitter(seconds: float) -> None:
    jitter = seconds * (0.7 + 0.6 * random.random())
    time.sleep(jitter)


# ------------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------------
def warm_up_models(models: List[str], base_url: Optional[str] = None) -> None:
    host = (base_url or BASE_VLLM_HOST).rstrip("/")
    url = f"{host}/v1/chat/completions"

    for m in models:
        try:
            payload = {
                "model": m,
                "messages": [
                    {"role": "system", "content": "Warm-up request."},
                    {"role": "user", "content": "hi"},
                ],
                "max_tokens": 4,
                "temperature": 0.0,
            }
            _session.post(url, json=payload, timeout=(10, 30))
        except Exception:
            pass


# ------------------------------------------------------------------
# Core vLLM call
# ------------------------------------------------------------------
def call_vllm_chat(
    model: str,
    prompt: str,
    content: str,
    options_override: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
) -> str:
    host = (base_url or BASE_VLLM_HOST).rstrip("/")
    url = f"{host}/v1/chat/completions"

    full_prompt = (
        f"{prompt}\n\n"
        "-----\n"
        "FILE CONTENT START\n"
        f"{content}\n"
        "FILE CONTENT END\n"
        "-----"
    )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Answer directly. Return only the required output.",
            },
            {
                "role": "user",
                "content": full_prompt,
            },
        ],
        "temperature": 0.2,
    }

    if options_override:
        if isinstance(options_override.get("num_predict"), int):
            payload["max_tokens"] = options_override["num_predict"]
        if isinstance(options_override.get("max_tokens"), int):
            payload["max_tokens"] = options_override["max_tokens"]
        if isinstance(options_override.get("temperature"), (float, int)):
            payload["temperature"] = options_override["temperature"]

    r = _session.post(url, json=payload, timeout=PER_REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Malformed vLLM response: {data!r}")

    text = (text or "").strip()
    if not text:
        raise RuntimeError("Empty response from vLLM")

    return text
