from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, SGLANG_API_KEY, SGLANG_BASE_URL


def chat_completion(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
    }
    if extra_body:
        payload.update(extra_body)

    headers = {
        "Authorization": f"Bearer {SGLANG_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{SGLANG_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def extract_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("No choices returned from SGLang response")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty message content returned from SGLang response")
    return content
