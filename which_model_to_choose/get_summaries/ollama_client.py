# ollama_client.py

from __future__ import annotations

import json
import random

from typing import Any, Dict, Optional, List

import requests

from requests.adapters import HTTPAdapter, Retry

from config import (
    MAX_RETRIES, RETRY_BACKOFF_SECS, PER_REQUEST_TIMEOUT, REQUEST_KEEP_ALIVE,
    OLLAMA_HOST, MODEL_OVERRIDES
)

# Shared session with retries
_session = requests.Session()
_retries = Retry(
    total=4,                                   # total attempts (1 + 3 retries)
    backoff_factor=0.7,
    status_forcelist=(408, 409, 425, 429, 499, 500, 502, 503, 504, 524),
    allowed_methods=frozenset(['POST']),
    raise_on_status=False,
)
_session.mount("https://", HTTPAdapter(max_retries=_retries))
_session.mount("http://",  HTTPAdapter(max_retries=_retries))

def _sleep_with_jitter(seconds: float) -> None:
    import time
    # +/- 30% jitter to de-synchronise retries through the proxy
    jitter = seconds * (0.7 + 0.6 * random.random())
    time.sleep(jitter)

def warm_up_models(models: List[str], base_url: Optional[str] = None) -> None:
    """Trigger a small streamed request to load graphs before main batch."""
    host = (base_url or OLLAMA_HOST).rstrip("/")
    url = f"{host}/api/generate"
    for m in models:
        try:
            payload = {
                "model": m,
                "prompt": "hi",
                "stream": True,
                "keep_alive": REQUEST_KEEP_ALIVE,
                "options": {"num_ctx": 16000},
            }
            with _session.post(url, json=payload, stream=True, timeout=(10, 60)) as r:
                r.raise_for_status()
                next(r.iter_lines(), None)  # read one chunk
        except Exception:
            pass  # non-fatal

def call_ollama(model: str, prompt: str, content: str,
                options_override: Optional[Dict[str, Any]] = None,
                base_url: Optional[str] = None) -> str:
    """
    Core client: handles /api/generate vs /api/chat, JSON schema toggle,
    retries and (streamed) accumulation.
    """
    host = (base_url or OLLAMA_HOST).rstrip("/")
    full_prompt = f"{prompt}\n\n-----\nFILE CONTENT START\n{content}\nFILE CONTENT END\n-----"

    ovr = MODEL_OVERRIDES.get(model, {})
    use_chat = bool(ovr.get("use_chat", False))
    no_schema = bool(ovr.get("no_schema", False))

    approx_tokens = max(1, int(len(content) / 4))

    def decide_ctx_and_predict(token_count: int) -> dict:
        if token_count <= 4_000:      return {"num_ctx": 8192,   "num_predict": 1024}
        if token_count <= 12_000:     return {"num_ctx": 16384,  "num_predict": 1536}
        if token_count <= 30_000:     return {"num_ctx": 32768,  "num_predict": 2048}
        if token_count <= 60_000:     return {"num_ctx": 65536,  "num_predict": 3072}
        if token_count <= 110_000:    return {"num_ctx": 131072, "num_predict": 4096}
        return {"num_ctx": 32768,     "num_predict": 2048}  # must chunk

    opts = decide_ctx_and_predict(approx_tokens)
    if options_override:
        opts.update(options_override)

    if "num_predict" in ovr:
        opts["num_predict"] = int(ovr["num_predict"])
    opts["temperature"] = 0.2

    if model.startswith("gpt-oss"):
        use_chat = True if ovr.get("use_chat", True) else False
        no_schema = True if ovr.get("no_schema", True) else False
        opts.setdefault("num_predict", 2048)

    def make_payload(schema: bool, chat: bool) -> tuple[str, dict]:
        common = {
            "model": model,
            "stream": False,
            "keep_alive": REQUEST_KEEP_ALIVE,
            "options": opts,
        }
        if schema:
            common["format"] = {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            }
        if chat:
            payload = {
                **common,
                "messages": [
                    {"role": "system", "content": "Answer directly. Return only the required JSON object."},
                    {"role": "user", "content": full_prompt},
                ],
            }
            return ("/api/chat", payload)
        else:
            payload = {
                **common,
                "system": "Answer directly. Return only the required JSON object.",
                "prompt": full_prompt,
            }
            return ("/api/generate", payload)

    attempts = [
        (not no_schema, False),
        (False, False),
        (False, True) if use_chat or model.startswith("gpt-oss") else None,
    ]
    attempts = [a for a in attempts if a is not None]

    last_err: Optional[Exception] = None
    for (use_schema, chat_mode) in attempts:
        endpoint, payload = make_payload(use_schema, chat_mode)
        url = f"{host}{endpoint}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = _session.post(url, json=payload, stream=True, timeout=PER_REQUEST_TIMEOUT)
                r.raise_for_status()

                # Accumulate streamed chunks
                chunks = []
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if chat_mode:
                        part = (obj.get("message") or {}).get("content", "")
                    else:
                        part = obj.get("response", "")
                    if part:
                        chunks.append(part)
                    if obj.get("done"):
                        break
                resp = "".join(chunks).strip()
                if isinstance(resp, str) and resp.strip():
                    return resp
                raise RuntimeError("Empty response from model")

            except requests.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                if code in (429, 500, 502, 503, 504, 524) and attempt < MAX_RETRIES:
                    _sleep_with_jitter(RETRY_BACKOFF_SECS)
                    last_err = e
                    if attempt == MAX_RETRIES - 1:
                        payload["options"]["num_ctx"] = min(16000, payload["options"].get("num_ctx", 16000))
                    continue
                raise
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < MAX_RETRIES:
                    _sleep_with_jitter(RETRY_BACKOFF_SECS)
                    last_err = e
                    if attempt == MAX_RETRIES - 1:
                        payload["options"]["num_ctx"] = min(16000, payload["options"].get("num_ctx", 16000))
                    continue
                last_err = e
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    _sleep_with_jitter(RETRY_BACKOFF_SECS)
                    last_err = e
                    continue
                last_err = e
                break

    raise RuntimeError(f"Ollama returned no text for '{model}': {last_err}")
