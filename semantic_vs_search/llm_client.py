"""OpenAI-compatible chat client with disk caching, retries and concurrency.

Kept separate from the experiment so the retrieval code stays readable, and so
the cache can be inspected or cleared on its own.

Credentials are read from a .env file rather than hardcoded; the default points
at whichever service in this workspace already has a working endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable

log = logging.getLogger("llm_client")

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def load_env_file(path: str) -> dict[str, str]:
    """Parse a .env file into a dict. Values are never logged."""
    values: dict[str, str] = {}
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def extract_json(text: str) -> Any:
    """Pull a JSON object out of a model reply, tolerating fences and prose."""
    if not text:
        raise ValueError("empty response")
    cleaned = _FENCE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    raise ValueError(f"no JSON found in response: {cleaned[:200]!r}")


class LLMClient:
    """Minimal OpenAI-compatible client.

    - Every call is cached on disk by (model, prompt, params), so re-running an
      experiment costs nothing and results stay reproducible.
    - Retries transient failures with jittered backoff.
    - Tracks token usage so the report can state what the run cost.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        cache_dir: str,
        reasoning_effort: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
        temperature: float = 0.0,
    ) -> None:
        base = base_url.rstrip("/")
        self.endpoint = base + ("" if base.endswith("/v1") else "/v1") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------
    def _cache_path(self, payload: dict[str, Any]) -> str:
        key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{digest}.json")

    def complete(self, prompt: str, max_tokens: int = 700, system: str | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        # A reasoning model burns the whole budget thinking and returns null
        # content unless reasoning is switched off explicitly.
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        path = self._cache_path(payload)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                with self._lock:
                    self.cache_hits += 1
                return json.load(fh)["content"]

        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(
                    self.endpoint,
                    data=body,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = (data["choices"][0]["message"].get("content") or "").strip()
                usage = data.get("usage") or {}
                with self._lock:
                    self.calls += 1
                    self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    self.completion_tokens += int(usage.get("completion_tokens") or 0)
                if not content:
                    raise ValueError("model returned empty content")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({"content": content, "usage": usage}, fh)
                return content
            except Exception as exc:  # noqa: BLE001 - retry anything transient
                last_error = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403):
                    raise
                delay = min(20.0, (2 ** attempt) * 0.8) * (0.6 + 0.8 * random.random())
                log.debug("LLM retry %d after %s (%.1fs)", attempt + 1, type(exc).__name__, delay)
                time.sleep(delay)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def complete_json(self, prompt: str, max_tokens: int = 700, system: str | None = None) -> Any:
        return extract_json(self.complete(prompt, max_tokens=max_tokens, system=system))

    # ------------------------------------------------------------------
    def map_concurrent(
        self,
        items: list[Any],
        worker: Callable[[Any], Any],
        workers: int = 8,
        label: str = "llm",
        log_every: int = 50,
    ) -> list[Any]:
        """Run `worker` over `items`, keeping order. Failures become None."""
        results: list[Any] = [None] * len(items)
        done = 0
        lock = threading.Lock()

        def run(index_item: tuple[int, Any]) -> None:
            nonlocal done
            index, item = index_item
            try:
                results[index] = worker(item)
            except Exception as exc:  # noqa: BLE001 - one bad item must not stop the run
                log.warning("[%s] item %d failed: %s: %s", label, index, type(exc).__name__, str(exc)[:120])
            with lock:
                done += 1
                if log_every and done % log_every == 0:
                    log.info("[%s] %d/%d", label, done, len(items))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, enumerate(items)))
        return results

    # ------------------------------------------------------------------
    def usage_summary(self) -> str:
        return (
            f"{self.calls} API calls ({self.cache_hits} served from cache), "
            f"{self.prompt_tokens:,} prompt tokens, {self.completion_tokens:,} completion tokens"
        )


def build_client_from_env(
    env_paths: Iterable[str],
    cache_dir: str,
    url_key: str = "VLLM_URL",
    model_key: str = "VLLM_MODEL",
    key_key: str = "VLLM_API_KEY",
    effort_key: str = "LLM_REASONING_EFFORT",
    overrides: dict[str, str] | None = None,
) -> LLMClient:
    """First .env that carries a usable endpoint wins."""
    merged: dict[str, str] = {}
    for path in env_paths:
        for key, value in load_env_file(path).items():
            merged.setdefault(key, value)
    merged.update({k: v for k, v in (overrides or {}).items() if v})

    url, model, key = merged.get(url_key), merged.get(model_key), merged.get(key_key, "")
    if not url or not model:
        raise RuntimeError(
            f"No LLM endpoint configured: set {url_key} and {model_key} in one of "
            + ", ".join(env_paths)
        )
    effort = (merged.get(effort_key) or "").strip() or None
    return LLMClient(url, key, model, cache_dir, reasoning_effort=effort)
