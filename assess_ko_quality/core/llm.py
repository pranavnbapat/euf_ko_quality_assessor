# assess_ko_quality/core/llm.py
"""
LLM access for the KO quality pipeline.

This is a thin adapter over `semantic_vs_search/llm_client.py`, which already solves
the hard parts and is documented in ../USING_LLMS.md:

  - disk caching keyed on a SHA-256 of the whole payload, so re-running an experiment
    costs nothing and stays reproducible
  - `reasoning_effort` sent optimistically and dropped on the specific 400 from models
    that reject it, remembered per client
  - token budget escalation, because a truncated reasoning model returns empty content
    rather than an error - the failure mode that once turned a real Fleiss kappa of
    0.60 into an apparent 0.75 by silently dropping the rows it could not finish
  - jittered retry backoff, concurrency, token accounting

Only one thing is added here: the model is discovered from the provider at runtime
instead of being pinned. USING_LLMS.md says to query /v1/models for the live list
rather than trusting a table, so `resolve_model` matches a preference expression
against what is actually served. "qwen3.5" resolves to whatever qwen3.5-* exists today,
and a comma-separated list falls through to the next entry when one is retired.

Defaults follow USING_LLMS.md: credentials come from farm_assistant_um/.env first
(the most complete, and on Scaleway), with this repo's .env as a fallback.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# semantic_vs_search is a sibling folder in this repo, not a separate service, and
# USING_LLMS.md points here explicitly as the client to reuse.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SVS = _REPO_ROOT / "semantic_vs_search"
if str(_SVS) not in sys.path:
    sys.path.insert(0, str(_SVS))

from llm_client import (  # noqa: E402
    LLMClient,
    build_client_from_env,
    extract_json,
    load_env_file,
)

# farm_assistant_um first: it is the most complete Scaleway config and the one
# USING_LLMS.md says to copy from. The repo .env (also Scaleway) is the fallback.
DEFAULT_ENV_PATHS: Sequence[str] = (
    str(_REPO_ROOT.parent / "farm_assistant_um" / ".env"),
    str(_REPO_ROOT / "scout" / ".env"),
    str(_REPO_ROOT / ".env"),
)

DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parent.parent / ".cache" / "llm")

# Preference expression, not a pinned id. Override with KOQ_LLM_MODEL.
DEFAULT_MODEL_PREFERENCE = "qwen3.5,gpt-oss,mistral-medium,glm"

# For anything where agreement matters, USING_LLMS.md is explicit: use several model
# FAMILIES, not several checkpoints of one, because two Qwen models share their
# mistakes and their agreement tells you very little.
DEFAULT_PANEL_PREFERENCE = "qwen3.5,gpt-oss,mistral-medium,glm"


def env_paths(extra: Sequence[str] = ()) -> List[str]:
    paths = [p for p in extra if p]
    if os.getenv("KOQ_ENV_FILES"):
        paths += os.environ["KOQ_ENV_FILES"].split(":")
    return paths + list(DEFAULT_ENV_PATHS)


def merged_env(extra: Sequence[str] = ()) -> Dict[str, str]:
    """Merge the candidate .env files, earliest winning, then the real environment."""
    merged: Dict[str, str] = {}
    for path in env_paths(extra):
        for k, v in load_env_file(path).items():
            merged.setdefault(k, v)
    for k, v in os.environ.items():
        if v:
            merged[k] = v
    return merged


def list_models(base_url: str, api_key: str, timeout: int = 20) -> List[str]:
    """Ask the provider what it actually serves. Empty list if it will not say."""
    base = base_url.rstrip("/")
    base = base if base.endswith("/v1") else base + "/v1"
    req = urllib.request.Request(base + "/models",
                                 headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
    except Exception:
        return []


def resolve_model(preference: str, available: Sequence[str]) -> Optional[str]:
    """First preference the provider serves. Substring match, shortest id wins."""
    for want in [p.strip() for p in (preference or "").split(",") if p.strip()]:
        if want in available:
            return want
        hits = [m for m in available if want.lower() in m.lower()]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def get_client(
    model: str = "",
    cache_dir: str = "",
    extra_env_paths: Sequence[str] = (),
    strict: bool = False,
) -> Optional[LLMClient]:
    """
    Build a cached, quirk-handling client whose model is resolved against the provider.

    `model` is a preference expression, not an id. Returns None when nothing is
    configured, so callers can degrade to heuristics only; pass strict=True to raise.
    """
    env = merged_env(extra_env_paths)
    url = env.get("LLM_URL") or env.get("VLLM_URL") or ""
    key = env.get("LLM_API_KEY") or env.get("VLLM_API_KEY") or ""
    if not url:
        if strict:
            raise RuntimeError("No LLM endpoint configured; see ../USING_LLMS.md")
        return None

    preference = model or os.getenv("KOQ_LLM_MODEL") or DEFAULT_MODEL_PREFERENCE
    available = list_models(url, key)
    chosen = resolve_model(preference, available)
    if chosen is None:
        # Provider would not list models, or nothing matched: fall back to configured id.
        chosen = env.get("LLM_MODEL") or env.get("VLLM_MODEL") or ""
        if not chosen:
            if strict:
                raise RuntimeError(f"No model matched {preference!r}; served: {available}")
            return None

    try:
        return build_client_from_env(
            env_paths(extra_env_paths),
            cache_dir=cache_dir or DEFAULT_CACHE_DIR,
            overrides={"LLM_URL": url, "LLM_API_KEY": key, "LLM_MODEL": chosen},
        )
    except Exception:
        if strict:
            raise
        return None


def get_panel(
    preference: str = "",
    cache_dir: str = "",
    extra_env_paths: Sequence[str] = (),
) -> List[LLMClient]:
    """
    One client per distinct model family, for tasks where agreement is the measurement.

    Panels exist to expose shared priors. Two checkpoints of the same family agreeing
    is not evidence, so each preference term contributes at most one client.
    """
    env = merged_env(extra_env_paths)
    url = env.get("LLM_URL") or env.get("VLLM_URL") or ""
    key = env.get("LLM_API_KEY") or env.get("VLLM_API_KEY") or ""
    if not url:
        return []
    available = list_models(url, key)
    wanted = preference or os.getenv("KOQ_LLM_PANEL") or DEFAULT_PANEL_PREFERENCE

    clients: List[LLMClient] = []
    seen: set = set()
    for term in [t.strip() for t in wanted.split(",") if t.strip()]:
        chosen = resolve_model(term, available)
        if not chosen or chosen in seen:
            continue
        seen.add(chosen)
        client = get_client(model=chosen, cache_dir=cache_dir, extra_env_paths=extra_env_paths)
        if client is not None:
            clients.append(client)
    return clients


def describe() -> Dict[str, Any]:
    """What this machine would use right now - for logging at the top of a run."""
    env = merged_env()
    url = env.get("LLM_URL") or env.get("VLLM_URL") or ""
    key = env.get("LLM_API_KEY") or env.get("VLLM_API_KEY") or ""
    available = list_models(url, key) if url else []
    return {
        "endpoint": url,
        "key_present": bool(key),
        "models_served": len(available),
        "available": available,
        "resolved": resolve_model(os.getenv("KOQ_LLM_MODEL") or DEFAULT_MODEL_PREFERENCE, available),
        "panel": [resolve_model(t, available)
                  for t in (os.getenv("KOQ_LLM_PANEL") or DEFAULT_PANEL_PREFERENCE).split(",")],
    }


def escape_stray_quotes(text: str) -> str:
    """
    Escape unescaped ASCII double quotes INSIDE JSON strings. Models writing
    non-English text close typographic quotations with a plain '"' („Horizontas
    2020"), which terminates the JSON string early. A '"' inside a string is kept as
    a terminator only when the next non-space character is structural (:,}] or the
    end); anything else is content and gets escaped.
    """
    out: List[str] = []
    in_str = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if not in_str:
            in_str = ch == '"'
            out.append(ch)
        elif ch == "\\":
            out.append(text[i:i + 2])
            i += 1
        elif ch == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in ":,}]":
                in_str = False
                out.append(ch)
            else:
                out.append('\\"')
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def complete_json_robust(client: LLMClient, content: Any, max_tokens: int,
                         system: str) -> Any:
    """
    complete_json with the two fallbacks every stage needs: a doubled token budget
    when the answer came back truncated, and a quote-repair pass when the answer is
    JSON broken by unescaped quotes inside strings. The repair re-reads the raw text
    from the same call (disk-cached, so free) and parses the repaired version. None
    when nothing usable came back.
    """
    budget = max_tokens
    for _ in range(2):
        try:
            raw = client.complete_json(content, max_tokens=budget, system=system)
            if raw is not None:
                return raw
        except ValueError:
            try:
                text = client.complete(content, max_tokens=budget, system=system)
                return extract_json(escape_stray_quotes(text))
            except Exception:
                pass                   # truncated or unrepairable: retry with more room
        budget *= 2
    return None


def transcribe(audio: bytes, model: str = "whisper-large-v3", filename: str = "audio.mp3",
               cache_dir: str = "", timeout: int = 600) -> Tuple[str, bool]:
    """
    Speech to text through the provider's OpenAI-compatible /v1/audio/transcriptions
    (whisper-large-v3 on Scaleway). Returns (text, served_from_cache). Cached on disk
    by audio bytes and model, like every chat call, so a rerun is free.
    """
    import hashlib

    import requests

    env = merged_env()
    url = (env.get("LLM_URL") or env.get("VLLM_URL") or "").rstrip("/")
    key = env.get("LLM_API_KEY") or env.get("VLLM_API_KEY") or ""
    if not url:
        raise RuntimeError("No LLM endpoint configured; see ../USING_LLMS.md")
    url = url if url.endswith("/v1") else url + "/v1"
    digest = hashlib.sha256(model.encode("utf-8") + b"\0" + audio).hexdigest()[:24]
    cdir = Path(cache_dir or DEFAULT_CACHE_DIR) / "transcribe"
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"{digest}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"], True
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            r = requests.post(url + "/audio/transcriptions",
                              headers={"Authorization": f"Bearer {key}"},
                              files={"file": (filename, audio, "application/octet-stream")},
                              data={"model": model, "response_format": "json"},
                              timeout=timeout)
            if r.status_code in (400, 401, 403, 413, 415):
                raise RuntimeError(f"transcription rejected ({r.status_code}): {r.text[:200]}")
            r.raise_for_status()
            text = str((r.json() or {}).get("text") or "").strip()
            path.write_text(json.dumps({"text": text, "model": model}, ensure_ascii=False),
                            encoding="utf-8")
            return text, False
        except RuntimeError:
            raise
        except Exception as exc:                  # transient: retry with a pause
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"transcription failed after 3 attempts: {last}")


__all__ = [
    "LLMClient", "get_client", "get_panel", "describe",
    "list_models", "resolve_model", "merged_env", "env_paths", "extract_json",
    "escape_stray_quotes", "complete_json_robust", "transcribe",
    "DEFAULT_MODEL_PREFERENCE", "DEFAULT_PANEL_PREFERENCE", "DEFAULT_CACHE_DIR",
]
