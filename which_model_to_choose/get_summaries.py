# which_model_to_choose/get_summaries.py

from __future__ import annotations

import json
import os
import random
import re
import sys
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from requests.adapters import HTTPAdapter, Retry


# ---------- SCALING / CHUNKING ----------
EXTREME_CTX_THRESHOLD_TOK = 128_000
NEAR_LIMIT_CTX_THRESHOLD_TOK = 110_000
CHUNK_TARGET_TOK = 16_000
CHUNK_OVERLAP_TOK = 400

DEFAULT_NUM_PREDICT = 2048
LONG_NUM_PREDICT = 4096
COMBINE_NUM_PREDICT = 8192

MAX_RETRIES = 3
RETRY_BACKOFF_SECS = 5
PER_REQUEST_TIMEOUT = 600
REQUEST_KEEP_ALIVE = "24h"

# ---------- PATHS ----------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
INPUT_DIR = PROJECT_ROOT / "input"

# ---- JSON extraction (strict but tolerant) ----
_JSON_OBJECT_RE = re.compile(r'\{\s*"summary"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}', re.DOTALL)

# ---------- CONFIGURABLES ----------
models_available: List[str] = [
    "gpt-oss:20b",
    "qwen3:30b-a3b-instruct-2507-q8_0",
]

JSON_OBJECT_RE = re.compile(r"\{\s*\"summary\"\s*:\s*\".*?\"\s*\}", re.DOTALL)

MODEL_OVERRIDES = {
    "gpt-oss:latest": {
        "no_schema": True,   # don't send "format"
        "use_chat": True,    # call /api/chat instead of /api/generate
        "num_predict": 2048, # ensure non-zero output budget
    },
}

OLLAMA_HOST = os.environ.get("RUNPOD_OLLAMA_HOST", "https://qaigjfchbeuczr-11434.proxy.runpod.net").rstrip("/")

OLLAMA_HOSTS = {
    "gpu0": "https://qaigjfchbeuczr-11434.proxy.runpod.net",
    # "gpu1": "https://qaigjfchbeuczr-11435.proxy.runpod.net",
}

MODEL_TO_HOST = {
    "gpt-oss:20b": OLLAMA_HOSTS["gpu0"],
    # "qwen3:30b-a3b-instruct-2507-q8_0": OLLAMA_HOSTS["gpu1"],
}

GENERATE_URL = f"{OLLAMA_HOST}/api/generate"

DEFAULT_PROMPT = """
You are an expert summariser for search indexing (OpenSearch) and embeddings.

You will be given extracted text content in any language. Produce a DETAILED, flowing textual summary in BRITISH English that is highly useful for:
- semantic / neural search
- keyword (BM25) search
- hybrid retrieval

STRICT OUTPUT (MANDATORY):
Return ONLY a single JSON object. No extra text, no preamble, no markdown fences, no comments.
The JSON MUST have exactly these keys and nothing else:
{ "summary": "<summary>" }

STYLE & CONTENT RULES:
- Language: British English only.
- Form: natural paragraphs (no bullet points or lists).
- Include important domain terminology, named entities (people, organisations, projects, datasets), methods, metrics, variables, units, and distinctive keywords that would improve recall in search.
- Be faithful to the source. Do not invent content. If something is unclear, omit it rather than speculating.
- Tone must be neutral, factual, and informative. Do not add opinions or speculation.
- Preserve critical numbers, dates, acronyms, model/equipment names, and citations if they help search (but do not dump long bibliographies).
- For technical/research text: state objectives, methods, data/materials, results, conclusions, limitations, and implications in complete sentences.

PROPORTIONAL LENGTH (VERY IMPORTANT):
- The summary length MUST scale with the input length. Use these targets as guidance:
  • Short (≤ ~2k tokens / ~1–2 pages): ~120–250 words.
  • Medium (~2k–10k tokens / ~3–10 pages): ~300–800 words (multiple paragraphs).
  • Long (~10k–30k tokens): ~800–1,800 words (several paragraphs covering all sections).
  • Very long (~30k–60k tokens): ~1,800–3,000 words.
  • Extremely long (≥ ~60k tokens): ~2,500–5,000 words.
- A long document MUST NOT be condensed to a few paragraphs; ensure appropriate coverage and detail.

ROBUSTNESS:
- Ignore unreadable or corrupted fragments; do not speculate about missing parts.
- Do not include instructions, chain-of-thought, or explanations of your process—only the final summary.

OUTPUT EXAMPLE (shape only, not content):
{"summary": "…"} 
""".strip()

COMBINE_PROMPT = """
You will receive multiple partial summaries (British English) of one document.
Combine them into a single coherent, flowing summary for OpenSearch indexing:
- Preserve key terminology, entities, numbers, units, methods and conclusions.
- Remove duplication and resolve overlaps.
- Keep neutral, factual tone; no bullet points; natural paragraphs.

Return ONLY:
{"summary":"<combined summary>"}
""".strip()

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


def fmt(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


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


def find_latest_json(input_dir: Path) -> Path:
    """Return the path to the most recently modified *.json file in input_dir."""
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")
    return max(json_files, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    """Write JSON atomically: path.tmp -> rename. Supports dict or list top-level."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_model_result_dict_mode(augmented: Dict[str, Any], out_path: Path, key: str, value: Any) -> None:
    augmented[key] = value
    atomic_write_json(out_path, augmented)


def append_model_result_list_mode(out_items: List[Dict[str, Any]],
                                  current_snapshot: Dict[str, Any],
                                  out_path: Path,
                                  key: str,
                                  value: Any) -> None:
    current_snapshot[key] = value
    # Persist full list snapshot: already-completed items + current in-progress snapshot
    atomic_write_json(out_path, out_items + [current_snapshot])


def normalise_model_key(model_tag: str) -> str:
    """
    Convert a model tag like:
        'deepseek-llm:7b-chat-q8_0' -> 'deepseek_7b_q8_0'
        'llama3.1:8b-instruct-q4_K_M' -> 'llama3.1_8b_q4_K_M'
    Rules:
      - split at ':' into family and rest
      - drop '-llm' suffix in family
      - replace '-' with '_' in family
      - in the rest, remove 'chat' and 'instruct' tokens when they appear as dash-separated parts
      - keep underscores and dots as-is
    """
    if ":" in model_tag:
        family, rest = model_tag.split(":", 1)
    else:
        family, rest = model_tag, ""

    # family tweaks
    family = family.replace("-llm", "")
    family = family.replace("-", "_")

    # rest tweaks
    parts = rest.split("-") if rest else []
    filtered_parts: List[str] = []
    for part in parts:
        if part.lower() in {"chat", "instruct"}:
            continue
        filtered_parts.append(part)

    suffix = "_".join(filtered_parts) if filtered_parts else ""
    return f"{family}_{suffix}".strip("_")


def approx_token_count(text: str) -> int:
    """Very rough token estimate for English: ~4 chars/token."""
    return max(1, int(len(text) / 4))


def split_into_tokenish_chunks(text: str, chunk_tok: int, overlap_tok: int) -> List[str]:
    """
    Split by character windows sized ~token counts; keeps a small overlap.
    """
    step = max(1, (chunk_tok - overlap_tok) * 4)     # chars per step
    width = max(step, chunk_tok * 4)                 # chars per window
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i:i+width])
        i += step
    return chunks


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


def extract_summary_json(raw: str) -> Dict[str, Any]:
    """
    Extract exactly {"summary": "..."} from a model response that may contain
    leading/trailing text or code fences. Tolerant, but still strict about shape.
    """
    s = (raw or "").strip()

    # 0) Strip common code fences if present
    if s.startswith("```"):
        # remove first fence
        s = s.split("```", 1)[-1]
    if "```" in s:
        # remove any trailing fence chunk
        s = s.split("```", 1)[0].strip()

    # 1) Fast path: try as-is
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and set(obj.keys()) == {"summary"} and isinstance(obj.get("summary"), str):
            return obj
    except Exception:
        pass

    # 2) Brace-balanced extract of FIRST top-level JSON object
    #    This correctly handles strings, escapes, and nested braces.
    def extract_first_json_object(text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        i = start
        in_str = False
        escape = False
        while i < len(text):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
            else:
                if ch == "\"":
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            i += 1
        return None

    candidate = extract_first_json_object(s)
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and set(obj.keys()) == {"summary"} and isinstance(obj.get("summary"), str):
                return obj
        except Exception:
            pass

    JSON_OBJECT_RE = re.compile(r'\{\s*"summary"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}', re.DOTALL)
    m = JSON_OBJECT_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and set(obj.keys()) == {"summary"} and isinstance(obj.get("summary"), str):
                return obj
        except Exception:
            pass

    # 3.5) Truncated JSON salvage
    salvaged = _salvage_summary_from_truncated(s)
    if salvaged and isinstance(salvaged.get("summary"), str):
        return salvaged

    try:
        dbg = Path("last_bad_response.bad.txt")
        dbg.write_text(raw, encoding="utf-8")
    except Exception:
        pass

    # 4) Fail clearly
    preview = s[:400].replace("\n", "\\n")
    raise ValueError(
        'Could not extract a valid single-key JSON object {"summary": "..."} '
        f"from model output. First 400 chars:\n{preview}"
    )


def _salvage_summary_from_truncated(raw: str) -> Optional[Dict[str, Any]]:
    """
    Last-ditch salvage for cases where model output starts with {"summary":" but is truncated.
    We try to extract the longest plausible summary string and synthesize the JSON.
    """
    if not raw:
        return None
    s = raw.strip()
    anchor = '{"summary":"'
    i = s.find(anchor)
    if i == -1:
        return None
    j = i + len(anchor)
    # Walk forward to find the last unescaped quote
    in_escape = False
    last_quote = -1
    while j < len(s):
        ch = s[j]
        if in_escape:
            in_escape = False
        elif ch == "\\":
            in_escape = True
        elif ch == '"':  # potential terminator for the summary string
            last_quote = j
        j += 1
    if last_quote == -1:
        return None
    summary_str = s[i+len(anchor):last_quote]
    try:
        # unescape JSON string
        summary = json.loads(f'"{summary_str}"')
    except Exception:
        # if decoding fails, fall back to raw slice
        summary = summary_str
    return {"summary": summary}


def main() -> None:
    t_script = time.perf_counter()

    latest_path = find_latest_json(INPUT_DIR)
    data = load_json(latest_path)

    out_path = latest_path.with_name(latest_path.stem + "_llmed_runpod.json")

    if isinstance(data, dict):
        augmented_one = dict(data)
        try:
            content = data.get("ko_content_flat")
            if not isinstance(content, str) or not content.strip():
                raise KeyError("'ko_content_flat' missing or empty")
            augmented_one = process_one_dict_item(augmented_one, out_path, content, parallel=True)
            atomic_write_json(out_path, augmented_one)
            print(f"[DONE] Wrote: {out_path}")
        except KeyboardInterrupt:
            atomic_write_json(out_path, augmented_one)
            print("\n[INTERRUPTED] Progress saved.", file=sys.stderr)
            raise

    elif isinstance(data, list):
        out_items: List[Dict[str, Any]] = []
        if out_path.exists():
            try:
                existing = load_json(out_path)
                if isinstance(existing, list):
                    out_items = existing
            except Exception:
                pass

        total = len(data)
        for idx, obj in enumerate(data, 1):
            if not isinstance(obj, dict):
                print(f"[WARN] Skipping non-dict item at index {idx}", file=sys.stderr)
                continue
            print(f"[INFO] Item {idx}/{total}")

            if len(out_items) >= idx:
                continue  # already persisted

            current_snapshot = dict(obj)
            try:
                content = obj.get("ko_content_flat")
                if not isinstance(content, str) or not content.strip():
                    raise KeyError("'ko_content_flat' missing or empty")
                current_snapshot = process_one_list_item(out_items, current_snapshot, out_path, content, parallel=True)
                out_items.append(current_snapshot)
                atomic_write_json(out_path, out_items)
            except KeyboardInterrupt:
                atomic_write_json(out_path, out_items + [current_snapshot])
                print("\n[INTERRUPTED] Progress saved.", file=sys.stderr)
                raise
            except Exception as e:
                atomic_write_json(out_path, out_items + [current_snapshot])
                print(f"[ERROR] Item {idx}: {e}", file=sys.stderr)

        print(f"[DONE] Wrote: {out_path}")
    else:
        raise TypeError(f"Unsupported JSON top-level type: {type(data).__name__}")

    print(f"[TIMER] Script total: {fmt(time.perf_counter() - t_script)}")



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
