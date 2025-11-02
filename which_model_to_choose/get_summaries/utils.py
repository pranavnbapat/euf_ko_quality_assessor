# utils.py

from __future__ import annotations

import json, re

from pathlib import Path
from typing import Any, Dict, List, Optional

def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


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

# ---- JSON extraction (strict but tolerant) ----
_JSON_OBJECT_RE = re.compile(r'\{\s*"summary"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}', re.DOTALL)


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
