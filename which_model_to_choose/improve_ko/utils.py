# which_model_to_choose/improve_ko/utils.py

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


def _salvage_single_field_from_truncated(raw: str, field: str) -> Optional[Dict[str, Any]]:
    """
    Last-ditch salvage for cases where model output starts with {"<field>":" but is truncated
    or contains invalid JSON (e.g. unescaped quotes). We try to extract the longest plausible
    string value and synthesise a minimal JSON object {field: "..."}.
    """
    if not raw:
        return None
    s = raw.strip()

    anchor = f'{{"{field}":"'
    i = s.find(anchor)
    if i == -1:
        return None

    # position just after the opening quote of the string value
    j = i + len(anchor)

    in_escape = False
    last_quote = -1
    while j < len(s):
        ch = s[j]
        if in_escape:
            in_escape = False
        elif ch == "\\":
            in_escape = True
        elif ch == '"':
            # treat as possible terminator for the string
            last_quote = j
        j += 1

    if last_quote == -1:
        return None

    value_raw = s[i + len(anchor): last_quote]

    try:
        # Let json handle unescaping if possible
        value = json.loads(f'"{value_raw}"')
    except Exception:
        # Fall back to raw slice if unescaping fails
        value = value_raw

    return {field: value}


def _extract_single_field_json(raw: str, field: str) -> Dict[str, Any]:
    """
    Generic version of extract_*_json:
    Extract exactly { "<field>": "..." } from model output.
    Same robustness as extract_summary_json, but parametrised by field name.
    """
    s = (raw or "").strip()

    # Strip code fences
    if s.startswith("```"):
        s = s.split("```", 1)[-1]
    if "```" in s:
        s = s.split("```", 1)[0].strip()

    # Fast path
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and set(obj.keys()) == {field} and isinstance(obj.get(field), str):
            return obj
    except Exception:
        pass

    # Helper: extract first JSON object (same as in extract_summary_json)
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
            if isinstance(obj, dict) and set(obj.keys()) == {field} and isinstance(obj.get(field), str):
                return obj
        except Exception:
            pass

    # Simple regex for this field
    pattern = re.compile(rf'\{{\s*"{field}"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}}', re.DOTALL)
    m = pattern.search(s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and set(obj.keys()) == {field} and isinstance(obj.get(field), str):
                return obj
        except Exception:
            pass

    # Truncated / invalid JSON salvage as a last resort
    salvaged = _salvage_single_field_from_truncated(s, field)
    if salvaged and isinstance(salvaged.get(field), str):
        return salvaged

    try:
        dbg = Path(f"last_bad_response_{field}.bad.txt")
        dbg.write_text(raw, encoding="utf-8")
    except Exception:
        pass

    preview = s[:400].replace("\n", "\\n")
    raise ValueError(
        f'Could not extract a valid single-key JSON object {{"{field}": "..."}} '
        f"from model output. First 400 chars:\\n{preview}"
    )


def extract_summary_json(raw: str) -> Dict[str, Any]:
    return _extract_single_field_json(raw, "summary")


def extract_cleaned_json(raw: str) -> Dict[str, Any]:
    """
    Parse cleaned text from model output.

    New default behaviour with CLEAN_PROMPT:
      - Model is expected to return *plain cleaned text* (no JSON).
      - We still support legacy JSON forms:
          {"cleaned": "..."}  or  {"summary": "..."}.
    Always returns: {"cleaned": "<cleaned_text>"}.
    """
    s = (raw or "").strip()

    # Strip code fences if some model still insists on them
    if s.startswith("```"):
        # Drop the first fence and keep the rest
        s = s.split("```", 1)[-1].strip()
    if "```" in s:
        # Drop anything after a closing fence
        s = s.split("```", 1)[0].strip()

    # 1) Try direct JSON first (old behaviour / misaligned models)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            # Preferred: {"cleaned": "..."}
            if isinstance(obj.get("cleaned"), str):
                return {"cleaned": obj["cleaned"].strip()}
            # Legacy: {"summary": "..."} reused as "cleaned"
            if isinstance(obj.get("summary"), str):
                return {"cleaned": obj["summary"].strip()}
    except Exception:
        # Not JSON – this is expected with the new CLEAN_PROMPT
        pass

    # 2) Non-JSON: treat the full body as already-cleaned text
    if s:
        return {"cleaned": s}

    # 3) Last-resort salvage for truncated JSON-like outputs
    #    (keeps robustness from older code paths)
    salvaged = (
        _salvage_single_field_from_truncated(raw, "cleaned")
        or _salvage_single_field_from_truncated(raw, "summary")
    )
    if salvaged:
        val = (salvaged.get("cleaned") or salvaged.get("summary") or "").strip()
        if val:
            return {"cleaned": val}

    # If we reach here, the model really gave us nothing usable
    raise ValueError("Empty cleaned text from model output")

    # try:
    #     # First, try the correct shape
    #     return _extract_single_field_json(raw, "cleaned")
    # except ValueError:
    #     # Fallback: accept {"summary": "..."} and remap to 'cleaned'
    #     obj = _extract_single_field_json(raw, "summary")
    #     cleaned = (obj.get("summary") or "").strip()
    #     if not cleaned:
    #         raise ValueError("Fallback summary-based cleaned text is empty")
    #     return {"cleaned": cleaned}


def extract_metadata_json(raw: str) -> Dict[str, Any]:
    """
    Parse metadata JSON for improved English metadata:

    Expected shape:
    {
      "title_improved": "<title>",
      "subtitle_improved": "<subtitle or empty string>",
      "description_improved": "<description>",
      "keywords_improved": ["kw1", "kw2", ...]
    }
    """
    s = (raw or "").strip()

    # Strip code fences
    if s.startswith("```"):
        s = s.split("```", 1)[-1]
    if "```" in s:
        s = s.split("```", 1)[0].strip()

    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        # try first JSON object using same brace walker
        from typing import Optional
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
            obj = json.loads(candidate)

    if not isinstance(obj, dict):
        raise ValueError("Metadata output is not a JSON object")

    required = {"title_improved", "subtitle_improved", "description_improved", "keywords_improved"}
    missing = required - set(obj.keys())
    if missing:
        raise ValueError(f"Metadata JSON missing required keys: {missing}")

    # Type checks
    if not isinstance(obj["title_improved"], str):
        raise ValueError("title_improved must be string")
    if not isinstance(obj["subtitle_improved"], str):
        raise ValueError("subtitle_improved must be string")
    if not isinstance(obj["description_improved"], str):
        raise ValueError("description_improved must be string")

    # Must be a list
    keywords = obj.get("keywords_improved")
    if not isinstance(keywords, list) or not all(isinstance(x, str) for x in keywords):
        raise ValueError("keywords_improved must be a list of strings")


    # if not isinstance(keywords, list):
    #     raise ValueError("keywords_improved must be a list of strings")
    # if not all(isinstance(x, str) for x in keywords):
    #     raise ValueError("keywords_improved must be a list of strings")

    return obj
