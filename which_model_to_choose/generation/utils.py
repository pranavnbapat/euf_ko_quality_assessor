from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def approx_token_count(text: str) -> int:
    return max(1, int(len(text) / 4))


def split_into_tokenish_chunks(text: str, chunk_tok: int, overlap_tok: int) -> List[str]:
    step = max(1, (chunk_tok - overlap_tok) * 4)
    width = max(step, chunk_tok * 4)
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i:i + width])
        i += step
    return chunks


_JSON_OBJECT_RE = re.compile(r'\{\s*"summary"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}', re.DOTALL)


def _salvage_summary_from_truncated(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    anchor = '{"summary":"'
    i = s.find(anchor)
    if i == -1:
        return None
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
            last_quote = j
        j += 1
    if last_quote == -1:
        return None
    summary_str = s[i + len(anchor):last_quote]
    try:
        summary = json.loads(f'"{summary_str}"')
    except Exception:
        summary = summary_str
    return {"summary": summary}


def extract_summary_json(raw: str) -> Dict[str, Any]:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("```", 1)[-1]
    if "```" in s:
        s = s.split("```", 1)[0].strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and set(obj.keys()) == {"summary"} and isinstance(obj.get("summary"), str):
            return obj
    except Exception:
        pass

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
                        return text[start:i + 1]
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

    m = _JSON_OBJECT_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and set(obj.keys()) == {"summary"} and isinstance(obj.get("summary"), str):
                return obj
        except Exception:
            pass

    salvaged = _salvage_summary_from_truncated(s)
    if salvaged and isinstance(salvaged.get("summary"), str):
        return salvaged

    try:
        Path("last_bad_response.bad.txt").write_text(raw, encoding="utf-8")
    except Exception:
        pass

    preview = s[:400].replace("\n", "\\n")
    raise ValueError(
        'Could not extract a valid single-key JSON object {"summary": "..."} '
        f"from model output. First 400 chars:\n{preview}"
    )


def extract_metadata_json(raw: str) -> Dict[str, Any]:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("```", 1)[-1]
    if "```" in s:
        s = s.split("```", 1)[0].strip()

    def is_valid(obj: Dict[str, Any]) -> bool:
        required = {"title", "subtitle", "description", "keywords"}
        if set(obj.keys()) != required:
            return False
        if not all(isinstance(obj[k], str) for k in ("title", "subtitle", "description")):
            return False
        kws = obj.get("keywords")
        if isinstance(kws, list):
            return all(isinstance(x, str) for x in kws)
        return isinstance(kws, str)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and is_valid(obj):
            return obj
    except Exception:
        pass

    def extract_first_json_object(text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        i = start
        in_str = False
        esc = False
        while i < len(text):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
            i += 1
        return None

    candidate = extract_first_json_object(s)
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and is_valid(obj):
                return obj
        except Exception:
            pass

    preview = s[:400].replace("\n", "\\n")
    raise ValueError(
        'Could not extract a valid metadata JSON object with keys '
        '{"title","subtitle","description","keywords"} '
        f"from model output. First 400 chars:\n{preview}"
    )


def clamp_metadata_lengths(obj: Dict[str, Any]) -> Dict[str, Any]:
    title = str(obj.get("title", "")).strip()[:90].strip()
    subtitle = str(obj.get("subtitle", "")).strip()[:140].strip()
    description = str(obj.get("description", "")).strip()[:600].strip()
    keywords = obj.get("keywords", [])

    if isinstance(keywords, str):
        keywords_list = [x.strip() for x in keywords.split(",") if x.strip()]
    elif isinstance(keywords, list):
        keywords_list = [str(x).strip() for x in keywords if str(x).strip()]
    else:
        keywords_list = []

    deduped: List[str] = []
    seen = set()
    for kw in keywords_list:
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(kw[:64])
    deduped = deduped[:10]

    return {
        "title": title,
        "subtitle": subtitle,
        "description": description,
        "keywords": deduped,
    }
