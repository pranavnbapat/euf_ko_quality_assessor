# text_utils.py

import re
import string

from typing import Optional, List

from langdetect import detect


_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_OCR_SPLIT = re.compile(r"([A-Za-z])\s*-\s*\n?\s*([A-Za-z])")
_NON_ASCII = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")

def norm_text(s: Optional[str]) -> str:
    if not s: return ""
    s = s.replace("\u00AD", "")
    s = _OCR_SPLIT.sub(r"\1\2", s)
    s = s.replace(" \n", " ").replace("\n", " ")
    s = _WS.sub(" ", s).strip()
    return s

def tokens(s: str) -> List[str]:
    s = _URL.sub(" ", s.lower())
    tb = s.translate(str.maketrans("", "", string.punctuation))
    return [t for t in tb.split() if len(t) >= 2]

def detect_lang_safe(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def token_overlap_ratio(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def regexes():
    return _URL, _OCR_SPLIT, _NON_ASCII
