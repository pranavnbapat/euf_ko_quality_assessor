# assess_ko_quality/quality_text_utils.py

import re
import string
from typing import Any, Iterable, List

from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords

DetectorFactory.seed = 42

# ---------- Regexes ----------
_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_ASCII = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")

# Full NLTK English stopword list (for lexical filtering)
_STOP_EN = set(stopwords.words("english"))


def norm_text(s: Any) -> str:
    """
    Basic normalisation: coalesce whitespace, strip, flatten newlines.
    """
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00AD", "")  # soft hyphen
    s = s.replace(" \n", " ").replace("\n", " ")
    s = _WS.sub(" ", s).strip()
    return s


def tokens(s: str) -> List[str]:
    """
    Very simple tokeniser: lowercase, strip punctuation, keep ≥2-char tokens.
    """
    s = s.lower()
    s = _URL.sub(" ", s)
    tb = s.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in tb.split() if len(t) >= 2]
    return toks


def strip_stops(toks: List[str]) -> List[str]:
    """
    Remove English stopwords from a list of tokens.
    """
    return [t for t in toks if t not in _STOP_EN]


def unique_ratio(toks: List[str]) -> float:
    """
    Ratio of unique tokens to total tokens (0.0 if empty).
    """
    return 0.0 if not toks else len(set(toks)) / len(toks)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """
    Jaccard similarity between two token iterables.
    """
    A, B = set(a), set(b)
    return 0.0 if not A or not B else len(A & B) / len(A | B)


def detect_lang_safe(text: str) -> str:
    """
    langdetect wrapper that never raises.
    """
    try:
        return detect(text)
    except Exception:
        return "unknown"


def sentence_lengths(text: str) -> List[int]:
    """
    Split text into sentences on .?! and compute token lengths.
    """
    parts = re.split(r"[.!?]\s+", text)
    lens = [len(tokens(p)) for p in parts if p.strip()]
    return lens


def count_urls(text: str) -> int:
    """
    Count URL-like patterns in text.
    """
    return len(_URL.findall(text))


def count_non_ascii(text: str) -> int:
    """
    Count non-ASCII characters (rough noise indicator).
    """
    return len(_NON_ASCII.findall(text))


def count_all_caps_runs(text: str) -> int:
    """
    Count long ALL-CAPS tokens (6+ chars).
    """
    return len(re.findall(r"\b[A-Z]{6,}\b", text))


def _ensure_str_list(value: Any) -> List[str]:
    """
    Ensure we always work with a list[str] for fields that might be string or list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for v in value:
            if v is None:
                continue
            out.append(str(v))
        return out
    return [str(value)]
