# scoring.py

from typing import Iterable, List

from nltk.corpus import stopwords

from text_utils import tokens

_STOP_EN = set(stopwords.words("english"))

def score_title(title: str) -> int:
    if not title: return 0
    toks = tokens(title)
    base = 8 if (8 <= len(title) <= 200) else 5
    sw = sum(1 for t in toks if t in _STOP_EN)
    ratio = 0 if not toks else sw/len(toks)
    if ratio > 0.5: base -= 3
    return max(0, min(10, base))

def score_description(desc: str) -> int:
    if not desc: return 0
    toks = tokens(desc)
    score = 7 if len(toks) >= 40 else 0
    d = desc.lower()
    if any(k in d for k in ("objective","aim","goal","purpose")): score += 3
    if any(k in d for k in ("method","approach","methodology")): score += 3
    if any(k in d for k in ("result","output","findings","outcome")): score += 2
    return min(15, score)

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    return 0.0 if not A or not B else len(A & B)/len(A | B)

def score_keyword_alignment(keywords: List[str], text_blobs: List[str]) -> int:
    if not keywords: return 0
    kws = tokens(" ".join(keywords))
    text = tokens(" ".join(text_blobs))
    sim = jaccard(set(kws), set(text))
    if sim >= 0.5: return 10
    if sim >= 0.35: return 8
    if sim >= 0.2: return 6
    if sim >= 0.1: return 3
    return 1

def _unique_ratio(toks: List[str]) -> float:
    return 0.0 if not toks else len(set(toks))/len(toks)

def score_content_depth(content: str) -> int:
    n = len(tokens(content))
    if n >= 1500: return 10
    if n >= 800: return 8
    if n >= 500: return 6
    if n >= 250: return 4
    if n >= 100: return 2
    return 0

def score_diversity(content: str, lang: str) -> int:
    toks = [t for t in tokens(content) if not (lang.startswith("en") and t in _STOP_EN)]
    r = _unique_ratio(toks)
    if r >= 0.35: return 5
    if r >= 0.28: return 4
    if r >= 0.22: return 3
    if r >= 0.18: return 2
    if r >= 0.14: return 1
    return 0

def score_duplication(content: str) -> int:
    # very light proxy, same thresholds
    raw = content.encode("utf-8", errors="ignore")
    counts = {}
    for i in range(0, len(raw)-3, 3):
        tri = raw[i:i+3]; counts[tri] = counts.get(tri, 0) + 1
    if not counts: return 5
    avg = sum(counts.values())/len(counts)
    cp = min(1.0, 0.5 + 0.5*(1.0/avg))
    if cp >= 0.95: return 5
    if cp >= 0.9:  return 4
    if cp >= 0.8:  return 3
    if cp >= 0.7:  return 2
    if cp >= 0.6:  return 1
    return 0

def score_topics_themes(topics: List[str], themes: List[str], text_blobs: List[str]) -> int:
    text = tokens(" ".join(text_blobs))
    tt = tokens(" ".join(topics + themes))
    sim = jaccard(set(tt), set(text))
    if sim >= 0.35: return 6
    if sim >= 0.2:  return 4
    if sim >= 0.1:  return 2
    return 0

def score_project_echo(acronym: str, name: str, text_blobs: List[str]) -> int:
    text = " ".join(text_blobs).lower()
    s = 0
    if acronym and acronym.lower() in text: s += 3
    if name:
        if name.lower() in text: s += 3
        else:
            from rapidfuzz import fuzz
            if fuzz.partial_ratio(name.lower(), text) >= 85: s += 2
    return min(6, s)

# EN-only spell score (unchanged)
try:
    from spellchecker import SpellChecker
    _SPELL = SpellChecker(language="en")
except Exception:
    _SPELL = None

def en_spell_score(title: str, desc: str, content: str) -> tuple[int, float]:
    if _SPELL is None: return 8, 0.0
    sample = " ".join([title, desc, content[:4000]])
    toks = [t for t in tokens(sample) if t.isalpha()]
    if not toks: return 8, 0.0
    miss = _SPELL.unknown(toks)
    ratio = len(miss) / len(toks)
    if ratio <= 0.01: return 8, ratio
    if ratio <= 0.02: return 7, ratio
    if ratio <= 0.03: return 6, ratio
    if ratio <= 0.05: return 4, ratio
    if ratio <= 0.08: return 2, ratio
    return 0, ratio
