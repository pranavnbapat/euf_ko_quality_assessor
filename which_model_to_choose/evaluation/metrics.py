from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple


_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9%]+(?:['-][A-Za-z0-9]+)?")
_NUM_RE = re.compile(r"\b\d{1,4}(?:[.,]\d{1,3})*(?:%|[A-Za-z]*)?\b")


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def words(s: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(s or "")]


def top_keywords(text: str, top_n: int = 50, stopset: Optional[set[str]] = None) -> set[str]:
    stopset = stopset or set()
    toks = [t for t in words(text) if t not in stopset and len(t) > 2]
    most = [w for w, _ in Counter(toks).most_common(top_n)]
    return set(most)


def coverage_recall(source: str, summary: str, top_n: int = 50, stopset: Optional[set[str]] = None) -> float:
    kw = top_keywords(source, top_n=top_n, stopset=stopset)
    if not kw:
        return 0.0
    sum_toks = set(words(summary))
    found = sum(1 for k in kw if k in sum_toks)
    return found / len(kw)


def repetition_ratio(summary: str, n: int = 3) -> float:
    toks = words(summary)
    if len(toks) < n:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    reps = sum(c - 1 for c in counts.values() if c > 1)
    return reps / max(1, len(grams))


def length_score(source: str, summary: str) -> float:
    s_len = len(source)
    m_len = len(summary)
    if s_len < 2000:
        tgt_min, tgt_max = 400, 1600
    elif s_len < 10000:
        tgt_min, tgt_max = 800, 3000
    else:
        tgt_min, tgt_max = 1500, 6000
    if m_len <= 0:
        return 0.0
    if m_len < tgt_min:
        return max(0.0, m_len / tgt_min * 0.7)
    if m_len > tgt_max:
        return max(0.0, tgt_max / m_len * 0.7)
    return 1.0


def salient_mismatch_penalty(source: str, summary: str) -> float:
    src = norm_space(source).lower()
    summ = norm_space(summary).lower()
    nums = set(_NUM_RE.findall(summ))
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", summ))
    bad = 0
    total = 0
    for m in nums:
        total += 1
        if m not in src:
            bad += 1
    for y in years:
        total += 1
        if y not in src:
            bad += 1
    if total == 0:
        return 0.0
    return min(1.0, bad / total)


def score_summary(source: str, summary: str) -> Dict[str, float | str]:
    if not summary or not summary.strip():
        return {
            "final_score": 0.0,
            "coverage": 0.0,
            "length_score": 0.0,
            "repetition": 0.0,
            "num_penalty": 0.0,
            "reason": "empty",
        }
    cov = coverage_recall(source, summary, top_n=75)
    ls = length_score(source, summary)
    rep = repetition_ratio(summary, n=3)
    pen = salient_mismatch_penalty(source, summary)
    final = (
        0.45 * cov +
        0.25 * ls +
        0.20 * max(0.0, 1.0 - rep) +
        0.10 * max(0.0, 1.0 - pen)
    )
    reason = "ok"
    if pen > 0.5:
        reason = "numeric_mismatch"
    return {
        "final_score": round(final, 4),
        "coverage": round(cov, 4),
        "length_score": round(ls, 4),
        "repetition": round(rep, 4),
        "num_penalty": round(pen, 4),
        "reason": reason,
    }


def keyword_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = {x.lower() for x in a if x}
    sb = {x.lower() for x in b if x}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_ratio(a: str, b: str) -> float:
    wa = set(words(a))
    wb = set(words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def metadata_field_score(text: str, min_len: int, max_len: int) -> float:
    s = norm_space(text)
    if not s:
        return 0.0
    n = len(s)
    if n < min_len:
        return max(0.0, n / min_len * 0.7)
    if n > max_len:
        return max(0.0, max_len / n * 0.7)
    return 1.0


def score_metadata(record: Dict, summary_text: str, metadata: Dict) -> Dict[str, float | str]:
    title = str(metadata.get("title", "")).strip()
    subtitle = str(metadata.get("subtitle", "")).strip()
    description = str(metadata.get("description", "")).strip()
    keywords = keyword_list(metadata.get("keywords", []))

    title_score = metadata_field_score(title, 8, 90)
    subtitle_score = metadata_field_score(subtitle, 12, 140)
    description_score = metadata_field_score(description, 60, 600)
    keywords_score = 1.0 if 4 <= len(keywords) <= 10 else (len(keywords) / 4 * 0.7 if keywords else 0.0)

    summary_overlap = overlap_ratio(description or title, summary_text)
    orig_title = str(record.get("title") or record.get("title_llm") or "").strip()
    orig_keywords = keyword_list(record.get("keywords_llm") or record.get("keywords") or [])
    title_consistency = overlap_ratio(title, orig_title) if orig_title else 0.5
    kw_consistency = jaccard(keywords, orig_keywords) if orig_keywords else 0.5

    final = (
        0.18 * title_score +
        0.12 * subtitle_score +
        0.24 * description_score +
        0.16 * keywords_score +
        0.20 * summary_overlap +
        0.05 * title_consistency +
        0.05 * kw_consistency
    )
    return {
        "final_score": round(final, 4),
        "title_score": round(title_score, 4),
        "subtitle_score": round(subtitle_score, 4),
        "description_score": round(description_score, 4),
        "keywords_score": round(keywords_score, 4),
        "summary_overlap": round(summary_overlap, 4),
        "title_consistency": round(title_consistency, 4),
        "keywords_consistency": round(kw_consistency, 4),
        "reason": "ok" if final > 0 else "empty",
    }
