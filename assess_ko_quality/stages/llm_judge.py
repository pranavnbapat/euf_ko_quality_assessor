# assess_ko_quality/stages/llm_judge.py
"""
LLM-as-judge assessment of a Knowledge Object.

The rubric is NOT written here. It is read from the human review instrument itself -
the question texts in the 2024 reviewer workbook - so that every question the judge
answers has a matching column of human answers to validate against. Change the
workbook, or point --rubric at a different one, and the judge follows.

This is the only way an LLM opinion earns a place in the score. The failure this whole
pipeline was rebuilt to avoid was components asserted into a weighted sum without ever
being checked against human judgement; a larger model does not exempt it. So the judge
emits answers, validation/validate_metrics.py measures them against the human labels,
and only what correlates gets weight.

Dimensions the heuristics cannot reach - whether the audience is right, whether jargon
is explained, whether the KO stands alone, whether sources are credited - are exactly
where a judge should help, because none of them is a surface statistic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.llm import LLMClient, get_client

# Cell values in the workbook's answer rows that mark a question as binary.
_YESNO_HINT = re.compile(r"yes\s*or\s*no", re.I)


def load_rubric(workbook: Path, sheet: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Extract the review questions from the reviewer workbook.

    Reviewer sheets carry the question text in the first row and the expected answer
    type ("Yes or No", or blank for the free numeric recommendation) in the second.
    Questions are discovered by scanning those two rows, so adding a question to the
    workbook adds it to the judge with no code change.
    """
    import pandas as pd

    xls = pd.ExcelFile(workbook)
    sheets = [sheet] if sheet else xls.sheet_names
    for name in sheets:
        df = pd.read_excel(workbook, sheet_name=name)
        if df.shape[0] < 2 or df.shape[1] < 20:
            continue
        questions: List[Dict[str, str]] = []
        for col in df.columns:
            q = df.loc[0, col]
            if not isinstance(q, str) or len(q.strip()) < 25:
                continue
            a = df.loc[1, col] if df.shape[0] > 1 else None
            kind = "yes_no" if isinstance(a, str) and _YESNO_HINT.search(a) else None
            if kind is None and "recommend" in q.lower():
                kind = "scale_1_5"
            if kind is None:
                continue
            questions.append({
                "id": _slug(q),
                "question": " ".join(q.split()),
                "type": kind,
                "column": str(col),
                "sheet": name,
            })
        if len(questions) >= 5:
            return questions
    raise ValueError(f"No reviewer-style question rows found in {workbook}")


def _slug(text: str) -> str:
    words = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return "_".join(words[:6])[:60] or "q"


def _ko_view(ko: Dict[str, Any], max_content_chars: int) -> str:
    """Render the KO the way a reviewer would see it."""
    def fmt(v: Any) -> str:
        if isinstance(v, (list, tuple)):
            return "; ".join(str(x) for x in v if str(x).strip())
        return str(v or "").strip()

    parts = [
        f"TITLE: {fmt(ko.get('title'))}",
        f"SUBTITLE: {fmt(ko.get('subtitle'))}",
        f"DESCRIPTION: {fmt(ko.get('description'))}",
        f"KEYWORDS: {fmt(ko.get('keywords'))}",
        f"LANGUAGES: {fmt(ko.get('languages'))}",
        f"TOPICS: {fmt(ko.get('topics'))}",
        f"LICENCE: {fmt(ko.get('license'))}",
        f"CREATORS: {fmt(ko.get('creators'))}",
        f"PROJECT: {fmt(ko.get('project_name')) or fmt(ko.get('project_acronym'))}",
        "",
        "CONTENT:",
        fmt(ko.get("ko_content_flat"))[:max_content_chars],
    ]
    return "\n".join(parts)


SYSTEM = (
    "You are reviewing entries in EU-FarmBook, a multilingual European platform of "
    "practical agricultural, forestry, environmental and rural knowledge for farmers, "
    "foresters and their advisors. You are applying a fixed review form. "
    "Judge only what the entry actually shows you. Content may be in any European "
    "language; judge it in its own language and never penalise it for not being English. "
    "An entry whose text failed to extract should be judged on what is present, not "
    "assumed to be bad."
)


def build_prompt(ko: Dict[str, Any], rubric: Sequence[Dict[str, str]], max_content_chars: int) -> str:
    lines = [
        "Review this knowledge object against the form below.",
        "",
        "=== KNOWLEDGE OBJECT ===",
        _ko_view(ko, max_content_chars),
        "",
        "=== REVIEW FORM ===",
    ]
    for q in rubric:
        expect = "yes or no" if q["type"] == "yes_no" else "an integer 1-5"
        lines.append(f'- "{q["id"]}" ({expect}): {q["question"]}')
    lines += [
        "",
        "Return a JSON object mapping each id to an object with:",
        '  "answer": "yes"/"no" for yes-or-no questions, or an integer 1-5 for scale questions',
        '  "why": one short sentence of justification grounded in the entry',
        "Answer every id. Use null for answer only if the entry gives you nothing to judge.",
    ]
    return "\n".join(lines)


def judge_one(
    ko: Dict[str, Any],
    rubric: Sequence[Dict[str, str]],
    client: LLMClient,
    max_content_chars: int = 6000,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Run the rubric over one KO. Returns {id: {answer, why}} plus bookkeeping.

    A reply cut off mid-JSON raises rather than returning nothing, so the budget is
    escalated rather than the row being recorded as a failure. This is the failure mode
    USING_LLMS.md warns about: a truncated answer looks like a broken row, and the rows
    that survive are the short, simple ones - which biases everything computed from them.
    """
    prompt = build_prompt(ko, rubric, max_content_chars)
    raw = None
    budget = max_tokens
    for attempt in range(3):
        try:
            raw = client.complete_json(prompt, max_tokens=budget, system=SYSTEM)
            if raw is not None:
                break
        except ValueError:
            pass          # unparseable, usually truncation: try again with more room
        budget *= 2
    out: Dict[str, Any] = {"llm_model": client.model, "llm_ok": raw is not None,
                           "llm_tokens_used": budget}
    if not isinstance(raw, dict):
        out["llm_answers"] = {}
        return out

    answers: Dict[str, Any] = {}
    for q in rubric:
        entry = raw.get(q["id"])
        if isinstance(entry, dict):
            value, why = entry.get("answer"), entry.get("why")
        else:
            value, why = entry, None
        answers[q["id"]] = {"answer": _normalise(value, q["type"]), "why": why}
    out["llm_answers"] = answers
    return out


def _normalise(value: Any, kind: str) -> Optional[float]:
    """Map an answer onto 0-1 so it lines up with the human columns."""
    if value is None:
        return None
    if kind == "yes_no":
        s = str(value).strip().lower()
        if s in {"yes", "y", "true", "1"}:
            return 1.0
        if s in {"no", "n", "false", "0"}:
            return 0.0
        return None
    try:
        v = float(str(value).strip())
    except Exception:
        return None
    return max(0.0, min(1.0, (v - 1.0) / 4.0))


def judge_many(
    kos: Sequence[Dict[str, Any]],
    rubric: Sequence[Dict[str, str]],
    client: Optional[LLMClient] = None,
    model: str = "",
    workers: int = 8,
    max_content_chars: int = 6000,
) -> List[Dict[str, Any]]:
    """
    Judge a batch in parallel. Returns one result per KO, in order.

    `model` is a preference expression resolved against what the provider serves;
    a per-model client is built so each one keeps its own cache and quirk state.
    """
    client = client or get_client(model=model)
    if client is None:
        return [{"llm_ok": False, "llm_answers": {}, "llm_error": "no LLM configured"} for _ in kos]

    def run(ko: Dict[str, Any]) -> Dict[str, Any]:
        return judge_one(ko, rubric, client, max_content_chars=max_content_chars)

    results = client.map_concurrent(list(kos), run, workers=workers, label="judge")
    return [r if r is not None else {"llm_ok": False, "llm_answers": {},
                                     "llm_error": "call failed"} for r in results]


def judge_panel(
    kos: Sequence[Dict[str, Any]],
    rubric: Sequence[Dict[str, str]],
    preference: str = "",
    workers: int = 8,
    max_content_chars: int = 6000,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Judge a batch with one model per family.

    Panels exist to expose shared priors, so this deliberately spreads across families
    rather than checkpoints: two models from the same family agreeing is not evidence.
    """
    from core.llm import get_panel

    out: Dict[str, List[Dict[str, Any]]] = {}
    for client in get_panel(preference):
        out[client.model] = judge_many(kos, rubric, client=client, workers=workers,
                                       max_content_chars=max_content_chars)
    return out


__all__ = ["load_rubric", "judge_one", "judge_many", "build_prompt", "SYSTEM"]
