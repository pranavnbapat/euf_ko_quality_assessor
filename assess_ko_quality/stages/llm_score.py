# assess_ko_quality/stages/llm_score.py
"""
The LLM pillar: an LLM's 0-100 score for one metadata field, read against the summary.

One call per field per version: the KO's summary, that one field and the criteria go in;
one overall score, a one-sentence reason and the LLM's own 0-100 reading of each
criterion (structural, semantic, functional) come out. Only the overall score enters
the field score; the per-criterion numbers are reported next to the measured pillars,
so every pillar has two sources - code and LLM - that can be compared (the agreement
is summarised per run in stages/analysis.py). Fields are never batched into one prompt,
because a long multi-field answer is where the output breaks. An empty field is not
sent: it scores 0 on every pillar, the LLM one included (stages/metadata_score.py).

The score is the fourth pillar of the field score, next to the three measured ones
(structural, semantic, functional), with the weight `pillars.llm` in the spec. Rating
named criteria with an LLM is the approach of G-Eval (Liu et al., EMNLP 2023). Scoring
each version on its own, rather than side by side, removes the position bias Wang et al.
(ACL 2024) found in pairwise comparison. The judge is a model that wrote neither version,
because LLM judges favour their own model's text (Panickssery et al., NeurIPS 2024). The
G-Eval authors also report that LLM evaluators may favour LLM-generated text, which here
means the *_llm version: a known bias, accepted for now.

The prompt names the KO's language - the document's own (file_language, read from the
pages during --resummarise) when known, else the first declared one - names the
summary's language (the document's own with --resummarise, else English), and asks
the judge to rate meaning across languages rather than
the language itself: LLM evaluators are known to be less reliable on non-English text
(Hada et al., Findings of EACL 2024), and the platform is multilingual by design.

    from stages import llm_score
    scores, info = llm_score.score_all(records, ["original", "llm"], spec)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.corpus_index import FIELDS, field_text, field_value, ko_id, summary_text

SYSTEM = (
    "You are an impartial reviewer of metadata for a European knowledge platform on "
    "agriculture, forestry, livestock, food and rural life. You score one metadata field "
    "of one knowledge object against the summary of its file. A longer field is not a "
    "better one."
)

# What the three measured pillars check, in words for the LLM. Only rules with a
# published source carry numbers (METHODS.md 5.3).
LENGTH_RULE = {
    "title": "at most 150 characters, not counting spaces; short and easily understandable",
    "description": "50 to 5,000 characters; an account of the file: topic, purpose, key results, audience",
}
FIELD_NOTE = {
    "title": "one key sentence or phrase naming what the file is",
    "subtitle": "one sentence that adds to the title",
    "description": "an account of the file",
    "keywords": "a list of terms, comma-separated",
}


def _language_hint(rec: Dict[str, Any]) -> str:
    """
    The language named in the prompt: the document's own (`file_language`, read from
    the pages during --resummarise) when known; else the first declared language, as
    an approximation of what the contributor wrote in.
    """
    lang = str((rec.get("_resummarised_from") or {}).get("file_language") or "").strip()
    if lang:
        return lang
    langs = rec.get("languages") or []
    if isinstance(langs, str):
        langs = [langs]
    names = [str(x).strip() for x in langs if str(x).strip()]
    return names[0] if names else ""


def prompt(field: str, text: str, summary: str, language: str = "",
           summary_language: str = "English") -> str:
    rule = LENGTH_RULE.get(field)
    return "\n".join([
        f"=== WHAT THE FILE SAYS (summary, in {summary_language}) ===", summary, "",
        f"=== THE {field.upper()} TO SCORE ({FIELD_NOTE[field]}) ===", text, "",
        f"The {field} may be written in the knowledge object's own language"
        + (f" ({language})" if language else "")
        + f"; the summary is in {summary_language}. Judge meaning across languages; "
          "do not penalise the field for the language it is written in.", "",
        "=== CRITERIA ===",
        "Structural: well-formed for this field"
        + (f" ({rule})" if rule else "")
        + "; not a placeholder, file name or internal project code (like \"D3.1\" or \"Task 4\"); "
          "free of markup, URLs and encoding debris; normal capitalisation"
        + ("; each keyword one concept, no duplicates" if field == "keywords" else "") + ".",
        "Semantic: everything it says is supported by the summary; specific rather than "
        "generic words; clear (not cut off, no unexplained acronyms); covers what the file "
        "is mainly about.",
        "Functional: someone searching for this file's topic, by keyword or by meaning, "
        "would find it with these words.",
        "",
        f"Score the {field} from 0 (useless) to 100 (excellent): each criterion on its "
        "own, then overall.",
        'Return JSON only: {"structural": <integer 0-100>, "semantic": <integer 0-100>, '
        '"functional": <integer 0-100>, "score": <integer 0-100, overall>, '
        '"reason": "<one sentence>"}',
    ])


def _complete(client, text, max_tokens: int) -> Any:
    from core.llm import complete_json_robust

    return complete_json_robust(client, text, max_tokens, SYSTEM)


def _num(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(min(100.0, max(0.0, v)), 1) if v == v else None   # NaN -> None


def parse(raw: Any) -> Dict[str, Any]:
    """
    {"score": 0-100 or None, "reason": str, "criteria": {structural, semantic,
    functional}}. Only "score" enters the field score; the per-criterion numbers are
    the LLM's own reading of the three measured pillars' criteria, reported next to
    the measured values so the two sources can be compared. A value that is not a
    number is None.
    """
    if not isinstance(raw, dict):
        return {"score": None, "reason": "", "criteria": {}}
    return {
        "score": _num(raw.get("score")),
        "reason": str(raw.get("reason") or "")[:300],
        "criteria": {k: _num(raw.get(k)) for k in ("structural", "semantic", "functional")},
    }


def score_all(records: Sequence[Dict[str, Any]], variants: Sequence[str], spec: Dict[str, Any],
              workers: int = 8, log=print
              ) -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, Any]]]], Dict[str, Any]]:
    """
    Score every field that has text, of every requested version, of every KO.

    Returns {ko_id: {variant: {field: {"score", "reason"}}}} and the run's model and call
    count. A KO without a summary gets no call: there is nothing to score against.
    """
    from core.llm import get_client

    cfg = spec["llm_score"]
    client = get_client(model=cfg["model"])
    if client is None:
        raise RuntimeError(f"no LLM available for {cfg['model']!r}; see ../USING_LLMS.md, "
                           "or pass --no-llm-score to leave the LLM pillar out")
    cap = int(cfg["max_summary_chars"])
    jobs: List[Tuple[str, str, str, str]] = []
    for rec in records:
        summary = summary_text(rec)[:cap]
        if not summary:
            continue
        lang = _language_hint(rec)
        slang = str(rec.get("_summary_language") or "English")   # document's own with --resummarise
        for v in variants:
            for f in FIELDS:
                text = field_text(f, field_value(rec, f, v))
                if text:
                    jobs.append((ko_id(rec), v, f, prompt(f, text, summary, lang, slang)))
    log(f"[llm] {len(jobs)} calls to {client.model}: one per field with text, per version, "
        f"for {len(records)} KOs (empty fields score 0 without a call; cached calls are free)")
    raws = client.map_concurrent([j[3] for j in jobs],
                                 lambda p, c=client: _complete(c, p, int(cfg["max_tokens"])),
                                 workers=workers, label=f"llm_score:{client.model}")
    out: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    failed = 0
    for (kid, v, f, _), raw in zip(jobs, raws):
        r = parse(raw)
        failed += r["score"] is None
        out.setdefault(kid, {}).setdefault(v, {})[f] = r
    if failed:
        log(f"[llm] {failed} of {len(jobs)} calls returned no usable score; those fields "
            "are scored on the three measured pillars and marked")
    return out, {"model": client.model, "calls": len(jobs), "failed": failed}


__all__ = ["score_all", "prompt", "parse", "SYSTEM"]
