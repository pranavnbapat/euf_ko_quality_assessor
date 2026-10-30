# assess_ko_quality/stages/generate_metadata.py
"""
Write the LLM version of the metadata fresh, from the reference summary.

Until version 11 the "llm" variant was the `*_llm` fields already in the export,
written by the platform's enrichment pipeline (data-prep-opensearch) with the model of
their day. Now the pipeline writes that variant itself: one call per KO, the current
model, the reference summary and the published field rules in, all four fields out as
JSON. The comparison is then "the contributor vs what a current LLM writes from the
summary", not "vs what an older model wrote years ago". --use-existing-llm restores the
old behaviour (and the zero-LLM-call path).

Three choices, recorded in validation/metadata_spec.json (_provenance):

- The generation is blind: the model sees the summary and the rules only, never the
  contributor's fields, so the LLM version is an independent rewrite rather than an
  edit of the contributor's wording.
- Each field is written in the document's own language: `file_language`, read from
  the pages during --resummarise, else the detected language of the stored text. The
  contributor's declared languages are input under test, not an instrument, so they
  are not used. The summary may be in another language (English for the export's
  summaries); the judge is told which and asked not to penalise it (stages/llm_score.py).
- The generator must be a different model from the judge (llm_score.model), because
  LLM judges favour their own model's text (Panickssery et al., NeurIPS 2024). The
  pipeline warns when they match.

The field rules in the prompt are the ones the measured pillars score (the published
length limits and the placeholder/noise/duplication rules), so the generator is asked
for exactly what the scorer measures.

With --resummarise this stage runs after the summary is regenerated, so the fields are
written from, and scored against, the same summary. A KO whose summary is empty, or
whose call fails, keeps the export's `*_llm` fields and is marked. Calls are counted
before any is made and disk-cached (core/llm.py), so a rerun is free.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.corpus_index import FIELDS, ko_id, summary_text

SYSTEM = (
    "You write metadata for entries on a multilingual European platform of practical "
    "knowledge on agriculture, forestry, livestock, food and rural life. Base everything "
    "on the file summary you are given; never add facts that are not in it."
)

def prompt(summary: str, language: str, summary_language: str = "English") -> str:
    """Plain concatenation: the JSON example carries literal braces."""
    same = summary_language.strip().lower() == language.strip().lower()
    return (
        f"=== WHAT THE FILE SAYS (summary, in {summary_language}) ===\n" + summary + "\n\n"
        "=== RULES ===\n"
        "- title: short and easily understandable, one key sentence, at most 150 "
        'characters not counting spaces; not an internal project code (like "D3.1" or '
        '"Task 4") and not a file name\n'
        "- subtitle: one sentence that adds to the title without repeating it\n"
        "- description: an account of the file - its topic, purpose, key results and "
        "who it is useful for (farmers, advisors); 50-5000 characters; explain "
        "technical terms and avoid unexplained abbreviations\n"
        "- keywords: at most 10 words or short phrases, each one concept, specific to "
        "this file, without duplicates\n"
        "- say nothing the summary does not support\n"
        f"- write every field in {language}, the language of the file itself"
        + ("" if same else f", even though the summary is in {summary_language}") + "\n\n"
        f"Write a title, subtitle, description and keywords for this file, in {language}.\n"
        'Return JSON only: {"title": "...", "subtitle": "...", "description": "...", '
        '"keywords": ["...", "..."]}'
    )


def target_language(rec: Dict[str, Any]) -> str:
    """
    The language the fields are written in: the document's own. `file_language` (read
    from the pages during --resummarise) first; else the detected language of the
    stored text (set by the pipeline as `_detected_lang_code`); English when neither
    is known. Never the contributor's declared languages.
    """
    lang = str((rec.get("_resummarised_from") or {}).get("file_language") or "").strip()
    if lang:
        return lang
    code = str(rec.get("_detected_lang_code") or "").strip()
    if code and code not in ("unknown", "un"):
        from core.corpus_index import language_name

        return language_name(code)
    return "English"


def _complete(client, text, max_tokens: int) -> Any:
    from core.llm import complete_json_robust

    return complete_json_robust(client, text, max_tokens, SYSTEM)


def parse(raw: Any) -> Optional[Dict[str, Any]]:
    """The four fields, or None when the answer carries none of them."""
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Any] = {}
    for f in FIELDS:
        v = raw.get(f)
        if f == "keywords" and isinstance(v, str):
            v = [x.strip() for x in v.replace(";", ",").split(",") if x.strip()]
        out[f] = v if v else None
    return out if any(out.values()) else None


def generate(records: Sequence[Dict[str, Any]], spec: Dict[str, Any],
             workers: int = 8, log=print
             ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Fresh metadata per record with a summary, each in its document's language.
    Returns {ko_id: {field: value}}, the run's model/call/failure counts, and
    {ko_id: language used}. A KO without a summary gets no call and keeps the
    export's `*_llm` fields; so does a KO whose call fails.
    """
    from core.llm import get_client

    cfg = spec.get("generate_metadata", {})
    client = get_client(model=cfg.get("model", ""))
    if client is None:
        raise RuntimeError("no LLM available to generate the *_llm fields; see "
                           "../USING_LLMS.md, or pass --use-existing-llm to score the "
                           "export's fields instead")

    cap = int(cfg.get("max_summary_chars", 24000))
    jobs: List[Tuple[str, str, str]] = []            # (ko_id, prompt, language)
    skipped = 0
    for rec in records:
        summary = summary_text(rec)[:cap]
        if summary:
            lang = target_language(rec)
            # The summary is in the document's language with --resummarise, else English.
            jobs.append((ko_id(rec), prompt(summary, lang, rec.get("_summary_language") or "English"),
                         lang))
        else:
            skipped += 1
    log(f"[generate] {len(jobs)} calls to {client.model}: one per KO with a summary, "
        "all four fields from the summary alone, in the document's language"
        + (f" ({skipped} without a summary keep the export's *_llm fields)" if skipped else "")
        + "; cached calls are free")

    raws = client.map_concurrent(
        [j[1] for j in jobs],
        lambda p, c=client: _complete(c, p, int(cfg.get("max_tokens", 2000))),
        workers=workers, label=f"generate:{client.model}")

    out: Dict[str, Dict[str, Any]] = {}
    langs: Dict[str, str] = {}
    failed = 0
    for (kid, _, lang), raw in zip(jobs, raws):
        values = parse(raw)
        if values:
            out[kid] = values
            langs[kid] = lang
        else:
            failed += 1
    if failed:
        log(f"[generate] {failed} of {len(jobs)} calls returned no usable fields; "
            "those KOs keep the export's *_llm fields and are marked")
    return out, {"model": client.model, "calls": len(jobs), "failed": failed,
                 "skipped_no_summary": skipped}, langs


__all__ = ["generate", "parse", "prompt", "target_language", "SYSTEM"]
