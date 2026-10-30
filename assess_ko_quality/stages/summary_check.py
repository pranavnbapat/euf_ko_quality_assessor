# assess_ko_quality/stages/summary_check.py
"""
Stage 3b - vet the reference summary against the file (reported, never scored).

The metadata score measures every field against the KO's summary, so a bad summary
silently poisons every field score of that KO. This stage measures the summary itself
against the file text and reports the result next to the scores, as a reliability
reading on the reference - it never enters the score, for the same reason the content
report does not: the reference text is the pipeline's instrument, not its subject.

Per KO, against the file text (the vision reading followed by the extracted text,
exactly what `reference_text(..., source="file")` reads):

  compression   summary words / file words. No published universal ratio exists, so no
                threshold is imposed per KO; the run's own distribution flags the
                outliers (stages/analysis.reference_flags, corpus-learned percentiles).
  coverage      the share of the file's key terms (the same TF-IDF machinery the
                usefulness sub-pillar uses, idx.key_terms) that appear in the summary.
  support       NLI faithfulness, the design of summary_evaluation/03: sampled summary
                sentences as hypotheses, their best-matching file chunks as premises,
                mean P(entailment). Chunking and premise counts reuse the spec's
                `reference` settings.
  language      the platform's summaries are English whatever the file's language;
                --resummarise writes them in the document's own language. A summary
                detected as any language other than the one it should be in is
                flagged (`summary_language_mismatch`).

Output: {"summary_present", "summary_words", "file_words", "compression_ratio",
"summary_language", "key_term_coverage", "nli_support", "flags", "reference_reliability"}.
Corpus-relative flags (compression outliers, weak support/coverage) are attached after
the run by stages/analysis.reference_flags, because a percentile needs the whole run.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from core.corpus_index import detect_languages, reference_text, summary_text, word_tokens

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str, min_chars: int = 25, cap: int = 40) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) >= min_chars]
    return sents[:cap]


def _sample_evenly(items: List[str], k: int) -> List[str]:
    """Deterministic spread over the whole summary, not just its opening."""
    if len(items) <= k:
        return items
    pos = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(pos.tolist()))]


def _chunks(text: str, chunk_words: int, max_chunks: int) -> List[str]:
    words = text.split()
    out = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
    return out[:max_chunks]


def _language_code(name: Any) -> Optional[str]:
    """ISO code of a language named in English ("Lithuanian" -> "lt"), None if unknown."""
    from core.corpus_index import language_code

    return language_code(name)


def check(rec: Dict[str, Any], idx, emb, nli, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Measure the KO's summary against its file text. Never part of the score."""
    ref_cfg = spec["reference"]
    cfg = spec.get("summary_check", {})
    summary = summary_text(rec)
    file_text = reference_text(rec, int(ref_cfg["max_chars"]), "file")

    out: Dict[str, Any] = {
        "summary_present": bool(summary),
        "summary_words": len(summary.split()),
        "file_words": len(file_text.split()),
        "compression_ratio": None,
        "summary_language": None,
        "key_term_coverage": None,
        "nli_support": None,
        "flags": [],
    }
    if not summary:
        out["flags"].append("no_summary")
        out["reference_reliability"] = "unreliable"
        return out

    if out["file_words"]:
        out["compression_ratio"] = round(out["summary_words"] / out["file_words"], 4)

    out["summary_language"] = detect_languages([summary], workers=1)[0]
    expected = _language_code(rec.get("_summary_language")) or "en"
    out["summary_language_expected"] = expected
    if out["summary_language"] not in (expected, "unknown"):
        out["flags"].append("summary_language_mismatch")

    # Coverage: the file's most characteristic words, carried literally. The semantic
    # match the usefulness sub-pillar adds is left out here on purpose: the summary is
    # long prose, where a literal miss of a key term is already informative and cheap.
    file_tokens = word_tokens(file_text)
    if file_tokens:
        key = idx.key_terms(file_tokens, int(spec["coverage"]["key_terms"]))
        if key:
            present = set(word_tokens(summary))
            out["key_term_coverage"] = round(sum(t in present for t in key) / len(key), 4)

    # Support: sampled summary sentences, each read against its best file chunks.
    sents = _sample_evenly(_sentences(summary), int(cfg.get("sentences", 6)))
    chunks = _chunks(file_text, int(ref_cfg["chunk_words"]), int(ref_cfg["max_chunks"]))
    if sents and chunks:
        n_prem = int(ref_cfg["premises_per_hypothesis"])
        c_vecs = emb.passages(chunks)
        s_vecs = emb.queries(sents)
        sims = s_vecs @ c_vecs.T                                   # sentences x chunks
        pairs, spans = [], []
        for i in range(len(sents)):
            top = np.argsort(-sims[i])[:n_prem]
            spans.append(len(top))
            pairs.extend((chunks[j], sents[i]) for j in top)
        probs = nli.probs(pairs)[:, 2]                             # P(entailment)
        best, k = [], 0
        for n in spans:
            best.append(float(probs[k:k + n].max()) if n else 0.0)
            k += n
        out["nli_support"] = round(float(np.mean(best)), 4)

    out["reference_reliability"] = "check" if out["flags"] else "ok"
    return out


__all__ = ["check"]
