# assess_ko_quality/stages/metadata_score.py
"""
The metadata score: how well the title, subtitle, description and keywords describe
the file they belong to.

This is the score. The content of the file - how well it is written, how long it is,
whether its text extracted cleanly - is the contributor's doing and is reported
separately; it never raises or lowers this number. What is scored is whether the
metadata tells a reader, and a search engine, what the file actually is.

    KO metadata score (0-100)
      = weighted mean over the four fields; an empty one scores 0   (validation/metadata_spec.json)
        field score
          = weighted mean of four pillars
              structural  = mean(length, completeness, noise, formatting)
              semantic    = mean(consistency, density, clarity, usefulness)
              functional  = mean(bm25_findability, semantic_findability)
              llm         = an LLM's 0-100 score of the field against the summary
                            (stages/llm_score.py, one call per field per version)

Every sub-pillar is defined in core/metadata_metrics.py. Weights default to equal, the
OECD/JRC composite-indicator default; change them in the spec, not here. A sub-pillar
that cannot be evaluated (no text to check against, say) is left out of its pillar's
mean rather than scored as zero, and the output says which ones were left out. The same
holds for the LLM pillar when it was not asked for (--no-llm-score) or its call failed.

What the fields are checked against is the KO's reference text, `reference.source` in
the spec: its summary (ko_content_flat_summarised) by default, or the file's own text.

The same function scores the contributor's metadata ("original") and the LLM version
("llm") on identical criteria, so they can be compared number for number. The LLM
version is written fresh from the summary by stages/generate_metadata.py; with
--use-existing-llm it is the *_llm fields already in the export.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core import metadata_metrics as mm
from core.corpus_index import FIELDS, CorpusIndex, Embedder, field_text, field_value

VARIANTS = ("original", "llm")
PILLARS = (*mm.SUBPILLARS, "llm")      # the three measured pillars, then the LLM's score


def _weighted(values: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    num = sum(weights[k] * v for k, v in values.items() if v is not None)
    den = sum(weights[k] for k, v in values.items() if v is not None)
    return num / den if den else None


def subpillar_calls(field: str, value: Any, others: Dict[str, Any], variant: str,
                    ctx: mm.KOContext, idx: CorpusIndex, emb: Embedder, nli: mm.NLI,
                    spec: Dict[str, Any]) -> Dict[str, Any]:
    """One deferred call per sub-pillar, so a caller can run just the one it needs."""
    return {
        "length": lambda: mm.length(field, value, spec),
        "completeness": lambda: mm.completeness(field, value, others, variant, idx, spec,
                                                ctx.file_name),
        "noise": lambda: mm.noise(field, value),
        "formatting": lambda: mm.formatting(field, value, idx),
        "consistency": lambda: mm.consistency(field, value, ctx, others, emb, nli, idx, spec),
        "density": lambda: mm.density(field, value, idx),
        "clarity": lambda: mm.clarity(field, value, ctx, others, spec),
        "usefulness": lambda: mm.usefulness(field, value, ctx, others, emb, idx, spec),
        "bm25_findability": lambda: mm.bm25_findability(field, value, ctx, idx),
        "semantic_findability": lambda: mm.semantic_findability(field, value, ctx, idx, emb),
    }


def score_field(field: str, value: Any, others: Dict[str, Any], variant: str,
                ctx: mm.KOContext, idx: CorpusIndex, emb: Embedder, nli: mm.NLI,
                spec: Dict[str, Any], llm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """All ten sub-pillars, the four pillars and the field score for one field."""
    calls = subpillar_calls(field, value, others, variant, ctx, idx, emb, nli, spec)
    sub: Dict[str, Optional[float]] = {}
    evidence: Dict[str, Any] = {}
    for name, fn in calls.items():
        v, ev = fn()
        sub[name] = None if v is None else round(float(v), 4)
        evidence[name] = ev

    pillars: Dict[str, Optional[float]] = {}
    for pillar, names in mm.SUBPILLARS.items():
        vals = [sub[n] for n in names if sub[n] is not None]
        pillars[pillar] = round(sum(vals) / len(vals), 4) if vals else None
    llm_score = (llm or {}).get("score")
    pillars["llm"] = None if llm_score is None else round(llm_score / 100.0, 4)

    score = _weighted(pillars, spec["pillars"])
    not_evaluable = sorted(n for n, v in sub.items() if v is None)
    if pillars["llm"] is None:
        not_evaluable.append("llm")
    return {
        "field": field,
        "variant": variant,
        "text": field_text(field, value)[:600],
        "subpillars": sub,
        "pillars": pillars,
        "score": None if score is None else round(100.0 * score, 1),
        "llm": llm,
        "not_evaluable": not_evaluable,
        "evidence": evidence,
    }


def score_variant(rec: Dict[str, Any], variant: str, ctx: mm.KOContext,
                  idx: CorpusIndex, emb: Embedder, nli: mm.NLI,
                  spec: Dict[str, Any],
                  values: Optional[Dict[str, Any]] = None,
                  llm: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """
    Score one set of metadata for one KO. Returns None when the variant has no fields
    at all (the LLM proposal is missing for about 0.7% of KOs).

    `values` overrides the record's own fields, so any candidate metadata - a perturbed
    field, say - is scored by exactly the same code. `llm` is this version's LLM scores,
    field -> {"score", "reason"} (stages/llm_score.py); a field without one is scored on
    the three measured pillars and says so in `not_evaluable`.
    """
    values = values if values is not None else {f: field_value(rec, f, variant) for f in FIELDS}
    if not any(field_text(f, v) for f, v in values.items()):
        return None

    fields: Dict[str, Dict[str, Any]] = {}
    for f in FIELDS:
        cfg = spec["fields"][f]
        present = bool(field_text(f, values[f]))
        if not present and not cfg["required"]:
            continue                       # optional and absent: not scored at all
        if not present:
            # An empty field scores 0 on every pillar, so the KO's pillar means agree with
            # its score; completeness is the one sub-pillar that can say why.
            fields[f] = {"field": f, "variant": variant, "text": "",
                         "subpillars": {"completeness": 0.0},
                         "pillars": {p: 0.0 for p in PILLARS}, "score": 0.0,
                         "llm": {"score": 0.0, "reason": "field is empty (no LLM call)", "criteria": {}},
                         "not_evaluable": [],
                         "evidence": {"missing": "field is empty",
                                      "completeness": {"present": False}}}
            continue
        fields[f] = score_field(f, values[f], values, variant, ctx, idx, emb, nli, spec,
                                llm=(llm or {}).get(f))

    weights = {f: spec["fields"][f]["weight"] for f in fields}
    scores = {f: (r["score"] / 100.0 if r["score"] is not None else None) for f, r in fields.items()}
    total = _weighted(scores, weights)

    pillar_means: Dict[str, Optional[float]] = {}
    for p in PILLARS:
        vals = {f: r["pillars"].get(p) for f, r in fields.items()}
        m = _weighted(vals, weights)
        pillar_means[p] = None if m is None else round(100.0 * m, 1)

    return {
        "variant": variant,
        "score": None if total is None else round(100.0 * total, 1),
        "pillars": pillar_means,
        "fields": fields,
    }


def score_ko(rec: Dict[str, Any], idx: CorpusIndex, emb: Embedder, nli: mm.NLI,
             spec: Dict[str, Any], variants=VARIANTS,
             llm_scores: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """
    Score every requested variant of one KO against the same reference text.
    `llm_scores` is variant -> field -> {"score", "reason"} for this KO, or None.
    """
    from core.corpus_index import ko_id

    ctx = mm.build_context(rec, idx, emb, spec)
    out: Dict[str, Any] = {
        "_orig_id": ko_id(rec),
        "reference_language": ctx.ref_lang,        # of the text the fields are measured against
        "corpus_row_language": ctx.lang,           # of the KO's row in the index (the export's)
        "reference_chars": len(ctx.ref_text),
        "reference_evaluable": ctx.evaluable,
        "notes": ctx.notes,
        "variants": {},
    }
    for v in variants:
        res = score_variant(rec, v, ctx, idx, emb, nli, spec, llm=(llm_scores or {}).get(v))
        if res is not None:
            out["variants"][v] = res
    return out


def recommend(variants: Dict[str, Any], tolerance: float = 0.02,
              llm_name: str = "generated_llm") -> Dict[str, Dict[str, Any]]:
    """
    Per field, the better of the contributor's version and the LLM one, when that
    variant was scored - no LLM call. The better one is the higher field score,
    provided it is not less faithful to the reference text than the contributor's (by
    more than `tolerance`); when the contributor's field is missing, the LLM one is
    taken if present. `llm_name` names the LLM variant in the output: "generated_llm"
    (written by stages/generate_metadata.py) or "existing_llm" (--use-existing-llm,
    the export's *_llm fields).
    """
    from core.corpus_index import FIELDS

    out: Dict[str, Dict[str, Any]] = {}
    for f in FIELDS:
        cand = {}
        for key, name in (("original", "original"), ("llm", llm_name)):
            fr = ((variants.get(key) or {}).get("fields") or {}).get(f)
            if fr and fr.get("score") is not None and fr.get("text"):
                acc = ((fr.get("evidence") or {}).get("consistency") or {}).get("parts", {}).get("accuracy")
                cand[name] = {"score": fr["score"], "accuracy": acc}
        if not cand:
            continue
        bar = (cand.get("original") or {}).get("accuracy")
        ok = {k: v for k, v in cand.items()
              if k == "original" or bar is None or v["accuracy"] is None or v["accuracy"] >= bar - tolerance}
        best = max(ok, key=lambda k: ok[k]["score"])
        out[f] = {"recommended": best, **{f"{k}_score": v["score"] for k, v in cand.items()},
                  **{f"{k}_accuracy": v["accuracy"] for k, v in cand.items()}}
    return out


__all__ = ["score_ko", "score_variant", "score_field", "subpillar_calls", "recommend", "VARIANTS",
           "PILLARS"]
