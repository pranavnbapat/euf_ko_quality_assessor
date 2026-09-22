# assess_ko_quality/stages/scoring.py
"""
Stage 2: the quality score.

This is the only number in the pipeline validated against human judgement. Nothing
enters it that has not been measured against the 2024 review, and the fitted model
lives in validation/score_model.json rather than in this file, so it can be refitted
when more review data arrives without editing code.

WHAT SURVIVED VALIDATION

Of sixteen sub-scores in the previous four-pillar design, eleven had no relationship
with reviewer judgement, one ran backwards, and the rest were dominated by two
measurements. Handing ridge regression all four pillar scores and letting it choose its
own weights produced a cross-validated Spearman of -0.170, below chance, so the failure
was never the 30/35/25/10 weighting - it was the contents.

    old four-pillar total                     rho 0.174   21% of ceiling
    four pillars, optimally reweighted        rho -0.170  below chance
    content depth + lexical variety           rho 0.535   63%
    ... + an LLM judge's overall verdict      rho 0.610   72%

The ceiling is 0.847: two reviewers agree with each other at rho 0.558, and no scorer
can exceed the square root of the criterion's reliability.

THE JUDGE

Added only after it was measured. glm-5.2's overall verdict reaches rho 0.612 against
the human consensus with no fitting at all - higher than two humans agree with each
other (0.558). Four model families were tried; more than one judge makes the score
WORSE (0.638 -> 0.600 CV) because the families correlate 0.61-0.71 with each other and
extra ones cost parameters without adding information. So: one judge, chosen by
measurement, and the score still works without it.

TWO THINGS NOT TO UNDO

  - Do not band the inputs. Discretising content length into 0-5 bins lost 53% of its
    signal, because the old band table penalised documents over 6,000 tokens while
    reviewers consistently preferred longer ones.
  - Do not use type-token ratio. It falls mechanically with length (rho -0.93), so
    93.7% of the corpus scored 5/5 and failed extractions scored best. MTLD is
    length-robust: -0.29 against reviewers, versus +0.47 for MTLD.

CALIBRATION. Fitted on the 91 KOs covered by the 2024 review. That is thin, and the
features were chosen after looking at the same data, so the score is provisional until
it holds on a held-out review batch. Refit with:

    python -m validation.validate_metrics --auto output/<run>.tsv
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import math

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "validation" / "score_model.json"

# Below this many content tokens extraction almost certainly failed. The score is still
# emitted, but flagged: a KO with no extractable text is an ingestion problem, not an
# editorial one, and conflating the two sends contributors chasing the wrong fix.
EXTRACTION_FLOOR_TOKENS = 80

_MODEL_CACHE: Dict[str, Any] = {}


def load_model(path: Path = MODEL_PATH) -> Optional[Dict[str, Any]]:
    key = str(path)
    if key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[key] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _MODEL_CACHE[key] = None
    return _MODEL_CACHE[key]


def _percentile(value: float, reference: Sequence[float]) -> float:
    """
    Where this value falls in the corpus distribution, as 0-1.

    Ties take the MID-rank, not the lower edge. The weights were fitted on features
    ranked with pandas' default, which averages tied ranks, so taking the lower edge
    here would apply the weights to a different quantity than they were fitted to.
    It matters most for the judge, whose answers take only five distinct values and
    are therefore almost entirely ties: a median verdict was scoring as though it sat
    at the bottom of its tie block.
    """
    if not reference:
        return 0.5
    n = len(reference)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if reference[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    first = lo
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if reference[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    last = lo                      # one past the final tied entry
    return ((first + last) / 2.0) / max(1, n - 1)


def score(
    content_tokens: int,
    mtld: float,
    judge_overall: Optional[float] = None,
    model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute the 0-100 quality score.

    Args:
        content_tokens: token count of the extracted content
        mtld: MTLD of the stopword-stripped content (core.text_utils.mtld)
        judge_overall: the judge's overall verdict on 0-1, or None. When absent the
            two-feature model is used, so the pipeline still produces a validated
            score with no LLM configured.
    """
    bundle = model if model is not None else load_model()
    if not bundle:
        return {
            "quality_score": None,
            "quality_score_model": "none",
            "quality_score_is_validated": False,
            "quality_score_error": f"no fitted model at {MODEL_PATH}",
            "quality_content_tokens": content_tokens,
            "quality_mtld": round(mtld, 2),
            "quality_extraction_suspect": content_tokens < EXTRACTION_FLOOR_TOKENS,
        }

    which = "with_judge" if judge_overall is not None else "no_judge"
    spec = bundle["models"].get(which) or bundle["models"]["no_judge"]
    if which == "with_judge" and spec is not bundle["models"].get("with_judge"):
        which = "no_judge"

    raw_values = {
        "log_content_tokens": math.log1p(max(0, content_tokens)),
        "log_mtld": math.log1p(max(0.0, mtld)),
        "judge_overall": judge_overall if judge_overall is not None else 0.0,
    }
    refs = bundle.get("reference_quantiles", {})

    total = float(spec["intercept"])
    contributions: Dict[str, float] = {}
    for name, weight in zip(spec["features"], spec["coef"]):
        pct = _percentile(raw_values[name], refs.get(name, []))
        contributions[name] = round(weight * pct, 4)
        total += weight * pct

    # Rescale to the range this model can actually produce. The ridge fit was never
    # constrained to span 0-1, so the two models covered different intervals:
    # with_judge 0.032-1.051, no_judge 0.102-0.750. Left alone, no_judge could never
    # exceed 75 and a 70 from one model would not mean a 70 from the other. Every
    # feature is a percentile in [0, 1], so the reachable interval is the intercept
    # plus the positive weights (top) and the negative ones (bottom).
    lo = float(spec["intercept"]) + sum(w for w in spec["coef"] if w < 0)
    hi = float(spec["intercept"]) + sum(w for w in spec["coef"] if w > 0)
    normalised = (total - lo) / max(1e-9, hi - lo)

    return {
        "quality_score": round(100.0 * min(1.0, max(0.0, normalised)), 1),
        "quality_score_raw": round(total, 4),
        "quality_score_range": [round(lo, 4), round(hi, 4)],
        "quality_score_model": which,
        "quality_score_is_validated": True,
        "quality_score_cv_spearman": spec.get("cv_spearman"),
        "quality_score_ceiling": bundle.get("ceiling"),
        "quality_score_contributions": contributions,
        "quality_content_tokens": content_tokens,
        "quality_mtld": round(mtld, 2),
        "quality_judge_overall": judge_overall,
        "quality_extraction_suspect": content_tokens < EXTRACTION_FLOOR_TOKENS,
    }


def retrieval_readiness(functional: Dict[str, Any]) -> Dict[str, Any]:
    """
    Report retrieval readiness WITHOUT claiming it is validated.

    The BM25, embedding and RAG proxies have never been checked against retrieval
    outcomes, and the one comparison available says their length assumptions are
    suspect: the index is chunk-per-doc, so a long document is not penalised by the
    retriever the way these proxies assume. Validating this means joining against the
    ClickHouse search logs - whether a KO is ever returned, and at what rank - which is
    a separate piece of work.
    """
    return {
        "retrieval_readiness": functional.get("Functional_Score_0_25"),
        "retrieval_readiness_is_validated": False,
        "retrieval_readiness_note": "proxy only; not checked against retrieval outcomes",
    }


__all__ = ["score", "retrieval_readiness", "load_model", "MODEL_PATH",
           "EXTRACTION_FLOOR_TOKENS"]
