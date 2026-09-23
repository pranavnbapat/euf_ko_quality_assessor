# assess_ko_quality/stages/diagnostics.py
"""
Stage 3: diagnostics.

Everything the metrics measure that is NOT in the score ends up here. That is most of
it, and for a reason: measured against the 2024 human review (Spearman, n=91), noise
scored -0.148, clarity -0.137, usefulness -0.048 and metadata consistency -0.009. None
of them predicts whether a reviewer rates a KO well.

That result dictates how they are reported. Calling a KO "noisy" implies noise is bad,
and we have no evidence for that on this corpus - the old noise score was largely a
length proxy (r = -0.55 with content length). So nothing here is phrased as a fault.
A metric is reported when this KO sits in the extreme tail of the CORPUS distribution,
described as unusual rather than wrong, with the percentile attached so the reader can
judge. Thresholds are learned from the corpus by `learn_thresholds`, never typed in.

Two things are exceptions, because they are not opinions about quality:

  - extraction failure, which is an ingestion fault rather than an editorial one
  - compliance failures, which are obligations, evaluated in stage 1

Plain-language explanation comes from the LLM judge's per-question justifications when
a judge has run. That is the right division of labour: the statistics say what is
unusual, the judge says what a reader would notice.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = HERE.parent / "validation" / "diagnostic_thresholds.json"

_CACHE: Dict[str, Any] = {}

# Metrics that describe the record rather than measure it; flagging a KO for being in
# the tail of "how many characters its title has" is noise, not a diagnostic.
_SKIP_SUFFIXES = ("_chars", "_count", "_tokens", "_len", "_words", "_sentences",
                  "_score", "_raw", "_max", "_used")


def _numeric_items(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out[k] = float(v)
    return out


# Two guards against flagging noise. With N metrics and a two-tailed cut at p, every
# KO lands in some tail about N*2p times by arithmetic alone: 85 metrics at p05/p95
# produced 8 "unusual" metrics for every single KO, which is not a diagnostic. The
# tails are therefore narrow, and metrics too coarse for a percentile to mean anything
# (0-5 sub-scores, mostly ties) are excluded rather than reported as outliers.
MIN_DISTINCT_VALUES = 12


def learn_thresholds(
    metric_rows: Iterable[Dict[str, Any]],
    low: float = 0.01,
    high: float = 0.99,
) -> Dict[str, Any]:
    """
    Learn what "unusual" means for this corpus, per metric.

    Everything numeric the metric modules emit is considered, so adding a metric
    upstream makes it flaggable with no change here. Metrics with too few distinct
    values are dropped: a percentile on a 0-5 integer score is not a tail, it is a tie.
    """
    import numpy as np

    rows = list(metric_rows)
    if not rows:
        raise ValueError("learn_thresholds needs at least one row")

    columns: Dict[str, List[float]] = {}
    for row in rows:
        for k, v in _numeric_items(row).items():
            columns.setdefault(k, []).append(v)

    cuts: Dict[str, Dict[str, float]] = {}
    for k, values in columns.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 20 or float(arr.std()) == 0.0:
            continue           # too few or constant: no meaningful tail
        if np.unique(arr).size < MIN_DISTINCT_VALUES:
            continue           # too coarse for a percentile to mean anything
        cuts[k] = {
            "low": float(np.percentile(arr, low * 100)),
            "high": float(np.percentile(arr, high * 100)),
            "median": float(np.median(arr)),
            "n": int(arr.size),
        }
    return {
        "built": datetime.now(timezone.utc).date().isoformat(),
        "rows": len(rows),
        "low_percentile": low,
        "high_percentile": high,
        "metrics_considered": len(columns),
        "metrics_kept": len(cuts),
        # What to expect by chance, so a reader can tell signal from arithmetic.
        "expected_flags_per_ko_by_chance": round(len(cuts) * (low + (1 - high)), 2),
        "cuts": cuts,
    }


def save_thresholds(profile: Dict[str, Any], path: Path = DEFAULT_THRESHOLDS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=1), encoding="utf-8")
    return path


def load_thresholds(path: Path = DEFAULT_THRESHOLDS) -> Optional[Dict[str, Any]]:
    key = str(path)
    if key not in _CACHE:
        try:
            _CACHE[key] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _CACHE[key] = None
    return _CACHE[key]


def diagnose(
    metrics: Dict[str, Any],
    compliance: Optional[Dict[str, Any]] = None,
    judge: Optional[Dict[str, Any]] = None,
    extraction_suspect: bool = False,
    thresholds: Optional[Dict[str, Any]] = None,
    max_unusual: int = 8,
) -> Dict[str, Any]:
    """
    Collect diagnostics for one KO.

    `metrics` is the merged output of the metric modules; anything numeric in it is
    eligible. `judge` is a result from stages.llm_judge, used only for its wording.
    """
    thresholds = thresholds if thresholds is not None else load_thresholds()
    cuts = (thresholds or {}).get("cuts", {})

    notes: List[str] = []
    unusual: List[Dict[str, Any]] = []

    if extraction_suspect:
        notes.append("Almost no text could be extracted from the resource. This is an "
                     "ingestion problem with the source file, not a metadata problem.")

    for name, value in _numeric_items(metrics).items():
        cut = cuts.get(name)
        if not cut or name.endswith(_SKIP_SUFFIXES):
            continue
        if value <= cut["low"]:
            side = "low"
        elif value >= cut["high"]:
            side = "high"
        else:
            continue
        unusual.append({
            "metric": name,
            "value": round(value, 4),
            "side": side,
            "corpus_median": round(cut["median"], 4),
        })

    # Most extreme first, so a truncated list keeps the most informative entries.
    def distance(item: Dict[str, Any]) -> float:
        cut = cuts[item["metric"]]
        span = max(1e-9, abs(cut["high"] - cut["low"]))
        return abs(item["value"] - cut["median"]) / span

    unusual.sort(key=distance, reverse=True)
    unusual = unusual[:max_unusual]

    failed = list((compliance or {}).get("compliance_failed", []))

    # The judge explains; the statistics only say what is unusual.
    if judge and judge.get("llm_ok"):
        for qid, entry in (judge.get("llm_answers") or {}).items():
            if entry.get("answer") == 0.0 and entry.get("why"):
                notes.append(str(entry["why"]))

    # The chance rate travels with the count. Without it a reader cannot tell three
    # flags from three findings, and with 57 metrics at p1/p99 a couple are expected
    # on every KO. It also exposes the case that matters: a batch flagging far above
    # chance means the batch differs from the corpus the thresholds were learned on,
    # not that every KO in it is defective.
    expected = (thresholds or {}).get("expected_flags_per_ko_by_chance")

    return {
        "diagnostics_unusual": unusual,
        "diagnostics_unusual_count": len(unusual),
        "diagnostics_expected_by_chance": expected,
        "diagnostics_compliance_failed": failed,
        "diagnostics_notes": notes,
        "diagnostics_thresholds_used": bool(cuts),
    }


__all__ = [
    "diagnose", "learn_thresholds", "save_thresholds", "load_thresholds",
    "DEFAULT_THRESHOLDS",
]
