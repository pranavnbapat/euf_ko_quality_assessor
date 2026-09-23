# assess_ko_quality/stages/analysis.py
"""
What a score means for one KO, and how much the scores depend on our choices.

explain()      the weakest sub-pillars of each field, with the evidence that produced
               them, so a low score comes with a reason a contributor can act on.

robustness()   the OECD/JRC Handbook's robustness step (2008; Saisana, Saltelli &
               Tarantola 2005): recompute the composite with other defensible weights
               and report how far each KO's rank moves. Three data-driven schemes:
                 entropy  - weight by how much an indicator varies (Shannon entropy)
                 CRITIC   - contrast and conflict: std x sum(1 - correlation)
                            (Diakoulaki, Mavrotas & Papayannakis 1995)
                 PCA      - squared loadings of the first principal component
                            (the handbook's factor-analysis weighting)
               If rankings barely move, the equal weights are not doing the work; if
               they move a lot, the weights matter and should be decided deliberately.

redundancy()   pairs of sub-pillars correlated above 0.9, the handbook's multivariate
               check: two indicators that move together double-count one property.

length_bias()  correlation of each sub-pillar with the field's length. A metric that
               mostly tracks length measures length (the reason type-token ratio was
               dropped), so it is reported next to every run.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.metadata_metrics import SUBPILLARS

SUBS = [s for names in SUBPILLARS.values() for s in names]


# --------------------------------------------------------------------- explanation
def _evidence_note(sub: str, ev: Dict[str, Any]) -> str:
    """A short, plain reading of the evidence behind one sub-pillar."""
    if not isinstance(ev, dict):
        return ""
    if sub in ("bm25_findability", "semantic_findability") and "rank" in ev:
        return f"ranks {ev['rank']:.0f} of {ev['of']} when used as a search query"
    if sub == "length":
        if "items" in ev:
            return f"{ev['items']} terms, limits {ev.get('range')}"
        size = f"{ev.get('chars')} characters"
        if ev.get("chars_no_spaces") is not None:
            size += f" ({ev['chars_no_spaces']} without spaces)"
        return f"{size}; published limits {ev.get('limits')}"
    if sub == "completeness":
        failed = [k for k, v in (ev.get("checks") or {}).items() if not v]
        return "fails: " + ", ".join(failed) if failed else ("missing" if not ev.get("present", True) else "")
    if sub == "formatting":
        failed = [k for k, v in (ev.get("checks") or {}).items() if not v]
        return "fails: " + ", ".join(failed) if failed else ""
    if sub == "noise":
        return f"noise tokens {ev.get('noise_tokens')}" if ev.get("noise_tokens") else ""
    if sub == "consistency":
        acc = (ev.get("accuracy") or {})
        bits = []
        if acc.get("unsupported"):
            bits.append("not supported by the file: " + ", ".join(acc["unsupported"][:5]))
        if acc.get("weakest"):
            bits.append(f"weakest claim: \"{acc['weakest'][:90]}\"")
        coh = ev.get("coherence") or {}
        low = [k for k, v in coh.items() if isinstance(v, (int, float)) and v < 0.5]
        if low:
            bits.append("no more related to " + "/".join(low) + " than chance")
        return "; ".join(bits)
    if sub == "density":
        return f"mean word specificity {ev.get('mean_specificity')} (corpus p5-p95 {ev.get('corpus_p5_p95')})"
    if sub == "clarity":
        bits = []
        if ev.get("unexplained_acronyms"):
            bits.append("unexplained acronyms: " + ", ".join(ev["unexplained_acronyms"][:5]))
        if (ev.get("parts") or {}).get("not_truncated") == 0.0:
            bits.append("looks cut off")
        if ev.get("lix") is not None:
            bits.append(f"LIX {ev['lix']}")
        return "; ".join(bits)
    if sub == "usefulness":
        bits = []
        if ev.get("missed"):
            bits.append("misses the file's key terms: " + ", ".join(ev["missed"][:5]))
        if ev.get("overlap_with_title") is not None and ev["overlap_with_title"] > 0.5:
            bits.append(f"repeats the title ({ev['overlap_with_title']:.0%} word overlap)")
        div = ev.get("diversity") or {}
        if div.get("dup_token_ratio"):
            bits.append(f"{div['dup_token_ratio']:.0%} repeated tokens across keywords")
        return "; ".join(bits)
    return ""


def explain(variant_result: Optional[Dict[str, Any]], threshold: float = 0.5,
            limit: int = 6) -> List[Dict[str, Any]]:
    """The weakest sub-pillars across a KO's fields, weakest first."""
    if not variant_result:
        return []
    items = []
    for f, fr in variant_result["fields"].items():
        for sub, v in (fr.get("subpillars") or {}).items():
            if v is not None and v < threshold:
                items.append({"field": f, "subpillar": sub, "value": v,
                              "why": _evidence_note(sub, (fr.get("evidence") or {}).get(sub))})
    items.sort(key=lambda x: x["value"])
    return items[:limit]


# --------------------------------------------------------------------- weights
def _entropy_weights(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    P = X / np.clip(X.sum(axis=0, keepdims=True), 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=0) / math.log(n)
    d = 1.0 - e
    return d / d.sum() if d.sum() > 0 else np.full(X.shape[1], 1.0 / X.shape[1])


def _critic_weights(X: np.ndarray) -> np.ndarray:
    std = X.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):   # a constant column has no correlation
        R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    c = std * (1.0 - R).sum(axis=0)
    return c / c.sum() if c.sum() > 0 else np.full(X.shape[1], 1.0 / X.shape[1])


def _pca_weights(X: np.ndarray) -> np.ndarray:
    Z = (X - X.mean(axis=0)) / np.clip(X.std(axis=0, ddof=1), 1e-12, None)
    vals, vecs = np.linalg.eigh(np.cov(Z, rowvar=False))
    load = vecs[:, np.argmax(vals)] ** 2
    return load / load.sum()


def _ranks(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(-x, method="average")


def robustness(matrix: Sequence[Sequence[float]], names: Sequence[str],
               nominal: Sequence[float]) -> Dict[str, Any]:
    """
    Rank stability of the composite under alternative weights.

    `matrix` is KOs x indicators in [0, 1]; `nominal` the weights actually used.
    """
    from scipy.stats import spearmanr

    X = np.asarray(matrix, dtype=float)
    keep = ~np.isnan(X).any(axis=1)
    X = X[keep]
    if X.shape[0] < 10:
        return {"note": f"only {X.shape[0]} complete KOs; at least 10 needed"}
    nominal = np.asarray(nominal, dtype=float) / np.sum(nominal)
    base = X @ nominal
    base_r = _ranks(base)
    out: Dict[str, Any] = {"kos": int(X.shape[0]), "indicators": list(names),
                           "weights": {"nominal": dict(zip(names, np.round(nominal, 3).tolist()))},
                           "rank_change": {}}
    for label, fn in [("entropy", _entropy_weights), ("critic", _critic_weights), ("pca", _pca_weights)]:
        w = fn(X)
        alt_r = _ranks(X @ w)
        shift = np.abs(alt_r - base_r)
        out["weights"][label] = dict(zip(names, np.round(w, 3).tolist()))
        out["rank_change"][label] = {
            "spearman_vs_nominal": round(float(spearmanr(base, X @ w).statistic), 3),
            "mean_rank_shift": round(float(shift.mean()), 2),
            "max_rank_shift": int(shift.max()),
            "share_moving_over_10pct": round(float((shift > 0.1 * len(shift)).mean()), 3),
        }
    return out


def _spearman_matrix(X: np.ndarray) -> np.ndarray:
    """
    Spearman correlations between columns, NaN for a constant column.

    Computed directly (ranks, then Pearson) because scipy collapses the whole result to
    a single NaN when any column is constant - and on a clean sample some sub-pillars
    are (noise is 1.0 for every KO when no field carries debris).
    """
    from scipy.stats import rankdata

    R = np.apply_along_axis(rankdata, 0, X)
    ok = R.std(axis=0) > 0
    rho = np.full((X.shape[1], X.shape[1]), np.nan)
    if ok.sum() >= 2:
        rho[np.ix_(ok, ok)] = np.corrcoef(R[:, ok], rowvar=False)
    return rho


def redundancy(matrix: Sequence[Sequence[float]], names: Sequence[str],
               cut: float = 0.9) -> Dict[str, Any]:
    X = np.asarray(matrix, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    if X.shape[0] < 10:
        return {"note": f"only {X.shape[0]} complete KOs; at least 10 needed"}
    rho = _spearman_matrix(X)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if not np.isnan(rho[i, j]) and abs(rho[i, j]) >= cut:
                pairs.append({"a": names[i], "b": names[j], "spearman": round(float(rho[i, j]), 3)})
    constant = [n for n, c in zip(names, X.std(axis=0) == 0) if c]
    return {"pairs": pairs, "constant_in_this_sample": constant}


def length_bias(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Spearman correlation of each sub-pillar with the field's length in words."""
    from scipy.stats import spearmanr

    out: Dict[str, Optional[float]] = {}
    for sub in SUBS:
        if sub == "length":
            continue
        xs, ys = [], []
        for r in rows:
            v = r["subpillars"].get(sub)
            if v is not None and r.get("words") is not None:
                xs.append(r["words"]); ys.append(v)
        out[sub] = round(float(spearmanr(xs, ys).statistic), 3) if len(xs) >= 10 and len(set(ys)) > 1 else None
    return out


__all__ = ["explain", "robustness", "redundancy", "length_bias", "SUBS"]
