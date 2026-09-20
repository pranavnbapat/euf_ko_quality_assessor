"""Bootstrap confidence intervals and paired significance tests.

Retrieval runs are compared on the *same* queries, so the comparison is paired
and the right test is a paired bootstrap over queries. Without this, a report
can only say "0.61 is bigger than 0.58", which on a few hundred queries may be
noise.
"""

from __future__ import annotations

import numpy as np


def mean_ci(
    per_query: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float, float]:
    """Mean plus a percentile bootstrap confidence interval."""
    values = np.asarray(per_query, dtype=float)
    if values.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(values.mean())
    if values.size < 5:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 13,
) -> dict[str, float]:
    """Two-sided paired bootstrap on the mean difference a - b.

    Returns the observed difference, its confidence interval, and a p-value for
    the null hypothesis that the two systems are equally good.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        raise ValueError("paired test needs equal-length score vectors")
    if a.size == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0, "n": 0}

    diff = a - b
    observed = float(diff.mean())
    if a.size < 5:
        return {"diff": observed, "lo": observed, "hi": observed, "p": 1.0, "n": int(a.size)}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))

    # Shift to the null (mean zero) and ask how often the null reaches the
    # observed effect in either direction.
    centred = boot - boot.mean()
    p = float((np.abs(centred) >= abs(observed)).mean())
    p = max(p, 1.0 / n_boot)
    return {"diff": observed, "lo": lo, "hi": hi, "p": p, "n": int(a.size)}


def significance_label(p: float) -> str:
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return "p<0.01"
    if p < 0.05:
        return "p<0.05"
    return f"not significant (p={p:.2f})"


def describe_comparison(name_a: str, name_b: str, result: dict[str, float]) -> str:
    """One line a non-statistician can read."""
    diff, p = result["diff"], result["p"]
    if p >= 0.05:
        return (
            f"{name_a} vs {name_b}: difference {diff:+.4f} "
            f"[{result['lo']:+.4f}, {result['hi']:+.4f}] - not statistically significant"
        )
    winner, loser = (name_a, name_b) if diff > 0 else (name_b, name_a)
    return (
        f"{winner} beats {loser} by {abs(diff):.4f} "
        f"[{min(abs(result['lo']), abs(result['hi'])):.4f}, {max(abs(result['lo']), abs(result['hi'])):.4f}] "
        f"- {significance_label(p)}"
    )
