"""Statistical helpers for experiment analysis."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(data, confidence: float = 0.95):
    """Compute mean and confidence interval.

    Returns (mean, ci_low, ci_high) using t-distribution for small n.
    """
    a = np.asarray(data, dtype=np.float64)
    n = len(a)
    if n < 2:
        m = float(a.mean())
        return m, m, m
    m = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n))
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    return m, m - t_val * se, m + t_val * se


def format_mean_std(data) -> str:
    """Format as 'mean +/- std' string."""
    a = np.asarray(data, dtype=np.float64)
    return f"{a.mean():.2f} +/- {a.std():.2f}"


def bootstrap_ci(data, statistic=np.mean, n_boot: int = 10000, confidence: float = 0.95):
    """Bootstrap confidence interval for a statistic.

    Returns (statistic_value, ci_low, ci_high).
    """
    a = np.asarray(data, dtype=np.float64)
    n = len(a)
    rng = np.random.default_rng(42)
    boot_stats = np.array([statistic(rng.choice(a, size=n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - confidence) / 2
    return (
        float(statistic(a)),
        float(np.percentile(boot_stats, 100 * alpha)),
        float(np.percentile(boot_stats, 100 * (1 - alpha))),
    )
