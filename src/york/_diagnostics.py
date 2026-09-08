"""Diagnostics: effective sample size and below-LOD sigma-floor checks.

These are library behavior, not documentation notes (SPEC.md §6).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DEFAULT_N_EFF_WARN_FRACTION = 0.5
"""Warn when n_eff / n falls below this."""

DEFAULT_FLOOR_WARN_FRACTION = 0.05
"""Warn when this fraction of sigma values is clamped to ``min_sigma``."""


def lag1_autocorrelation(resid: NDArray[np.float64]) -> float:
    """Lag-1 autocorrelation of a residual series, in the order given."""
    r = resid - resid.mean()
    denom = float(np.sum(r * r))
    if denom == 0.0 or r.size < 2:
        return 0.0
    return float(np.sum(r[1:] * r[:-1]) / denom)


def effective_n(n: int, rho1: float) -> float:
    """Effective sample size for AR(1) errors: ``n (1 - rho) / (1 + rho)``.

    Negative autocorrelation would give more than ``n`` effective samples;
    that is clamped to ``n`` so the diagnostic only ever reduces confidence.
    """
    if not np.isfinite(rho1) or rho1 <= 0.0:
        return float(n)
    rho = min(rho1, 1.0 - 1e-12)
    return float(n * (1.0 - rho) / (1.0 + rho))


def autocorrelation_is_significant(rho1: float, n: int, z: float = 1.96) -> bool:
    """Is ``rho1`` outside the Bartlett white-noise band ``±z / sqrt(n)``?

    A lag-1 estimate from a handful of points is mostly noise; without this
    gate the warning fires on roughly a third of small white-noise samples.
    """
    if n < 2 or not np.isfinite(rho1):
        return False
    return bool(abs(rho1) > z / np.sqrt(n))


def floor_fraction(sigma: NDArray[np.float64], min_sigma: float) -> float:
    """Fraction of sigma values sitting at the ``min_sigma`` floor."""
    if min_sigma <= 0.0 or sigma.size == 0:
        return 0.0
    return float(np.mean(sigma <= min_sigma))
