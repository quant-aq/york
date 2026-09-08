"""Distribution functions needed by :class:`york.YorkFit`.

INVARIANT: numpy and the standard library only. These are hand-rolled so that
p-values and confidence intervals never pull scipy into the runtime
dependency set. Each function is cross-checked against ``scipy.stats`` in the
dev test suite.

Algorithms follow Numerical Recipes (Press et al.), 3rd ed., §6.2 and §6.4:
series and Lentz continued fractions for the incomplete gamma and beta
functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-15
_FPMIN = 1e-300
_MAX_ITER = 500


def _vectorize(
    fn: Callable[[float], float], x: NDArray[np.float64] | float
) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    out = np.vectorize(fn, otypes=[np.float64])(arr)
    return np.asarray(out, dtype=np.float64)


# --- incomplete gamma -------------------------------------------------------


def _gamma_series(a: float, x: float) -> float:
    """P(a, x) by series; good for x < a + 1."""
    term = 1.0 / a
    total = term
    ap = a
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    """Q(a, x) by Lentz continued fraction; good for x >= a + 1."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _gammainc_p(a: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        return _gamma_series(a, x)
    return 1.0 - _gamma_cf(a, x)


def _gammainc_q(a: float, x: float) -> float:
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_cf(a, x)


def regularized_lower_gamma(
    a: float, x: NDArray[np.float64] | float
) -> NDArray[np.float64]:
    """P(a, x): regularized lower incomplete gamma function."""
    return _vectorize(lambda v: _gammainc_p(a, v), x)


# --- incomplete beta --------------------------------------------------------


def _beta_cf(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def regularized_incomplete_beta(
    a: float, b: float, x: NDArray[np.float64] | float
) -> NDArray[np.float64]:
    """I_x(a, b): regularized incomplete beta function."""
    return _vectorize(lambda v: _betainc(a, b, v), x)


# --- distributions ----------------------------------------------------------


def chi2_sf(x: NDArray[np.float64] | float, dof: int) -> NDArray[np.float64]:
    """Survival function (upper tail) of the chi-square distribution."""
    half = 0.5 * dof
    return _vectorize(lambda v: _gammainc_q(half, 0.5 * v), x)


def _t_two_sided_scalar(t: float, dof: int) -> float:
    if math.isinf(t):
        return 0.0
    x = dof / (dof + t * t)
    return _betainc(0.5 * dof, 0.5, x)


def t_sf_two_sided(t: NDArray[np.float64] | float, dof: int) -> NDArray[np.float64]:
    """Two-sided p-value for a t statistic: P(|T| >= |t|)."""
    return _vectorize(lambda v: _t_two_sided_scalar(v, dof), t)


def _t_cdf_scalar(t: float, dof: int) -> float:
    p_two = _t_two_sided_scalar(t, dof)
    return 1.0 - 0.5 * p_two if t >= 0.0 else 0.5 * p_two


def t_cdf(t: NDArray[np.float64] | float, dof: int) -> NDArray[np.float64]:
    """CDF of Student's t distribution."""
    return _vectorize(lambda v: _t_cdf_scalar(v, dof), t)


def t_ppf(q: float, dof: int) -> float:
    """Quantile of Student's t distribution (inverse CDF), by bisection."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    if q == 0.5:
        return 0.0
    # Bracket, then bisect. The t distribution has heavy tails, so grow the
    # bracket geometrically until it contains the quantile.
    lo, hi = -1.0, 1.0
    while _t_cdf_scalar(lo, dof) > q:
        lo *= 2.0
    while _t_cdf_scalar(hi, dof) < q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf_scalar(mid, dof) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)
