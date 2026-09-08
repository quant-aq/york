"""x/y symmetry (SPEC.md §7.3).

York's solution is the maximum-likelihood line, so swapping x and y (with
their uncertainties) must give the same line: reciprocal slope, intercept
``-a/b``, identical MSWD, and slope error transformed as ``sigma_b / b**2``.
The error correlation ``r`` is symmetric and carries over unchanged.

This is the single most common bug in third-party York code.
"""

from __future__ import annotations

import numpy as np
import pytest

import york


def _assert_swap_symmetric(x, y, sx, sy, r=0.0):
    f = york.fit(x, y, sx=sx, sy=sy, r=r)
    g = york.fit(y, x, sx=sy, sy=sx, r=r)
    assert f.converged and g.converged
    assert f.slope * g.slope == pytest.approx(1.0, rel=1e-10)
    assert g.intercept == pytest.approx(-f.intercept / f.slope, rel=1e-10)
    assert g.mswd == pytest.approx(f.mswd, rel=1e-10)
    assert g.slope_err == pytest.approx(f.slope_err / f.slope**2, rel=1e-8)


def test_pearson_york_swap(pearson_york):
    d = pearson_york
    _assert_swap_symmetric(d.x, d.y, d.sx, d.sy)


@pytest.mark.parametrize("r", [0.0, 0.4, -0.6])
def test_pearson_york_swap_with_correlated_errors(pearson_york, r):
    d = pearson_york
    _assert_swap_symmetric(d.x, d.y, d.sx, d.sy, r=r)


def test_heteroscedastic_swap():
    rng = np.random.default_rng(99)
    n = 60
    x = np.linspace(0.5, 30.0, n) + rng.normal(0.0, 0.5, n)
    y = 1.2 + 0.9 * x + rng.normal(0.0, 1.0, n)
    sx = rng.uniform(0.2, 0.8, n)
    sy = rng.uniform(0.5, 1.5, n)
    _assert_swap_symmetric(x, y, sx, sy)


def test_swap_with_scalar_sigmas():
    rng = np.random.default_rng(5)
    x = np.linspace(0.0, 10.0, 25)
    y = -3.0 + 2.2 * x + rng.normal(0.0, 0.3, x.size)
    _assert_swap_symmetric(x, y, 0.4, 0.3)
