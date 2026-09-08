"""Pearson-York benchmark (SPEC.md §7.1). Hard merge gate; never loosen.

Reference values for the ten-point Pearson (1901) data with York's (1966)
weights, verified independently against a direct transcription of the
York et al. (2004) equations and against ``scipy.odr``:

    slope      -0.480533   internal err 0.05799   external err 0.07062
    intercept   5.479910   internal err 0.29497   external err 0.35925
    chi2 11.8664   MSWD 1.4833   dof 8

"External" errors are internal errors scaled by sqrt(MSWD). That is what
``scipy.odr`` reports as ``sd_beta`` and what ``scale_errors=True`` produces.
The default (``scale_errors=False``) reports internal errors.
"""

from __future__ import annotations

import pytest

import york

SLOPE = -0.480533
INTERCEPT = 5.479910
SLOPE_ERR_INTERNAL = 0.05799
INTERCEPT_ERR_INTERNAL = 0.29497
SLOPE_ERR_EXTERNAL = 0.07062
INTERCEPT_ERR_EXTERNAL = 0.35925
CHI2 = 11.8664
MSWD = 1.4833


@pytest.fixture(scope="module")
def result(pearson_york):
    d = pearson_york
    return york.fit(d.x, d.y, sx=d.sx, sy=d.sy)


@pytest.fixture(scope="module")
def result_scaled(pearson_york):
    d = pearson_york
    return york.fit(d.x, d.y, sx=d.sx, sy=d.sy, scale_errors=True)


def test_slope_and_intercept(result):
    assert result.slope == pytest.approx(SLOPE, abs=1e-6)
    assert result.intercept == pytest.approx(INTERCEPT, abs=1e-6)


def test_internal_uncertainties(result):
    assert result.slope_err == pytest.approx(SLOPE_ERR_INTERNAL, abs=1e-5)
    assert result.intercept_err == pytest.approx(INTERCEPT_ERR_INTERNAL, abs=1e-5)


def test_external_uncertainties(result_scaled):
    assert result_scaled.slope_err == pytest.approx(SLOPE_ERR_EXTERNAL, abs=1e-5)
    assert result_scaled.intercept_err == pytest.approx(
        INTERCEPT_ERR_EXTERNAL, abs=1e-5
    )


def test_scaling_does_not_move_the_line(result, result_scaled):
    assert result_scaled.slope == result.slope
    assert result_scaled.intercept == result.intercept
    assert result_scaled.mswd == result.mswd


def test_goodness_of_fit(result):
    assert result.n == 10
    assert result.dof == 8
    assert result.chi2 == pytest.approx(CHI2, abs=1e-4)
    assert result.mswd == pytest.approx(MSWD, abs=1e-4)
    assert result.mswd_is_constrained is False


def test_converges_quickly(result):
    assert result.converged is True
    assert result.n_iter < 20


def test_york_alias_is_fit():
    assert york.york is york.fit
