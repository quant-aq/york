"""Diagnostics (SPEC.md §6) and the default sigma policy (SPEC.md §5)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import york
from york import _diagnostics


def _ar1(rng, n, rho, sigma=1.0):
    e = np.empty(n)
    e[0] = rng.normal(0.0, sigma)
    innov = sigma * np.sqrt(1.0 - rho**2)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0.0, innov)
    return e


# --- helpers ----------------------------------------------------------------


def test_lag1_autocorrelation_recovers_ar1_coefficient():
    rng = np.random.default_rng(0)
    e = _ar1(rng, 5000, 0.7)
    assert _diagnostics.lag1_autocorrelation(e) == pytest.approx(0.7, abs=0.05)


def test_lag1_autocorrelation_of_white_noise_is_near_zero():
    rng = np.random.default_rng(1)
    e = rng.normal(0.0, 1.0, 5000)
    assert abs(_diagnostics.lag1_autocorrelation(e)) < 0.05


def test_lag1_autocorrelation_degenerate_inputs():
    assert _diagnostics.lag1_autocorrelation(np.array([2.0, 2.0, 2.0])) == 0.0
    assert _diagnostics.lag1_autocorrelation(np.array([1.0])) == 0.0


def test_effective_n():
    assert _diagnostics.effective_n(100, 0.5) == pytest.approx(100 * 0.5 / 1.5)
    assert _diagnostics.effective_n(100, 0.0) == 100.0
    assert _diagnostics.effective_n(100, -0.4) == 100.0
    assert 0.0 < _diagnostics.effective_n(100, 1.0) < 1e-6
    assert _diagnostics.effective_n(100, float("nan")) == 100.0


def test_autocorrelation_significance_band():
    assert _diagnostics.autocorrelation_is_significant(0.85, 400)
    assert not _diagnostics.autocorrelation_is_significant(0.05, 400)
    # 0.5 from five points is inside the white-noise band 1.96 / sqrt(5).
    assert not _diagnostics.autocorrelation_is_significant(0.5, 5)
    assert not _diagnostics.autocorrelation_is_significant(float("nan"), 50)


def test_small_white_noise_samples_rarely_warn():
    """Without the significance gate this fires on ~a third of small samples.

    With it, the false-positive rate is the band's nominal 5%.
    """
    rng = np.random.default_rng(11)
    trials = 400
    fired = 0
    for _ in range(trials):
        n = int(rng.integers(5, 12))
        x = rng.uniform(0.0, 10.0, n)
        y = 1.0 + 0.5 * x + rng.normal(0.0, 0.3, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = york.fit(x, y, sx=0.2, sy=0.3)
        fired += bool(fit.warnings)
    assert fired / trials < 0.10


def test_floor_fraction():
    sigma = np.array([0.1, 0.1, 0.5])
    assert _diagnostics.floor_fraction(sigma, 0.1) == pytest.approx(2 / 3)
    assert _diagnostics.floor_fraction(sigma, 0.0) == 0.0
    assert _diagnostics.floor_fraction(np.array([]), 0.1) == 0.0


# --- autocorrelation through fit() -------------------------------------------


@pytest.fixture(scope="module")
def autocorrelated_series():
    rng = np.random.default_rng(42)
    n = 400
    x = np.linspace(0.0, 50.0, n) + rng.normal(0.0, 0.5, n)
    y = 1.0 + 0.9 * x + _ar1(rng, n, 0.85, sigma=2.0)
    return x, y


def test_autocorrelated_residuals_warn_and_reduce_n_eff(autocorrelated_series):
    x, y = autocorrelated_series
    with pytest.warns(UserWarning, match="autocorrelated"):
        fit = york.fit(x, y, sx=0.5, sy=2.0)
    assert fit.n_eff < 0.5 * fit.n
    assert any("autocorrelated" in w for w in fit.warnings)
    assert "autocorrelated" in fit.summary()


def test_autocorrelation_check_can_be_disabled(autocorrelated_series):
    x, y = autocorrelated_series
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = york.fit(x, y, sx=0.5, sy=2.0, check_autocorrelation=False)
    assert np.isnan(fit.n_eff)
    assert fit.warnings == ()
    assert "n_eff = n/a" in fit.summary()


def test_independent_residuals_do_not_warn():
    rng = np.random.default_rng(7)
    n = 400
    x = np.linspace(0.0, 50.0, n) + rng.normal(0.0, 0.5, n)
    y = 1.0 + 0.9 * x + rng.normal(0.0, 2.0, n)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = york.fit(x, y, sx=0.5, sy=2.0)
    assert fit.n_eff >= 0.5 * n
    assert fit.warnings == ()


# --- default sigma policy ---------------------------------------------------


def test_default_policy_warns_and_constrains_mswd(pearson_york):
    d = pearson_york
    with pytest.warns(UserWarning, match="MSWD was forced to 1") as record:
        fit = york.fit(d.x, d.y)
    assert fit.mswd_is_constrained
    assert fit.mswd == pytest.approx(1.0)
    assert fit.chi2 == pytest.approx(fit.dof)
    msg = str(record[0].message)
    assert "sigma_x / sigma_y" in msg
    assert "mswd_is_constrained=True" in msg
    assert any("forced to 1" in w for w in fit.warnings)
    assert "constrained" in fit.summary().lower()


def test_default_policy_slope_matches_explicit_ratio(pearson_york):
    """Only the sigma ratio matters for the line; the scale only sets MSWD."""
    d = pearson_york
    with pytest.warns(UserWarning):
        auto = york.fit(d.x, d.y)
    lam = float(auto.sx[0] / auto.sy[0])
    explicit = york.fit(d.x, d.y, sx=lam, sy=1.0, scale_errors=True)
    assert auto.slope == pytest.approx(explicit.slope, rel=1e-10)
    assert auto.intercept == pytest.approx(explicit.intercept, rel=1e-10)
    # Forcing MSWD = 1 makes the internal errors equal the external ones.
    assert auto.slope_err == pytest.approx(explicit.slope_err, rel=1e-8)


def test_one_sided_sigma_is_ambiguous(pearson_york):
    d = pearson_york
    with pytest.raises(ValueError, match="sx was given but sy was not"):
        york.fit(d.x, d.y, sx=d.sx)
    with pytest.raises(ValueError, match="sy was given but sx was not"):
        york.fit(d.x, d.y, sy=d.sy)


def test_non_convergence_warns(pearson_york):
    d = pearson_york
    with pytest.warns(UserWarning, match="did not converge"):
        fit = york.fit(d.x, d.y, sx=d.sx, sy=d.sy, max_iter=1)
    assert fit.converged is False
    assert fit.n_iter == 1
    assert "NOT converged" in fit.summary()
