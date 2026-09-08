"""``YorkFit`` methods and ``_stats`` (SPEC.md §4).

Hypothesis tests and confidence bands are checked against ``scipy.stats``;
the OLS limit of the confidence band is checked against the textbook
formula with known sigma.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import york
from york import _stats


@pytest.fixture(scope="module")
def fit(pearson_york):
    d = pearson_york
    return york.fit(d.x, d.y, sx=d.sx, sy=d.sy)


# --- _stats vs scipy --------------------------------------------------------


@pytest.mark.parametrize("dof", [1, 2, 3, 8, 30, 200])
def test_chi2_sf_matches_scipy(dof):
    xs = np.linspace(0.01, 80.0, 300)
    np.testing.assert_allclose(
        _stats.chi2_sf(xs, dof), stats.chi2.sf(xs, dof), atol=1e-10
    )


@pytest.mark.parametrize("dof", [1, 2, 3, 8, 30, 200])
def test_t_distribution_matches_scipy(dof):
    ts = np.linspace(-12.0, 12.0, 301)
    np.testing.assert_allclose(_stats.t_cdf(ts, dof), stats.t.cdf(ts, dof), atol=1e-10)
    np.testing.assert_allclose(
        _stats.t_sf_two_sided(ts, dof), 2 * stats.t.sf(np.abs(ts), dof), atol=1e-10
    )
    for q in (0.005, 0.025, 0.5, 0.975, 0.995):
        assert _stats.t_ppf(q, dof) == pytest.approx(stats.t.ppf(q, dof), rel=1e-9)


def test_t_ppf_rejects_out_of_range():
    with pytest.raises(ValueError):
        _stats.t_ppf(1.0, 5)


# --- hypothesis tests -------------------------------------------------------


def test_chi2_p_matches_scipy(fit):
    assert fit.chi2_p == pytest.approx(stats.chi2.sf(fit.chi2, fit.dof), rel=1e-10)


def test_test_slope_against_scipy(fit):
    t = fit.test_slope(value=1.0, level=0.95)
    stat = (fit.slope - 1.0) / fit.slope_err
    assert t.statistic == pytest.approx(stat)
    assert t.p_value == pytest.approx(2 * stats.t.sf(abs(stat), fit.dof), rel=1e-9)
    half = stats.t.ppf(0.975, fit.dof) * fit.slope_err
    assert t.ci_low == pytest.approx(fit.slope - half, rel=1e-9)
    assert t.ci_high == pytest.approx(fit.slope + half, rel=1e-9)
    assert t.dof == fit.dof
    assert t.null_value == 1.0
    assert t.estimate == fit.slope
    assert t.std_err == fit.slope_err


def test_test_intercept_against_scipy(fit):
    t = fit.test_intercept(value=0.0, level=0.99)
    stat = fit.intercept / fit.intercept_err
    assert t.statistic == pytest.approx(stat)
    assert t.p_value == pytest.approx(2 * stats.t.sf(abs(stat), fit.dof), rel=1e-9)
    half = stats.t.ppf(0.995, fit.dof) * fit.intercept_err
    assert t.ci_low == pytest.approx(fit.intercept - half, rel=1e-9)
    assert t.ci_high == pytest.approx(fit.intercept + half, rel=1e-9)


def test_test_against_own_estimate_is_not_significant(fit):
    t = fit.test_slope(value=fit.slope)
    assert t.statistic == 0.0
    assert t.p_value == pytest.approx(1.0)


def test_hypothesis_test_rejects_bad_level(fit):
    with pytest.raises(ValueError):
        fit.test_slope(level=1.0)


# --- predict / residuals / band ---------------------------------------------


def test_predict(fit):
    xs = np.array([0.0, 1.0, 2.5])
    np.testing.assert_allclose(fit.predict(xs), fit.intercept + fit.slope * xs)
    assert float(fit.predict(1.0)) == pytest.approx(fit.intercept + fit.slope)


def test_weighted_residuals_sum_of_squares_is_chi2(fit):
    r = fit.residuals()
    assert r.shape == fit.x.shape
    assert np.sum(r**2) == pytest.approx(fit.chi2, rel=1e-12)


def test_confidence_band_brackets_the_line_and_widens_outward(fit):
    xs = np.linspace(-2.0, 10.0, 50)
    lo, hi = fit.confidence_band(xs, level=0.95)
    line = fit.predict(xs)
    assert np.all(lo < line) and np.all(line < hi)
    half = hi - lo
    # Narrowest near the data centroid, wider toward the extremes.
    centre = np.argmin(half)
    assert half[0] > half[centre] and half[-1] > half[centre]
    lo99, hi99 = fit.confidence_band(xs, level=0.99)
    assert np.all(hi99 - lo99 > half)


def test_confidence_band_matches_ols_formula_in_ols_limit():
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 10.0, 30)
    sigma = 0.7
    y = 1.0 + 2.0 * x + rng.normal(0.0, sigma, x.size)
    f = york.fit(x, y, sx=1e-7, sy=sigma)
    xs = np.array([-1.0, 3.0, 5.0, 12.0])
    lo, hi = f.confidence_band(xs, level=0.95)
    # Known-sigma OLS band: t * sigma * sqrt(1/n + (x - xbar)^2 / Sxx).
    sxx = np.sum((x - x.mean()) ** 2)
    se = sigma * np.sqrt(1.0 / x.size + (xs - x.mean()) ** 2 / sxx)
    half = stats.t.ppf(0.975, x.size - 2) * se
    np.testing.assert_allclose((hi - lo) / 2, half, rtol=1e-5)


def test_confidence_band_scales_with_errors(pearson_york):
    d = pearson_york
    a = york.fit(d.x, d.y, sx=d.sx, sy=d.sy)
    b = york.fit(d.x, d.y, sx=d.sx, sy=d.sy, scale_errors=True)
    xs = np.linspace(0.0, 8.0, 9)
    wa = np.subtract(*a.confidence_band(xs)[::-1])
    wb = np.subtract(*b.confidence_band(xs)[::-1])
    np.testing.assert_allclose(wb / wa, np.sqrt(a.mswd), rtol=1e-10)


# --- summary / repr ---------------------------------------------------------


def test_repr_is_one_line(fit):
    r = repr(fit)
    assert "\n" not in r
    assert r.startswith("YorkFit(y = ")
    assert "MSWD=1.48" in r
    assert "n=10" in r


def test_summary_contents(fit):
    s = fit.summary()
    for needle in ("slope", "intercept", "MSWD", "1.483", "sx", "sy", "p ="):
        assert needle in s
    assert "slope = 1" in s or "slope == 1" in s
    assert "intercept = 0" in s or "intercept == 0" in s


def test_summary_states_constrained_mswd(pearson_york):
    d = pearson_york
    with pytest.warns(UserWarning):
        f = york.fit(d.x, d.y)
    s = f.summary()
    assert f.mswd_is_constrained
    assert "constrained" in s.lower()
    assert "goodness" in s.lower()
