"""Limiting cases (SPEC.md §7.2).

Every expected value here comes from an independent closed form or from
``scipy.odr``. Nothing is asserted against ``york.fit`` itself.

- x error -> 0 collapses to OLS y-on-x (``np.polyfit``).
- y error -> 0 collapses to OLS x-on-y, inverted.
- sx == sy gives orthogonal regression (total least squares via SVD).
- Fixed sy/sx gives Deming regression (closed form).
- Heteroscedastic errors match ``scipy.odr``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import odr

import york


@pytest.fixture(scope="module")
def synthetic():
    """Well-conditioned line with modest noise in both variables."""
    rng = np.random.default_rng(20240)
    n = 40
    x_true = np.linspace(1.0, 20.0, n)
    y = 2.5 + 0.8 * x_true + rng.normal(0.0, 0.4, n)
    x = x_true + rng.normal(0.0, 0.3, n)
    return x, y


def _deming(x, y, delta):
    """Closed-form Deming regression.

    ``delta`` is var(y error) / var(x error). ``delta == 1`` is orthogonal
    regression; ``delta -> inf`` is OLS y-on-x.
    """
    xm, ym = x.mean(), y.mean()
    sxx = np.mean((x - xm) ** 2)
    syy = np.mean((y - ym) ** 2)
    sxy = np.mean((x - xm) * (y - ym))
    slope = (
        syy - delta * sxx + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy**2)
    ) / (2 * sxy)
    return ym - slope * xm, slope


def _tls_svd(x, y):
    """Orthogonal regression from the smallest singular vector."""
    a = np.column_stack([x - x.mean(), y - y.mean()])
    _, _, vt = np.linalg.svd(a)
    nx, ny = vt[-1]
    slope = -nx / ny
    return y.mean() - slope * x.mean(), slope


def test_negligible_x_error_is_ols_y_on_x(synthetic):
    x, y = synthetic
    sigma_y = 0.5
    fit = york.fit(x, y, sx=1e-6, sy=sigma_y)
    slope_ols, intercept_ols = np.polyfit(x, y, 1)
    assert fit.slope == pytest.approx(slope_ols, rel=1e-8)
    assert fit.intercept == pytest.approx(intercept_ols, rel=1e-8)
    # Known constant sigma_y: internal slope error is the textbook OLS value.
    expected_slope_err = sigma_y / np.sqrt(np.sum((x - x.mean()) ** 2))
    assert fit.slope_err == pytest.approx(expected_slope_err, rel=1e-6)


def test_negligible_y_error_is_ols_x_on_y(synthetic):
    x, y = synthetic
    fit = york.fit(x, y, sx=0.5, sy=1e-6)
    slope_xy, intercept_xy = np.polyfit(y, x, 1)
    assert fit.slope == pytest.approx(1.0 / slope_xy, rel=1e-8)
    assert fit.intercept == pytest.approx(-intercept_xy / slope_xy, rel=1e-8)


def test_equal_errors_is_orthogonal_regression(synthetic):
    x, y = synthetic
    fit = york.fit(x, y, sx=0.5, sy=0.5)
    intercept_svd, slope_svd = _tls_svd(x, y)
    assert fit.slope == pytest.approx(slope_svd, rel=1e-9)
    assert fit.intercept == pytest.approx(intercept_svd, rel=1e-9)


@pytest.mark.parametrize("delta", [0.25, 1.0, 4.0, 25.0])
def test_constant_error_ratio_is_deming(synthetic, delta):
    x, y = synthetic
    sx = 1.0
    sy = np.sqrt(delta) * sx
    fit = york.fit(x, y, sx=sx, sy=sy)
    intercept_d, slope_d = _deming(x, y, delta)
    assert fit.slope == pytest.approx(slope_d, rel=1e-9)
    assert fit.intercept == pytest.approx(intercept_d, rel=1e-9)


def test_common_scale_of_sigmas_does_not_move_the_line(synthetic):
    x, y = synthetic
    a = york.fit(x, y, sx=0.3, sy=0.6)
    b = york.fit(x, y, sx=3.0, sy=6.0)
    assert a.slope == pytest.approx(b.slope, rel=1e-10)
    assert a.intercept == pytest.approx(b.intercept, rel=1e-10)
    # MSWD scales as 1 / factor**2; internal errors scale as factor.
    assert a.mswd == pytest.approx(100.0 * b.mswd, rel=1e-8)
    assert b.slope_err == pytest.approx(10.0 * a.slope_err, rel=1e-8)


def test_heteroscedastic_matches_scipy_odr(synthetic):
    x, y = synthetic
    rng = np.random.default_rng(7)
    sx = rng.uniform(0.1, 0.5, x.size)
    sy = rng.uniform(0.2, 0.8, x.size)

    fit = york.fit(x, y, sx=sx, sy=sy, scale_errors=True)

    data = odr.RealData(x, y, sx=sx, sy=sy)
    beta0 = np.polyfit(x, y, 1)
    out = odr.ODR(data, odr.unilinear, beta0=beta0).run()
    slope_odr, intercept_odr = out.beta

    assert fit.slope == pytest.approx(slope_odr, rel=1e-6)
    assert fit.intercept == pytest.approx(intercept_odr, rel=1e-6)
    assert fit.mswd == pytest.approx(out.res_var, rel=1e-6)
    # scipy.odr's sd_beta is scaled by residual variance, i.e. external.
    assert fit.slope_err == pytest.approx(out.sd_beta[0], rel=1e-4)
    assert fit.intercept_err == pytest.approx(out.sd_beta[1], rel=1e-4)
