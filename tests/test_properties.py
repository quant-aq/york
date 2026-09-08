"""Hypothesis property tests (SPEC.md §7.5) and degenerate inputs (§7.6)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import york

# Random small samples trip the autocorrelation band at its nominal 5% rate;
# that behaviour is tested in test_diagnostics, not here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Residuals are autocorrelated:UserWarning"
)

# --- data strategy ----------------------------------------------------------


@st.composite
def datasets(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    n = draw(st.integers(5, 60))
    slope = draw(st.floats(-5.0, 5.0))
    intercept = draw(st.floats(-50.0, 50.0))
    ratio = draw(st.floats(0.1, 10.0))
    rng = np.random.default_rng(seed)
    x_true = rng.uniform(0.0, 10.0, n)
    assume(np.ptp(x_true) > 1.0)
    sx = rng.uniform(0.1, 0.3, n)
    sy = rng.uniform(0.1, 0.3, n) * ratio
    x = x_true + rng.normal(0.0, sx)
    y = intercept + slope * x_true + rng.normal(0.0, sy)
    return x, y, sx, sy


nonzero_scale = st.one_of(st.floats(-100.0, -0.01), st.floats(0.01, 100.0))


# --- properties -------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(datasets())
def test_converges_quickly_and_never_silently_fails(data):
    x, y, sx, sy = data
    fit = york.fit(x, y, sx=sx, sy=sy)
    assert fit.converged is True
    assert fit.n_iter < 20
    assert np.isfinite(fit.slope) and np.isfinite(fit.slope_err)
    assert np.isfinite(fit.intercept) and np.isfinite(fit.intercept_err)
    assert fit.mswd >= 0.0
    assert 0.0 <= fit.chi2_p <= 1.0


@settings(max_examples=200, deadline=None)
@given(
    datasets(),
    nonzero_scale,
    nonzero_scale,
    st.floats(-1000.0, 1000.0),
    st.floats(-1000.0, 1000.0),
)
def test_affine_equivariance(data, cx, cy, dx, dy):
    """x' = cx x + dx, y' = cy y + dy transforms the fit predictably."""
    x, y, sx, sy = data
    base = york.fit(x, y, sx=sx, sy=sy)
    moved = york.fit(cx * x + dx, cy * y + dy, sx=abs(cx) * sx, sy=abs(cy) * sy)

    slope_expected = base.slope * cy / cx
    intercept_expected = cy * base.intercept + dy - slope_expected * dx
    scale = abs(dy) + abs(dx * slope_expected) + abs(cy * base.intercept) + 1.0

    assert moved.slope == pytest.approx(slope_expected, rel=1e-7)
    assert moved.intercept == pytest.approx(
        intercept_expected, rel=1e-7, abs=1e-9 * scale
    )
    assert moved.slope_err == pytest.approx(base.slope_err * abs(cy / cx), rel=1e-7)
    assert moved.mswd == pytest.approx(base.mswd, rel=1e-7)
    assert moved.chi2_p == pytest.approx(base.chi2_p, rel=1e-6, abs=1e-12)


@settings(max_examples=200, deadline=None)
@given(datasets())
def test_swap_symmetry(data):
    x, y, sx, sy = data
    f = york.fit(x, y, sx=sx, sy=sy)
    g = york.fit(y, x, sx=sy, sy=sx)
    assert f.slope * g.slope == pytest.approx(1.0, rel=1e-9)
    assert g.mswd == pytest.approx(f.mswd, rel=1e-9)


@settings(max_examples=100, deadline=None)
@given(datasets(), st.floats(0.1, 100.0))
def test_common_sigma_scale_only_changes_mswd(data, k):
    x, y, sx, sy = data
    a = york.fit(x, y, sx=sx, sy=sy)
    b = york.fit(x, y, sx=k * sx, sy=k * sy)
    assert b.slope == pytest.approx(a.slope, rel=1e-9)
    assert b.mswd == pytest.approx(a.mswd / k**2, rel=1e-8)
    assert b.slope_err == pytest.approx(k * a.slope_err, rel=1e-8)


# --- degenerate inputs ------------------------------------------------------


def test_too_few_points():
    with pytest.raises(ValueError, match="at least 3"):
        york.fit([0.0, 1.0], [0.0, 1.0], sx=0.1, sy=0.1)


def test_nan_and_inf_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        york.fit([0.0, 1.0, np.nan], [0.0, 1.0, 2.0], sx=0.1, sy=0.1)
    with pytest.raises(ValueError, match="finite"):
        york.fit([0.0, 1.0, 2.0], [0.0, np.inf, 2.0], sx=0.1, sy=0.1)


def test_zero_variance_x_and_identical_points():
    with pytest.raises(ValueError, match="zero variance"):
        york.fit([2.0, 2.0, 2.0], [0.0, 1.0, 2.0], sx=0.1, sy=0.1)
    with pytest.raises(ValueError, match="zero variance"):
        york.fit([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], sx=0.1, sy=0.1)


def test_length_and_shape_mismatches():
    with pytest.raises(ValueError, match="same length"):
        york.fit([0.0, 1.0, 2.0], [0.0, 1.0], sx=0.1, sy=0.1)
    with pytest.raises(ValueError, match="one-dimensional"):
        york.fit(np.zeros((3, 2)), [0.0, 1.0, 2.0], sx=0.1, sy=0.1)
    with pytest.raises(ValueError, match="length"):
        york.fit([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], sx=[0.1, 0.1], sy=0.1)
    with pytest.raises(ValueError, match="length"):
        york.fit([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], sx=0.1, sy=0.1, r=[0.0, 0.0])


def test_column_vector_input_is_accepted():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.1, 1.0, 2.1, 2.9])
    fit = york.fit(x, y, sx=0.1, sy=0.1)
    assert fit.x.shape == (4,)


def test_bad_sigma_and_correlation():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [0.1, 1.0, 2.1, 2.9]
    with pytest.raises(ValueError, match="strictly positive"):
        york.fit(x, y, sx=-0.1, sy=0.1)
    with pytest.raises(ValueError, match="strictly positive"):
        york.fit(x, y, sx=0.1, sy=[0.1, 0.0, 0.1, 0.1])
    with pytest.raises(ValueError, match="NaN"):
        york.fit(x, y, sx=0.1, sy=[0.1, np.nan, 0.1, 0.1])
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        york.fit(x, y, sx=0.1, sy=0.1, r=1.5)


def test_perfectly_collinear_data():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 + 0.5 * x
    fit = york.fit(x, y, sx=0.1, sy=0.1)
    assert fit.converged
    assert fit.slope == pytest.approx(0.5)
    assert fit.intercept == pytest.approx(2.0)
    assert fit.chi2 == pytest.approx(0.0, abs=1e-20)
    assert fit.chi2_p == pytest.approx(1.0)


def test_constant_y_gives_zero_slope():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.full(5, 3.0)
    fit = york.fit(x, y, sx=0.1, sy=0.1)
    assert fit.converged
    assert fit.slope == pytest.approx(0.0, abs=1e-15)
    assert fit.intercept == pytest.approx(3.0)


def test_correlated_errors_change_the_fit(pearson_york):
    d = pearson_york
    a = york.fit(d.x, d.y, sx=d.sx, sy=d.sy, r=0.0)
    b = york.fit(d.x, d.y, sx=d.sx, sy=d.sy, r=0.5)
    c = york.fit(d.x, d.y, sx=d.sx, sy=d.sy, r=np.full(d.x.size, 0.5))
    assert a.slope != b.slope
    assert b.slope == c.slope
    assert b.converged and c.converged
