"""Plotting (SPEC.md §10, 0.2.0). Checks the artists, not pixels."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import york

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from york import plotting  # noqa: E402


@pytest.fixture(scope="module")
def fit(pearson_york):
    d = pearson_york
    return york.fit(d.x, d.y, sx=d.sx, sy=d.sy)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _labels(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_plot_fit_draws_all_layers(fit):
    ax = plotting.plot_fit(fit, xlabel="reference", ylabel="candidate")
    labels = _labels(ax)
    assert any("data" in lab for lab in labels)
    assert any("confidence band" in lab for lab in labels)
    assert any("1:1" == lab for lab in labels)
    assert any(lab.startswith("y = ") and "MSWD" in lab for lab in labels)
    assert ax.containers, "error bars should be drawn"
    assert ax.collections, "confidence band should be drawn"
    assert ax.get_xlabel() == "reference"
    assert ax.get_ylabel() == "candidate"


def test_plot_fit_toggles(fit):
    ax = plotting.plot_fit(fit, one_to_one=False, band=False, errorbars=False)
    labels = _labels(ax)
    assert "1:1" not in labels
    assert not any("confidence" in lab for lab in labels)
    assert not ax.containers
    assert not ax.collections


def test_plot_fit_line_matches_result(fit):
    ax = plotting.plot_fit(fit, one_to_one=False, band=False, errorbars=False)
    (line,) = [ln for ln in ax.get_lines() if ln.get_label().startswith("y = ")]
    xs, ys = line.get_xdata(), line.get_ydata()
    np.testing.assert_allclose(ys, fit.predict(xs))
    assert xs.min() < fit.x.min() and xs.max() > fit.x.max()


def test_plot_fit_on_existing_axes(fit):
    _, ax = plt.subplots()
    out = plotting.plot_fit(fit, ax)
    assert out is ax


def test_plot_residuals(fit):
    ax = plotting.plot_residuals(fit, xlabel="reference")
    (pts,) = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
    np.testing.assert_allclose(pts.get_ydata(), fit.residuals())
    np.testing.assert_allclose(pts.get_xdata(), fit.x)
    guides = sorted(ln.get_ydata()[0] for ln in ax.get_lines() if ln is not pts)
    assert guides == [-2.0, 0.0, 2.0]
    assert ax.get_xlabel() == "reference"


def test_plot_combined(fit):
    fig = plotting.plot(fit, xlabel="reference", ylabel="candidate", level=0.9)
    assert len(fig.axes) == 2
    ax_fit, ax_res = fig.axes
    assert any("90%" in lab for lab in _labels(ax_fit))
    assert ax_fit.get_shared_x_axes().joined(ax_fit, ax_res)
    assert ax_res.get_xlabel() == "reference"
    assert ax_fit.get_ylabel() == "candidate"


def test_yorkfit_plot_method(fit):
    fig = fit.plot()
    assert len(fig.axes) == 2


def test_missing_matplotlib_message(fit, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"york\[plot\]"):
        plotting.plot_fit(fit)
