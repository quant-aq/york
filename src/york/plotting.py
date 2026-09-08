"""Plotting helpers. Requires the ``[plot]`` extra (matplotlib).

Three entry points:

- :func:`plot_fit` — scatter with error bars, fitted line, confidence band,
  and the 1:1 reference line that method comparison is judged against.
- :func:`plot_residuals` — weighted residuals against the reference axis,
  with ±2 sigma guides. Structure here means the error model or the
  straight-line assumption is wrong.
- :func:`plot` — both, stacked with a shared x axis.

All accept an existing ``Axes`` so they compose with your own figures, and
all return what they drew on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from york._result import YorkFit

__all__ = ["plot", "plot_fit", "plot_residuals"]


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it with `pip install york[plot]`."
        ) from exc
    return plt


def _line_label(result: YorkFit) -> str:
    sign = "-" if result.intercept < 0 else "+"
    return (
        f"y = {result.slope:.3g}x {sign} {abs(result.intercept):.3g}"
        f"  (MSWD {result.mswd:.2f})"
    )


def plot_fit(
    result: YorkFit,
    ax: Any = None,
    *,
    level: float = 0.95,
    one_to_one: bool = True,
    band: bool = True,
    errorbars: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Any:
    """Scatter with error bars, fitted line, confidence band, and 1:1 line.

    Parameters
    ----------
    result
        A :class:`york.YorkFit`.
    ax
        Matplotlib ``Axes`` to draw on. A new figure is created if omitted.
    level
        Confidence level for the band.
    one_to_one
        Draw the ``y = x`` reference line. On for intercomparison because
        "is it 1:1?" is the question being asked.
    band, errorbars
        Toggle the confidence band and the per-point error bars.
    xlabel, ylabel
        Axis labels.

    Returns
    -------
    The ``Axes`` drawn on.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    x, y = result.x, result.y
    if errorbars:
        ax.errorbar(
            x,
            y,
            xerr=result.sx,
            yerr=result.sy,
            fmt="o",
            ms=3,
            lw=0.8,
            alpha=0.7,
            capsize=0,
            label="data (±1σ)",  # noqa: RUF001
            zorder=2,
        )
    else:
        ax.plot(x, y, "o", ms=3, alpha=0.7, label="data", zorder=2)

    lo, hi = float(np.min(x)), float(np.max(x))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, 200)

    if band:
        lower, upper = result.confidence_band(grid, level=level)
        ax.fill_between(
            grid,
            lower,
            upper,
            alpha=0.2,
            lw=0,
            label=f"{level:.0%} confidence band",
            zorder=1,
        )
    ax.plot(
        grid, result.predict(grid), "-", lw=1.5, label=_line_label(result), zorder=3
    )

    if one_to_one:
        ax.plot(grid, grid, "--", color="0.4", lw=1, label="1:1", zorder=0)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_residuals(
    result: YorkFit,
    ax: Any = None,
    *,
    xlabel: str | None = None,
) -> Any:
    """Weighted residuals against the reference (x) axis.

    Residuals are in units of their own sigma, so ±2 lines mark where about
    95% of points should fall if the error model is right. Trends or
    fanning mean the model or the straight line is wrong.

    Returns
    -------
    The ``Axes`` drawn on.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    resid = result.residuals()
    ax.axhline(0.0, color="0.4", lw=1, zorder=0)
    for level in (-2.0, 2.0):
        ax.axhline(level, color="0.4", lw=0.8, ls=":", zorder=0)
    ax.plot(result.x, resid, "o", ms=3, alpha=0.7, zorder=2)
    ax.set_ylabel("weighted residual (σ)")  # noqa: RUF001
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    return ax


def plot(
    result: YorkFit,
    *,
    level: float = 0.95,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (6.0, 7.0),
    **fit_kwargs: Any,
) -> Any:
    """Fit on top, residuals below, sharing the x axis.

    Extra keyword arguments go to :func:`plot_fit`.

    Returns
    -------
    The ``Figure``. Its two ``Axes`` are ``fig.axes[0]`` (fit) and
    ``fig.axes[1]`` (residuals).
    """
    plt = _require_matplotlib()
    fig, (ax_fit, ax_res) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    plot_fit(result, ax_fit, level=level, ylabel=ylabel, **fit_kwargs)
    plot_residuals(result, ax_res, xlabel=xlabel)
    return fig
