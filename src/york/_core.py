"""Iterative York solver.

INVARIANT: this module imports numpy and the standard library only. No scipy,
no pandas, no other ``york`` modules. Keep it readable against
Wehr & Saleska (2017) and York et al. (2004).

Notation follows York et al. (2004), eqs. 13a-13c and the error formulas
that follow them:

    W_i   = w_x w_y / (w_x + b^2 w_y - 2 b r alpha),   alpha = sqrt(w_x w_y)
    U_i   = x_i - x_bar,  V_i = y_i - y_bar            (W-weighted means)
    beta_i = W_i [ U_i / w_y + b V_i / w_x - (b U_i + V_i) r / alpha ]
    b     = sum(W beta V) / sum(W beta U)              (iterate to convergence)
    a     = y_bar - b x_bar
    x_adj = x_bar + beta,  u = x_adj - mean_W(x_adj)
    sigma_b^2 = 1 / sum(W u^2)
    sigma_a^2 = 1 / sum(W) + mean_W(x_adj)^2 sigma_b^2
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class CoreResult(NamedTuple):
    """Raw solver output, before diagnostics and hypothesis tests are attached."""

    slope: float
    intercept: float
    slope_err: float
    intercept_err: float
    mswd: float
    chi2: float
    x_bar: float
    """Weight-averaged x, using the final combined weights W."""
    y_bar: float
    """Weight-averaged y, using the final combined weights W."""
    x_adj: NDArray[np.float64]
    """Adjusted x values (the fitted points on the line), for residuals."""
    y_adj: NDArray[np.float64]
    """Adjusted y values (the fitted points on the line), for residuals."""
    n_iter: int
    converged: bool


class _Step(NamedTuple):
    W: NDArray[np.float64]
    x_bar: float
    y_bar: float
    U: NDArray[np.float64]
    V: NDArray[np.float64]
    beta: NDArray[np.float64]


def _ols_slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    dx = x - x.mean()
    return float(np.sum(dx * (y - y.mean())) / np.sum(dx * dx))


def _step(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    wx: NDArray[np.float64],
    wy: NDArray[np.float64],
    r: NDArray[np.float64],
    alpha: NDArray[np.float64],
    b: float,
) -> _Step:
    W = wx * wy / (wx + b * b * wy - 2.0 * b * r * alpha)
    sum_W = np.sum(W)
    x_bar = float(np.sum(W * x) / sum_W)
    y_bar = float(np.sum(W * y) / sum_W)
    U = x - x_bar
    V = y - y_bar
    beta = W * (U / wy + b * V / wx - (b * U + V) * r / alpha)
    return _Step(W, x_bar, y_bar, U, V, beta)


def york_solve(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sx: NDArray[np.float64],
    sy: NDArray[np.float64],
    r: NDArray[np.float64],
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> CoreResult:
    """Fit ``y = a + b x`` by York's method.

    Parameters
    ----------
    x, y
        Observations, shape ``(n,)``, already validated (finite, n >= 3).
    sx, sy
        One-sigma uncertainties, shape ``(n,)``, strictly positive.
    r
        Correlation between the x and y errors of each point, shape ``(n,)``.
    tol
        Convergence tolerance on the relative change in slope.
    max_iter
        Iteration cap. Reaching it on clean data means the weight update is
        wrong; do not loosen ``tol`` to compensate.

    Returns
    -------
    CoreResult
        Slope, intercept, their (unscaled, internal) standard errors, MSWD and
        chi-square, the weighted centroid, adjusted points, iteration count
        and convergence flag.
    """
    wx = 1.0 / (sx * sx)
    wy = 1.0 / (sy * sy)
    alpha = np.sqrt(wx * wy)

    b = _ols_slope(x, y)
    converged = False
    n_iter = 0
    for i in range(1, max_iter + 1):
        n_iter = i
        s = _step(x, y, wx, wy, r, alpha, b)
        b_new = float(np.sum(s.W * s.beta * s.V) / np.sum(s.W * s.beta * s.U))
        done = abs(b_new - b) <= tol * abs(b_new)
        b = b_new
        if done:
            converged = True
            break

    # Final quantities, all evaluated at the converged slope.
    s = _step(x, y, wx, wy, r, alpha, b)
    a = s.y_bar - b * s.x_bar

    x_adj = s.x_bar + s.beta
    y_adj = s.y_bar + b * s.beta
    sum_W = float(np.sum(s.W))
    x_adj_bar = float(np.sum(s.W * x_adj) / sum_W)
    u = x_adj - x_adj_bar

    slope_var = 1.0 / float(np.sum(s.W * u * u))
    intercept_var = 1.0 / sum_W + x_adj_bar * x_adj_bar * slope_var

    resid = y - a - b * x
    chi2 = float(np.sum(s.W * resid * resid))
    mswd = chi2 / (x.size - 2)

    return CoreResult(
        slope=b,
        intercept=a,
        slope_err=float(np.sqrt(slope_var)),
        intercept_err=float(np.sqrt(intercept_var)),
        mswd=mswd,
        chi2=chi2,
        x_bar=s.x_bar,
        y_bar=s.y_bar,
        x_adj=x_adj,
        y_adj=y_adj,
        n_iter=n_iter,
        converged=converged,
    )
