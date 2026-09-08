"""Public :func:`fit` entry point: input resolution and orchestration.

The solver itself lives in :mod:`york._core`; this module turns user inputs
(arrays, scalars, error models, or ``None``) into sigma arrays, runs the
solver, attaches diagnostics, and builds the :class:`york.YorkFit`.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from york import _diagnostics, _stats
from york._core import york_solve
from york._result import YorkFit, weighted_residuals
from york.errors import ErrorModel

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    SigmaSpec = ArrayLike | float | ErrorModel | None


def fit(
    x: ArrayLike,
    y: ArrayLike,
    sx: SigmaSpec = None,
    sy: SigmaSpec = None,
    r: ArrayLike | float = 0.0,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
    scale_errors: bool = False,
    check_autocorrelation: bool = True,
) -> YorkFit:
    """Fit a straight line when both x and y have uncertainty (York 2004).

    Parameters
    ----------
    x, y
        Array-likes of equal length, n >= 3.
    sx, sy
        One-sigma uncertainties for x and y, specified independently. Each
        accepts an array, a scalar (constant absolute uncertainty), an
        :class:`york.errors.ErrorModel`, or ``None``. If both are ``None``,
        the ratio sigma_x / sigma_y is estimated from local variability and
        both are scaled so MSWD = 1; a ``UserWarning`` states exactly what
        was done and the result carries ``mswd_is_constrained=True``. Giving
        one but not the other is ambiguous and raises ``ValueError``.
    r
        Correlation between x and y errors; scalar or array. Default 0.
    tol, max_iter
        Solver convergence controls.
    scale_errors
        If True, inflate parameter uncertainties by ``sqrt(MSWD)``. Off by
        default: internal vs external errors is an explicit choice.
    check_autocorrelation
        Compute lag-1 residual autocorrelation and ``n_eff``; warn if
        ``n_eff`` is well below n.

    Raises
    ------
    ValueError
        On ambiguous or degenerate input (n < 3, NaN, zero-variance x, shape
        mismatch). Ambiguity is never silently resolved.
    """
    xa = _as_1d(x, "x")
    ya = _as_1d(y, "y")
    if xa.shape != ya.shape:
        raise ValueError(
            f"x and y must have the same length; got {xa.size} and {ya.size}"
        )
    n = xa.size
    if n < 3:
        raise ValueError(f"York regression needs at least 3 points; got {n}")
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        raise ValueError("x and y must be finite; remove NaN/inf values before fitting")
    if np.ptp(xa) == 0.0:
        raise ValueError("x has zero variance; a straight line in y-on-x is undefined")

    active: list[str] = []
    mswd_is_constrained = False

    if sx is None and sy is None:
        sx_arr, sy_arr, description = _default_sigmas(xa, ya, r, tol, max_iter)
        mswd_is_constrained = True
        msg = (
            "sx and sy were not given. " + description + " MSWD was forced to 1 by "
            "construction and carries no goodness-of-fit information "
            "(mswd_is_constrained=True). The slope and its uncertainty remain "
            "meaningful. Pass sx and sy to get a real MSWD."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        active.append(msg)
    elif sx is None or sy is None:
        missing, given = ("sx", "sy") if sx is None else ("sy", "sx")
        raise ValueError(
            f"{given} was given but {missing} was not. Pass both uncertainties, "
            "or neither to use the default ratio-estimation policy. For a "
            f"negligible {missing}, pass a small positive value explicitly."
        )
    else:
        sx_arr, sx_desc, sx_floor = _resolve_sigma(sx, xa, "sx")
        sy_arr, sy_desc, sy_floor = _resolve_sigma(sy, ya, "sy")
        description = f"{sx_desc}; {sy_desc}"
        for name, frac in (("sx", sx_floor), ("sy", sy_floor)):
            if frac > _diagnostics.DEFAULT_FLOOR_WARN_FRACTION:
                msg = (
                    f"{frac:.0%} of {name} values sit at the min_sigma floor. "
                    "Values near or below the detection limit are being held "
                    "at the floor; check that the floor reflects real precision."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
                active.append(msg)

    r_arr = _resolve_r(r, n)

    core = york_solve(xa, ya, sx_arr, sy_arr, r_arr, tol=tol, max_iter=max_iter)
    if not core.converged:
        msg = (
            f"Solver did not converge in {max_iter} iterations "
            f"(final slope {core.slope:.6g}). Results are unreliable."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        active.append(msg)

    slope_err = core.slope_err
    intercept_err = core.intercept_err
    if scale_errors:
        scale = float(np.sqrt(core.mswd))
        slope_err *= scale
        intercept_err *= scale

    dof = n - 2
    chi2_p = float(_stats.chi2_sf(core.chi2, dof))

    n_eff = float("nan")
    if check_autocorrelation:
        resid = weighted_residuals(
            xa, ya, sx_arr, sy_arr, r_arr, core.slope, core.intercept
        )
        rho1 = _diagnostics.lag1_autocorrelation(resid)
        n_eff = _diagnostics.effective_n(n, rho1)
        if (
            n_eff < _diagnostics.DEFAULT_N_EFF_WARN_FRACTION * n
            and _diagnostics.autocorrelation_is_significant(rho1, n)
        ):
            msg = (
                f"Residuals are autocorrelated (lag-1 rho = {rho1:.2f}); effective "
                f"sample size is about {n_eff:.0f} of {n}. Reported uncertainties "
                "assume independent points and are too small."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            active.append(msg)

    return YorkFit(
        slope=core.slope,
        intercept=core.intercept,
        slope_err=slope_err,
        intercept_err=intercept_err,
        mswd=core.mswd,
        chi2=core.chi2,
        chi2_p=chi2_p,
        n=n,
        dof=dof,
        x_bar=core.x_bar,
        y_bar=core.y_bar,
        n_iter=core.n_iter,
        converged=core.converged,
        mswd_is_constrained=mswd_is_constrained,
        n_eff=n_eff,
        error_model_description=description,
        scale_errors=scale_errors,
        x=xa,
        y=ya,
        sx=sx_arr,
        sy=sy_arr,
        r=r_arr,
        x_adj=core.x_adj,
        y_adj=core.y_adj,
        warnings=tuple(active),
    )


def _as_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {arr.shape}")
    return arr


def _resolve_sigma(
    spec: SigmaSpec, values: NDArray[np.float64], name: str
) -> tuple[NDArray[np.float64], str, float]:
    """Turn a user sigma spec into an array, a description, and a floor fraction."""
    floor_frac = 0.0
    if isinstance(spec, ErrorModel):
        sigma = np.asarray(spec(values), dtype=np.float64)
        description = f"{name}: {spec.describe()}"
        floor_frac = _diagnostics.floor_fraction(sigma, spec.min_sigma)
    else:
        arr = np.asarray(spec, dtype=np.float64)
        if arr.ndim == 0:
            sigma = np.full(values.shape, float(arr))
            description = f"{name}: constant {float(arr):g}"
        else:
            sigma = _as_1d(arr, name)
            if sigma.shape != values.shape:
                raise ValueError(
                    f"{name} has length {sigma.size} but the data have length "
                    f"{values.size}"
                )
            description = f"{name}: per-point array"
    if not np.all(np.isfinite(sigma)):
        raise ValueError(f"{name} contains NaN or inf")
    if np.any(sigma <= 0.0):
        raise ValueError(
            f"{name} must be strictly positive everywhere. For a negligible "
            "uncertainty pass a small positive value; to guard against zero "
            "sigma from a relative error model, set its min_sigma."
        )
    return sigma, description, floor_frac


def _resolve_r(r: ArrayLike | float, n: int) -> NDArray[np.float64]:
    arr = np.asarray(r, dtype=np.float64)
    out: NDArray[np.float64]
    if arr.ndim == 0:
        out = np.full(n, float(arr))
    else:
        out = _as_1d(arr, "r")
        if out.size != n:
            raise ValueError(f"r has length {out.size} but the data have length {n}")
    if not np.all(np.isfinite(out)) or np.any(np.abs(out) > 1.0):
        raise ValueError("r must be finite and within [-1, 1]")
    return out


def _local_scale(values: NDArray[np.float64]) -> float:
    """Robust local noise estimate from successive differences.

    ``1.4826 * MAD(diff) / sqrt(2)`` estimates the per-point noise sigma if
    the series were signal plus white noise. Falls back to the standard
    deviation of the differences, then to the global standard deviation.
    """
    d = np.diff(values)
    mad = float(np.median(np.abs(d - np.median(d))))
    scale = 1.4826 * mad / math.sqrt(2.0)
    if scale <= 0.0:
        scale = float(np.std(d) / np.sqrt(2.0))
    if scale <= 0.0:
        scale = float(np.std(values))
    return scale


def _default_sigmas(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    r: ArrayLike | float,
    tol: float,
    max_iter: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], str]:
    """Default policy: estimate lambda = sigma_x / sigma_y, scale to MSWD = 1.

    Only the ratio matters for the slope. The common scale is then fixed by
    requiring MSWD = 1, which makes the reported parameter uncertainties the
    sqrt(MSWD)-scaled ("external") ones.
    """
    sx0 = _local_scale(x)
    sy0 = _local_scale(y)
    if sx0 <= 0.0 and sy0 <= 0.0:
        sx0 = sy0 = 1.0
    elif sx0 <= 0.0:
        sx0 = sy0
    elif sy0 <= 0.0:
        sy0 = sx0
    lam = sx0 / sy0

    n = x.size
    r_arr = _resolve_r(r, n)
    sx_arr = np.full(n, sx0)
    sy_arr = np.full(n, sy0)
    core = york_solve(x, y, sx_arr, sy_arr, r_arr, tol=tol, max_iter=max_iter)
    scale = float(np.sqrt(core.mswd)) if core.mswd > 0.0 else 1.0
    sx_arr *= scale
    sy_arr *= scale
    description = (
        f"Estimated sigma_x / sigma_y = {lam:.3g} from the median absolute "
        f"successive difference of each series, then scaled both so that "
        f"MSWD = 1 (sigma_x = {sx0 * scale:.3g}, sigma_y = {sy0 * scale:.3g})."
    )
    return sx_arr, sy_arr, description
