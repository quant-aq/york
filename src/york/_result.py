"""The :class:`YorkFit` result object and hypothesis-test container."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from york import _stats


def _fmt_pm(value: float, err: float) -> str:
    """Format ``value(±err)`` with the error to two significant figures."""
    if not (math.isfinite(err) and err > 0.0):
        return f"{value:.4g}(±{err:.2g})"
    decimals = max(0, 1 - math.floor(math.log10(err)))
    return f"{value:.{decimals}f}(±{err:.{decimals}f})"


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")


def weighted_residuals(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sx: NDArray[np.float64],
    sy: NDArray[np.float64],
    r: NDArray[np.float64],
    slope: float,
    intercept: float,
) -> NDArray[np.float64]:
    """Residuals in units of their own sigma: ``sqrt(W) (y - a - b x)``.

    ``W = 1 / (sy^2 + b^2 sx^2 - 2 b r sx sy)`` is York's combined weight, so
    the sum of squares of these residuals is the fit's chi-square.
    """
    var = sy * sy + slope * slope * sx * sx - 2.0 * slope * r * sx * sy
    return np.asarray((y - intercept - slope * x) / np.sqrt(var), dtype=np.float64)


@dataclass(frozen=True)
class HypothesisTest:
    """Result of a one-parameter two-sided t-test against a null value."""

    name: str
    null_value: float
    estimate: float
    std_err: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    level: float
    dof: int

    @property
    def significant(self) -> bool:
        """True if the null value lies outside the confidence interval."""
        return self.p_value < 1.0 - self.level

    def __str__(self) -> str:
        return (
            f"{self.name} = {self.null_value:g}: t = {self.statistic:.3f}, "
            f"p = {self.p_value:.3g}, {self.level:.0%} CI "
            f"[{self.ci_low:.4g}, {self.ci_high:.4g}]"
        )


@dataclass(frozen=True)
class YorkFit:
    """Result of a York fit. See SPEC.md §4."""

    slope: float
    intercept: float
    slope_err: float
    intercept_err: float
    mswd: float
    chi2: float
    chi2_p: float
    n: int
    dof: int
    x_bar: float
    y_bar: float
    n_iter: int
    converged: bool
    mswd_is_constrained: bool
    n_eff: float
    error_model_description: str
    scale_errors: bool
    x: NDArray[np.float64] = field(repr=False)
    y: NDArray[np.float64] = field(repr=False)
    sx: NDArray[np.float64] = field(repr=False)
    sy: NDArray[np.float64] = field(repr=False)
    r: NDArray[np.float64] = field(repr=False)
    x_adj: NDArray[np.float64] = field(repr=False)
    y_adj: NDArray[np.float64] = field(repr=False)
    warnings: tuple[str, ...] = ()
    """Active diagnostic warnings (constrained MSWD, autocorrelation, LOD)."""

    # --- presentation -------------------------------------------------------

    def __repr__(self) -> str:
        slope = _fmt_pm(self.slope, self.slope_err)
        sign = "-" if self.intercept < 0 else "+"
        intercept = _fmt_pm(abs(self.intercept), self.intercept_err)
        mswd = (
            "MSWD=1 (constrained)"
            if self.mswd_is_constrained
            else f"MSWD={self.mswd:.3g}"
        )
        return f"YorkFit(y = {slope}x {sign} {intercept}, {mswd}, n={self.n})"

    def summary(self, level: float = 0.95) -> str:
        """Printable summary: parameters, MSWD, error model, tests, warnings."""
        _check_level(level)
        n_eff = "n/a" if math.isnan(self.n_eff) else f"{self.n_eff:.1f}"
        conv = (
            f"converged in {self.n_iter} iterations"
            if self.converged
            else f"NOT converged after {self.n_iter} iterations"
        )
        kind = "external (scaled by sqrt(MSWD))" if self.scale_errors else "internal"

        lines = [
            "York regression (errors in both variables)",
            f"  n = {self.n}   n_eff = {n_eff}   dof = {self.dof}   {conv}",
            "",
            "  parameters",
            f"    slope     = {self.slope:12.6g} ± {self.slope_err:.4g}",
            f"    intercept = {self.intercept:12.6g} ± {self.intercept_err:.4g}",
            f"    uncertainties are {kind}",
            "",
            "  goodness of fit",
        ]
        if self.mswd_is_constrained:
            lines.append(
                "    MSWD = 1 (constrained): sigmas were scaled to force MSWD = 1,"
            )
            lines.append(
                "    so MSWD and chi-square carry no goodness-of-fit information."
            )
        else:
            lines.append(
                f"    chi2 = {self.chi2:.4g}   MSWD = {self.mswd:.4g}   "
                f"p = {self.chi2_p:.3g}"
            )
        lines += ["", "  error model"]
        lines += [f"    {part}" for part in self.error_model_description.split("; ")]
        lines += [
            "",
            f"  hypothesis tests ({level:.0%} CI)",
            f"    {self.test_slope(1.0, level)}",
            f"    {self.test_intercept(0.0, level)}",
            "",
            "  warnings",
        ]
        if self.warnings:
            lines += [f"    - {w}" for w in self.warnings]
        else:
            lines.append("    (none)")
        return "\n".join(lines)

    # --- predictions --------------------------------------------------------

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        """Fitted line evaluated at ``x``."""
        xs = np.asarray(x, dtype=np.float64)
        return np.asarray(self.intercept + self.slope * xs, dtype=np.float64)

    def _weights(self) -> NDArray[np.float64]:
        var = (
            self.sy * self.sy
            + self.slope * self.slope * self.sx * self.sx
            - 2.0 * self.slope * self.r * self.sx * self.sy
        )
        return np.asarray(1.0 / var, dtype=np.float64)

    def confidence_band(
        self, x: ArrayLike, level: float = 0.95
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Lower and upper confidence bounds on the fitted line at ``x``.

        Uses ``var(y_hat) = 1/sum(W) + (x - x_adj_bar)^2 sigma_b^2``, which
        follows from York's expressions for ``sigma_a`` and ``cov(a, b)``.
        When ``scale_errors`` is set the whole variance is scaled by MSWD,
        consistent with the reported parameter uncertainties.
        """
        _check_level(level)
        xs = np.asarray(x, dtype=np.float64)
        W = self._weights()
        sum_W = float(np.sum(W))
        x_adj_bar = float(np.sum(W * self.x_adj) / sum_W)
        factor = self.mswd if self.scale_errors else 1.0
        var = factor / sum_W + (xs - x_adj_bar) ** 2 * self.slope_err**2
        half = _stats.t_ppf(0.5 * (1.0 + level), self.dof) * np.sqrt(var)
        line = self.predict(xs)
        return (
            np.asarray(line - half, dtype=np.float64),
            np.asarray(line + half, dtype=np.float64),
        )

    def plot(self, **kwargs: Any) -> Any:
        """Fit and residual panels; see :func:`york.plotting.plot`.

        Requires matplotlib (``pip install york[plot]``).
        """
        from york.plotting import plot

        return plot(self, **kwargs)

    def residuals(self) -> NDArray[np.float64]:
        """Weighted residuals, one per point; their squares sum to ``chi2``."""
        return weighted_residuals(
            self.x, self.y, self.sx, self.sy, self.r, self.slope, self.intercept
        )

    # --- hypothesis tests ---------------------------------------------------

    def _t_test(
        self, name: str, estimate: float, std_err: float, value: float, level: float
    ) -> HypothesisTest:
        _check_level(level)
        statistic = (estimate - value) / std_err
        p_value = float(_stats.t_sf_two_sided(statistic, self.dof))
        half = _stats.t_ppf(0.5 * (1.0 + level), self.dof) * std_err
        return HypothesisTest(
            name=name,
            null_value=value,
            estimate=estimate,
            std_err=std_err,
            statistic=statistic,
            p_value=p_value,
            ci_low=estimate - half,
            ci_high=estimate + half,
            level=level,
            dof=self.dof,
        )

    def test_slope(self, value: float = 1.0, level: float = 0.95) -> HypothesisTest:
        """Is the slope different from ``value``? Default null: slope = 1."""
        return self._t_test("slope", self.slope, self.slope_err, value, level)

    def test_intercept(self, value: float = 0.0, level: float = 0.95) -> HypothesisTest:
        """Is the intercept different from ``value``? Default null: intercept = 0."""
        return self._t_test(
            "intercept", self.intercept, self.intercept_err, value, level
        )
