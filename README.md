# york

Straight-line fitting when **both** x and y have measurement uncertainty
(York 2004), built for instrument intercomparison.

```python
import york
from york.errors import InstrumentSpec

# Reference monitor: ±(0.5 µg/m³ + 5% of reading). Candidate: ±(1.0 + 10%).
result = york.fit(
    reference, candidate,
    sx=InstrumentSpec(absolute=0.5, relative=0.05),
    sy=InstrumentSpec(absolute=1.0, relative=0.10),
)
print(result)
# YorkFit(y = 0.94(±0.02)x + 1.3(±0.6), MSWD=1.1, n=847)

print(result.test_slope(1.0))       # is the slope different from 1?
print(result.test_intercept(0.0))   # is the intercept different from 0?
print(result.summary())
```

Ordinary least squares assumes x is exact. When you compare two instruments
that both have real uncertainty, that assumption biases the slope toward
zero and understates the intercept. York's method is the maximum-likelihood
straight line when every point has its own uncertainty in both variables,
optionally correlated. Orthogonal, Deming, and weighted least squares are
all special cases of it.

`york` is aimed at the questions a method-comparison study actually asks:
is the slope 1, is the intercept 0, and is the scatter consistent with the
stated instrument precision?

## Why not OLS?

A low-cost PM2.5 sensor co-located with a beta-attenuation monitor for a
month, simulated so the truth is known. The sensor reads 35 percent high.
Hourly reference noise of 2.4 µg/m³ against a concentration spread of about
6 µg/m³ is enough to pull OLS well off:

| Method | Slope | Intercept |
|---|---|---|
| Truth | 1.35 | 1.0 |
| OLS, sensor on reference | 1.17 | 2.8 |
| OLS, reference on sensor, inverted | 1.49 | −0.6 |
| York, with each instrument's precision | 1.37 ± 0.03 | 0.6 |

OLS assumes the reference is exact. When it isn't, the slope is attenuated
by the reliability ratio `var(true x) / (var(true x) + σx²)`, a bias that
more data makes more precisely wrong. York treats both instruments as
imperfect and recovers the line. OLS remains the right tool when the goal
is to *predict* reference values from future sensor readings.

> **EPA guidance recommends OLS.** The *Air Sensor Guidebook* and the NSIM
> performance-targets reports evaluate sensors by OLS of sensor against a
> co-located FRM/FEM monitor, with PM2.5 targets of slope 1.0 ± 0.35,
> intercept within ±5 µg/m³ and R² ≥ 0.70 on 24-hour averages. That is
> defensible for what it is: at daily averaging the reference noise is
> small relative to the spread of the data, so attenuation is a few
> percent, and the targets are loose screening criteria that need no
> uncertainty model. It stops being defensible on hourly data, at clean
> sites, or between instruments of similar precision, and the OLS slope
> should never be reported as the sensor's response. Compute both: OLS
> where the protocol asks, York for what the sensor actually does.

[Sensor versus reference: York or OLS?](https://quant-aq.github.io/york/sensor-vs-reference/)
walks through the example with sources, and
[`examples/sensor_vs_reference.py`](https://github.com/quant-aq/york/blob/main/examples/sensor_vs_reference.py) reproduces it.

## Install

```
pip install york
```

The only runtime dependency is numpy. Optional extras: `york[plot]` for
matplotlib, `york[pandas]` for pandas helpers.

## Error models

Uncertainties are objects, not strings. `sx` and `sy` are specified
independently because two instruments rarely share an error structure.

| Model | Sigma | Use when |
|---|---|---|
| `InstrumentSpec(absolute, relative)` | `sqrt(a² + (b·x)²)` | You have a manufacturer spec. Start here. |
| `Replicate(sd)` | measured SD | Averaged data already carries a within-window SD. Prefer this when available. |
| `Poisson(interval)` | `sqrt(N) / t` | Counting instruments (CPCs). |
| `Relative(b)` | `b·|x|` | Fractional precision only. Set `min_sigma`. |
| `Absolute(a)` | `a` | Fixed precision. |
| `RollingStd(window)` | local SD, global fallback | Unknown error structure. Last resort. |

`sx` and `sy` also accept a scalar or a per-point array. Every model takes a
`min_sigma` floor to stop relative errors collapsing to zero near the
detection limit; `fit` warns when a non-trivial fraction of points sit on
the floor.

`york.errors.REFERENCE_SPECS` holds starting-point specs for a few common
reference monitors. They are transcribed from datasheets, are not
authoritative, and should be overridden with your own numbers.

### Custom uncertainties

The built-in models cover the common cases, but you are never limited to
them. There are three ways to supply your own.

**A per-point array.** Compute sigma however you like and pass the array.
This is the simplest escape hatch and needs no library machinery:

```python
sigma_ref = np.sqrt(0.5**2 + (0.05 * reference) ** 2 + (0.02 * rh) ** 2)
result = york.fit(reference, candidate, sx=sigma_ref, sy=0.8)
```

**A scalar.** A plain number is a constant absolute uncertainty:

```python
result = york.fit(reference, candidate, sx=0.5, sy=1.2)
```

**Your own model class.** Subclass `ErrorModel`, implement `_sigma` and
`describe`, and the result works everywhere a built-in model does: the
`min_sigma` floor is applied for you, the detection-limit warning fires when
too many points hit it, and `describe()` is printed in `summary()` so the
fit records how its weights were chosen.

```python
from dataclasses import dataclass
import numpy as np
from york.errors import ErrorModel

@dataclass(frozen=True)
class HumidityDependent(ErrorModel):
    """Sigma grows with relative humidity: base + per_rh * RH."""

    base: float
    per_rh: float
    rh: np.ndarray
    min_sigma: float = 0.0

    def _sigma(self, values):
        return self.base + self.per_rh * self.rh

    def describe(self):
        return f"HumidityDependent(±{self.base} + {self.per_rh} per %RH)"

result = york.fit(
    reference, candidate,
    sx=HumidityDependent(base=0.5, per_rh=0.02, rh=rh),
    sy=InstrumentSpec(absolute=1.0, relative=0.10),
)
```

`_sigma` receives the data as a float array and must return something
broadcastable to the same shape. Anything beyond that, such as parameter
validation in `__post_init__` or a length check against side data like `rh`
above, is up to you. Models are ordinary Python objects, so they can be
built from configuration, stored, and reused across fits.

If a custom sigma can reach zero, set `min_sigma` or `fit` will raise
rather than silently give those points infinite weight.

### If you do not know the uncertainties

Calling `fit(x, y)` with no `sx` or `sy` estimates the *ratio* of the two
sigmas from local variability and scales both so that MSWD = 1. The slope
and its uncertainty remain meaningful because they depend only on the ratio.
What is lost is any goodness-of-fit test, because MSWD was forced to 1. The
result carries `mswd_is_constrained=True`, a `UserWarning` says exactly what
was done, and `summary()` refuses to present the MSWD as evidence of fit.

Giving one sigma but not the other is ambiguous and raises `ValueError`.

## What you get back

`YorkFit` is a frozen dataclass with `slope`, `intercept`, their standard
errors, `mswd`, `chi2`, `chi2_p`, `n_eff`, the error model description, and
any active warnings. Methods:

- `test_slope(value=1.0, level=0.95)` and `test_intercept(value=0.0, level=0.95)`
  return the t statistic, two-sided p-value, and confidence interval.
- `predict(x)`, `confidence_band(x, level)`, and `residuals()` (weighted, so
  their squares sum to `chi2`).
- `summary()` prints all of the above plus the error model and warnings.

### Plotting

With the `[plot]` extra installed, `result.plot()` draws the data with
error bars, the fitted line and its confidence band, the 1:1 line, and a
residual panel underneath. `york.plotting.plot_fit` and
`york.plotting.plot_residuals` draw the panels separately onto any
matplotlib axes you give them.

```python
fig = result.plot(xlabel="reference PM2.5 (µg/m³)", ylabel="candidate PM2.5 (µg/m³)")
fig.savefig("intercomparison.png", dpi=150)
```

### scikit-learn-style estimator

`YorkRegressor` wraps `fit` in the `fit` / `predict` / `score` /
`get_params` / `set_params` protocol, so it works with `cross_val_score`,
`GridSearchCV` and pipelines. scikit-learn is not a dependency; the
protocol is implemented by hand.

```python
from york import YorkRegressor

est = YorkRegressor(sx=InstrumentSpec(0.5, 0.05), sy=InstrumentSpec(1.0, 0.10))
est.fit(reference, candidate)
est.slope_, est.slope_err_, est.mswd_
est.result_.test_slope(1.0)     # the full YorkFit is on result_
```

`score` is R², which is there for scikit-learn's benefit rather than yours;
the method-comparison answers are on `result_`. One caution: a transformer
in front of the estimator changes the units of `X` while `sx` stays in the
original units, so a `StandardScaler` in a pipeline silently makes the x
uncertainties wrong unless you rescale `sx` to match.

### Internal versus external uncertainties

By default the reported parameter uncertainties are the *internal* ones
implied by your stated sigmas. If MSWD is well above 1 your sigmas are too
small or the model is wrong. Pass `scale_errors=True` to multiply the
uncertainties by `sqrt(MSWD)`, which is what `scipy.odr` reports. The choice
is explicit because the two answer different questions.

## Diagnostics

Hourly air-quality series are strongly autocorrelated, so the effective
number of independent points is well below n. `fit` computes the lag-1
autocorrelation of the weighted residuals, reports `n_eff`, and warns when
it falls below half of n and the autocorrelation is statistically
significant. Reported uncertainties assume independent points; treat them
as lower bounds when this warning fires.

## Validation

- The Pearson-York ten-point benchmark reproduces slope −0.480533 and
  intercept 5.479910 with the published standard errors, both internal and
  MSWD-scaled.
- Limiting cases are checked against independent implementations: OLS in
  both directions, orthogonal regression via SVD, Deming via its closed form,
  and heteroscedastic data against `scipy.odr`.
- Swapping x and y gives the reciprocal slope and the correctly transformed
  slope error to machine precision, including with correlated errors.
- Hypothesis property tests cover affine equivariance, convergence, and
  sigma rescaling.
- The hand-rolled t and chi-square distribution functions agree with scipy
  to 1e-10.


## References

- York, D., Evensen, N. M., Martínez, M. L., De Basabe Delgado, J. (2004).
  Unified equations for the slope, intercept, and standard errors of the best
  straight line. *American Journal of Physics* 72(3). doi:10.1119/1.1632486
- Wehr, R. & Saleska, S. R. (2017). The long-solved problem of the best-fit
  straight line: application to isotopic mixing lines. *Biogeosciences* 14(1).

## License

MIT
