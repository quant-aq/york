# Guide

A walkthrough of a typical instrument intercomparison, from choosing error
models to reading the result.

## 1. State what you know about each instrument

Every point carries its own one-sigma uncertainty on each axis, and the two
axes are specified independently. Start from whatever you actually know.

**You have a manufacturer spec.** Most datasheets quote precision as an
absolute term plus a percentage of reading. That maps directly onto
`InstrumentSpec`:

```python
from york.errors import InstrumentSpec

ref = InstrumentSpec(absolute=0.5, relative=0.05)   # ±(0.5 + 5% of reading)
cand = InstrumentSpec(absolute=1.0, relative=0.10)
```

**Your data are averages with a reported SD.** Hourly means from a
one-minute instrument usually come with the within-hour standard deviation.
That is a measurement of sigma, not an estimate, so prefer it:

```python
from york.errors import Replicate

ref = Replicate(df["ref_sd"].to_numpy())
```

**You have a counting instrument.** A CPC reports a rate; its uncertainty
is Poisson in the counts:

```python
from york.errors import Poisson

cand = Poisson(interval=60.0)   # counts accumulated over 60 s
```

**You have nothing.** Leave `sx` and `sy` out. `york` estimates the ratio of
the two sigmas from local variability and forces MSWD to 1. The slope stays
meaningful; the goodness-of-fit test does not exist. See
[the default policy](#when-you-do-not-know-the-uncertainties).

## 2. Fit

```python
import york

result = york.fit(reference, candidate, sx=ref, sy=cand)
print(result)
# YorkFit(y = 0.92(±0.02)x + 1.5(±0.6), MSWD=1.03, n=120)
```

`sx` and `sy` also accept a scalar or a per-point array, and you can write
your own model class. See [custom uncertainties](index.md#custom-uncertainties).

## 3. Read the summary

```python
print(result.summary())
```

```
York regression (errors in both variables)
  n = 120   n_eff = 120.0   dof = 118   converged in 6 iterations

  parameters
    slope     =     0.921654 ± 0.01823
    intercept =      1.48762 ± 0.5964
    uncertainties are internal

  goodness of fit
    chi2 = 121.4   MSWD = 1.029   p = 0.398

  error model
    sx: InstrumentSpec(±0.5 + 5% of reading)
    sy: InstrumentSpec(±1 + 10% of reading)

  hypothesis tests (95% CI)
    slope = 1: t = -4.298, p = 3.55e-05, 95% CI [0.8856, 0.9578]
    intercept = 0: t = 2.494, p = 0.014, 95% CI [0.3066, 2.669]

  warnings
    (none)
```

Read it top to bottom:

- **Convergence** in a handful of iterations is normal. Reaching the cap
  means something is badly wrong with the inputs.
- **MSWD** (the reduced chi-square) near 1 means the scatter matches the
  stated uncertainties. Well above 1 means the sigmas are too small or the
  relationship is not a straight line. Well below 1 means the sigmas are
  too large. The p-value is the chi-square probability of the observed
  scatter given the sigmas.
- **The error model** is recorded so the fit is self-documenting.
- **The hypothesis tests** answer the actual questions. Here the slope is
  significantly below 1 and the intercept is significantly above 0: the
  candidate reads low by about 8 percent with a small positive offset.

## 4. Decide whether to scale the uncertainties

By default the parameter uncertainties are the *internal* ones, propagated
from the sigmas you stated. If MSWD is far from 1 those sigmas are wrong,
and so are the internal uncertainties.

`scale_errors=True` multiplies them by the square root of MSWD. This is
what `scipy.odr` reports and what many geochronology packages call
"external" errors. It answers "given the scatter I actually observed, how
well do I know the slope?" rather than "given the precision I claimed".

The choice is explicit because the two answers differ whenever your error
model is imperfect, which is always.

## 5. Look at the residuals

```python
result.plot(xlabel="reference", ylabel="candidate")
```

The top panel is the data with error bars, the fitted line, its confidence
band, and the 1:1 line. The bottom panel is the weighted residuals: each
residual divided by its own combined sigma, so about 95 percent should fall
within the dotted ±2 lines if the error model is right.

Look for:

- **A trend** in the residuals: the relationship is not a straight line.
- **Fanning** with x: the relative term in the error model is wrong.
- **A cluster at low x** outside the band: points near the detection limit
  have too much weight. Set `min_sigma` or use an absolute term.

## 6. Watch the diagnostics

`fit` raises `UserWarning` and records the message on `result.warnings`
whenever something undermines the numbers:

- **Autocorrelation.** Hourly series are strongly autocorrelated, so the
  effective number of independent points is below n. `fit` estimates it
  from the lag-1 autocorrelation of the residuals, reports it as `n_eff`,
  and warns when it drops below half of n. Reported uncertainties assume
  independence; treat them as lower bounds when this fires.
- **Detection-limit floor.** When more than five percent of the sigmas from
  an error model are held at `min_sigma`, the fit is being shaped by the
  floor rather than by the instrument spec.
- **Constrained MSWD.** The default policy fabricated the sigma scale, so
  MSWD carries no information.

## When you do not know the uncertainties

```python
result = york.fit(reference, candidate)
# UserWarning: sx and sy were not given. Estimated sigma_x / sigma_y = 0.42
# from the median absolute successive difference of each series, then
# scaled both so that MSWD = 1 ...
```

The slope of a York fit depends only on the *ratio* of the two sigmas, so
estimating the ratio and fixing the scale by MSWD = 1 gives a defensible
slope and a defensible slope uncertainty. What you lose is any ability to
say whether the scatter is consistent with the instruments, because the
scale was chosen to make it so. The result carries
`mswd_is_constrained=True` and the summary says so in place of the
goodness-of-fit line.

Giving one sigma but not the other is ambiguous and raises `ValueError`.
Pass a small positive value for a negligible uncertainty.
