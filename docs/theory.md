# Theory

Enough of the mathematics to read the code and know what the numbers mean.
For the derivation see York et al. (2004); for the clearest presentation
of the algorithm see Wehr and Saleska (2017).

## The problem

Each observation is a pair `(x_i, y_i)` with known one-sigma uncertainties
`sx_i` and `sy_i`, and optionally a correlation `r_i` between the two
errors. We want the line `y = a + b x` that maximises the likelihood of the
data under Gaussian errors.

Ordinary least squares minimises the vertical residuals and assumes `sx = 0`.
When x has error, OLS biases the slope toward zero (attenuation) and the
intercept away from it. York's solution treats both variables
symmetrically: swapping x and y gives exactly the reciprocal slope.

## The solution

Define the per-point weights `wx = 1 / sx²` and `wy = 1 / sy²`, and their
geometric mean `alpha = sqrt(wx wy)`. For a trial slope `b`, the combined
weight of each point is

```
W = wx wy / (wx + b² wy - 2 b r alpha)
```

This is the inverse variance of the residual `y - a - b x` once the error
in x is projected onto it. With `x̄` and `ȳ` the W-weighted means and
`U = x - x̄`, `V = y - ȳ`, York's update is

```
beta = W [ U / wy + b V / wx - (b U + V) r / alpha ]
b_new = sum(W beta V) / sum(W beta U)
```

Iterate until `b` stops changing. Convergence takes a handful of steps on
sensible data. The intercept is `a = ȳ - b x̄`.

## Uncertainties

The adjusted points `x_adj = x̄ + beta` are where each observation would
sit on the fitted line. With `u = x_adj - mean_W(x_adj)`,

```
sigma_b² = 1 / sum(W u²)
sigma_a² = 1 / sum(W) + mean_W(x_adj)² sigma_b²
cov(a, b) = -mean_W(x_adj) sigma_b²
```

These are the *internal* uncertainties: what the stated sigmas imply. The
confidence band on the line at any `x` follows from the same covariance:

```
var(ŷ) = 1 / sum(W) + (x - mean_W(x_adj))² sigma_b²
```

## Goodness of fit

The weighted sum of squared residuals

```
chi² = sum( W (y - a - b x)² )
MSWD = chi² / (n - 2)
```

is chi-square distributed with `n - 2` degrees of freedom if the sigmas are
right and the line is the right model. MSWD (mean square weighted
deviation, the geochronologists' name for reduced chi-square) is therefore
about 1 for a good fit. `scale_errors=True` multiplies the parameter
uncertainties by `sqrt(MSWD)`, which is the standard way to say "believe
the scatter rather than the stated sigmas".

## Special cases

All of these fall out of the same equations by choosing the weights:

| Weights | Result |
|---|---|
| `wx → ∞` | Ordinary least squares, y on x |
| `wy → ∞` | Ordinary least squares, x on y |
| `wx = wy` | Orthogonal (total least squares) regression |
| `wy / wx` constant | Deming regression with that variance ratio |

The test suite checks each of these against an independent implementation.

## Hypothesis tests

`test_slope(value)` and `test_intercept(value)` form the statistic
`(estimate - value) / sigma` and compare it with Student's t on `n - 2`
degrees of freedom, two-sided. The confidence interval uses the same t
quantile. Both use the uncertainties as reported, so `scale_errors` changes
them.

## Effective sample size

If the residuals follow a first-order autoregressive process with lag-1
correlation `rho`, the number of effectively independent points is

```
n_eff = n (1 - rho) / (1 + rho)
```

`fit` estimates `rho` from the weighted residuals in the order given and
warns when `n_eff` falls below half of `n` and `rho` is outside the
white-noise band `±1.96 / sqrt(n)`. The parameter uncertainties are not
adjusted, because the right adjustment depends on the error structure; the
warning is there so the numbers are not oversold.
