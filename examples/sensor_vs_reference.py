"""Low-cost PM2.5 sensor co-located with a regulatory monitor: York vs OLS.

Simulates a month of hourly co-location between a beta-attenuation
reference monitor and a low-cost optical sensor, then fits the sensor
against the reference three ways: ordinary least squares in each direction,
and York regression with each instrument's stated uncertainty.

The data are simulated so that the true relationship is known. Run it:

    python examples/sensor_vs_reference.py [--plot out.png]

Parameters are chosen to be typical, not to flatter york:

- Hourly reference noise of 2.4 µg/m³ (one sigma) is the Met One BAM-1020
  hourly detection limit of 4.8 µg/m³ at two sigma. Check the current
  datasheet before relying on it.
- Hourly PM2.5 in a US city has a median near 9 µg/m³ and a standard
  deviation near 6 µg/m³, with strong hour-to-hour autocorrelation.
- Uncorrected low-cost optical sensors commonly over-read PM2.5 by tens of
  percent relative to FEM monitors (Barkjohn et al., 2021).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import york
from york.errors import InstrumentSpec

TRUE_SLOPE = 1.35
TRUE_INTERCEPT = 1.0
REF_SIGMA = 2.4  # µg/m³, hourly, one sigma
SENSOR_SPEC = InstrumentSpec(absolute=1.0, relative=0.10)


def simulate(hours: int = 24 * 30, seed: int = 2024):
    """Hourly truth, reference reading, and sensor reading."""
    rng = np.random.default_rng(seed)
    # Log-space AR(1) truth: median 9 µg/m³, geometric SD ~1.7, rho 0.9.
    rho, s = 0.9, np.log(1.7)
    z = np.empty(hours)
    z[0] = rng.normal()
    for i in range(1, hours):
        z[i] = rho * z[i - 1] + rng.normal(0.0, np.sqrt(1.0 - rho**2))
    truth = np.exp(np.log(9.0) + s * z)

    reference = truth + rng.normal(0.0, REF_SIGMA, hours)
    sensor_true = TRUE_INTERCEPT + TRUE_SLOPE * truth
    sensor = sensor_true + rng.normal(0.0, SENSOR_SPEC(sensor_true))
    return truth, reference, sensor


def ols(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def daily_means(*series, hours_per_day: int = 24):
    return [s.reshape(-1, hours_per_day).mean(axis=1) for s in series]


def fit_all(reference, sensor, ref_sigma, sensor_spec):
    b_yx, a_yx = ols(reference, sensor)
    b_xy, a_xy = ols(sensor, reference)  # reference on sensor, then invert
    yk = york.fit(reference, sensor, sx=ref_sigma, sy=sensor_spec)
    return {
        "OLS, sensor on reference": (b_yx, a_yx, None),
        "OLS, reference on sensor (inverted)": (1.0 / b_xy, -a_xy / b_xy, None),
        "York": (yk.slope, yk.intercept, yk),
    }


def report(title, reference, truth, fits):
    reliability = np.var(truth) / np.var(reference)
    print(f"\n{title}")
    print(
        f"  n = {reference.size}   "
        f"reliability ratio var(truth)/var(reference) = {reliability:.2f}"
    )
    print(f"  {'method':38s} {'slope':>8s} {'intercept':>10s}")
    print(f"  {'truth':38s} {TRUE_SLOPE:8.3f} {TRUE_INTERCEPT:10.2f}")
    for name, (b, a, yk) in fits.items():
        extra = ""
        if yk is not None:
            t = yk.test_slope(1.0)
            extra = (
                f"   ±{yk.slope_err:.3f}   MSWD {yk.mswd:.2f}   "
                f"slope≠1: p = {t.p_value:.2g}"
            )
        print(f"  {name:38s} {b:8.3f} {a:10.2f}{extra}")
    print(
        f"  expected OLS slope from attenuation alone: {TRUE_SLOPE * reliability:.3f}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", metavar="PNG", help="save a figure to this path")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args(argv)

    truth, reference, sensor = simulate(seed=args.seed)
    hourly = fit_all(reference, sensor, REF_SIGMA, SENSOR_SPEC)
    report("Hourly data", reference, truth, hourly)

    # Averaging 24 independent hours shrinks random precision by sqrt(24) on
    # both instruments. It does nothing to a calibration bias, which is why
    # the slope and intercept come out the same.
    truth_d, reference_d, sensor_d = daily_means(truth, reference, sensor)
    k = np.sqrt(24)
    daily_spec = InstrumentSpec(SENSOR_SPEC.absolute / k, SENSOR_SPEC.relative / k)
    daily = fit_all(reference_d, sensor_d, REF_SIGMA / k, daily_spec)
    report("Daily means of the same data", reference_d, truth_d, daily)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from york.plotting import plot_fit

        yk = hourly["York"][2]
        fig, ax = plt.subplots(figsize=(6.5, 6))
        plot_fit(yk, ax, one_to_one=True, band=True)
        grid = np.linspace(0, reference.max(), 2)
        b, a, _ = hourly["OLS, sensor on reference"]
        ax.plot(
            grid,
            a + b * grid,
            "-",
            color="C3",
            lw=1.5,
            label=f"OLS: y = {b:.2f}x + {a:.1f}",
        )
        ax.plot(
            grid,
            TRUE_INTERCEPT + TRUE_SLOPE * grid,
            ":",
            color="k",
            lw=1.5,
            label=f"truth: y = {TRUE_SLOPE:.2f}x + {TRUE_INTERCEPT:.1f}",
        )
        ax.set_xlabel("reference PM2.5, hourly (µg/m³)")
        ax.set_ylabel("sensor PM2.5, hourly (µg/m³)")
        ax.legend(loc="upper left", fontsize="small")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nfigure written to {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
