"""The shipped examples run, and their headline claims hold."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def sensor_example():
    return runpy.run_path(str(EXAMPLES / "sensor_vs_reference.py"))


def test_sensor_vs_reference_claims(sensor_example):
    ns = sensor_example
    truth, reference, sensor = ns["simulate"]()
    fits = ns["fit_all"](reference, sensor, ns["REF_SIGMA"], ns["SENSOR_SPEC"])

    ols_slope = fits["OLS, sensor on reference"][0]
    york_slope, _, result = fits["York"]
    reliability = np.var(truth) / np.var(reference)

    # OLS is attenuated by roughly the reliability ratio; York is not.
    assert ols_slope == pytest.approx(ns["TRUE_SLOPE"] * reliability, rel=0.05)
    assert ols_slope < ns["TRUE_SLOPE"] - 4 * result.slope_err
    assert abs(york_slope - ns["TRUE_SLOPE"]) < 2 * result.slope_err
    # Stated precisions are consistent with the scatter.
    assert 0.8 < result.mswd < 1.25


def test_sensor_vs_reference_main_runs(sensor_example, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "fig.png"
    assert sensor_example["main"](["--plot", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 0
    text = capsys.readouterr().out
    assert "Hourly data" in text and "Daily means" in text
