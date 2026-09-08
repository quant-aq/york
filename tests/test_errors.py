"""Error models (SPEC.md §5, §7.7). Each model against hand-computed values."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import york
from york import errors


def test_instrument_spec_formula():
    m = errors.InstrumentSpec(absolute=0.5, relative=0.05)
    got = m(np.array([0.0, 10.0, 100.0]))
    expected = [0.5, np.sqrt(0.25 + 0.25), np.sqrt(0.25 + 25.0)]
    np.testing.assert_allclose(got, expected)


def test_instrument_spec_handles_negative_readings_symmetrically():
    m = errors.InstrumentSpec(absolute=0.5, relative=0.05)
    np.testing.assert_allclose(m(np.array([-10.0])), m(np.array([10.0])))


def test_poisson_formula():
    # sigma of a rate N/t is sqrt(N)/t. With rate 100/s over 1 s, N = 100.
    np.testing.assert_allclose(
        errors.Poisson(interval=1.0)(np.array([100.0, 400.0])), [10.0, 20.0]
    )
    # Same rate over 4 s: N = 400, sigma = 20 / 4 = 5.
    np.testing.assert_allclose(errors.Poisson(interval=4.0)(np.array([100.0])), [5.0])


def test_poisson_negative_rate_is_treated_as_zero():
    np.testing.assert_allclose(errors.Poisson(interval=1.0)(np.array([-3.0])), [0.0])


def test_replicate_returns_measured_sd():
    sd = np.array([0.1, 0.2, 0.3])
    m = errors.Replicate(sd)
    np.testing.assert_array_equal(m(np.array([5.0, 6.0, 7.0])), sd)


def test_replicate_length_mismatch_raises():
    m = errors.Replicate(np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="length"):
        m(np.array([1.0, 2.0, 3.0]))


def test_relative_uses_magnitude():
    np.testing.assert_allclose(
        errors.Relative(0.05)(np.array([10.0, -20.0, 0.0])), [0.5, 1.0, 0.0]
    )


def test_absolute_is_constant():
    np.testing.assert_array_equal(
        errors.Absolute(0.3)(np.array([1.0, 100.0, -5.0])), [0.3, 0.3, 0.3]
    )


def test_rolling_std_local_and_fallback():
    values = np.array([1.0, 2.0, 4.0, 4.0, 4.0, 10.0, 12.0])
    m = errors.RollingStd(window=3)
    got = m(values)
    global_sd = np.std(values, ddof=1)
    # Centered window of 3: interior points use their neighbours.
    assert got[1] == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))
    assert got[2] == pytest.approx(np.std([2.0, 4.0, 4.0], ddof=1))
    # A flat window has zero local SD: fall back to the global SD.
    assert got[3] == pytest.approx(global_sd)
    # Edges have no full window: fall back to the global SD.
    assert got[0] == pytest.approx(global_sd)
    assert got[-1] == pytest.approx(global_sd)


def test_rolling_std_window_larger_than_series_is_global():
    values = np.array([1.0, 3.0, 2.0])
    got = errors.RollingStd(window=10)(values)
    np.testing.assert_allclose(got, np.full(3, np.std(values, ddof=1)))


def test_rolling_std_rejects_bad_window():
    with pytest.raises(ValueError):
        errors.RollingStd(window=1)


@pytest.mark.parametrize(
    "model",
    [
        errors.Relative(0.05, min_sigma=0.2),
        errors.InstrumentSpec(absolute=0.0, relative=0.05, min_sigma=0.2),
        errors.Poisson(interval=1.0, min_sigma=0.2),
        errors.Absolute(0.01, min_sigma=0.2),
    ],
)
def test_min_sigma_floor_is_enforced(model):
    got = model(np.array([0.0, 1.0, 100.0]))
    assert np.all(got >= 0.2)
    assert got[0] == 0.2


def test_min_sigma_default_is_zero_and_zero_sigma_is_rejected_by_fit():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = x + 0.1
    with pytest.raises(ValueError, match="strictly positive"):
        york.fit(x, y, sx=errors.Relative(0.05), sy=0.1)


def test_describe_is_informative():
    assert "0.5" in errors.InstrumentSpec(0.5, 0.05).describe()
    assert "5%" in errors.InstrumentSpec(0.5, 0.05).describe()
    assert "min_sigma" in errors.Relative(0.05, min_sigma=0.2).describe()
    for m in (
        errors.Poisson(1.0),
        errors.Replicate(np.ones(3)),
        errors.Relative(0.1),
        errors.Absolute(0.1),
        errors.RollingStd(5),
    ):
        assert isinstance(m.describe(), str) and m.describe()


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        errors.InstrumentSpec(absolute=-1.0, relative=0.05)
    with pytest.raises(ValueError):
        errors.Relative(-0.1)
    with pytest.raises(ValueError):
        errors.Absolute(0.0)
    with pytest.raises(ValueError):
        errors.Poisson(interval=0.0)
    with pytest.raises(ValueError):
        errors.Replicate(np.array([0.1, -0.2]))


def test_reference_specs_registry():
    for key in ("teom", "bam-1020", "t640", "tsi-3783"):
        assert isinstance(errors.REFERENCE_SPECS[key], errors.InstrumentSpec)


def test_error_model_and_equivalent_scalar_give_same_fit(pearson_york):
    d = pearson_york
    a = york.fit(d.x, d.y, sx=0.3, sy=0.5)
    b = york.fit(d.x, d.y, sx=errors.Absolute(0.3), sy=errors.Absolute(0.5))
    assert a.slope == b.slope
    assert a.slope_err == b.slope_err
    assert "Absolute" in b.error_model_description
    assert errors.Absolute(0.3).describe() in b.error_model_description


def test_independent_models_per_axis(pearson_york):
    d = pearson_york
    fit = york.fit(
        d.x,
        d.y,
        sx=errors.InstrumentSpec(0.2, 0.05),
        sy=errors.Replicate(np.full(d.x.size, 0.4)),
    )
    assert "InstrumentSpec" in fit.error_model_description
    assert "Replicate" in fit.error_model_description
    assert fit.converged


def test_lod_floor_warning_fires_when_many_values_hit_floor():
    rng = np.random.default_rng(1)
    n = 50
    x = np.concatenate([np.zeros(20), rng.uniform(5, 50, n - 20)])
    y = 0.9 * x + rng.normal(0, 0.5, n)
    with pytest.warns(UserWarning, match="min_sigma floor"):
        fit = york.fit(x, y, sx=errors.Relative(0.05, min_sigma=0.1), sy=0.5)
    assert any("floor" in w for w in fit.warnings)


def test_custom_error_model_subclass_is_first_class(pearson_york):
    """The subclassing contract from the README: implement _sigma and describe."""
    from dataclasses import dataclass

    d = pearson_york
    rh = np.linspace(20.0, 90.0, d.x.size)

    @dataclass(frozen=True, eq=False)
    class HumidityDependent(errors.ErrorModel):
        base: float
        per_rh: float
        rh: np.ndarray
        min_sigma: float = 0.0

        def _sigma(self, values):
            return self.base + self.per_rh * self.rh

        def describe(self):
            return f"HumidityDependent(±{self.base} + {self.per_rh} per %RH)"

    model = HumidityDependent(base=0.1, per_rh=0.01, rh=rh)
    expected = 0.1 + 0.01 * rh
    np.testing.assert_allclose(model(d.x), expected)

    # Explicit array and custom model give identical fits, and the model's
    # description reaches the result.
    via_model = york.fit(d.x, d.y, sx=model, sy=d.sy)
    via_array = york.fit(d.x, d.y, sx=expected, sy=d.sy)
    assert via_model.slope == via_array.slope
    assert via_model.slope_err == via_array.slope_err
    assert model.describe() in via_model.error_model_description
    assert model.describe() in via_model.summary()

    # The floor is applied by the base class without the subclass doing anything.
    floored = HumidityDependent(base=0.0, per_rh=0.001, rh=rh, min_sigma=0.5)
    assert np.all(floored(d.x) >= 0.5)


def test_custom_model_returning_scalar_is_broadcast():
    class Flat(errors.ErrorModel):
        def _sigma(self, values):
            return 0.25

        def describe(self):
            return "Flat"

    np.testing.assert_array_equal(Flat()(np.zeros(4)), np.full(4, 0.25))


def test_no_lod_warning_when_floor_rarely_hit():
    rng = np.random.default_rng(2)
    x = rng.uniform(5, 50, 40)
    y = 0.9 * x + rng.normal(0, 0.5, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        york.fit(x, y, sx=errors.Relative(0.05, min_sigma=0.1), sy=0.5)
