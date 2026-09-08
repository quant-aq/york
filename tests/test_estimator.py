"""``YorkRegressor`` (SPEC.md §3.2).

The estimator protocol is hand-written; scikit-learn is imported only in
this file, to prove the duck-typing works with its real tooling.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

import york
from york import errors
from york.estimator import NotFittedError, YorkRegressor


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(12)
    n = 80
    x = np.linspace(1.0, 40.0, n) + rng.normal(0.0, 0.4, n)
    y = 2.0 + 0.85 * x + rng.normal(0.0, 1.0, n)
    return x, y


# --- protocol (no scikit-learn needed) -------------------------------------


def test_fitted_attributes_match_functional_api(data):
    x, y = data
    est = YorkRegressor(sx=0.4, sy=1.0).fit(x, y)
    ref = york.fit(x, y, sx=0.4, sy=1.0)
    assert est.slope_ == ref.slope
    assert est.intercept_ == ref.intercept
    assert est.slope_err_ == ref.slope_err
    assert est.intercept_err_ == ref.intercept_err
    assert est.mswd_ == ref.mswd
    assert est.n_iter_ == ref.n_iter
    assert est.converged_ is ref.converged
    assert est.result_.slope == ref.slope
    assert est.n_features_in_ == 1


def test_fit_returns_self(data):
    x, y = data
    est = YorkRegressor(sx=0.4, sy=1.0)
    assert est.fit(x, y) is est


def test_accepts_1d_and_column_vector(data):
    x, y = data
    a = YorkRegressor(sx=0.4, sy=1.0).fit(x, y)
    b = YorkRegressor(sx=0.4, sy=1.0).fit(x.reshape(-1, 1), y)
    assert a.slope_ == b.slope_
    np.testing.assert_array_equal(a.predict(x[:5]), b.predict(x[:5].reshape(-1, 1)))


def test_accepts_column_vector_y(data):
    x, y = data
    a = YorkRegressor(sx=0.4, sy=1.0).fit(x, y)
    b = YorkRegressor(sx=0.4, sy=1.0).fit(x, y.reshape(-1, 1))
    assert a.slope_ == b.slope_


def test_rejects_multiple_features(data):
    x, y = data
    X = np.column_stack([x, x**2])
    with pytest.raises(ValueError, match="exactly 1 feature"):
        YorkRegressor(sx=0.4, sy=1.0).fit(X, y)


def test_rejects_inconsistent_lengths(data):
    x, y = data
    with pytest.raises(ValueError, match="inconsistent"):
        YorkRegressor(sx=0.4, sy=1.0).fit(x, y[:-1])


def test_predict_and_score(data):
    x, y = data
    est = YorkRegressor(sx=0.4, sy=1.0).fit(x, y)
    pred = est.predict(x)
    np.testing.assert_allclose(pred, est.intercept_ + est.slope_ * x)
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert est.score(x, y) == pytest.approx(r2)
    assert est.score(x, y) > 0.99


def test_unfitted_predict_raises(data):
    x, _ = data
    with pytest.raises(NotFittedError):
        YorkRegressor().predict(x)
    with pytest.raises(NotFittedError):
        YorkRegressor().score(x, x)
    # NotFittedError is both, so either handler style works.
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(NotFittedError, AttributeError)


def test_predict_checks_feature_count(data):
    x, y = data
    est = YorkRegressor(sx=0.4, sy=1.0).fit(x, y)
    with pytest.raises(ValueError, match="expecting 1 feature"):
        est.predict(np.column_stack([x, x]))


def test_get_and_set_params_round_trip():
    model = errors.InstrumentSpec(0.5, 0.05)
    est = YorkRegressor(sx=model, sy=0.3, r=0.1, scale_errors=True, tol=1e-9)
    params = est.get_params()
    assert params == {
        "sx": model,
        "sy": 0.3,
        "r": 0.1,
        "scale_errors": True,
        "tol": 1e-9,
        "max_iter": 200,
    }
    assert est.set_params(max_iter=50, scale_errors=False) is est
    assert est.max_iter == 50 and est.scale_errors is False
    with pytest.raises(ValueError, match="Invalid parameter"):
        est.set_params(bogus=1)


def test_repr_shows_params():
    r = repr(YorkRegressor(sx=0.4, sy=1.0))
    assert r.startswith("YorkRegressor(")
    assert "sx=0.4" in r and "sy=1.0" in r


def test_error_models_as_params(data):
    x, y = data
    est = YorkRegressor(
        sx=errors.InstrumentSpec(0.3, 0.01), sy=errors.Absolute(1.0)
    ).fit(x, y)
    assert "InstrumentSpec" in est.result_.error_model_description


def test_default_policy_warns_through_estimator(data):
    x, y = data
    with pytest.warns(UserWarning, match="MSWD was forced to 1"):
        est = YorkRegressor().fit(x, y)
    assert est.result_.mswd_is_constrained


def test_scale_errors_param(data):
    x, y = data
    a = YorkRegressor(sx=0.4, sy=1.0).fit(x, y)
    b = YorkRegressor(sx=0.4, sy=1.0, scale_errors=True).fit(x, y)
    assert b.slope_err_ == pytest.approx(a.slope_err_ * np.sqrt(a.mswd_))


def test_hypothesis_tests_reachable_from_result(data):
    x, y = data
    est = YorkRegressor(sx=0.4, sy=1.0).fit(x, y)
    assert est.result_.test_slope(1.0).p_value < 0.05


# --- duck typing into scikit-learn ------------------------------------------

sklearn = pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def test_clone_and_pickle(data):
    x, y = data
    est = YorkRegressor(sx=0.4, sy=1.0)
    cloned = clone(est)
    assert cloned is not est and cloned.get_params() == est.get_params()
    est.fit(x, y)
    restored = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(restored.predict(x), est.predict(x))


def test_cross_val_score(data):
    x, y = data
    # Shuffle: the data are sorted by x, and R² inside a contiguous held-out
    # block is low for any estimator because the block has little variance.
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    scores = cross_val_score(YorkRegressor(sx=0.4, sy=1.0), x.reshape(-1, 1), y, cv=cv)
    assert scores.shape == (4,)
    assert np.all(scores > 0.95)


def test_pipeline_with_scaler_requires_rescaled_sigma(data):
    """A scaler changes the units of X, so sx must be given in scaled units."""
    x, y = data
    X = x.reshape(-1, 1)
    scale = X.std()
    pipe = make_pipeline(StandardScaler(), YorkRegressor(sx=0.4 / scale, sy=1.0))
    pipe.fit(X, y)
    direct = YorkRegressor(sx=0.4, sy=1.0).fit(x, y)
    # Slopes differ by the scale factor; MSWD is invariant to it.
    fitted = pipe[-1]
    assert fitted.slope_ == pytest.approx(direct.slope_ * scale, rel=1e-8)
    assert fitted.mswd_ == pytest.approx(direct.mswd_, rel=1e-8)
    assert pipe.score(X, y) > 0.99


def test_grid_search(data):
    x, y = data
    grid = GridSearchCV(
        YorkRegressor(sy=1.0), {"sx": [0.1, 0.4, 1.0]}, cv=3, scoring="r2"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid.fit(x.reshape(-1, 1), y)
    assert grid.best_params_["sx"] in (0.1, 0.4, 1.0)
    assert hasattr(grid.best_estimator_, "result_")
