"""scikit-learn-style estimator, implemented without scikit-learn.

:class:`YorkRegressor` wraps :func:`york.fit` in the ``fit`` / ``predict`` /
``score`` / ``get_params`` / ``set_params`` protocol so it duck-types into
``cross_val_score``, ``GridSearchCV`` and pipelines. scikit-learn is never
imported at runtime; it is a dev dependency used only to test that the
duck-typing holds.

The interesting output of a York fit is the hypothesis tests and MSWD, not
``predict``. They live on the ``result_`` attribute, which is the full
:class:`york.YorkFit`.

.. warning::

   A transformer in front of this estimator changes the units of ``X``, but
   ``sx`` is still whatever you passed in. Put a ``StandardScaler`` in a
   pipeline and the x uncertainties are silently wrong unless you rescale
   ``sx`` by the same factor. Prefer fitting on raw units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from york._api import fit as _fit

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from york._api import SigmaSpec
    from york._result import YorkFit

__all__ = ["NotFittedError", "YorkRegressor"]

_PARAM_NAMES = ("sx", "sy", "r", "scale_errors", "tol", "max_iter")


class NotFittedError(ValueError, AttributeError):
    """Raised by ``predict`` or ``score`` before ``fit``.

    Inherits from both ``ValueError`` and ``AttributeError``, as
    scikit-learn's does, so either handler style works.
    """


class YorkRegressor:
    """York regression with a scikit-learn-style interface.

    Parameters mirror :func:`york.fit`. ``X`` may be shape ``(n,)`` or
    ``(n, 1)``; the regressor takes exactly one feature by design.

    Fitted attributes: ``slope_``, ``intercept_``, ``slope_err_``,
    ``intercept_err_``, ``mswd_``, ``n_iter_``, ``converged_``,
    ``n_features_in_``, and ``result_`` (the full :class:`york.YorkFit`,
    with ``summary()``, ``test_slope()`` and the rest).
    """

    slope_: float
    intercept_: float
    slope_err_: float
    intercept_err_: float
    mswd_: float
    n_iter_: int
    converged_: bool
    n_features_in_: int
    result_: YorkFit

    def __init__(
        self,
        sx: SigmaSpec = None,
        sy: SigmaSpec = None,
        r: ArrayLike | float = 0.0,
        scale_errors: bool = False,
        tol: float = 1e-12,
        max_iter: int = 200,
    ) -> None:
        self.sx = sx
        self.sy = sy
        self.r = r
        self.scale_errors = scale_errors
        self.tol = tol
        self.max_iter = max_iter

    # --- estimator protocol -------------------------------------------------

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _PARAM_NAMES}

    def set_params(self, **params: Any) -> YorkRegressor:
        for name, value in params.items():
            if name not in _PARAM_NAMES:
                raise ValueError(
                    f"Invalid parameter {name!r} for estimator YorkRegressor. "
                    f"Valid parameters are: {list(_PARAM_NAMES)}."
                )
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"YorkRegressor({args})"

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "result_")

    def __sklearn_tags__(self) -> Any:
        # Called only by scikit-learn >= 1.6, so the import is safe here.
        from sklearn.utils import InputTags, RegressorTags, Tags, TargetTags

        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True),
            regressor_tags=RegressorTags(),
            input_tags=InputTags(one_d_array=True, two_d_array=True),
        )

    # --- fit / predict / score ----------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> YorkRegressor:
        """Fit the York line. Returns ``self``."""
        x = self._one_feature(X, reset=True)
        ya = np.asarray(y, dtype=np.float64)
        if ya.ndim == 2 and ya.shape[1] == 1:
            ya = ya[:, 0]
        if ya.ndim != 1:
            raise ValueError(f"y must be one-dimensional; got shape {ya.shape}")
        if ya.size != x.size:
            raise ValueError(
                f"X and y have inconsistent numbers of samples: {x.size} and {ya.size}"
            )
        result = _fit(
            x,
            ya,
            self.sx,
            self.sy,
            self.r,
            tol=self.tol,
            max_iter=self.max_iter,
            scale_errors=self.scale_errors,
        )
        self.result_ = result
        self.slope_ = result.slope
        self.intercept_ = result.intercept
        self.slope_err_ = result.slope_err
        self.intercept_err_ = result.intercept_err
        self.mswd_ = result.mswd
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Fitted line evaluated at ``X``."""
        self._check_fitted()
        return self.result_.predict(self._one_feature(X, reset=False))

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Coefficient of determination R² of ``predict(X)`` against ``y``.

        This is here for scikit-learn's benefit. It is not the question a
        method comparison asks; use ``result_.test_slope()`` and
        ``result_.mswd`` for that.
        """
        self._check_fitted()
        ya = np.asarray(y, dtype=np.float64).ravel()
        pred = self.predict(X)
        ss_res = float(np.sum((ya - pred) ** 2))
        ss_tot = float(np.sum((ya - ya.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    # --- helpers ------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                "This YorkRegressor instance is not fitted yet. Call 'fit' first."
            )

    def _one_feature(self, X: ArrayLike, *, reset: bool) -> NDArray[np.float64]:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            n_features = 1
            col = arr
        elif arr.ndim == 2:
            n_features = arr.shape[1]
            col = arr[:, 0] if n_features else arr[:, :0].reshape(-1)
        else:
            raise ValueError(f"X must be 1-D or 2-D; got shape {arr.shape}")

        if reset:
            if n_features != 1:
                raise ValueError(
                    f"X has {n_features} features, but YorkRegressor accepts "
                    "exactly 1 feature (shape (n,) or (n, 1))"
                )
            self.n_features_in_ = 1
        elif n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but YorkRegressor is expecting "
                f"{self.n_features_in_} feature as input"
            )
        return col
