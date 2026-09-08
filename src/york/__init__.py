"""Errors-in-variables straight-line regression (York 2004).

Straight-line fitting when both x and y carry measurement uncertainty, built
for instrument intercomparison. See SPEC.md for the full design.
"""

from york import errors
from york._api import fit
from york._result import HypothesisTest, YorkFit
from york.estimator import YorkRegressor

__version__ = "0.1.0"

york = fit
"""Alias for :func:`york.fit`, for discoverability."""

__all__ = [
    "HypothesisTest",
    "YorkFit",
    "YorkRegressor",
    "__version__",
    "errors",
    "fit",
    "york",
]
