"""Error models: first-class descriptions of instrument uncertainty.

Each model is a callable ``model(values) -> ndarray`` of one-sigma
uncertainties, plus ``describe() -> str`` which propagates into
:meth:`york.YorkFit.summary` so a fit is self-documenting.

``sx`` and ``sy`` are always specified independently. Two instruments rarely
share an error structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "REFERENCE_SPECS",
    "REFERENCE_SPEC_NOTES",
    "Absolute",
    "ErrorModel",
    "InstrumentSpec",
    "Poisson",
    "Relative",
    "Replicate",
    "RollingStd",
]


def _floor_suffix(min_sigma: float) -> str:
    return f", min_sigma={min_sigma:g}" if min_sigma > 0.0 else ""


class ErrorModel(ABC):
    """Base class for error models.

    Subclasses implement :meth:`_sigma` and :meth:`describe`. The base
    ``__call__`` applies the ``min_sigma`` floor uniformly, so no model can
    return a sigma below it.

    To define your own model, subclass this and implement the two methods.
    ``_sigma`` receives the data as a float array and must return something
    broadcastable to the same shape; ``describe`` is printed in
    :meth:`york.YorkFit.summary`::

        @dataclass(frozen=True)
        class HumidityDependent(ErrorModel):
            base: float
            per_rh: float
            rh: np.ndarray
            min_sigma: float = 0.0

            def _sigma(self, values):
                return self.base + self.per_rh * self.rh

            def describe(self):
                return f"HumidityDependent(±{self.base} + {self.per_rh} per %RH)"

        york.fit(x, y, sx=HumidityDependent(0.5, 0.02, rh), sy=0.8)

    A plain per-point array or a scalar is also accepted anywhere a model is,
    so subclassing is only needed when the description and the floor are
    worth having.
    """

    min_sigma: float = 0.0

    def __call__(self, values: ArrayLike) -> NDArray[np.float64]:
        arr = np.asarray(values, dtype=np.float64)
        sigma = np.broadcast_to(
            np.asarray(self._sigma(arr), dtype=np.float64), arr.shape
        ).copy()
        if self.min_sigma > 0.0:
            np.maximum(sigma, self.min_sigma, out=sigma)
        return sigma

    @abstractmethod
    def _sigma(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def describe(self) -> str: ...

    def _check_floor(self) -> None:
        if self.min_sigma < 0.0:
            raise ValueError(f"min_sigma must be >= 0, got {self.min_sigma}")


@dataclass(frozen=True)
class InstrumentSpec(ErrorModel):
    """Manufacturer-style spec: ``sqrt(absolute**2 + (relative * x)**2)``.

    "±(0.5 µg/m³ + 5% of reading)" is ``InstrumentSpec(absolute=0.5,
    relative=0.05)``. This is the flagship model: the absolute term keeps
    sigma finite near zero and the relative term captures span error.
    """

    absolute: float
    relative: float
    min_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.absolute < 0.0 or self.relative < 0.0:
            raise ValueError("absolute and relative must be >= 0")
        if self.absolute == 0.0 and self.relative == 0.0:
            raise ValueError("at least one of absolute or relative must be > 0")
        self._check_floor()

    def _sigma(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(self.absolute**2 + (self.relative * values) ** 2)

    def describe(self) -> str:
        return (
            f"InstrumentSpec(±{self.absolute:g} + {self.relative * 100:g}% of reading"
            f"{_floor_suffix(self.min_sigma)})"
        )


@dataclass(frozen=True)
class Poisson(ErrorModel):
    """Counting statistics for a rate ``N / interval``: ``sqrt(N) / interval``.

    ``values`` are rates (counts per unit time); ``interval`` is the counting
    time in the same units, so ``N = rate * interval``. Negative rates are
    treated as zero counts.
    """

    interval: float
    min_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.interval <= 0.0:
            raise ValueError("interval must be > 0")
        self._check_floor()

    def _sigma(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        counts = np.maximum(values, 0.0) * self.interval
        return np.sqrt(counts) / self.interval

    def describe(self) -> str:
        return (
            f"Poisson(counting, interval={self.interval:g}"
            f"{_floor_suffix(self.min_sigma)})"
        )


@dataclass(frozen=True, eq=False)
class Replicate(ErrorModel):
    """Measured within-window standard deviation supplied alongside the data.

    Ignores ``values`` except to check length. Prefer this whenever averaged
    data already carries an SD: it is a measurement of sigma, not an estimate.
    """

    sd: NDArray[np.float64] = field(repr=False)
    min_sigma: float = 0.0

    def __post_init__(self) -> None:
        sd = np.asarray(self.sd, dtype=np.float64).ravel()
        if sd.size == 0 or not np.all(np.isfinite(sd)) or np.any(sd < 0.0):
            raise ValueError("sd must be a non-empty array of finite values >= 0")
        object.__setattr__(self, "sd", sd)
        self._check_floor()

    def _sigma(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.sd.shape != values.shape:
            raise ValueError(
                f"Replicate sd has length {self.sd.size} but the data have "
                f"length {values.size}"
            )
        return self.sd

    def describe(self) -> str:
        return (
            f"Replicate(measured SD, n={self.sd.size}{_floor_suffix(self.min_sigma)})"
        )


@dataclass(frozen=True)
class Relative(ErrorModel):
    """Fractional precision: ``b * |x|``.

    Sigma goes to zero as x goes to zero, so set ``min_sigma`` when the data
    approach the detection limit.
    """

    b: float
    min_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.b <= 0.0:
            raise ValueError("b must be > 0")
        self._check_floor()

    def _sigma(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.b * np.abs(values)

    def describe(self) -> str:
        return f"Relative({self.b * 100:g}% of reading{_floor_suffix(self.min_sigma)})"


@dataclass(frozen=True)
class Absolute(ErrorModel):
    """Fixed precision: a constant ``a``."""

    a: float
    min_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.a <= 0.0:
            raise ValueError("a must be > 0")
        self._check_floor()

    def _sigma(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(values.shape, self.a)

    def describe(self) -> str:
        return f"Absolute(±{self.a:g}{_floor_suffix(self.min_sigma)})"


@dataclass(frozen=True)
class RollingStd(ErrorModel):
    """Local standard deviation over a centred rolling window.

    Positions without a full window (the edges, or every position when the
    window exceeds the series length) and windows with zero spread fall back
    to the global standard deviation. Last resort for unknown error
    structure: it measures local variability, not instrument precision.
    """

    window: int
    min_sigma: float = 0.0

    def __post_init__(self) -> None:
        if int(self.window) != self.window or self.window < 2:
            raise ValueError("window must be an integer >= 2")
        self._check_floor()

    def _sigma(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        n = values.size
        global_sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
        out = np.full(values.shape, global_sd)
        w = int(self.window)
        if n >= w:
            local = np.lib.stride_tricks.sliding_window_view(values, w).std(
                axis=1, ddof=1
            )
            start = (w - 1) // 2
            sl = slice(start, start + local.size)
            out[sl] = np.where(local > 0.0, local, global_sd)
        return out

    def describe(self) -> str:
        return (
            f"RollingStd(window={int(self.window)}, global fallback"
            f"{_floor_suffix(self.min_sigma)})"
        )


REFERENCE_SPECS: dict[str, InstrumentSpec] = {
    "teom": InstrumentSpec(absolute=2.0, relative=0.0),
    "bam-1020": InstrumentSpec(absolute=2.4, relative=0.0),
    "t640": InstrumentSpec(absolute=0.5, relative=0.05),
    "tsi-3783": InstrumentSpec(absolute=0.0, relative=0.10),
}
"""Convenience specs for common reference instruments.

These are starting points transcribed from public datasheet precision
figures and are NOT authoritative. Datasheets change, hourly and daily
precision differ, and a deployed instrument rarely meets its datasheet.
Override with your own ``InstrumentSpec`` whenever you have better numbers,
for example from co-located replicates via :class:`Replicate`.
"""

REFERENCE_SPEC_NOTES: dict[str, str] = {
    "teom": "Thermo TEOM 1405: datasheet precision ±2.0 µg/m³ for 1-h averages.",
    "bam-1020": (
        "Met One BAM-1020: hourly lower detection limit < 4.8 µg/m³ at 2 sigma, "
        "taken as 2.4 µg/m³ at 1 sigma. 24-h precision is much better."
    ),
    "t640": (
        "Teledyne API T640: datasheet precision ±0.5 µg/m³ with ±5% span; "
        "verify against the current datasheet revision."
    ),
    "tsi-3783": (
        "TSI 3783 EPC: concentration accuracy ±10%. Counting statistics are not "
        "included here; use Poisson(interval) when count rates are low."
    ),
}
"""Where each entry in ``REFERENCE_SPECS`` came from, for the summary."""
