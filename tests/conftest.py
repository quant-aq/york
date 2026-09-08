"""Shared fixtures.

The Pearson-York dataset (Pearson 1901; weights from York 1966) is the
canonical benchmark. Expected results from York et al. (2004):
slope -0.48053 ± 0.0156, intercept 5.47991 ± 0.0794 (see SPEC.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass(frozen=True)
class PearsonYork:
    x: np.ndarray
    y: np.ndarray
    wx: np.ndarray
    wy: np.ndarray

    @property
    def sx(self) -> np.ndarray:
        return 1.0 / np.sqrt(self.wx)

    @property
    def sy(self) -> np.ndarray:
        return 1.0 / np.sqrt(self.wy)


@pytest.fixture(scope="session")
def pearson_york() -> PearsonYork:
    return PearsonYork(
        x=np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4]),
        y=np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5]),
        wx=np.array([1000.0, 1000.0, 500.0, 800.0, 200.0, 80.0, 60.0, 20.0, 1.8, 1.0]),
        wy=np.array([1.0, 1.8, 4.0, 8.0, 20.0, 20.0, 70.0, 70.0, 100.0, 500.0]),
    )
