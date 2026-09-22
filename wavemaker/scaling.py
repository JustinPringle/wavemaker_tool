"""Froude scaling between prototype and model.

Length scale lambda = L_p / L_m; time scale sqrt(lambda) (Hughes 1993,
*Physical Models and Laboratory Techniques in Coastal Engineering*, ch. 2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FroudeScale:
    """Froude similitude for length scale lam (prototype / model)."""

    lam: float

    def __post_init__(self):
        if self.lam <= 0:
            raise ValueError("Length scale must be positive.")

    @property
    def length(self) -> float:
        return self.lam

    @property
    def time(self) -> float:
        return math.sqrt(self.lam)

    def to_model(self, H: float, T: float, h: float) -> tuple[float, float, float]:
        """Prototype (H, T, h) -> model (H, T, h)."""
        return H / self.length, T / self.time, h / self.length

    def to_prototype(self, H: float, T: float, h: float) -> tuple[float, float, float]:
        """Model (H, T, h) -> prototype (H, T, h)."""
        return H * self.length, T * self.time, h * self.length

    def describe(self) -> str:
        return f"length scale 1:{self.length:g}, time scale 1:{self.time:.4g}"
