"""The single place where paddle positions convert to PLC units.

Numerics run in metres. The PLC buffer `aPos` takes millimetres relative to
the paddle centre (Section 6 of the project instructions).
"""
from __future__ import annotations

import numpy as np

M_TO_MM = 1000.0
PADDLE_SIGN = +1   # [TBC] +1 if positive drive travel moves the paddle toward the beach


def paddle_m_to_plc_mm(x_m) -> np.ndarray:
    """Convert paddle displacement (m, + toward beach) to PLC mm from centre."""
    return PADDLE_SIGN * M_TO_MM * np.asarray(x_m, dtype=float)
