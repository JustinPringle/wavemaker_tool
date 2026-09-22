"""First-order wavemaker transfer functions H/S (wave height over stroke).

Because both H and S are double amplitudes, the same ratio links wave
amplitude a to paddle amplitude X: a / X = H / S.

Reference
---------
Biesel, F. & Suquet, F. (1951); as given in Dean & Dalrymple (1991),
*Water Wave Mechanics for Engineers and Scientists*, eqs. 6.20 and 6.23.
"""
from __future__ import annotations

import numpy as np

_KH_ASYM = 20.0  # beyond this, use the deep-water asymptotes (avoids overflow)


def piston_tf(kh):
    """Piston H/S = 2(cosh 2kh - 1) / (sinh 2kh + 2kh).

    Written as 4 sinh^2 kh / (sinh 2kh + 2kh) to avoid cancellation at small
    kh. Limits: H/S -> kh as kh -> 0; H/S -> 2 as kh -> infinity.
    """
    kh = np.asarray(kh, dtype=float)
    x = np.clip(kh, 1e-300, _KH_ASYM)
    tf = 4 * np.sinh(x) ** 2 / (np.sinh(2 * x) + 2 * x)
    tf = np.where(kh > _KH_ASYM, 2.0, tf)
    tf = np.where(kh <= 0, 0.0, tf)
    return float(tf) if tf.ndim == 0 else tf


def flap_tf(kh):
    """Flap (hinged at the bed) H/S.

    H/S = (4 sinh kh / kh) (kh sinh kh - cosh kh + 1) / (sinh 2kh + 2kh).
    Limits: H/S -> kh/2 as kh -> 0; H/S -> 2(kh - 1)/kh as kh -> infinity.
    """
    kh = np.asarray(kh, dtype=float)
    x = np.clip(kh, 1e-300, _KH_ASYM)
    num = x * np.sinh(x) - 2 * np.sinh(x / 2) ** 2      # kh sinh kh - (cosh kh - 1)
    tf = 4 * np.sinh(x) / x * num / (np.sinh(2 * x) + 2 * x)
    tf = np.where(kh > _KH_ASYM, 2 * (kh - 1) / np.maximum(kh, 1.0), tf)
    tf = np.where(kh <= 0, 0.0, tf)
    return float(tf) if tf.ndim == 0 else tf


def transfer_function(kh, paddle_type: str = "piston"):
    """Dispatch to the piston or flap transfer function."""
    if paddle_type == "piston":
        return piston_tf(kh)
    if paddle_type == "flap":
        return flap_tf(kh)
    raise ValueError(f"Unknown paddle type '{paddle_type}'.")
