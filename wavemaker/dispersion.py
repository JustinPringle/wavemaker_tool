"""Linear dispersion relation and derived wave properties.

All quantities are SI: omega in rad/s, k in rad/m, h, L in m, T in s.

Reference
---------
Dean, R.G. & Dalrymple, R.A. (1991) *Water Wave Mechanics for Engineers and
Scientists*. World Scientific, ch. 3.
"""
from __future__ import annotations

import numpy as np

G = 9.81
_EPS = np.finfo(float).eps


def _out(a: np.ndarray):
    """Return a Python float for 0-d arrays, else the array."""
    return float(a) if np.ndim(a) == 0 else a


def wavenumber(omega, h, g: float = G, max_iter: int = 100):
    """Solve omega^2 = g k tanh(k h) for k by Newton iteration.

    Works in the dimensionless form x tanh x = y, with x = k h and
    y = omega^2 h / g, starting from x0 = y / sqrt(tanh y), and iterates to
    machine precision (Dean & Dalrymple 1991, eq. 3.34).

    Parameters
    ----------
    omega : array_like  Angular frequency (rad/s), >= 0.
    h : array_like      Water depth (m), > 0.
    g : float           Gravitational acceleration (m/s^2).

    Returns
    -------
    k : float or ndarray  Wavenumber (rad/m).
    """
    omega = np.asarray(omega, dtype=float)
    h = np.asarray(h, dtype=float)
    if np.any(h <= 0):
        raise ValueError("Water depth must be positive.")
    if np.any(omega < 0):
        raise ValueError("Angular frequency must be non-negative.")
    omega, h = np.broadcast_arrays(omega, h)

    y = omega**2 * h / g
    x = np.zeros_like(y)
    pos = y > 0
    yp = y[pos]
    xp = yp / np.sqrt(np.tanh(yp))
    for _ in range(max_iter):
        t = np.tanh(xp)
        dx = (xp * t - yp) / (t + xp * (1.0 - t * t))
        xp = xp - dx
        if np.all(np.abs(dx) <= 8 * _EPS * np.maximum(xp, 1.0)):
            break
    else:  # pragma: no cover - Newton on this convex function always converges
        raise RuntimeError("Dispersion solver did not converge.")
    x[pos] = xp
    return _out(x / h)


def wavelength(T, h, g: float = G):
    """Wavelength L (m) for period T (s) in depth h (m)."""
    return _out(2 * np.pi / np.asarray(wavenumber(2 * np.pi / np.asarray(T, float), h, g)))


def period_from_wavelength(L, h, g: float = G):
    """Period T (s) of a wave of length L (m) in depth h (m), from dispersion."""
    k = 2 * np.pi / np.asarray(L, float)
    omega = np.sqrt(g * k * np.tanh(k * np.asarray(h, float)))
    return _out(2 * np.pi / omega)


def celerity(T, h, g: float = G):
    """Phase speed C = L / T (m/s)."""
    return _out(np.asarray(wavelength(T, h, g)) / np.asarray(T, float))


def group_velocity_ratio(kh):
    """n = Cg / C = 0.5 (1 + 2kh / sinh 2kh)."""
    kh = np.asarray(kh, float)
    two = np.minimum(2 * kh, 700.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        term = np.where(two > 0, two / np.sinh(two), 1.0)
    term = np.where(2 * kh >= 700.0, 0.0, term)
    return _out(0.5 * (1.0 + term))


def group_velocity(T, h, g: float = G):
    """Group velocity Cg (m/s)."""
    k = np.asarray(wavenumber(2 * np.pi / np.asarray(T, float), h, g))
    return _out(np.asarray(celerity(T, h, g)) * group_velocity_ratio(k * np.asarray(h, float)))
