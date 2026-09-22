"""Frequency spectra S(f) in m^2/Hz, parameterised by Hs = Hm0 and Tp.

References
----------
Goda, Y. (2010) *Random Seas and Design of Maritime Structures*, 3rd ed.,
World Scientific, sec. 2.3.
Hasselmann, K. et al. (1973) JONSWAP. Dtsch. Hydrogr. Z. A8(12).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad


def pierson_moskowitz(f, hs: float, tp: float):
    """Pierson-Moskowitz (Bretschneider-Mitsuyasu form, Goda 2010 eq. 2.10).

    S(f) = (5/16) Hs^2 fp^4 f^-5 exp[-1.25 (fp/f)^4], which integrates to
    exactly Hs^2 / 16.
    """
    f = np.atleast_1d(np.asarray(f, float))
    fp = 1.0 / tp
    S = np.zeros_like(f)
    p = f > 0
    S[p] = 5 / 16 * hs**2 * fp**4 * f[p] ** -5 * np.exp(-1.25 * (fp / f[p]) ** 4)
    return S


def _jonswap_shape(f, fp: float, gamma: float):
    f = np.atleast_1d(np.asarray(f, float))
    S = np.zeros_like(f)
    p = f > 0
    fx = f[p]
    sigma = np.where(fx <= fp, 0.07, 0.09)
    r = np.exp(-((fx - fp) ** 2) / (2 * sigma**2 * fp**2))
    S[p] = fx**-5 * np.exp(-1.25 * (fp / fx) ** 4) * gamma**r
    return S


def jonswap(f, hs: float, tp: float, gamma: float = 3.3):
    """JONSWAP spectrum, normalised numerically so that m0 = Hs^2 / 16.

    sigma = 0.07 below fp and 0.09 above (Hasselmann et al. 1973). The
    normalisation integrates the untruncated spectrum; truncation to a
    generation band is handled in :func:`wavemaker.signals.irregular`.
    """
    fp = 1.0 / tp
    shape = lambda x: float(_jonswap_shape(x, fp, gamma)[0])
    m0 = sum(quad(shape, a, b, limit=200)[0]
             for a, b in [(0.0, fp), (fp, 3 * fp), (3 * fp, 20 * fp)])
    m0 += fp**-4 * 20.0**-4 / 4                 # analytic f^-5 tail beyond 20 fp
    return hs**2 / 16 / m0 * _jonswap_shape(f, fp, gamma)


@dataclass
class SpectrumSpec:
    """Definition of a target spectrum.

    kind : "jonswap", "pm" or "csv". For "csv", the file holds two columns,
    f (Hz) and S (m^2/Hz); if hs is also given, S is rescaled to that Hm0.
    """

    kind: str = "jonswap"
    hs: float | None = None
    tp: float | None = None
    gamma: float = 3.3
    csv_path: str | None = None

    def __post_init__(self):
        if self.kind not in ("jonswap", "pm", "csv"):
            raise ValueError(f"Unknown spectrum kind '{self.kind}'.")
        if self.kind == "csv":
            data = np.loadtxt(Path(self.csv_path), delimiter=",", comments="#", ndmin=2)
            if data.shape[1] < 2:
                raise ValueError("Spectrum CSV needs two columns: f (Hz), S (m^2/Hz).")
            order = np.argsort(data[:, 0])
            self._f, self._S = data[order, 0], data[order, 1]
            m0 = np.trapezoid(self._S, self._f)
            if self.hs is not None:
                self._S = self._S * (self.hs**2 / 16) / m0
            else:
                self.hs = 4 * np.sqrt(m0)
            self.tp = 1.0 / self._f[np.argmax(self._S)]
        elif self.hs is None or self.tp is None:
            raise ValueError("Hs and Tp are required for parametric spectra.")

    @property
    def fp(self) -> float:
        """Peak frequency (Hz)."""
        return 1.0 / self.tp

    def density(self, f):
        """Spectral density S(f) in m^2/Hz."""
        if self.kind == "jonswap":
            return jonswap(f, self.hs, self.tp, self.gamma)
        if self.kind == "pm":
            return pierson_moskowitz(f, self.hs, self.tp)
        return np.interp(f, self._f, self._S, left=0.0, right=0.0)
