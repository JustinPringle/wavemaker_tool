"""Paddle displacement time series from first-order wavemaker theory.

Sign and phase convention (Dean & Dalrymple 1991, sec. 6.2): a piston moving
as x(t) = (S/2) sin(wt) produces eta(0, t) = (H/2) cos(wt) in the far field.
In complex form, X = -i A / TF, where A and X are the complex amplitudes of
eta and x, and TF = H/S.

x is positive toward the beach, in metres from the paddle centre. Conversion
to PLC millimetres happens only in :mod:`wavemaker.units`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dispersion import G, wavenumber
from .spectra import SpectrumSpec
from .transfer import transfer_function

DEFAULT_RECORD_S = 1200.0   # [TBC] 20 min at model scale ...
DEFAULT_MIN_WAVES = 500     # [TBC] ... or at least 500 waves, whichever is longer
DEFAULT_RAMP_PERIODS = 5    # [TBC] minimum 3


@dataclass
class WaveSeries:
    """A paddle signal and its target surface elevation.

    t : s; x : paddle displacement (m); eta : target surface elevation at the
    paddle (m, linear far-field); steady : slice excluding the ramps.
    """

    t: np.ndarray
    x: np.ndarray
    eta: np.ndarray
    dt: float
    steady: slice
    meta: dict = field(default_factory=dict)
    f_target: np.ndarray | None = None
    S_target: np.ndarray | None = None


def cosine_ramp(n: int, n_up: int, n_down: int) -> np.ndarray:
    """Weight that rises 0->1 over n_up samples and falls 1->0 over n_down.

    First and last samples are exactly zero, so the paddle starts and ends at
    centre.
    """
    w = np.ones(n)
    if n_up > 0:
        i = np.arange(n_up)
        w[:n_up] = 0.5 * (1 - np.cos(np.pi * i / n_up))
    if n_down > 0:
        i = np.arange(1, n_down + 1)
        w[n - n_down:] = 0.5 * (1 + np.cos(np.pi * i / n_down))
    return w


# ---------------------------------------------------------------------------
def monochromatic(H: float, T: float, h: float, dt: float,
                  paddle_type: str = "piston", n_steady: int = 30,
                  n_ramp: int = DEFAULT_RAMP_PERIODS, g: float = G) -> WaveSeries:
    """Regular wave: x(t) = (S/2) sin(wt) with S = H / TF(kh), cosine ramps.

    Parameters
    ----------
    H : wave height (m); T : period (s); h : depth (m); dt : sample interval (s)
    n_steady : full-amplitude periods; n_ramp : periods in each ramp
    """
    if min(H, T, h, dt) <= 0:
        raise ValueError("H, T, h and dt must be positive.")
    omega = 2 * np.pi / T
    k = wavenumber(omega, h, g)
    kh = k * h
    tf = transfer_function(kh, paddle_type)
    S = H / tf

    n_up = int(round(n_ramp * T / dt))
    N = int(round((n_steady + 2 * n_ramp) * T / dt)) + 1
    t = np.arange(N) * dt
    w = cosine_ramp(N, n_up, n_up)
    x = 0.5 * S * np.sin(omega * t) * w
    eta = 0.5 * H * np.cos(omega * t) * w

    meta = dict(kind="regular", paddle_type=paddle_type, H=H, T=T, h=h,
                k=k, kh=kh, L=2 * np.pi / k, tf=tf, stroke_S=S,
                n_steady=n_steady, n_ramp=n_ramp, duration_s=t[-1])
    return WaveSeries(t, x, eta, dt, slice(n_up, N - n_up), meta)


# ---------------------------------------------------------------------------
def spectral_moments(eta: np.ndarray, dt: float, smooth_frac: float = 0.0,
                     fp_hint: float | None = None) -> tuple[float, float]:
    """Hm0 = 4 sqrt(m0) and Tp from the periodogram of a periodic record.

    smooth_frac : width of a boxcar smoother on the periodogram, as a fraction
    of fp_hint; use it for random-amplitude records, whose raw periodogram is
    noisy. Tp is refined by a parabola through the peak and its neighbours.
    """
    N = len(eta)
    P = np.abs(np.fft.rfft(eta - eta.mean())) ** 2
    f = np.fft.rfftfreq(N, dt)
    if smooth_frac > 0 and fp_hint:
        nb = max(1, int(round(smooth_frac * fp_hint * N * dt)))
        if nb > 1:
            P = np.convolve(P, np.ones(nb) / nb, mode="same")
    i = int(np.argmax(P[1:])) + 1
    fp = f[i]
    if 1 <= i < len(P) - 1:
        a, b, c = P[i - 1], P[i], P[i + 1]
        denom = a - 2 * b + c
        if denom != 0:
            fp = f[i] + 0.5 * (a - c) / denom * (f[1] - f[0])
    hm0 = 4 * np.std(eta)
    return float(hm0), float(1.0 / fp)


def irregular(spec: SpectrumSpec, h: float, dt: float,
              paddle_type: str = "piston",
              duration_s: float | None = None,
              f_min_factor: float = 0.5, f_max_factor: float = 3.0,
              amplitude_mode: str = "deterministic",
              seed: int | None = None,
              n_ramp: int = DEFAULT_RAMP_PERIODS,
              match_hm0_in_band: bool = True,
              g: float = G) -> WaveSeries:
    """Irregular sea by inverse FFT.

    Steps (Frigaard et al.; Goda 2010, sec. 2.3):
    1. N samples at dt; frequency resolution df = 1/(N dt), so the signal
       repeats only after the full record.
    2. Components between f_min_factor*fp and f_max_factor*fp.
    3. a_n = sqrt(2 S(f_n) df). "deterministic": fixed a_n, uniform random
       phase. "random_complex": real and imaginary parts Gaussian, so |A_n| is
       Rayleigh with E|A_n|^2 = a_n^2.
    4. X_n = -i A_n / TF(k_n h); x and eta by irfft.
    5. Remove the mean, apply cosine ramps of n_ramp * Tp.

    match_hm0_in_band : if True (default), rescale S over the generation band
    so that its m0 equals Hs^2/16. If False, the band holds slightly less
    energy than the untruncated spectrum. [TBC: choose the lab convention.]

    The seed is always stored in meta so that any run can be repeated.
    """
    if amplitude_mode not in ("deterministic", "random_complex"):
        raise ValueError("amplitude_mode must be 'deterministic' or 'random_complex'.")
    fp = spec.fp
    if duration_s is None:
        duration_s = max(DEFAULT_RECORD_S, DEFAULT_MIN_WAVES * spec.tp)
    N = int(round(duration_s / dt))
    N += N % 2
    T_rec = N * dt
    df = 1.0 / T_rec

    f_all = np.fft.rfftfreq(N, dt)
    f_lo, f_hi = f_min_factor * fp, f_max_factor * fp
    if f_hi >= 0.5 / dt:
        raise ValueError("f_max exceeds the Nyquist frequency; reduce dt or f_max_factor.")
    band = (f_all >= f_lo) & (f_all <= f_hi) & (f_all > 0)
    f = f_all[band]

    S = spec.density(f)
    if match_hm0_in_band:
        S = S * (spec.hs**2 / 16) / (S.sum() * df)
    a = np.sqrt(2 * S * df)

    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**32)
    rng = np.random.default_rng(seed)
    if amplitude_mode == "deterministic":
        A = a * np.exp(1j * rng.uniform(0, 2 * np.pi, a.size))
    else:
        A = a * (rng.standard_normal(a.size) + 1j * rng.standard_normal(a.size)) / np.sqrt(2)

    k = wavenumber(2 * np.pi * f, h, g)
    tf = transfer_function(k * h, paddle_type)
    X = -1j * A / tf

    Y_eta = np.zeros(f_all.size, complex)
    Y_x = np.zeros(f_all.size, complex)
    Y_eta[band] = A * N / 2
    Y_x[band] = X * N / 2
    eta = np.fft.irfft(Y_eta, n=N)
    x = np.fft.irfft(Y_x, n=N)
    eta -= eta.mean()
    x -= x.mean()

    smooth = 0.05 if amplitude_mode == "random_complex" else 0.0
    hm0_r, tp_r = spectral_moments(eta, dt, smooth, fp)

    n_up = int(round(n_ramp * spec.tp / dt))
    w = cosine_ramp(N, n_up, n_up)
    t = np.arange(N) * dt

    kp = wavenumber(2 * np.pi * fp, h, g)
    meta = dict(kind="irregular", paddle_type=paddle_type, spectrum=spec.kind,
                Hs=spec.hs, Tp=spec.tp, gamma=spec.gamma, h=h,
                kp=kp, kph=kp * h, Lp=2 * np.pi / kp,
                f_min=f_lo, f_max=f_hi, df=df, n_components=int(band.sum()),
                amplitude_mode=amplitude_mode, seed=seed,
                match_hm0_in_band=match_hm0_in_band,
                repeat_period_s=T_rec, duration_s=T_rec, n_ramp=n_ramp,
                Hm0_realised=hm0_r, Tp_realised=tp_r)
    return WaveSeries(t, x * w, eta * w, dt, slice(n_up, N - n_up), meta,
                      f_target=f, S_target=S)


# ---------------------------------------------------------------------------
def zero_downcrossing(eta: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Individual wave heights and periods by zero-downcrossing analysis."""
    i = np.flatnonzero((eta[:-1] >= 0) & (eta[1:] < 0))
    if i.size < 2:
        return np.array([]), np.array([])
    tc = (i + eta[i] / (eta[i] - eta[i + 1])) * dt
    seg = [eta[i[j] + 1:i[j + 1] + 1] for j in range(i.size - 1)]
    H = np.array([s.max() - s.min() for s in seg])
    return H, np.diff(tc)
