"""Plots of paddle signal, target surface elevation and spectrum."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .signals import WaveSeries


def plot_series(s: WaveSeries, path=None, window_s: float | None = None):
    """Three panels: x(t) in mm, eta(t) in mm, and S(f) (irregular only)."""
    irregular = s.meta["kind"] == "irregular"
    fig, axes = plt.subplots(3 if irregular else 2, 1, figsize=(9, 8 if irregular else 5.5),
                             constrained_layout=True)
    n = s.t.size if window_s is None else min(s.t.size, int(window_s / s.dt))
    axes[0].plot(s.t[:n], 1e3 * s.x[:n], lw=0.8)
    axes[0].set_ylabel("Paddle x (mm)")
    axes[1].plot(s.t[:n], 1e3 * s.eta[:n], lw=0.8, color="C1")
    axes[1].set_ylabel(r"Target $\eta$ (mm)")
    axes[1].set_xlabel("t (s)")
    m = s.meta
    if irregular:
        title = (f"{m['spectrum']}: Hs = {m['Hs']:.3f} m, Tp = {m['Tp']:.2f} s, "
                 f"h = {m['h']:.2f} m, seed {m['seed']}")
        seg = s.eta[s.steady]
        P = np.abs(np.fft.rfft(seg - seg.mean())) ** 2 * 2 * s.dt / seg.size
        f = np.fft.rfftfreq(seg.size, s.dt)
        nb = max(1, int(0.05 * (1 / m["Tp"]) * seg.size * s.dt))
        axes[2].plot(f, np.convolve(P, np.ones(nb) / nb, "same"), lw=0.8, label="realised (smoothed)")
        axes[2].plot(s.f_target, s.S_target, "k--", lw=1, label="target")
        axes[2].set_xlim(0, 1.2 * m["f_max"])
        axes[2].set_xlabel("f (Hz)")
        axes[2].set_ylabel(r"S (m$^2$/Hz)")
        axes[2].legend()
    else:
        title = (f"Regular: H = {m['H']:.3f} m, T = {m['T']:.2f} s, h = {m['h']:.2f} m, "
                 f"S = {1e3*m['stroke_S']:.1f} mm")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(alpha=0.3)
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig
