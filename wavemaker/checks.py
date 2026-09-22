"""Limit and validity checks. Any 'fail' or 'unconfigured' result blocks upload.

References
----------
Miche, R. (1944), steepness limit H/L = 0.142 tanh(kh); McCowan (1894),
depth limit H/h = 0.78; Ursell number U = H L^2 / h^3 (Dean & Dalrymple
1991, sec. 11); transverse seiche modes (Dean & Dalrymple 1991, sec. 4.9).
Cross-waves at half the paddle frequency: Garrett (1970), J. Fluid Mech. 41.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Facility
from .dispersion import wavenumber
from .signals import WaveSeries, zero_downcrossing

PASS, WARN, FAIL, UNCONF = "pass", "warn", "fail", "unconfigured"
HM0_TOL = 0.02          # realised vs target Hm0 and Tp
MODE_TOL = 0.05         # proximity to a transverse mode
MIN_PTS_PER_WAVE = 20   # [TBC] samples per shortest generated period


@dataclass
class CheckResult:
    name: str
    status: str
    message: str

    @property
    def blocks(self) -> bool:
        return self.status in (FAIL, UNCONF)

    def __str__(self) -> str:
        return f"[{self.status.upper():^12}] {self.name}: {self.message}"


def upload_allowed(results: list[CheckResult]) -> bool:
    """True only if no check failed and none lacks a configured limit."""
    return not any(r.blocks for r in results)


def transverse_mode_frequencies(width: float, h: float, g: float, n: int = 3) -> np.ndarray:
    """Natural frequencies (Hz) of the first n cross-tank sloshing modes.

    k_m = m pi / B; omega_m^2 = g k_m tanh(k_m h).
    """
    k = np.arange(1, n + 1) * np.pi / width
    return np.sqrt(g * k * np.tanh(k * h)) / (2 * np.pi)


# ---------------------------------------------------------------------------
def _limit(name, value, limit, unit, fmt=".3f"):
    if limit is None:
        return CheckResult(name, UNCONF,
                           f"peak {value:{fmt}} {unit}; limit not configured [TBC]")
    st = PASS if value <= limit else FAIL
    return CheckResult(name, st, f"peak {value:{fmt}} {unit} vs limit {limit:{fmt}} {unit}")


def run_checks(s: WaveSeries, fac: Facility) -> list[CheckResult]:
    """Run every check on a generated series."""
    m, g, h = s.meta, fac.g, s.meta["h"]
    irregular = m["kind"] == "irregular"
    out: list[CheckResult] = []

    # --- Depth -------------------------------------------------------------
    out.append(CheckResult(
        "Water depth", PASS if h <= fac.max_water_depth_m else FAIL,
        f"h = {h:.3f} m vs max working depth {fac.max_water_depth_m:.3f} m"))

    # --- Stroke ------------------------------------------------------------
    xmax = float(np.max(np.abs(s.x)))
    if xmax > fac.half_stroke_m:
        out.append(CheckResult("Stroke", FAIL,
                               f"peak |x| {xmax*1e3:.1f} mm exceeds half-stroke "
                               f"{fac.half_stroke_m*1e3:.0f} mm"))
    elif fac.stroke_margin_m is None:
        out.append(CheckResult("Stroke", UNCONF,
                               f"peak |x| {xmax*1e3:.1f} mm of {fac.half_stroke_m*1e3:.0f} mm; "
                               "stroke margin not configured [TBC]"))
    else:
        lim = fac.half_stroke_m - fac.stroke_margin_m
        out.append(CheckResult("Stroke", PASS if xmax <= lim else FAIL,
                               f"peak |x| {xmax*1e3:.1f} mm vs usable {lim*1e3:.1f} mm"))

    # --- Velocity and acceleration -----------------------------------------
    v = np.gradient(s.x, s.dt)
    a = np.gradient(v, s.dt)
    out.append(_limit("Paddle velocity", float(np.max(np.abs(v))), fac.max_velocity_m_s, "m/s"))
    out.append(_limit("Paddle acceleration", float(np.max(np.abs(a))),
                      fac.max_acceleration_m_s2, "m/s^2"))

    # --- Representative wave -----------------------------------------------
    if irregular:
        Hrep, Trep, kh = m["Hs"], m["Tp"], m["kph"]
        tag = "Hs, Tp"
    else:
        Hrep, Trep, kh = m["H"], m["T"], m["kh"]
        tag = "H, T"
    L = 2 * np.pi * h / kh

    # --- Breaking: depth-limited and steepness -----------------------------
    r_depth = Hrep / h
    r_steep = (Hrep / L) / (0.142 * np.tanh(kh))
    out.append(CheckResult("Depth-limited breaking (" + tag + ")",
                           PASS if r_depth < 0.78 else FAIL,
                           f"H/h = {r_depth:.3f} vs 0.78"))
    out.append(CheckResult("Steepness breaking, Miche (" + tag + ")",
                           PASS if r_steep < 1 else FAIL,
                           f"H/L = {Hrep/L:.4f} vs {0.142*np.tanh(kh):.4f} "
                           f"({100*r_steep:.0f} % of limit)"))

    if irregular:
        # Policy [TBC]: individual waves beyond the limits warn; Hs, Tp beyond fail.
        Hi, Ti = zero_downcrossing(s.eta[s.steady], s.dt)
        if Hi.size:
            ki = wavenumber(2 * np.pi / Ti, h, g)
            n_d = int(np.sum(Hi / h >= 0.78))
            n_s = int(np.sum(Hi * ki / (2 * np.pi) >= 0.142 * np.tanh(ki * h)))
            st = WARN if (n_d or n_s) else PASS
            out.append(CheckResult(
                "Individual-wave breaking", st,
                f"{Hi.size} waves; {n_d} exceed H/h = 0.78, {n_s} exceed Miche; "
                f"Hmax = {Hi.max():.3f} m"))

    # --- Freeboard ---------------------------------------------------------
    crest = float(np.max(s.eta))
    top = h + crest
    if fac.freeboard_margin_m is None:
        st = FAIL if top >= fac.tank_depth_m else UNCONF
        out.append(CheckResult("Freeboard", st,
                               f"h + crest = {top:.3f} m of {fac.tank_depth_m:.2f} m tank; "
                               "margin not configured [TBC]"))
    else:
        lim = fac.tank_depth_m - fac.freeboard_margin_m
        out.append(CheckResult("Freeboard", PASS if top < lim else FAIL,
                               f"h + crest = {top:.3f} m vs {lim:.3f} m "
                               "(linear crest; nonlinear crests are higher)"))

    # --- Theory range ------------------------------------------------------
    out.append(CheckResult("Shallow-water range", WARN if kh < 0.3 else PASS,
                           f"kh = {kh:.3f}" + (" < 0.3: first-order generation will "
                                               "produce spurious long waves" if kh < 0.3 else "")))
    ur = Hrep * L**2 / h**3
    out.append(CheckResult("Ursell number", WARN if ur > 25 else PASS,
                           f"Ur = {ur:.1f}" + (" > 25: nonlinear, linear theory unreliable"
                                               if ur > 25 else "")))

    # --- Transverse modes and cross-waves ----------------------------------
    fm = transverse_mode_frequencies(fac.tank_width_m, h, g)
    fw = 1 / Trep
    near = [f"mode {i+1} ({f:.3f} Hz)" for i, f in enumerate(fm)
            if abs(fw - f) / f < MODE_TOL]
    near += [f"2 x mode {i+1} ({2*f:.3f} Hz, cross-wave)" for i, f in enumerate(fm)
             if abs(fw - 2 * f) / (2 * f) < MODE_TOL]
    out.append(CheckResult("Cross-tank modes", WARN if near else PASS,
                           f"f = {fw:.3f} Hz" + ("; near " + ", ".join(near) if near else
                                                 f"; modes at {', '.join(f'{f:.3f}' for f in fm)} Hz")))

    # --- Sampling ----------------------------------------------------------
    f_top = m["f_max"] if irregular else fw
    ppw = 1 / (f_top * s.dt)
    out.append(CheckResult("Sampling", PASS if ppw >= MIN_PTS_PER_WAVE else WARN,
                           f"{ppw:.0f} samples per shortest period at dt = {s.dt*1e3:.1f} ms"))

    # --- Realised spectrum -------------------------------------------------
    if irregular:
        eh = abs(m["Hm0_realised"] / m["Hs"] - 1)
        et = abs(m["Tp_realised"] / m["Tp"] - 1)
        bad = eh > HM0_TOL or et > HM0_TOL
        # Random-complex amplitudes vary by chance; mismatch warns rather than fails.
        st = (FAIL if m["amplitude_mode"] == "deterministic" else WARN) if bad else PASS
        out.append(CheckResult("Realised Hm0, Tp", st,
                               f"Hm0 {m['Hm0_realised']:.4f} m ({100*eh:.2f} %), "
                               f"Tp {m['Tp_realised']:.3f} s ({100*et:.2f} %); tolerance 2 %"))
    return out
