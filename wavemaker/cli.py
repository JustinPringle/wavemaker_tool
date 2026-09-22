"""Command-line front end.

Examples
--------
python -m wavemaker new runs/demo --name "Demo"
python -m wavemaker regular runs/demo reg_H05_T12 --H 0.05 --T 1.2 --h 0.5
python -m wavemaker spectrum runs/demo js_Hs04_Tp15 --Hs 0.04 --Tp 1.5 --h 0.5
python -m wavemaker generate runs/demo js_Hs04_Tp15 --plot
python -m wavemaker info --T 1.2 --h 0.5
python -m wavemaker scale --lam 25 --H 2.0 --T 10 --h 12
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .checks import upload_allowed
from .dispersion import G, wavenumber
from .project import Project, RegularCase, SpectrumCase
from .scaling import FroudeScale
from .transfer import flap_tf, piston_tf


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wavemaker", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("new", help="create a project folder")
    s.add_argument("root"); s.add_argument("--name", required=True); s.add_argument("--notes", default="")

    s = sub.add_parser("regular", help="add a monochromatic case")
    s.add_argument("root"); s.add_argument("case")
    s.add_argument("--H", type=float, required=True); s.add_argument("--T", type=float, required=True)
    s.add_argument("--h", type=float, required=True)
    s.add_argument("--n-steady", type=int, default=30); s.add_argument("--ramp", type=int, default=5)

    s = sub.add_parser("spectrum", help="add an irregular case")
    s.add_argument("root"); s.add_argument("case")
    s.add_argument("--type", default="jonswap", choices=["jonswap", "pm", "csv"])
    s.add_argument("--Hs", type=float); s.add_argument("--Tp", type=float)
    s.add_argument("--h", type=float, required=True)
    s.add_argument("--gamma", type=float, default=3.3); s.add_argument("--csv")
    s.add_argument("--duration", type=float, help="record length (s)")
    s.add_argument("--fmin", type=float, default=0.5, help="f_min / fp")
    s.add_argument("--fmax", type=float, default=3.0, help="f_max / fp")
    s.add_argument("--mode", default="deterministic", choices=["deterministic", "random_complex"])
    s.add_argument("--seed", type=int); s.add_argument("--ramp", type=int, default=5)

    s = sub.add_parser("generate", help="generate, check and store a case")
    s.add_argument("root"); s.add_argument("case"); s.add_argument("--plot", action="store_true")

    s = sub.add_parser("list", help="list cases in a project"); s.add_argument("root")

    s = sub.add_parser("info", help="dispersion and transfer function for T, h")
    s.add_argument("--T", type=float, required=True); s.add_argument("--h", type=float, required=True)
    s.add_argument("--H", type=float, help="wave height, to report stroke")

    s = sub.add_parser("scale", help="Froude-scale prototype conditions to the model")
    s.add_argument("--lam", type=float, required=True)
    s.add_argument("--H", type=float, required=True); s.add_argument("--T", type=float, required=True)
    s.add_argument("--h", type=float, required=True)
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)

    if a.cmd == "new":
        p = Project.create(a.root, a.name, a.notes)
        print(f"Created {p.root}. Unconfigured facility values: {', '.join(p.facility.unconfigured())}")
    elif a.cmd == "regular":
        Project.open(a.root).save_case(RegularCase(a.case, a.H, a.T, a.h, a.n_steady, a.ramp))
        print(f"Saved case {a.case}")
    elif a.cmd == "spectrum":
        Project.open(a.root).save_case(SpectrumCase(
            a.case, a.h, a.type, a.Hs, a.Tp, a.gamma, a.csv, a.duration,
            a.fmin, a.fmax, a.mode, a.seed, a.ramp))
        print(f"Saved case {a.case}")
    elif a.cmd == "list":
        print("\n".join(Project.open(a.root).list_cases()) or "(no cases)")
    elif a.cmd == "generate":
        p = Project.open(a.root)
        s, res = p.generate(a.case)
        for r in res:
            print(r)
        ok = upload_allowed(res)
        print(f"\nUPLOAD {'ALLOWED' if ok else 'BLOCKED'}")
        if a.plot:
            from .plotting import plot_series
            out = p.root / "series" / f"{a.case}.png"
            plot_series(s, out, window_s=None if s.meta["kind"] == "regular" else 120)
            print(f"Plot: {out}")
        return 0 if ok else 2
    elif a.cmd == "info":
        k = wavenumber(2 * np.pi / a.T, a.h, G)
        kh = k * a.h
        print(f"L = {2*np.pi/k:.4f} m, k = {k:.4f} rad/m, kh = {kh:.4f}, h/L = {a.h*k/(2*np.pi):.4f}")
        print(f"H/S piston = {piston_tf(kh):.4f}, flap = {flap_tf(kh):.4f}")
        if a.H:
            print(f"Stroke for H = {a.H} m: piston {a.H/piston_tf(kh)*1e3:.1f} mm, "
                  f"flap {a.H/flap_tf(kh)*1e3:.1f} mm (stroke at still-water level)")
    elif a.cmd == "scale":
        fs = FroudeScale(a.lam)
        H, T, h = fs.to_model(a.H, a.T, a.h)
        print(fs.describe())
        print(f"Model: H = {H:.4f} m, T = {T:.4f} s, h = {h:.4f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
