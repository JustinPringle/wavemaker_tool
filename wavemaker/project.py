"""Project storage: one folder per project.

<project>/
    project.json        name, created, notes
    facility.json       tank and paddle parameters (see config.py)
    cases/<case>.json   wave case definitions
    series/<case>.csv   generated t, x, eta
    series/<case>_plc.csv   x in PLC millimetres, with dt in the header
    series/<case>_checks.txt
    run_log.jsonl       one JSON line per event
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .checks import run_checks, upload_allowed
from .config import Facility
from .signals import WaveSeries, irregular, monochromatic
from .spectra import SpectrumSpec
from .units import paddle_m_to_plc_mm


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RegularCase:
    """Monochromatic case at model scale (m, s)."""

    name: str
    H: float
    T: float
    h: float
    n_steady: int = 30
    n_ramp: int = 5
    kind: str = "regular"


@dataclass
class SpectrumCase:
    """Irregular case at model scale (m, s)."""

    name: str
    h: float
    spectrum: str = "jonswap"          # jonswap | pm | csv
    Hs: float | None = None
    Tp: float | None = None
    gamma: float = 3.3
    csv_path: str | None = None
    duration_s: float | None = None
    f_min_factor: float = 0.5
    f_max_factor: float = 3.0
    amplitude_mode: str = "deterministic"
    seed: int | None = None
    n_ramp: int = 5
    match_hm0_in_band: bool = True
    kind: str = "irregular"


def case_from_dict(d: dict):
    return (RegularCase if d.get("kind") == "regular" else SpectrumCase)(**d)


@dataclass
class Project:
    root: Path
    facility: Facility = field(default_factory=Facility)

    # ------------------------------------------------------------------
    @classmethod
    def create(cls, root: str | Path, name: str, notes: str = "") -> "Project":
        root = Path(root)
        if (root / "project.json").exists():
            raise FileExistsError(f"A project already exists at {root}.")
        for sub in ("cases", "series"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        (root / "project.json").write_text(json.dumps(
            dict(name=name, created=_now(), notes=notes, tool_version=__version__), indent=2))
        p = cls(root)
        p.facility.save(root / "facility.json")
        p.log("project_created", name=name)
        return p

    @classmethod
    def open(cls, root: str | Path) -> "Project":
        root = Path(root)
        if not (root / "project.json").exists():
            raise FileNotFoundError(f"No project at {root}.")
        return cls(root, Facility.load(root / "facility.json"))

    # ------------------------------------------------------------------
    def log(self, event: str, **data) -> None:
        with open(self.root / "run_log.jsonl", "a") as fh:
            fh.write(json.dumps(dict(time=_now(), event=event, **data), default=float) + "\n")

    def save_case(self, case) -> Path:
        path = self.root / "cases" / f"{case.name}.json"
        path.write_text(json.dumps(asdict(case), indent=2))
        self.log("case_saved", case=case.name)
        return path

    def load_case(self, name: str):
        return case_from_dict(json.loads((self.root / "cases" / f"{name}.json").read_text()))

    def list_cases(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "cases").glob("*.json"))

    # ------------------------------------------------------------------
    def generate(self, name: str) -> tuple[WaveSeries, list]:
        """Generate, check and store a case. The seed is written back to the case."""
        c = self.load_case(name)
        fac = self.facility
        if c.kind == "regular":
            s = monochromatic(c.H, c.T, c.h, fac.sample_interval_s, fac.paddle_type,
                              c.n_steady, c.n_ramp, fac.g)
        else:
            spec = SpectrumSpec(c.spectrum, c.Hs, c.Tp, c.gamma, c.csv_path)
            s = irregular(spec, c.h, fac.sample_interval_s, fac.paddle_type, c.duration_s,
                          c.f_min_factor, c.f_max_factor, c.amplitude_mode, c.seed,
                          c.n_ramp, c.match_hm0_in_band, fac.g)
            if c.seed is None:
                c.seed = s.meta["seed"]
                self.save_case(c)
        results = run_checks(s, fac)
        ok = upload_allowed(results)

        d = self.root / "series"
        np.savetxt(d / f"{name}.csv", np.column_stack([s.t, s.x, s.eta]),
                   delimiter=",", header="t_s,x_m,eta_m", comments="", fmt="%.6f")
        np.savetxt(d / f"{name}_plc.csv", paddle_m_to_plc_mm(s.x), fmt="%.4f",
                   header=f"dt_s={s.dt}\nn_samples={s.x.size}\nx_mm", comments="# ")
        report = "\n".join(str(r) for r in results)
        (d / f"{name}_checks.txt").write_text(
            report + f"\n\nUPLOAD {'ALLOWED' if ok else 'BLOCKED'}\n")
        self.log("series_generated", case=name, upload_allowed=ok,
                 meta={k: v for k, v in s.meta.items() if np.isscalar(v)},
                 blocking=[r.name for r in results if r.blocks])
        return s, results
