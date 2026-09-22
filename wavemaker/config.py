"""Facility configuration for the EFML concrete wave tank.

Confirmed values carry defaults. Values still awaited from Festo or KS
Industrial Automation are ``None``; every limit check that depends on an
unset value reports ``unconfigured`` and blocks upload. Never replace a
``None`` with a guess: enter the manufacturer's figure in ``facility.json``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Facility:
    """Physical and control parameters of the tank and paddle (SI units)."""

    # Tank geometry (confirmed)
    tank_length_m: float = 20.0
    tank_width_m: float = 1.2
    tank_depth_m: float = 0.8
    max_water_depth_m: float = 0.6          # chosen for freeboard

    # Paddle (confirmed)
    paddle_type: str = "piston"             # "piston" or "flap"
    half_stroke_m: float = 0.300            # 600 mm total, 300 mm each side of home

    # [TBC] safety choices -- set by the lab, not by the drive
    stroke_margin_m: float | None = None    # kept clear of the mechanical end stops
    freeboard_margin_m: float | None = None # kept between the highest crest and the tank rim

    # [TBC] drive and axis limits -- from Festo / KS only
    max_velocity_m_s: float | None = None
    max_acceleration_m_s2: float | None = None

    # [TBC] depends on the CODESYS MainTask cycle and playback method
    sample_interval_s: float = 0.01         # 100 Hz, placeholder

    g: float = 9.81                          # m/s^2

    # ------------------------------------------------------------------
    def unconfigured(self) -> list[str]:
        """Names of fields still awaiting confirmation."""
        return [f.name for f in fields(self) if getattr(self, f.name) is None]

    def save(self, path: str | Path) -> None:
        """Write the configuration to JSON."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Facility":
        """Read a configuration from JSON; unknown keys raise an error."""
        data = json.loads(Path(path).read_text())
        names = {f.name for f in fields(cls)}
        extra = set(data) - names
        if extra:
            raise ValueError(f"Unknown facility keys: {sorted(extra)}")
        return cls(**data)
