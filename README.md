# wavemaker — EFML concrete wave tank

Phase 1 of the wave-maker tool: linear wavemaker numerics, limit checks,
project storage and a command-line front end. Phases 3–4 (the `WaveGen` POU
and OPC UA link) will read the `*_plc.csv` files this tool writes.

## Install and test

    pip install -e ".[test]"
    pytest

Needs NumPy 1.26+ (uses `np.trapezoid`); tested on NumPy 2.4.

## Workflow

    wavemaker new runs/2026-09 --name "Commissioning"
    wavemaker regular  runs/2026-09 reg_H05_T12  --H 0.05 --T 1.2 --h 0.5
    wavemaker spectrum runs/2026-09 js_Hs04_Tp15 --Hs 0.04 --Tp 1.5 --h 0.5
    wavemaker generate runs/2026-09 js_Hs04_Tp15 --plot
    wavemaker info  --T 1.2 --h 0.5 --H 0.05     # L, kh, H/S, stroke
    wavemaker scale --lam 25 --H 2 --T 10 --h 12  # Froude to model

`generate` writes `series/<case>.csv` (t, x, eta in m), `<case>_plc.csv`
(x in mm, dt in the header), `<case>_checks.txt` and a plot, and appends to
`run_log.jsonl`. Exit code 2 means upload is blocked.

## Modules

| Module | Contents |
|---|---|
| `dispersion` | Newton solver for ω² = gk tanh kh; L, T from L, C, Cg |
| `transfer` | Piston and flap H/S (Biesel), overflow-safe |
| `spectra` | JONSWAP, Pierson–Moskowitz, CSV spectrum |
| `signals` | Monochromatic and IFFT irregular series, ramps, zero-crossing |
| `checks` | Stroke, velocity, acceleration, breaking, freeboard, kh, Ursell, cross-tank modes, sampling, realised Hm0/Tp |
| `scaling` | Froude scaling (time scale √λ) |
| `units` | The only metre → PLC-millimetre conversion |
| `project` | Project folders, case JSON, series export, run log |
| `plotting`, `cli` | Plots and command line |

## Values still to confirm ([TBC])

Unset values live in each project's `facility.json`. Any check that depends
on one reports `unconfigured` and blocks upload.

| Item | Where | Source |
|---|---|---|
| Max paddle velocity, acceleration | `facility.json` | Festo / KS |
| Stroke margin from end stops | `facility.json` | Lab decision |
| Freeboard margin | `facility.json` | Lab decision |
| Sample interval (100 Hz placeholder) | `facility.json` | MainTask cycle / playback method |
| Drive sign convention | `units.PADDLE_SIGN` | KS |
| Default record length, ramp periods | `signals.py` | Lab decision |
| Band normalisation of Hm0 | `SpectrumCase.match_hm0_in_band` | Lab decision |
| Irregular breaking policy (per-wave warns) | `checks.py` | Lab decision |
| Minimum samples per wave (20) | `checks.py` | After Phase 3 tracking test |
