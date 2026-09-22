"""Tests against limits, published tables and hand calculations."""
import numpy as np
import pytest

from wavemaker.checks import FAIL, UNCONF, WARN, run_checks, transverse_mode_frequencies, upload_allowed
from wavemaker.config import Facility
from wavemaker.dispersion import G, group_velocity_ratio, period_from_wavelength, wavelength, wavenumber
from wavemaker.scaling import FroudeScale
from wavemaker.signals import irregular, monochromatic, zero_downcrossing
from wavemaker.spectra import SpectrumSpec, jonswap, pierson_moskowitz
from wavemaker.transfer import flap_tf, piston_tf
from wavemaker.units import paddle_m_to_plc_mm


# --- Dispersion -------------------------------------------------------------
def test_dispersion_residual_machine_precision():
    omega = np.linspace(0.1, 30, 500)
    for h in (0.05, 0.3, 0.6, 5.0):
        k = wavenumber(omega, h)
        np.testing.assert_allclose(G * k * np.tanh(k * h), omega**2, rtol=1e-14)


def test_dispersion_known_root():
    # x tanh x = 1  ->  x = 1.19967864...
    assert wavenumber(np.sqrt(G), 1.0) == pytest.approx(1.1996786402577, rel=1e-12)


def test_dispersion_table_dean_dalrymple():
    # Dean & Dalrymple (1991) App. C: h/L0 = 0.10 -> h/L = 0.1410
    T = 1.0
    h = 0.10 * G * T**2 / (2 * np.pi)
    assert h / wavelength(T, h) == pytest.approx(0.1410, abs=5e-5)


def test_dispersion_limits():
    omega = 2.0
    assert wavenumber(omega, 100.0) == pytest.approx(omega**2 / G, rel=1e-12)
    h = 1e-4
    assert wavenumber(omega, h) == pytest.approx(omega / np.sqrt(G * h), rel=1e-4)
    assert wavenumber(0.0, 1.0) == 0.0


def test_period_from_wavelength_spreadsheet_case():
    # waveDesign.xlsx issue 2: L = 2 m, h = 0.3 m -> T = 1.32 s (sheet gave 2.04 s)
    assert period_from_wavelength(2.0, 0.3) == pytest.approx(1.319, abs=1e-3)
    assert wavelength(period_from_wavelength(2.0, 0.3), 0.3) == pytest.approx(2.0, rel=1e-12)


def test_group_velocity_limits():
    assert group_velocity_ratio(1e-6) == pytest.approx(1.0)
    assert group_velocity_ratio(50.0) == pytest.approx(0.5)


# --- Transfer functions -----------------------------------------------------
def test_piston_hand_value_and_limits():
    # 2(cosh 2 - 1)/(sinh 2 + 2) = 0.98179
    assert piston_tf(1.0) == pytest.approx(0.981789, abs=1e-6)
    assert piston_tf(1e-4) == pytest.approx(1e-4, rel=1e-4)
    assert piston_tf(30.0) == 2.0
    assert piston_tf(19.999) == pytest.approx(2.0, rel=1e-12)


def test_flap_hand_value_and_limits():
    assert flap_tf(1.0) == pytest.approx(0.528088, abs=1e-6)
    assert flap_tf(1e-3) == pytest.approx(5e-4, rel=1e-3)
    assert flap_tf(20.0 - 1e-9) == pytest.approx(flap_tf(20.0 + 1e-9), rel=1e-8)


def test_transfer_functions_monotonic():
    kh = np.linspace(0.01, 12, 2000)   # beyond ~15, piston H/S equals 2 to machine precision
    assert np.all(np.diff(piston_tf(kh)) > 0)
    assert np.all(np.diff(flap_tf(kh)) > 0)
    assert np.all(flap_tf(kh) < piston_tf(kh))


# --- Spectra ----------------------------------------------------------------
@pytest.mark.parametrize("fn,kw", [(pierson_moskowitz, {}), (jonswap, {"gamma": 3.3}),
                                   (jonswap, {"gamma": 7.0})])
def test_spectrum_m0_and_peak(fn, kw):
    hs, tp = 0.05, 1.4
    f = np.linspace(1e-3, 30, 400_001)
    S = fn(f, hs, tp, **kw)
    assert np.trapezoid(S, f) == pytest.approx(hs**2 / 16, rel=2e-4)
    assert f[np.argmax(S)] == pytest.approx(1 / tp, rel=1e-3)


def test_jonswap_gamma1_equals_pm():
    f = np.linspace(0.2, 5, 500)
    np.testing.assert_allclose(jonswap(f, 0.1, 1.5, 1.0), pierson_moskowitz(f, 0.1, 1.5), rtol=1e-6)


def test_csv_spectrum(tmp_path):
    f = np.linspace(0.1, 3, 300)
    S = jonswap(f, 0.04, 1.2)
    p = tmp_path / "s.csv"
    np.savetxt(p, np.column_stack([f, S]), delimiter=",")
    spec = SpectrumSpec("csv", csv_path=str(p))
    assert spec.hs == pytest.approx(0.04, rel=5e-3)
    assert spec.tp == pytest.approx(1.2, rel=1e-2)


# --- Monochromatic ----------------------------------------------------------
def test_monochromatic_amplitude_and_ends():
    H, T, h, dt = 0.05, 1.2, 0.5, 0.005
    s = monochromatic(H, T, h, dt)
    kh = wavenumber(2 * np.pi / T, h) * h
    assert np.max(np.abs(s.x[s.steady])) == pytest.approx(H / piston_tf(kh) / 2, rel=1e-3)
    assert s.x[0] == 0 and abs(s.x[-1]) < 1e-12
    assert np.max(np.abs(s.eta[s.steady])) == pytest.approx(H / 2, rel=1e-3)


# --- Irregular --------------------------------------------------------------
@pytest.fixture(scope="module")
def js():
    spec = SpectrumSpec("jonswap", 0.04, 1.5)
    return spec, irregular(spec, 0.5, 0.01, seed=42)


def test_irregular_realised_hm0_tp(js):
    spec, s = js
    assert s.meta["Hm0_realised"] == pytest.approx(0.04, rel=0.02)
    assert s.meta["Tp_realised"] == pytest.approx(1.5, rel=0.02)


def test_irregular_reproducible_and_seed_stored(js):
    spec, s = js
    s2 = irregular(spec, 0.5, 0.01, seed=42)
    np.testing.assert_array_equal(s.x, s2.x)
    s3 = irregular(spec, 0.5, 0.01)
    assert isinstance(s3.meta["seed"], int)
    np.testing.assert_array_equal(s3.x, irregular(spec, 0.5, 0.01, seed=s3.meta["seed"]).x)


def test_irregular_starts_and_ends_at_centre(js):
    _, s = js
    assert s.x[0] == 0 and abs(s.x[-1]) < 1e-9


def test_irregular_record_length_default(js):
    _, s = js
    assert s.meta["repeat_period_s"] >= 1200


def test_irregular_paddle_spectrum_follows_transfer_function(js):
    # |X_n| / |A_n| must equal 1 / TF(k_n h) at every component
    spec, s = js
    s0 = irregular(spec, 0.5, 0.01, seed=42, n_ramp=0)
    X = np.fft.rfft(s0.x); A = np.fft.rfft(s0.eta)
    f = np.fft.rfftfreq(s0.x.size, s0.dt)
    band = (f >= s0.meta["f_min"]) & (f <= s0.meta["f_max"])
    kh = wavenumber(2 * np.pi * f[band], 0.5) * 0.5
    np.testing.assert_allclose(np.abs(X[band]) / np.abs(A[band]), 1 / piston_tf(kh), rtol=1e-8)
    # x lags eta by 90 degrees
    np.testing.assert_allclose(np.angle(X[band] / A[band]), -np.pi / 2, atol=1e-8)


def test_random_complex_mode_runs():
    spec = SpectrumSpec("pm", 0.03, 1.2)
    s = irregular(spec, 0.5, 0.01, amplitude_mode="random_complex", seed=1)
    assert s.meta["Hm0_realised"] == pytest.approx(0.03, rel=0.06)


def test_zero_downcrossing_regular():
    t = np.arange(0, 20, 0.001)
    H, Ti = zero_downcrossing(0.02 * np.cos(2 * np.pi * t / 1.25), 0.001)
    np.testing.assert_allclose(H, 0.04, rtol=1e-4)
    np.testing.assert_allclose(Ti, 1.25, rtol=1e-4)


# --- Checks -----------------------------------------------------------------
def _fac(**kw):
    base = dict(stroke_margin_m=0.02, freeboard_margin_m=0.1,
                max_velocity_m_s=1.0, max_acceleration_m_s2=10.0)   # test values only
    base.update(kw)
    return Facility(**base)


def _status(res, name):
    return next(r.status for r in res if r.name.startswith(name))


def test_default_facility_blocks_upload():
    s = monochromatic(0.03, 1.2, 0.5, 0.01)
    res = run_checks(s, Facility())
    assert _status(res, "Paddle velocity") == UNCONF
    assert not upload_allowed(res)


def test_configured_small_wave_passes():
    res = run_checks(monochromatic(0.03, 1.5, 0.5, 0.01), _fac())
    assert upload_allowed(res), "\n".join(map(str, res))


def test_stroke_failure():
    # Very long wave in shallow water needs a large stroke
    res = run_checks(monochromatic(0.18, 4.0, 0.3, 0.01), _fac())
    assert _status(res, "Stroke") == FAIL


def test_breaking_failures():
    res = run_checks(monochromatic(0.40, 2.0, 0.45, 0.01), _fac())
    assert _status(res, "Depth-limited") == FAIL
    res = run_checks(monochromatic(0.12, 0.7, 0.5, 0.01), _fac())
    assert _status(res, "Steepness") == FAIL


def test_depth_over_working_limit():
    res = run_checks(monochromatic(0.03, 1.2, 0.7, 0.01), _fac())
    assert _status(res, "Water depth") == FAIL


def test_cross_mode_warning():
    h = 0.5
    f1 = transverse_mode_frequencies(1.2, h, G)[0]
    res = run_checks(monochromatic(0.02, 1 / f1, h, 0.01), _fac())
    assert _status(res, "Cross-tank") == WARN
    res = run_checks(monochromatic(0.02, 1 / (2 * f1), h, 0.01), _fac())
    assert _status(res, "Cross-tank") == WARN


def test_shallow_and_ursell_warnings():
    res = run_checks(monochromatic(0.02, 5.0, 0.2, 0.01), _fac(stroke_margin_m=0.0))
    assert _status(res, "Shallow") == WARN
    assert _status(res, "Ursell") == WARN


# --- Scaling and units ------------------------------------------------------
def test_froude_time_scale_spreadsheet_case():
    # waveDesign.xlsx issue 1: lambda = 10, T = 2.04 s -> 6.45 s (sheet gave 11.47 s)
    assert FroudeScale(10).to_prototype(0.1, 2.04, 0.5)[1] == pytest.approx(6.451, abs=1e-3)


def test_plc_units():
    np.testing.assert_allclose(paddle_m_to_plc_mm([0.1, -0.25]), [100.0, -250.0])
