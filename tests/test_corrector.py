"""The CEOS-like aberration corrector model (de_twin.column.corrector) and its Column integration."""

import json

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.column import Column, Corrector, CorrectorError
from de_twin.column.ceos import (
    DONE,
    FAILED,
    IDLE,
    RUNNING,
    induced_c1a1,
    native_aberrations,
    pi4_angle_mrad,
)
from de_twin.optics.aberrations import Aberrations


def stem_column(seed=0, corrector="probe", **opts):
    clk = ManualClock()
    col = Column(clock=clk, corrector=corrector, seed=seed, allow_ht=True, corrector_options=opts or None)
    col.set_tem_stem(1)  # the probe corrector measures on the Ronchigram
    return col, clk


# ------------------------------------------------------------------ physics
def test_induced_aberrations_textbook():
    t = 0.02 * np.exp(1j * 0.3)
    c1, a1 = induced_c1a1({"C3": 1e6}, t)
    assert c1 == pytest.approx(2 * 1e6 * abs(t) ** 2)
    assert a1 == pytest.approx(1e6 * t**2)
    c1, a1 = induced_c1a1({"C1": 5.0, "A1": 3 + 4j}, t)
    assert c1 == pytest.approx(5.0) and a1 == pytest.approx(3 + 4j)


@pytest.mark.parametrize("seed", range(4))
def test_native_vs_corrected_pi4_angle(seed):
    col = Column(corrector="probe", seed=seed, clock=ManualClock())
    unit = col.corrector.served
    native = pi4_angle_mrad(unit.native)
    tuned = col.corrector.pi4_angle_mrad()
    assert 4.5 < native < 7.5  # C3 ~1.2 mm at 200 kV
    assert 22.0 < tuned < 33.0  # a good CEOS tune
    r = col.corrector.get()
    assert abs(r["C3"]) < 1500 and abs(r["B2"]) < 50 and abs(r["A2"]) < 50
    assert 1e6 < abs(r["C5"]) < 4e6  # C5 is not corrected: a few mm
    assert abs(unit.native["C3"]) == pytest.approx(1.2e6, rel=0.05)


def test_state_carries_the_residual():
    col = Column(corrector="both", seed=3, clock=ManualClock())
    s = col.state()
    assert s.corrector == "both"
    assert Aberrations(s.probe_aberrations)["C3"] != 0 and Aberrations(s.image_aberrations)["C3"] != 0
    assert Aberrations(s.probe_aberrations).key() != Aberrations(s.image_aberrations).key()
    s = Column(corrector="image", clock=ManualClock()).state()
    assert s.corrector == "image" and s.probe_aberrations == {} and s.image_aberrations
    s = Column(clock=ManualClock()).state()
    assert s.corrector == "none" and s.probe_aberrations == {} and s.image_aberrations == {}
    assert Column(clock=ManualClock()).corrector is None


def test_deterministic_from_seed():
    a = Column(corrector="probe", seed=5, clock=ManualClock()).state().probe_aberrations
    b = Column(corrector="probe", seed=5, clock=ManualClock()).state().probe_aberrations
    c = Column(corrector="probe", seed=6, clock=ManualClock()).state().probe_aberrations
    assert a == b and a != c


# ------------------------------------------------------------------ drift
def test_drift_grows_and_is_read_pattern_independent():
    def run(read_every):
        clk = ManualClock()
        col = Column(corrector="probe", seed=2, clock=clk)
        r0 = col.corrector.get()
        t = 0.0
        while t < 3600:
            step = read_every or 3600
            clk.advance(step)
            t += step
            if read_every:
                col.state()
        return r0, col.corrector.get()

    r0, r1 = run(0)
    _, r1b = run(36.0)
    for n in ("C1", "A1", "B2", "A2", "C3"):
        assert r1[n] == pytest.approx(r1b[n], abs=1e-9)  # same path however often it is read
    d = r1 - r0
    assert abs(d["C1"]) > 2.0 and abs(d["B2"]) > 5.0  # an hour of drift is worth retuning
    assert abs(d["C5"]) == 0  # C5 does not drift
    assert pi4_angle_mrad(r1) < pi4_angle_mrad(r0) + 1.0


def test_drift_rates_configurable():
    clk = ManualClock()
    col = Column(corrector="probe", seed=2, clock=clk, corrector_options={"drift_rates": {}})
    r0 = col.corrector.get()
    clk.advance(10_000)
    assert (col.corrector.get() - r0).coeffs == {}


# ------------------------------------------------------------------ tune perturbations
def test_ht_change_detunes():
    col, _clk = stem_column(4)
    before = col.corrector.get()
    angle = col.corrector.pi4_angle_mrad()
    col.set_ht_kv(300)
    after = col.corrector.get()
    d = after - before
    assert abs(d["C3"]) > 1000 and abs(d["B2"]) > 100
    assert col.corrector.pi4_angle_mrad() < angle / 2
    assert "300kV" in col.corrector.served.info()


def test_mode_change_kicks_low_orders():
    clk = ManualClock()
    col = Column(corrector="probe", seed=4, clock=clk)
    before = col.corrector.get()
    col.set_probe_mode("NBD")
    d = col.corrector.get() - before
    assert abs(d["C1"]) > 1 and abs(d["B2"]) > 1
    assert abs(d["C3"]) == pytest.approx(0, abs=1e-6)
    before = col.corrector.get()
    col.set_defocus_um(0.1)  # focus is the operator's, not the corrector's
    assert (col.corrector.get() - before).coeffs == {}


# ------------------------------------------------------------------ measure
def test_tableau_status_timing_and_result():
    col, clk = stem_column(1)
    u = col.corrector.served
    assert u.status() == IDLE
    ok, _why = u.post_tableau("Standard")
    assert ok and u.status() == RUNNING
    assert u.post_measure_c1a1() == (False, "the corrector is busy")
    clk.advance(1.0)
    assert u.status() == RUNNING
    clk.advance(10.0)
    assert u.status() == DONE
    r = u.last_measurement
    assert r.names == ("C1", "A1", "A2", "B2", "C3", "A3", "S3", "A4")  # CEOS order, up to A4
    assert u.last_result.startswith("C1=") and ";A4=" in u.last_result
    truth = col.corrector.get()
    for n in ("C1", "A1", "B2", "A2"):
        assert abs(r.aberrations[n] - truth[n]) < 5 * r.sigma_nm[n] + 1.0, n
    x, y, st = u.aberration("B2")
    assert st == 1 and complex(x, y) * 1e9 == pytest.approx(r.aberrations["B2"])
    assert u.aberration("C5")[2] == 0  # not fitted by a Standard tableau


def test_measurement_includes_operator_focus_and_stig():
    col, _clk = stem_column(1)
    truth = col.corrector.get()
    col.set_defocus_um(0.05)
    col.set_condenser_stig(0.01, -0.02)
    r = col.corrector.measure("C1A1")
    assert r.names == ("C1", "A1")
    assert r.aberrations["C1"].real == pytest.approx(truth["C1"].real + 50.0, abs=3.0)
    assert r.aberrations["A1"] == pytest.approx(truth["A1"] + complex(10, -20), abs=3.0)


def test_measure_refused_without_beam_or_wrong_mode():
    col = Column(corrector="probe", seed=1, clock=ManualClock())  # TEM mode
    with pytest.raises(CorrectorError, match="STEM"):
        col.corrector.measure("Fast")
    assert col.corrector.status() == FAILED
    col.set_tem_stem(1)
    col.set_beam_blank(True)
    with pytest.raises(CorrectorError, match="no beam"):
        col.corrector.measure("Fast")
    img = Column(corrector="image", seed=1, clock=ManualClock())
    assert img.corrector.measure("Fast").side == "image"


def test_out_of_tune_needs_small_tableau():
    col, _clk = stem_column(7)
    col.corrector.set("C3", 5e5)  # 0.5 mm of C3 left: outer tilts defocus beyond range
    with pytest.raises(CorrectorError, match="measurement range"):
        col.corrector.measure("Enhanced")
    r = col.corrector.measure("Fast")  # 9 mrad still works
    assert r.images_lost == 0


# ------------------------------------------------------------------ correct
def test_correct_needs_an_estimate_and_is_imperfect():
    col, _clk = stem_column(2)
    corr = col.corrector
    with pytest.raises(CorrectorError, match="No value for aberration B2 available"):
        corr.correct("B2")
    corr.set("B2", 500 + 0j)
    before = corr.get()
    after = corr.correct("B2", value_nm=500 + 0j)  # the exact value: only element errors remain
    assert 0.1 < abs(after["B2"]) < 75  # under/overshoot and rotation: a few % of 500 nm
    assert abs(after["C1"] - before["C1"]) > 0.5  # parasitic coupling B2 -> C1
    with pytest.raises(CorrectorError, match="No value"):  # the state of correction is consumed
        corr.correct("B2")
    with pytest.raises(CorrectorError, match='Unknown aberration "XX"'):
        corr.correct("XX", value_nm=1.0)


def test_partial_correction_fraction():
    col, _clk = stem_column(2)
    corr = col.corrector
    corr.set("A2", 400 + 0j)
    corr.correct("A2", fraction=0.5, value_nm=400 + 0j)
    assert abs(corr.get()["A2"]) == pytest.approx(200, rel=0.25)
    assert corr.served.soc["A2"] == pytest.approx(200 + 0j)  # the corrector's own belief


def auto_tune(corr, rounds=6):
    """What an automated tuning routine does: small tableau first, then larger ones,
    correcting what is significant (partially for the higher orders)."""
    plan = (
        ("Fast", ("C1", "A1", "A2", "B2")),
        ("Standard", ("C1", "A1", "A2", "B2", "C3", "A3", "S3")),
        ("Enhanced", ("C1", "A1", "A2", "B2", "C3", "A3", "S3")),
    )
    for _ in range(rounds):
        for tab, names in plan:
            try:
                r = corr.measure(tab)
            except CorrectorError:
                continue  # outer images out of range: the next round will get there
            for n in names:
                v = r.aberrations[n]
                if abs(v) > 2 * r.sigma_nm.get(n, 0.0):
                    corr.correct(n, 1.0 if n in ("C1", "A1") else 0.8)


@pytest.mark.parametrize("seed", [1, 3])
def test_auto_tune_converges_after_ht_change(seed):
    col, clk = stem_column(seed)
    corr = col.corrector
    col.set_ht_kv(300)
    col.set_ht_kv(200)
    assert corr.pi4_angle_mrad() < 12
    auto_tune(corr)
    r = corr.get()
    assert corr.pi4_angle_mrad() > 24.0
    assert abs(r["B2"]) < 15 and abs(r["A2"]) < 15 and abs(r["S3"]) < 300
    # the last correction moves C1 via coupling; one C1A1 pass trims it like an operator would
    corr.measure("C1A1")
    corr.correct("C1")
    corr.correct("A1")
    assert abs(corr.get()["C1"]) < 3 and abs(corr.get()["A1"]) < 3
    assert clk.now() > 100  # tableaux take time on the clock


# ------------------------------------------------------------------ misc ops
def test_alignment_save_and_load_restores_the_tune():
    col, _clk = stem_column(3)
    corr = col.corrector
    good = corr.get()
    data = corr.save_alignment()
    assert json.loads(data)["correctorType"] == "CESCOR"
    col.set_ht_kv(300)
    col.set_ht_kv(200)
    assert abs((corr.get() - good)["C3"]) > 1000
    corr.load_alignment(data)
    d = corr.get() - good  # only the drift of the few seconds of save / load remains
    assert all(abs(d[n]) < 5 for n in ("C1", "A1", "B2", "A2")) and abs(d["C3"]) < 100
    with pytest.raises(CorrectorError, match="empty"):
        corr.load_alignment("")


def test_beam_tilt_relative_and_absolute():
    col, clk = stem_column(3)
    corr = col.corrector
    assert corr.beam_tilt == (0.0, 0.0, False)
    u = corr.served
    # absolute with no WD estimate refuses, like CEOS
    completed, _running, why = u.set_beam_tilt(1e-3, 0.0, relative=False, wait_s=1.0, sleeper=clk.sleep)
    assert not completed and "No value for aberration WD available" in why
    corr.set_beam_tilt(1e-3, 0.0)
    corr.set_beam_tilt(1e-3, 2e-3)
    assert corr.beam_tilt == pytest.approx((2e-3, 2e-3, True))
    assert u.wd == pytest.approx(2e-3 + 2e-3j)
    # Now there is an estimate -- but a correction leaves its *target* as the state of
    # correction, so after relative moves the estimate is the last step (1e-3, 2e-3),
    # not the true tilt: an absolute set is only as good as that (the CEOS pitfall).
    corr.set_beam_tilt(0.0, 0.0, relative=False)
    assert corr.beam_tilt[:2] == (0.0, 0.0)
    assert u.wd == pytest.approx(1e-3 + 0j)


def test_config_options():
    col, clk = stem_column(3)
    u = col.corrector.served
    assert u.post_get_config("detilt")[0]
    assert u.wait(sleeper=clk.sleep) == DONE and u.last_result == "true"
    u.post_set_config("use2ndOrder", "false")
    u.wait(sleeper=clk.sleep)
    u.post_get_config("use2ndOrder")
    u.wait(sleeper=clk.sleep)
    assert u.last_result == "false"
    u.post_get_config("bogus")
    assert u.wait(sleeper=clk.sleep) == FAILED
    assert u.last_error == 'Server error: Unknown configuration option "bogus" (-32000)'


def test_corrector_needs_a_kind():
    with pytest.raises(ValueError):
        Corrector("none")
    with pytest.raises(ValueError):
        Column(corrector="sideways")
    assert native_aberrations(1, "probe").key() != native_aberrations(1, "image").key()
