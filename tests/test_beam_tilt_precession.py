"""Beam tilt in diffraction, and precession: the tilt coils swept round a cone."""

from __future__ import annotations

import math

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.column import Column
from de_twin.crystal import library_for
from de_twin.render import Renderer
from de_twin.render.diffraction import beam_tilt_rotation, bucket_patterns, tilt_samples
from de_twin.render.renderer import _precession_frame
from de_twin.render.testing import SyntheticSpecimen, on_zone_grains
from de_twin.specimen.materials import GRAINS_PER_MATERIAL, MaterialId
from de_twin.state import Vec2

from test_render_diffraction import AU0, OPTICS, _saed_optics, bragg_spots

G200 = 2.0 / 0.4078
LAM = OPTICS.wavelength_nm


def _optics(**kw):
    from types import SimpleNamespace

    base = dict(vars(OPTICS), beam_tilt_mrad=(0.0, 0.0), precession_mrad=0.0, precession_hz=0.0,
                precession_descan=True, precession_phase_rad=0.0, precession_arc_rad=2 * math.pi)
    base.update(kw)
    return SimpleNamespace(**base)


def _pattern(zone=(0, 0, 1), t=20.0, **kw):
    gid = AU0 + 1
    return bucket_patterns([MaterialId.GOLD], [gid], [t], [1.0], on_zone_grains(gid, zone), _optics(**kw))[0]


def test_a_beam_tilted_by_the_bragg_angle_towards_minus_g_excites_g():
    lib = library_for(int(MaterialId.GOLD), 25.0)
    theta_b = LAM * G200 / 2.0  # rad
    m = np.eye(3)[None]
    j = int(np.argmin(np.abs(lib.g[:, 0] - G200) + np.abs(lib.g[:, 1]) + np.abs(lib.g[:, 2])))  # +g = (200)
    s = lambda tilt: float((lambda gl: 1 / LAM - math.sqrt(1 / LAM ** 2 - gl[0] ** 2 - gl[1] ** 2) - gl[2])(
        beam_tilt_rotation(tilt) @ lib.g[j]))
    assert abs(s((0.0, 0.0))) > 1e-3
    assert s((-theta_b, 0.0)) == pytest.approx(0.0, abs=1e-6), "towards -g: +g on the sphere"
    assert abs(s((+theta_b, 0.0))) > abs(s((0.0, 0.0))), "towards +g: further off"


def test_beam_tilt_moves_the_saed_pattern():
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM)
    r = Renderer(spec)
    a = r.render(_saed_optics(120.0))
    o = _saed_optics(120.0, beam_tilt_mrad=Vec2(1.0, -0.5))
    b = r.render(o)
    y, x = np.indices(a.shape)
    shift = ((b * x).sum() / b.sum() - (a * x).sum() / a.sum(),
             (b * y).sum() / b.sum() - (a * y).sum() / a.sum())
    mpp = o.extras["mrad_per_px"]
    assert shift == pytest.approx((1.0 / mpp, -0.5 / mpp), abs=0.05)


def test_beam_tilt_changes_which_reflections_are_excited():
    zone = _pattern()
    tilted = _pattern(beam_tilt_mrad=(1000 * LAM * G200 / 2.0, 0.0))  # Bragg for (-200)
    w = lambda p, gx: float(p.spots[np.isclose(p.spots[:, 0], gx, atol=0.05) & np.isclose(p.spots[:, 1], 0, atol=0.05), 2].sum())
    assert w(tilted, -G200) > 2 * w(zone, -G200)
    assert w(tilted, +G200) < w(zone, +G200)


def test_precession_excites_more_reflections_and_keeps_the_pattern_centred():
    zone = bragg_spots(_pattern(t=40.0))
    prec = _pattern(t=40.0, precession_mrad=20.0)
    ps = bragg_spots(prec)
    g = np.hypot(ps[:, 0], ps[:, 1])
    assert g.max() > np.hypot(zone[:, 0], zone[:, 1]).max() + 1.0, "reflections further out"
    # descanned: the direct beam of every tilt lands on the centre, none on the cone's ring
    r = np.hypot(prec.spots[:, 0], prec.spots[:, 1])
    ring = 20.0e-3 / LAM
    assert prec.spots[r < 1e-9, 2].sum() > 0.1
    assert prec.spots[np.abs(r - ring) < 0.02 * ring, 2].sum() < 1e-3
    assert prec.total == pytest.approx(1.0, rel=1e-6)


def test_precession_without_descan_sweeps_the_pattern_round_a_ring():
    theta = 10.0
    p = _pattern(t=5.0, precession_mrad=theta, precession_descan=False)
    radius = theta * 1e-3 / LAM  # 1/nm
    d = p.spots[p.spots[:, 2] > 0.01]
    assert np.allclose(np.hypot(d[:, 0], d[:, 1])[np.argsort(-d[:, 2])][:4], radius, rtol=0.05)


def test_a_frame_shorter_than_the_period_sees_part_of_the_cone():
    o = _saed_optics(120.0, precession_on=True, precession_mrad=10.0, precession_hz=100.0)
    full = _precession_frame(o, 0.0)
    assert full.precession_arc_rad == pytest.approx(2 * math.pi)  # 25 ms frames: 2.5 turns
    fast = o.__class__(**{**o.__dict__, "extras": {**o.extras, "exposure_s": 0.002}})
    f1, f2 = _precession_frame(fast, 0.0), _precession_frame(fast, 0.0025)
    assert f1.precession_arc_rad < math.pi and f1.precession_phase_rad != f2.precession_phase_rad
    assert len(tilt_samples(f1)) < len(tilt_samples(full))


def test_column_precession_properties():
    c = Column(clock=ManualClock())
    assert c.get("Precession") is False
    c.set("Precession", True)
    c.set("PrecessionAngle", 17.5)
    c.set("PrecessionFrequency", 250)
    c.set("PrecessionDescan", "false")
    s = c.state()
    assert (s.precession_on, s.precession_mrad, s.precession_hz, s.precession_descan) == (True, 17.5, 250.0, False)
    from de_twin.column import ColumnRefused

    with pytest.raises(ColumnRefused):
        c.set("PrecessionAngle", -1)


def test_precession_leaves_the_tem_image_cache_alone():
    """Precession changes diffraction, not the image: TEM frames still come from the cache."""
    from de_twin.render.testing import StubCamera

    from test_render_diffraction import _gold_specimen

    r = Renderer(_gold_specimen())
    from de_twin.optics import Calibration, OpticsConfig, derive_optics
    from de_twin.state import AcquisitionRequest, MicroscopeState

    s = MicroscopeState()
    s.precession_on, s.precession_mrad, s.precession_hz = True, 10.0, 100.0
    o = derive_optics(s, AcquisitionRequest(frame_time_s=0.0025), StubCamera(sensor_shape=(512, 512)),
                      Calibration.default(), OpticsConfig())
    r.render(o, time_s=0.0)
    before = r.frames_from_cache
    for k in range(1, 6):
        r.render(o, time_s=k * 0.0025)
    assert r.frames_from_cache - before == 5


def test_a_partial_cone_reuses_the_cached_tilts(monkeypatch):
    import de_twin.render.diffraction as D

    D._CONE_CACHE.clear()
    calls = []
    real = D._bucket_patterns_at
    monkeypatch.setattr(D, "_bucket_patterns_at", lambda *a, **k: calls.append(1) or real(*a, **k))
    gid = AU0 + 1
    grains = on_zone_grains(gid, (0, 0, 1))  # one specimen, as a renderer holds it
    step = 2 * math.pi / 24
    for j in range(24):  # a frame of 1/8 cone, starting at each phase in turn
        bucket_patterns([MaterialId.GOLD], [gid], [20.0], [1.0], grains,
                        _optics(precession_mrad=10.0, precession_arc_rad=3 * step, precession_phase_rad=j * step))
    assert len(calls) == 24, "each of the cone's 24 tilts excited once"


def test_stem_tables_are_not_rebuilt_as_the_precession_phase_moves(monkeypatch):
    import de_twin.render.stem as st
    from de_twin.clock import ManualClock as MC
    from de_twin.state import TemStem
    from de_twin.twin import DigitalTwin

    builds = []
    real = st.build_tables
    monkeypatch.setattr(st, "build_tables", lambda *a, **k: builds.append(1) or real(*a, **k))
    tw = DigitalTwin("Dense Au on holey C", camera="DESim", clock=MC(), seed=3)
    tw.column.set_tem_stem(TemStem.STEM)
    tw.column.set("Precession", True)
    req = tw.request(frame_time_s=0.001)
    req.scan.enabled = True
    req.scan.size = (16, 16)
    for i in range(12):
        tw.flux(req, frame_index=i, time_s=i * 0.001)
    assert len(builds) == 1
