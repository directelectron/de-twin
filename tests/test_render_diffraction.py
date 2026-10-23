"""Diffraction patterns from the crystal library, drawing, the LRU cache and SAED."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from de_twin.optics import Calibration, OpticsConfig, derive_optics
from de_twin.render import RenderConfig, Renderer
from de_twin.render.diffraction import (
    DiffractionCache,
    Pattern,
    PatternOptions,
    amorphous_fraction,
    annulus_fraction,
    bucket_patterns,
    powder_pattern,
    render_pattern,
)
from de_twin.render.testing import Particle, StubCamera, SyntheticSpecimen, on_zone_grains
from de_twin.specimen.materials import GRAINS_PER_MATERIAL, MaterialId
from de_twin.state import AcquisitionRequest, MicroscopeState, Projection, Vec2

A_AU = 0.4078
G111 = math.sqrt(3) / A_AU
G200 = 2.0 / A_AU
LAM200 = 0.0025079
OPTICS = SimpleNamespace(alpha_rad=0.0, beta_rad=0.0, wavelength_nm=LAM200, ht_kv=200.0, convergence_mrad=0.0)
AU0 = MaterialId.GOLD * GRAINS_PER_MATERIAL


def _radial_peak(img, cx, cy, r_guess, window=0.06):
    """Radius of the strongest ring near r_guess, on a background-detrended radial profile."""
    from scipy.ndimage import uniform_filter1d

    y, x = np.indices(img.shape)
    r = np.hypot(x - cx, y - cy)
    rb = np.round(r * 4).astype(int)
    prof = np.bincount(rb.ravel(), weights=img.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
    prof = prof - uniform_filter1d(prof, 161)  # remove halos / diffuse background (~40 px)
    rr = np.arange(len(prof)) / 4.0
    sel = (rr > r_guess * (1 - window)) & (rr < r_guess * (1 + window))
    return float(rr[sel][np.argmax(prof[sel])])


def zone_pattern(mid, zone, t=20.0, cryst=1.0, k=1):
    gid = mid * GRAINS_PER_MATERIAL + k
    return bucket_patterns([mid], [gid], [t], [cryst], on_zone_grains(gid, zone), OPTICS)[0]


def bragg_spots(p: Pattern, rel=1e-3):
    s = p.spots[1:]
    return s[s[:, 2] > rel * s[:, 2].max()]


def is_symmetric(spots, n, tol=1e-6):
    c, s = math.cos(2 * math.pi / n), math.sin(2 * math.pi / n)
    rot = spots[:, :2] @ np.array([[c, s], [-s, c]])
    d = np.hypot(rot[:, None, 0] - spots[None, :, 0], rot[:, None, 1] - spots[None, :, 1])
    j = d.argmin(axis=1)
    return bool(np.all(d[np.arange(len(j)), j] < tol) and np.allclose(spots[j, 2], spots[:, 2], rtol=1e-6))


# ------------------------------------------------------------ synthesis
def test_patterns_sum_to_one():
    grains = on_zone_grains(AU0 + 1, (0, 1, 1))
    mats = [
        MaterialId.VACUUM,
        MaterialId.AMORPHOUS_CARBON,
        MaterialId.GOLD,
        MaterialId.GOLD,
        MaterialId.IRON,
        MaterialId.SILICON,
    ]
    gids = [-1, -1, AU0 + 1, -1, MaterialId.IRON * 512 + 3, MaterialId.SILICON * 512 + 7]
    for t in (0.0, 8.0, 40.0, 200.0):
        for c in (0.0, 0.4, 1.0):
            for p in bucket_patterns(mats, gids, [t] * 6, [c] * 6, grains, OPTICS):
                assert p.total == pytest.approx(1.0, abs=1e-9)
                assert p.spots[0, 2] > 0  # the direct beam survives
    assert bucket_patterns([0], [-1], [50], [1], grains, OPTICS)[0].spots.shape == (1, 3)


@pytest.mark.parametrize("zone,fold,not_fold", [((0, 0, 1), 4, 3), ((0, 1, 1), 2, 4), ((1, 1, 1), 6, 4)])
def test_zone_axis_symmetry(zone, fold, not_fold):
    s = bragg_spots(zone_pattern(MaterialId.GOLD, zone))
    assert len(s) >= 4
    assert is_symmetric(s, fold)
    assert not is_symmetric(s, not_fold)


def test_extinction_rules_and_spacings():
    # FCC [001]: {200} and {220} but no {100}/{110}
    g = np.hypot(*bragg_spots(zone_pattern(MaterialId.GOLD, (0, 0, 1)))[:, :2].T)
    assert np.isclose(g.min(), G200)
    assert not np.any(np.isclose(g, 1 / A_AU)) and not np.any(np.isclose(g, math.sqrt(2) / A_AU))
    # FCC [011]: {111} and {200}, spacing ratio 2/sqrt(3)
    g = np.hypot(*bragg_spots(zone_pattern(MaterialId.GOLD, (0, 1, 1)))[:, :2].T)
    assert np.isclose(g.min(), G111) and np.any(np.isclose(g, G200))
    assert g[np.isclose(g, G200)][0] / g.min() == pytest.approx(2 / math.sqrt(3), rel=1e-9)
    # diamond [011]: {111} present, {200} forbidden
    g = np.hypot(*bragg_spots(zone_pattern(MaterialId.SILICON, (0, 1, 1)))[:, :2].T)
    assert np.isclose(g.min(), math.sqrt(3) / 0.5431)
    assert not np.any(np.isclose(g, 2 / 0.5431))
    # BCC [001]: {110} first
    g = np.hypot(*bragg_spots(zone_pattern(MaterialId.IRON, (0, 0, 1)))[:, :2].T)
    assert np.isclose(g.min(), math.sqrt(2) / 0.2866)


def test_thickness_moves_intensity_out_of_direct_beam():
    thin = zone_pattern(MaterialId.GOLD, (0, 1, 1), t=0.0)
    thick = zone_pattern(MaterialId.GOLD, (0, 1, 1), t=40.0)
    assert thin.spots[0, 2] == pytest.approx(1.0 - 0.08, abs=1e-9)  # only the film halo
    assert thick.spots[0, 2] < 0.6 and thick.spots[1:, 2].sum() > 0.4
    # per-material saturation: 20 nm of carbon scatters far less than 20 nm of gold
    assert amorphous_fraction(MaterialId.AMORPHOUS_CARBON, 20.0) == pytest.approx(0.75 * 0.85 * 20 / 120)
    assert amorphous_fraction(
        MaterialId.AMORPHOUS_CARBON, 20.0, PatternOptions(scaled_saturation=False)
    ) == pytest.approx(0.75 * 0.85)


def test_off_zone_grain_scatters_less_than_on_zone():
    on = zone_pattern(MaterialId.GOLD, (0, 1, 1), t=20.0)
    gid = AU0 + 1
    grains = on_zone_grains(gid, (0, 1, 1))
    tilted = SimpleNamespace(**{**vars(OPTICS), "alpha_rad": math.radians(4.0)})
    off = bucket_patterns([MaterialId.GOLD], [gid], [20.0], [1.0], grains, tilted)[0]
    n_on = int((on.spots[1:, 2] > 0.01).sum())
    n_off = int((off.spots[1:, 2] > 0.01).sum())
    assert n_on >= 6 and n_off <= 4  # only the systematic row along the tilt axis stays excited


def test_crystallinity_moves_bragg_into_glassy_halo():
    glass = zone_pattern(MaterialId.GOLD, (0, 1, 1), t=40.0, cryst=0.0)
    xtal = zone_pattern(MaterialId.GOLD, (0, 1, 1), t=40.0, cryst=1.0)
    assert glass.spots[1:, 2].sum() == pytest.approx(0.0)
    assert xtal.spots[1:, 2].sum() > 0.3
    assert glass.rings[:, 2].sum() > xtal.rings[:, 2].sum()


def test_powder_ring_radii_and_au_ratio():
    s = powder_pattern(MaterialId.GOLD, 40.0, 1.0, OPTICS)
    recip = 0.02
    img = render_pattern(s, (1024, 1024), recip, 3.0)
    assert 0.85 < img.sum() <= 1.0 + 1e-5  # the outermost rings fall off the detector
    r111 = _radial_peak(img, 512, 512, G111 / recip)
    r200 = _radial_peak(img, 512, 512, G200 / recip)
    assert r111 == pytest.approx(G111 / recip, abs=0.5)
    assert r200 == pytest.approx(G200 / recip, abs=0.5)
    assert r200 / r111 == pytest.approx(2 / math.sqrt(3), rel=0.005)
    # {111} (multiplicity 8) is the strongest ring
    rings = s.rings[s.rings[:, 2] > 0]
    assert rings[np.argmax(rings[:, 2]), 0] == pytest.approx(G111, rel=1e-3)


def test_render_normalisation_and_centre():
    p = bucket_patterns([0], [-1], [0], [1], None, OPTICS)[0]
    for radius in (0.3, 5.0, 40.0):
        img = render_pattern(p, (128, 128), 0.02, radius, center=(60.0, 70.0))
        assert img.sum() == pytest.approx(1.0, rel=1e-5)
        y, x = np.indices(img.shape)
        assert (img * x).sum() == pytest.approx(60.0, abs=1e-3)
        assert (img * y).sum() == pytest.approx(70.0, abs=1e-3)


def test_annulus_fraction():
    p = bucket_patterns([0], [-1], [0], [1], None, OPTICS)[0]
    assert annulus_fraction(p, 10.0, LAM200, 0.0, 10.0) == pytest.approx(1.0)
    assert annulus_fraction(p, 10.0, LAM200, 0.0, 5.0) == pytest.approx(0.25, abs=0.02)
    assert annulus_fraction(p, 10.0, LAM200, 20.0, 200.0) == 0.0
    au = zone_pattern(MaterialId.GOLD, (0, 1, 1), t=40.0)
    bragg_mrad = 1000 * LAM200 * G111
    assert annulus_fraction(au, 1.0, LAM200, bragg_mrad - 2, bragg_mrad + 2) > 0.05


# ---------------------------------------------------------------- cache
def test_cache_lru_budget():
    cache = DiffractionCache(budget_mb=1)  # 1 MiB = four 256x256 float32 patterns

    def render(k):
        return np.zeros(k[1], np.float32)

    keys = [("dp", (256, 256), i) for i in range(6)]
    for k in keys:
        cache.get(k, render)
    assert len(cache) == 4 and cache.stats.evictions == 2 and cache.stats.misses == 6
    assert keys[0] not in cache and keys[5] in cache
    cache.get(keys[2], render)  # refresh -> most recent
    cache.get(keys[0], render)  # miss, evicts the LRU (keys[3])
    assert keys[2] in cache and keys[3] not in cache
    assert cache.stats.hits == 1
    big = ("dp", (1024, 1024), 99)  # 4 MiB > budget: kept alone
    cache.get(big, render)
    assert len(cache) == 1 and big in cache
    with pytest.raises(ValueError):
        cache.get(big, render)[0, 0] = 1.0  # read-only


# ----------------------------------------------------------------- SAED
def _gold_specimen(n=400, seed=3):
    rng = np.random.default_rng(seed)
    parts = [
        Particle(float(x), float(y), 10.0, MaterialId.GOLD, int(g))
        for x, y, g in zip(rng.uniform(-0.2, 0.2, n), rng.uniform(-0.2, 0.2, n), range(n))
    ]
    return SyntheticSpecimen(film_thickness_nm=10.0, particles=parts)


def _saed_optics(cl_mm=120.0, cam=StubCamera(sensor_shape=(1024, 1024)), config=None, **kw):
    s = MicroscopeState()
    s.projection = Projection.DIFFRACTION
    s.camera_length_mm = cl_mm
    for k, v in kw.items():
        setattr(s, k, v)
    return derive_optics(
        s, AcquisitionRequest(), cam, Calibration.default(), config or OpticsConfig(sa_aperture_um=0.5)
    )


def test_saed_rings_are_camera_length_consistent():
    spec = _gold_specimen()
    r = Renderer(spec)
    for cl in (120.0, 240.0):
        o = _saed_optics(cl)
        img = r.render(o)
        assert img.shape == (1024, 1024) and img.min() >= 0
        cx, cy = o.diffraction_center_px
        for g in (G111, G200):
            rp = _radial_peak(img, cx, cy, g / o.recip_pixel_inv_nm)
            assert rp * o.recip_pixel_inv_nm == pytest.approx(g, rel=0.01)
    a, b = _saed_optics(120.0), _saed_optics(240.0)
    assert a.recip_pixel_inv_nm / b.recip_pixel_inv_nm == pytest.approx(2.0)


def test_saed_au_ring_ratio():
    o = _saed_optics(120.0)
    img = Renderer(_gold_specimen()).render(o)
    cx, cy = o.diffraction_center_px
    r111 = _radial_peak(img, cx, cy, G111 / o.recip_pixel_inv_nm)
    r200 = _radial_peak(img, cx, cy, G200 / o.recip_pixel_inv_nm)
    assert r200 / r111 == pytest.approx(2 / math.sqrt(3), rel=0.01)


def test_saed_folds_excess_grains_into_powder(monkeypatch):
    from de_twin.render import saed

    o = _saed_optics(120.0)
    spec = _gold_specimen()
    exact = Renderer(spec).render(o)
    monkeypatch.setattr(saed, "MAX_EXACT_GRAINS", 5)
    r = Renderer(spec)
    folded = r.render(o)
    assert r.stats["saed_exact_grains"] == 5
    cx, cy = o.diffraction_center_px
    assert _radial_peak(folded, cx, cy, G111 / o.recip_pixel_inv_nm) * o.recip_pixel_inv_nm == pytest.approx(
        G111, rel=0.01
    )
    assert folded.sum() == pytest.approx(exact.sum(), rel=0.02)  # same electrons, redistributed


def test_saed_total_electrons_and_vacuum():
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM)
    o = _saed_optics(120.0)
    img = Renderer(spec).render(o)
    assert img.sum() == pytest.approx(o.pattern_e_per_s, rel=1e-3)
    cx, cy = o.diffraction_center_px
    y, x = np.indices(img.shape)
    assert img[np.hypot(x - cx, y - cy) > o.disk_radius_px + 12].sum() < 1e-6 * img.sum()


def test_saed_diffraction_shift_moves_pattern():
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM)
    r = Renderer(spec)
    a = r.render(_saed_optics(120.0))
    o = _saed_optics(120.0, diffraction_shift_mrad=Vec2(1.0, 0.5))
    b = r.render(o)
    y, x = np.indices(a.shape)
    shift = (
        (b * x).sum() / b.sum() - (a * x).sum() / a.sum(),
        (b * y).sum() / b.sum() - (a * y).sum() / a.sum(),
    )
    mpp = o.extras["mrad_per_px"]
    assert shift == pytest.approx((1.0 / mpp, 0.5 / mpp), abs=0.05)


def test_saed_beam_stop():
    spec = _gold_specimen()
    o = _saed_optics(120.0)
    r = Renderer(spec, RenderConfig(beam_stop_radius_px=20.0))
    img = r.render(o)
    ax, ay = o.extras["axis_center_px"]
    assert img[int(ay), int(ax)] == 0.0
    assert img[int(ay) + 200, int(ax)] == 0.0  # the shaft runs to +y
    assert img[int(ay) - 200, int(ax) + 60].sum() >= 0.0
    assert img.sum() < 0.5 * Renderer(spec).render(o).sum()


def test_saed_composite_is_cached_per_view():
    spec = _gold_specimen()
    r = Renderer(spec)
    o = _saed_optics(120.0)
    a = r.render(o)
    b = r.render(o)
    assert np.array_equal(a, b) and spec.rasterize_calls == 1
