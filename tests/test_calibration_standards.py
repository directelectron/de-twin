"""Ted Pella calibration standards: every product is a preset, and each shows its published
numbers (grating period, sphere size) and the physics of a shadowed replica."""

from __future__ import annotations

import math

import numpy as np
import pytest

from de_twin.specimen import CALIBRATION_STANDARDS, from_name
from de_twin.specimen.fieldmap import ViewWindow
from de_twin.specimen.materials import MaterialId
from de_twin.specimen.standards import (
    PRODUCTS,
    ShadowedReplicaStructure,
    WaffleGrating,
    build_structure,
    island_film,
    preset_name,
)


@pytest.fixture
def make_standard_twin(make_twin):
    """A twin showing calibration standard ``number`` at ``magnification``."""

    def build(number, magnification):
        return make_twin(preset_name(number), magnification=magnification)

    return build


def _centre_on(tw, kind, pixel_um, n=512):
    """Put the densest spot of a feature (``latex``, on top or under a metal island, or
    ``moo3``) on axis: rasterize coarse windows around home, then move the stage."""
    from scipy import ndimage

    from de_twin.optics.derive import stage_for_view_center

    home = tw.specimen.home_um()
    for k in range(25):
        r = 0.0 if k == 0 else n * pixel_um * (0.5 + 0.5 * k ** 0.5)
        c = (home[0] + r * math.cos(2.4 * k), home[1] + r * math.sin(2.4 * k))
        fm = tw.specimen.rasterize(ViewWindow(center_um=c, pixel_um=pixel_um, shape=(n, n)))
        m = fm.material_id
        um = fm.under_material if fm.under_material is not None else np.zeros_like(m)
        hit = {"latex": (m == int(MaterialId.PROTEIN)) | (um == int(MaterialId.PROTEIN)),
               "moo3": m == int(MaterialId.MOLYBDENUM_TRIOXIDE)}[kind]
        if hit.any():
            i, j = np.unravel_index(np.argmax(ndimage.uniform_filter(hit.astype(float), 9) * hit), hit.shape)
            x, y = stage_for_view_center(fm.view.pixel_to_world(i, j), tw.column.state(), tw.optics_config)
            tw.column.set_stage(x=x, y=y)
            tw.clock.advance(30.0)  # let the stage get there
            return fm
    raise AssertionError(f"no {kind} found")


def test_every_product_is_a_preset():
    for number in PRODUCTS:
        name = preset_name(number)
        assert name in CALIBRATION_STANDARDS
        assert from_name(name, seed=1).options["standard"] == number
        build_structure(number)


def _period_nm(img, pixel_nm, near_nm):
    """The strongest spatial period of *img* within 30% of *near_nm* (the fundamental, not a
    harmonic or the crumple), refined by a zoomed DFT around the FFT peak."""
    n = img.shape[0]
    w = np.hanning(n)
    f = (img - img.mean()) * np.outer(w, w)
    F = np.abs(np.fft.rfft2(f))
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.rfftfreq(n)[None, :]
    k = np.hypot(kx, ky) * near_nm / pixel_nm
    F[(k < 0.7) | (k > 1.3)] = 0.0
    iy, ix = np.unravel_index(np.argmax(F), F.shape)
    k0x, k0y = kx[0, ix], ky[iy, 0]
    d = np.linspace(-1.0, 1.0, 41) / n
    x = np.arange(n)
    Ex = np.exp(-2j * np.pi * np.outer(k0x + d, x))  # (41, n)
    Ey = np.exp(-2j * np.pi * np.outer(k0y + d, x))
    Z = np.abs(Ey @ f @ Ex.T)  # (ky, kx)
    jy, jx = np.unravel_index(np.argmax(Z), Z.shape)
    return pixel_nm / math.hypot(k0x + d[jx], k0y + d[jy])


@pytest.mark.parametrize("number", ["607", "607-A", "606", "677"])
def test_the_grating_period_is_the_published_one(number, make_standard_twin):
    tw = make_standard_twin(number, 2000)
    tw.column.set("Intensity", 0.95)  # a beam spread past the field: no disc edge
    tw.column.set_defocus_um(-3.0)
    req = tw.request()
    img = tw.flux(req).astype(float)
    published = PRODUCTS[number].numbers["period_nm"]
    assert _period_nm(img, tw.optics(req).specimen_pixel_nm, published) == pytest.approx(published, rel=0.005)


@pytest.mark.parametrize("number", ["603", "673", "628-B"])
def test_latex_spheres_have_the_published_diameter(number, make_standard_twin):
    """The spheres on the gratings (262 nm, the tech notes) and 628-B's 204 nm: their chords
    peak at the published size (for 628-B under its gold: the layer under the metal)."""
    d = {"603": 262.0, "673": 262.0, "628-B": 204.0}[number]
    tw = make_standard_twin(number, 4000)
    _centre_on(tw, "latex", 0.004)
    gt = tw.ground_truth(as_arrays=True)
    latex = gt["material_id"] == int(MaterialId.PROTEIN)
    chord = gt["thickness_nm"][latex]
    if "under_material" in gt:  # latex under a metal island
        under = gt["under_material"] == int(MaterialId.PROTEIN)
        chord = np.concatenate([chord, gt["under_thickness_nm"][under]])
    assert chord.size > 100
    assert 0.94 * d < chord.max() < 1.15 * d


def test_the_aluminium_rings_are_at_the_published_spacings(make_standard_twin):
    """619: the SAED rings of the evaporated aluminium at Ted Pella's d-spacings (camera
    length calibration: K = S d must agree across the rings)."""
    from scipy import ndimage

    tw = make_standard_twin("619", 20000)
    tw.column.set("Intensity", 0.95)  # SAED: a parallel beam
    tw.column.set("ProjectionMode", 2)
    req = tw.request()
    img = tw.flux(req).astype(float)
    recip = tw.optics(req).recip_pixel_inv_nm
    m = img > np.percentile(img, 99.99)
    cy, cx = ndimage.center_of_mass(img * m)
    yy, xx = np.indices(img.shape)
    r2 = (np.hypot(yy - cy, xx - cx) * 2).astype(int)
    prof = np.bincount(r2.ravel(), img.ravel()) / np.maximum(np.bincount(r2.ravel()), 1)
    rr = np.arange(prof.size) / 2.0
    for hkl in ("111", "200", "220", "311"):
        e = 1.0 / PRODUCTS["619"].numbers["d_nm"][hkl] / recip
        win = (rr > 0.95 * e) & (rr < 1.05 * e)
        found = rr[np.argmax(np.where(win, prof, 0))]
        assert found == pytest.approx(e, rel=0.005), hkl
        assert prof[win].max() > 1.5 * np.median(prof[win])


def test_the_island_film_coverage_follows_the_deposit():
    g = np.mgrid[0:300, 0:300] * 0.0008
    prev = 0.0
    for c in (0.15, 0.3, 0.5, 0.65, 0.8):
        rel = (c / 0.6) ** 1.25
        inside, t, grain = island_film(5, g[1], g[0], np.full(g[0].shape, rel), 1.0, 8.0, 0.6,
                                       int(MaterialId.GOLD))
        assert inside.mean() == pytest.approx(c, abs=0.06)
        assert inside.mean() > prev
        prev = inside.mean()
        # the metal is conserved: islands carry the deposit of their surroundings
        assert t[inside].mean() * inside.mean() == pytest.approx(rel, rel=0.1)
        assert grain.size == inside.sum()


def test_shadowing_coats_the_facing_side_and_leaves_a_shadow_behind_a_ridge():
    """Source along +x at 25 deg: behind a 20 nm ridge (x just below it) lies a shadow ~43 nm
    long, with a soft edge; the ridge top and the open ground are lit."""
    g = WaffleGrating(1.0, depth_nm=20.0, line_fraction=0.2, ramp_nm=10.0, edge_nm=0.0, wavy_nm=0.0,
                      both=False)
    s = ShadowedReplicaStructure(g, rough_nm=0.0, crumple_nm=0.0, azimuth_deg=0.0, elevation_deg=25.0)
    x = np.linspace(0.0, 1.0, 2001)
    _h, gx, _gy, lit = s.tile().sample(x, np.full_like(x, 0.3))
    # the ridge spans [0.4, 0.6] um; its shadow falls on x < 0.4 (the source is at +x)
    assert np.all(lit[(x > 0.372) & (x < 0.393)] < 0.1)
    assert np.all(lit[(x > 0.1) & (x < 0.3)] > 0.99)
    assert np.all(lit[(x > 0.45) & (x < 0.55)] > 0.99), "the ridge top is lit"
    assert gx[(x > 0.39) & (x < 0.41)].max() > 1.0 and gx[(x > 0.59) & (x < 0.61)].min() < -1.0


def test_the_shadowed_replica_has_gold_islands_at_high_magnification(make_standard_twin):
    tw = make_standard_twin("607", 16000)
    gt = tw.ground_truth(as_arrays=True)
    au = gt["material_id"] == int(MaterialId.GOLD)
    assert 0.2 < au.mean() < 0.8
    assert np.unique(gt["grain_id"][au]).size > 200, "nanocrystalline islands"


def test_a_moo3_lath_long_edge_is_its_crystal_long_axis(make_standard_twin):
    """The image / diffraction rotation standard: each lath's crystal [010] (Pnma; [001] Pbnm)
    lies along its long edge, so the edge in the image and the spots of its pattern are tied."""
    from scipy import ndimage

    from de_twin.specimen.fieldmap import _direct_basis

    tw = make_standard_twin("625", 2500)
    fm = _centre_on(tw, "moo3", 0.02)
    mid = int(MaterialId.MOLYBDENUM_TRIOXIDE)
    lab, n = ndimage.label(fm.material_id == mid)
    checked = 0
    for k in range(1, n + 1):
        one = lab == k
        g = fm.grain_id[one]
        if one.sum() < 300 or (g == g[0]).mean() < 0.99:
            continue  # small, or laths overlapping
        ring = ndimage.binary_dilation(one, iterations=3) & ~one
        if ((fm.material_id[ring] == mid) & (fm.grain_id[ring] != g[0])).any() or (fm.grain_id == g[0]).sum() > one.sum():
            continue  # another lath touches it (and may hide part of it)
        rows, cols = np.nonzero(one)
        wx, wy = fm.view.pixel_to_world(rows, cols)  # world coordinates: no raster flips
        evals, evecs = np.linalg.eigh(np.cov(np.vstack([wx, wy])))
        if evals[1] < 6.0 * evals[0]:
            continue  # cut by the window's edge
        box = 12.0 * math.sqrt(evals[0] * evals[1])  # the area of a rectangle with these moments
        if one.sum() * fm.view.pixel_um ** 2 < 0.85 * box:
            continue  # not one clean lath (overlapping laths)
        b = tw.specimen.grains.matrices[int(g[0])] @ _direct_basis(mid) @ np.array([0.0, 1.0, 0.0])
        b = b[:2] / np.linalg.norm(b[:2])
        assert abs(float(np.dot(evecs[:, 1], b))) > math.cos(math.radians(2.0))
        checked += 1
    assert checked >= 1


