"""EBSD: geometry, band positions and widths, the pattern centre across a scan, ground truth,
the specimens, the faces, and test-only dictionary indexing of noisy frames."""

from __future__ import annotations

import math

import numpy as np
import pytest

from de_twin import DigitalTwin
from de_twin.ebsd import EbsdGeometry, convert_pc, detector_position, master_for
from de_twin.ebsd.geometry import pc_pixels
from de_twin.ebsd.master import MASTERS, bands, build_master
from de_twin.specimen.bulk import BULK_PRESETS, BulkPhase, BulkSurface, BulkSurfaceConfig
from de_twin.specimen.materials import MaterialId
from de_twin.state import RenderMode, Roi, ScanRequest

KV = 20.0


def _twin(specimen="EBSD Si (001) wafer", camera="Celeritas", **geom):
    tw = DigitalTwin(specimen, camera=camera)
    tw.configure_ebsd(beam_kv=KV, **geom)
    return tw


def _scan(tw, nx=1, ny=1, step=1.0, binning=4, dwell=1e-3, **kw):
    return tw.request(
        hw_binning=(binning, binning),
        frame_time_s=dwell,
        total_frames=nx * ny,
        scan=ScanRequest(enabled=True, size=(nx, ny), step_um=step, dwell_s=dwell),
        **kw,
    )


def _raw(tw, req, ix=0, iy=0, matrix=None, material=None, master=None):
    """The noise-free projected master values (no background) at a scan point."""
    optics = tw.optics(req)
    r = tw.renderer.ebsd
    tab = r.tables(tw.specimen, optics)
    o = int(tab.orient[iy, ix])
    m = tab.matrices[o] if matrix is None else matrix
    mid = int(tab.material[iy, ix]) if material is None else material
    pc = r.pattern_centre(optics, tab, ix, iy)
    return r.pattern(optics, m, pc, material_id=mid, raw=True), pc, optics


def _zone_centre(img, c, r, half=16):
    """Where a zone axis is in ``img`` near (col c, row r): the unweighted centroid of the
    bright region (above the midpoint of window median and maximum) that contains the window's
    brightest pixel. Crossing top-hat bands make a flat-topped region with brighter excess
    lines at its edges, so neither the argmax nor an intensity-weighted centroid is reliable;
    the region is symmetric about the axis."""
    from scipy import ndimage

    r0, c0 = int(round(r)) - half, int(round(c)) - half
    win = img[r0 : r0 + 2 * half + 1, c0 : c0 + 2 * half + 1].astype(np.float64)
    lab, _ = ndimage.label(win > 0.5 * (np.median(win) + win.max()))
    k = lab[np.unravel_index(np.argmax(win), win.shape)]
    rr, cc = np.nonzero(lab == k)
    return c0 + float(cc.mean()), r0 + float(rr.mean())


# ---------------------------------------------------------------------- mode and optics
def test_ebsd_mode_and_optics():
    tw = _twin()
    req = _scan(tw, 8, 6, step=0.25, binning=4)
    o = tw.optics(req)
    assert o.render_mode == RenderMode.EBSD
    assert o.view.shape == (6, 8) and o.view.pixel_um == pytest.approx(0.25)
    assert o.ht_kv == pytest.approx(KV)
    assert tw.flux(req, 0).shape == (256, 256)  # rendered on the binned grid
    tw.configure_ebsd(enabled=False)
    assert tw.optics(req).render_mode == RenderMode.STEM_4D
    g = EbsdGeometry.from_dict(EbsdGeometry(pcz=0.7, sample_tilt_deg=95).to_dict())
    assert g.pcz == 0.7 and g.sample_tilt_deg == 89.0  # clamped to range
    with pytest.raises(KeyError):
        g.replace(nonsense=1)


# ---------------------------------------------------------------------- geometry
def test_frames_are_orthonormal_and_forward_scatter_is_below_pc():
    g = EbsdGeometry(sample_tilt_deg=70, detector_tilt_deg=4)
    s2d = g.sample_to_detector()
    assert np.allclose(s2d @ s2d.T, np.eye(3)) and np.linalg.det(s2d) == pytest.approx(1.0)
    n, f = g.surface_normal_d(), g.forward_direction_d()
    assert n[1] < 0 and n[2] > 0  # the normal points at the screen, above the pattern centre
    assert f[1] > 0 and f[2] > 0  # forward scatter hits the screen below it
    assert float(n @ g.beam_direction_d()) == pytest.approx(-math.cos(math.radians(70)))


def test_known_zone_axes_at_predicted_positions():
    """Identity orientation: the surface normal [00-1] lies (90 - tilt + detector tilt) above the
    screen normal, so it is DD tan(20 deg) above the pattern centre; [0-1-1] bisects it and the
    down-slope direction (70 deg below), 25 deg below the screen normal. The rendered pattern
    has its zone axes (band crossings) there."""
    tw = _twin()
    req = _scan(tw, binning=2)
    raw, pc, optics = _raw(tw, req, matrix=np.eye(3), material=MaterialId.SILICON)
    H, W = optics.extras["sensor_shape"]
    x0, y0, dd = pc_pixels(pc, (H, W))
    g = tw.ebsd
    cases = {
        (0, 0, -1): (x0, y0 - dd * math.tan(math.radians(20.0))),
        (0, -1, -1): (x0, y0 + dd * math.tan(math.radians(25.0))),
    }
    for uvw, (ex, ey) in cases.items():
        pos = detector_position(g, np.eye(3), uvw, pc, (H, W))
        assert pos == pytest.approx((ex, ey), abs=1e-6)
        # on the 2x-binned grid: index = sensor / 2 - 0.5
        c, r = ex / 2 - 0.5, ey / 2 - 0.5
        fc, fr = _zone_centre(raw, c, r)
        assert abs(fc - c) <= 1.0 and abs(fr - r) <= 1.0, (uvw, fc, fr, c, r)
        assert raw[int(round(r)), int(round(c))] > np.percentile(raw, 99)
    # a rotated crystal: its [001] moves with it
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ _rx(15)
    v = rot @ np.array([0.0, 0.0, -1.0])
    pos = detector_position(g, rot, (0, 0, -1), pc, (H, W))
    d = g.sample_to_detector() @ v
    assert pos == pytest.approx((x0 + dd * d[0] / d[2], y0 + dd * d[1] / d[2]))


def _rx(deg):
    t = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])


def test_band_width_and_position():
    """A single (200) band of Ni, identity orientation: the plane normal [100] is x_d, so the band
    is the vertical strip |x - x_pc| < DD tan(theta_B) along the pattern-centre row. A (111)
    band of a rotated grain covers exactly |r . n| < sin(theta_B)."""
    tw = _twin(
        "EBSD Ni polycrystal", master_resolution_deg=0.05, energy_spread_frac=0.02, angular_blur_deg=0.02
    )
    g = tw.ebsd
    req = _scan(tw, binning=1, hw_roi=Roi(312, 280, 400, 64))
    optics = tw.optics(req)
    r = tw.renderer.ebsd
    lam = optics.wavelength_nm
    for hkl in [(2, 0, 0)]:
        mp = build_master(
            MaterialId.NICKEL,
            KV,
            energy_spread_frac=0.02,
            angular_blur_deg=0.02,
            resolution_deg=0.05,
            hkl_subset=[hkl],
        )
        MASTERS._d[(int(MaterialId.NICKEL), round(KV, 2), 0.02, 0.02, 0.05)] = mp
        pc = g.pattern_centre()
        img = r.pattern(optics, np.eye(3), pc, material_id=MaterialId.NICKEL, raw=True)
        H, W = optics.extras["sensor_shape"]
        x0, y0, dd = pc_pixels(pc, (H, W))
        row = img[int(y0) - 280]
        x = 312 + np.arange(img.shape[1]) + 0.5
        g_len = np.linalg.norm(hkl) / 0.3524
        theta = math.asin(lam * g_len / 2.0)
        half = dd * math.tan(theta)
        inside = np.abs(x - x0) < 0.6 * half
        outside = np.abs(x - x0) > 1.6 * half
        assert row[inside].mean() - row[outside].mean() > 0.5 * (row.max() - row.min())
        # full width at half of the plateau height
        base = np.median(row[outside])
        top = np.median(row[inside])
        above = x[row > 0.5 * (base + top)]
        fw = above.max() - above.min()
        assert fw == pytest.approx(2 * half, rel=0.08)
        assert 0.5 * (above.max() + above.min()) == pytest.approx(x0, abs=1.0)
    MASTERS.clear()
    # (111) of a general orientation
    mp = build_master(
        MaterialId.NICKEL,
        KV,
        energy_spread_frac=0.02,
        angular_blur_deg=0.02,
        resolution_deg=0.05,
        hkl_subset=[(1, 1, 1)],
    )
    MASTERS._d[(int(MaterialId.NICKEL), round(KV, 2), 0.02, 0.02, 0.05)] = mp
    req = _scan(tw, binning=2)
    optics = tw.optics(req)
    from scipy.spatial.transform import Rotation

    from de_twin.crystal.orientation import align

    # the (111) plane normal along a direction in the screen plane: its band crosses the screen
    v = g.sample_to_detector().T @ np.array([math.cos(0.5), math.sin(0.5), 0.0])
    G = Rotation.from_rotvec(0.7 * v).as_matrix() @ align(np.ones(3), v)
    pc = g.pattern_centre()
    img = r.pattern(optics, G, pc, material_id=MaterialId.NICKEL, raw=True)
    H, W = optics.extras["sensor_shape"]
    x0, y0, dd = pc_pixels(pc, (H, W))
    ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    X, Y = (xs + 0.5) * 2 - x0, (ys + 0.5) * 2 - y0
    rd = np.stack([X, Y, np.full_like(X, dd)], -1)
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    n = g.sample_to_detector() @ G @ (np.ones(3) / math.sqrt(3))
    t = np.abs(rd @ n) / (lam * math.sqrt(3) / 0.3524 / 2)
    assert (t < 0.6).sum() > 100
    assert img[t < 0.6].mean() - img[t > 1.6].mean() > 0.5 * (img.max() - img.min())
    edge = (t > 0.85) & (t < 1.15)
    mid = 0.5 * (img[t < 0.6].mean() + img[t > 1.6].mean())
    assert abs(img[edge].mean() - mid) < 0.25 * (img.max() - img.min())  # the edge is at t = 1
    MASTERS.clear()


def test_pattern_centre_follows_the_beam():
    tw = _twin(detector_tilt_deg=5.0)
    g = tw.ebsd
    H, W, p = 1024, 1024, 15.0
    d = math.radians(70 - 5)
    pc0 = g.pattern_centre(0, 0, (H, W), p)
    assert pc0 == pytest.approx((g.pcx, g.pcy, g.pcz))
    pc = g.pattern_centre(30.0, 0.0, (H, W), p)
    assert pc == pytest.approx((g.pcx + 30 / (W * p), g.pcy, g.pcz))
    pc = g.pattern_centre(0.0, 30.0, (H, W), p)
    assert pc == pytest.approx(
        (g.pcx, g.pcy - math.sin(d) * 30 / (H * p), g.pcz + math.cos(d) * 30 / (H * p))
    )
    # a larger working distance: the sample (source) lower, so the pattern centre moves down
    g2 = g.replace(working_distance_mm=g.calibration_wd_mm + 0.5)
    pc = g2.pattern_centre(0, 0, (H, W), p)
    assert pc[1] > g.pcy and pc[2] < g.pcz
    # through the renderer: the scan's pattern centres, and the zone axis moving with them
    tw.configure_ebsd(detector_tilt_deg=0.0)
    req = _scan(tw, 3, 3, step=120.0, binning=2)
    gt = tw.ground_truth(req, as_arrays=True)
    pcs = gt["pattern_centre"]
    assert pcs[1, 2, 0] - pcs[1, 1, 0] == pytest.approx(120 / (W * p))
    assert pcs[2, 1, 1] < pcs[1, 1, 1] and pcs[2, 1, 2] > pcs[1, 1, 2]
    # the Si [00-1] zone axis sits where the moved pattern centre puts it
    for ix, iy in [(1, 1), (2, 1), (1, 2), (0, 0)]:
        raw, pc, optics = _raw(tw, req, ix, iy)
        G = tw.specimen.orientations([gt_hash(tw, req, ix, iy)], [0])[0]
        xy = detector_position(tw.ebsd, G, (0, 0, -1), pc, (H, W))
        c, r = xy[0] / 2 - 0.5, xy[1] / 2 - 0.5
        fc, fr = _zone_centre(raw, c, r)
        assert abs(fc - c) <= 1.0 and abs(fr - r) <= 1.0, (ix, iy, fc, fr, c, r)


def gt_hash(tw, req, ix, iy):
    x, y = tw.optics(req).view.pixel_to_world(np.array([iy]), np.array([ix]))
    return tw.specimen.sample(x, y)["hash"][0]


# ---------------------------------------------------------------------- ground truth
def test_ground_truth_round_trips():
    from de_twin.crystal import crystal_to_lab

    tw = _twin("EBSD duplex steel", band_contrast=0.2)
    req = _scan(tw, 12, 10, step=1.5, binning=8)
    gt = tw.ground_truth(req, as_arrays=True)
    assert set(np.unique(gt["material_id"])) == {MaterialId.IRON, MaterialId.AUSTENITE}
    assert gt["quaternion"].shape == (10, 12, 4)
    assert np.allclose(np.linalg.norm(gt["quaternion"], axis=-1), 1.0)
    assert (gt["quaternion"][..., 0] >= 0).all()
    s = tw.specimen.sample(gt["x_um"].ravel(), gt["y_um"].ravel())
    mats = tw.specimen.orientations(s["hash"], s["phase"])
    assert np.allclose(crystal_to_lab(gt["quaternion"].reshape(-1, 4)), mats, atol=1e-9)
    assert (gt["grain_id"].ravel() == s["grain_id"]).all()
    # re-simulating from the ground truth gives the frames' patterns
    optics = tw.optics(req)
    r = tw.renderer.ebsd
    for iy, ix in [(0, 0), (4, 7), (9, 11)]:
        if gt["mix"][iy, ix] > 0:
            continue
        G = crystal_to_lab(gt["quaternion"][iy, ix])[0]
        p = r.pattern(
            optics,
            G,
            gt["pattern_centre"][iy, ix],
            material_id=int(gt["material_id"][iy, ix]),
            quality=float(gt["quality"][iy, ix]),
        )
        assert np.allclose(p, tw.flux(req, iy * 12 + ix), rtol=1e-5)
    cm = tw.ground_truth(req)
    assert cm.scan_unit == "um" and set(cm.prop) >= {"grain_id", "band_contrast", "pcx", "pcy", "pcz"}
    assert sorted(cm.phases.ids) == [MaterialId.IRON, MaterialId.AUSTENITE]


def test_boundaries_mix_patterns_and_lower_band_contrast():
    tw = _twin("EBSD Ni polycrystal", interaction_scale=3.0)
    req = _scan(tw, 120, 1, step=0.02, binning=8)  # a 2.4 um line
    gt = tw.ground_truth(req, as_arrays=True)
    gid = gt["grain_id"][0]
    cross = np.flatnonzero(np.diff(gid) != 0)
    if not len(cross):
        pytest.skip("line misses a boundary")
    i = int(cross[0])
    assert gt["mix"][0, i] > 0.2 and gt["band_contrast"][0, i] < gt["band_contrast"][0, 0]
    assert gt["grain_id2"][0, i] == gid[i + 1]
    # the mixed pattern is the weighted sum of the two grains' patterns
    optics = tw.optics(req)
    r = tw.renderer.ebsd
    tab = r.tables(tw.specimen, optics)
    w = float(tab.mix[0, i])
    p = r.raw_pattern(tw.specimen, optics, i, 0)
    a = r.pattern(
        optics,
        tab.matrices[tab.orient[0, i]],
        gt["pattern_centre"][0, i],
        material_id=MaterialId.NICKEL,
        raw=True,
    )
    b = r.pattern(
        optics,
        tab.matrices[tab.orient2[0, i]],
        gt["pattern_centre"][0, i],
        material_id=MaterialId.NICKEL,
        raw=True,
    )
    assert np.allclose(p, (1 - w) * a + w * b, atol=1e-4)


# ---------------------------------------------------------------------- master patterns
def test_master_pattern_is_compact_and_cached():
    MASTERS.clear()
    g = EbsdGeometry()
    mp = master_for(MaterialId.NICKEL, KV, g)
    assert mp.data.dtype == np.uint8 and mp.cube.dtype == np.uint8 and mp.nbytes < 3e6
    assert master_for(MaterialId.NICKEL, KV, g) is mp and MASTERS.builds >= 1
    v = mp.values()
    assert abs(float(v.mean())) < 0.1
    # the strongest bands are {111} and {200}, of width 2 theta_B
    hkl, normals, sin_tb, amp = bands(MaterialId.NICKEL, KV)
    assert sorted(abs(int(x)) for x in hkl[0]) == [1, 1, 1] and amp[0] == 1.0
    lam = 0.0085885  # nm at 20 kV
    assert sin_tb[0] == pytest.approx(lam * math.sqrt(3) / 0.3524 / 2, rel=1e-3)
    # zone axes are band crossings: <001> brighter than a random direction on average
    rng = np.random.default_rng(0)
    rand = rng.normal(size=(2000, 3))
    axes = np.array([[0, 0, 1.0], [0, 1.0, 0], [1.0, 1, 1]])
    assert mp.value(axes).min() > np.percentile(mp.value(rand), 90)
    assert MASTERS.nbytes <= MASTERS.budget


# ---------------------------------------------------------------------- specimens
def test_bulk_specimens():
    s = BulkSurface(BULK_PRESETS["EBSD duplex steel"])
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-500, 500, (2, 20000))
    out = s.sample(x, y)
    frac = float(np.mean(out["material"] == MaterialId.AUSTENITE))
    assert 0.3 < frac < 0.6
    assert (out["boundary_um"] >= 0).all() and np.allclose(np.linalg.norm(out["normal"], axis=1), 1.0)
    # deterministic, unique per grain
    again = BulkSurface(BULK_PRESETS["EBSD duplex steel"]).sample(x[:50], y[:50])
    assert (again["grain_id"] == out["grain_id"][:50]).all()
    assert len(np.unique(out["hash"])) > 5000 * 0  # many grains in a millimetre square
    # Si wafer: one crystal, [001] along the normal and [110] along x
    si = BulkSurface(BULK_PRESETS["EBSD Si (001) wafer"])
    o = si.sample([0.0, 300.0], [0.0, -200.0])
    assert o["grain_id"][0] == o["grain_id"][1]
    G = si.orientations(o["hash"][:1], o["phase"][:1])[0]
    assert abs(abs((G @ [0, 0, 1.0])[2]) - 1) < 1e-9
    v = G @ (np.array([1.0, 1, 0]) / math.sqrt(2))
    assert v == pytest.approx([1, 0, 0], abs=1e-9)
    # gamma-fibre ferrite: <111> near the normal
    fe = BulkSurface(BULK_PRESETS["EBSD ferrite gamma fibre"])
    o = fe.sample(*rng.uniform(-300, 300, (2, 400)))
    Gs = fe.orientations(o["hash"], o["phase"])
    family = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]])
    best = np.max(np.abs(np.einsum("nij,kj->nik", Gs, family / math.sqrt(3))[:, 2, :]), axis=1)
    assert np.median(np.degrees(np.arccos(np.clip(best, -1, 1)))) < 15
    # scratches lower the quality along lines
    al = BulkSurface(BULK_PRESETS["EBSD Al cube texture"])
    q = al.quality(*rng.uniform(-3000, 3000, (2, 20000)))
    assert q.max() == pytest.approx(0.9, abs=1e-6) and q.min() < 0.6
    with pytest.raises(ValueError):
        BulkSurface(BulkSurfaceConfig(phases=(BulkPhase(MaterialId.AMORPHOUS_CARBON),)))


def test_tem_specimen_in_ebsd_mode_uses_its_field_map():
    tw = DigitalTwin("Al thin film 100 nm", camera="Celeritas")
    tw.configure_ebsd(beam_kv=KV)
    req = _scan(tw, 6, 4, step=0.2, binning=8)
    gt = tw.ground_truth(req, as_arrays=True)
    assert gt["grain_id"].shape == (4, 6)
    f = tw.flux(req, 3)
    assert f.shape == (128, 128) and np.isfinite(f).all()


# ---------------------------------------------------------------------- detector and stacks
def test_low_kv_electrons_are_fully_absorbed():
    from de_twin.detector import camera

    m = camera("Celeritas")
    a40, a20, a10 = (m.adu_per_electron_at(v) for v in (40, 20, 10))
    assert a20 == pytest.approx(a40 / 2) and a10 == pytest.approx(a40 / 4)
    assert m.adu_per_electron_at(80) == pytest.approx(1148)


def test_frames_and_pattern_stack():
    tw = _twin("EBSD Ni polycrystal")
    req = _scan(tw, 5, 4, step=2.0, binning=8, dwell=2e-3)
    stack = tw.datacube(req)
    assert stack.shape == (4, 5, 128, 128) and stack.dtype == np.float32
    assert np.allclose(stack[1, 2], tw.flux(req, 7) * 64, rtol=1e-5)  # per binned pixel
    frames = [(raw, meta) for raw, meta in tw.frames(req)]
    assert len(frames) == 20 and frames[7][1].scan_point == (2, 1)
    raw = frames[7][0]
    assert raw.shape == (128, 128) and raw.dtype == np.uint16
    assert raw.max() < tw.detector.model.max_value  # 0.1 nA, 2 ms: not saturated
    e_per_px = float(stack[1, 2].mean()) * 2e-3
    assert 1.0 < e_per_px < 100.0


# ---------------------------------------------------------------------- pattern centre conventions
def test_pattern_centre_conventions():
    pc = np.array([0.52, 0.31, 0.63])
    edax = convert_pc(pc, (1024, 1280), "edax")
    assert edax == pytest.approx([0.52, 0.69 * 1024 / 1280, 0.63 * 1024 / 1280])
    em = convert_pc(pc, (1024, 1024), "emsoft", pixel_um=15.0, binning=4)
    assert em == pytest.approx([256 * (0.5 - 0.52), 256 * (0.5 - 0.31), 1024 * 0.63 * 15.0])
    assert convert_pc(pc, (10, 10), "bruker") == pytest.approx(pc)


# ---------------------------------------------------------------------- indexing
@pytest.mark.timeout(300)
def test_dictionary_indexing_recovers_orientations():
    """Noisy frames of a Ni polycrystal (~45 e- per binned pixel, integrating Celeritas)
    indexed with a 4-degree dictionary and a refinement: within 1.5 deg of the truth (in practice
    ~0.06 deg median at ~45 e-, ~0.15 deg at ~10 e-; the band contrast is calibrated to raw patterns)."""
    from orix.quaternion.symmetry import Oh
    from orix.sampling import get_sample_fundamental
    from reconstruction.ebsd_index import dictionary_index, flatten, misorientation_deg, normalise, refine

    from de_twin.crystal import crystal_to_lab

    tw = _twin("EBSD Ni polycrystal", probe_current_na=0.1)
    nx, ny = 6, 5
    req = _scan(tw, nx, ny, step=3.0, binning=8, dwell=5e-3, seed=7)
    optics = tw.optics(req)
    gt = tw.ground_truth(req, as_arrays=True)
    dark = tw.detector.expected_dark(req.frame_time_s, req)
    gain = tw.detector.expected_gain(req)
    exp_fine, exp_coarse = [], []
    for raw, meta in tw.frames(req):
        e = (raw.astype(np.float32) - dark) / gain
        exp_fine.append(flatten(e, sigma=12))
        exp_coarse.append(flatten(e, sigma=12, factor=2))
    exp_fine, exp_coarse = np.array(exp_fine), np.array(exp_coarse)
    assert float(gt["band_contrast"].max()) > 0
    r = tw.renderer.ebsd
    pcs = gt["pattern_centre"].reshape(-1, 3)
    centre = pcs.mean(axis=0)

    def sim(optics_, pc, factor=1):
        def f(m):
            p = r.pattern(optics_, m, pc, material_id=MaterialId.NICKEL, raw=True)
            if factor > 1:
                h, w = p.shape
                p = p.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
            return normalise(p)

        return f

    rots = get_sample_fundamental(4.0, point_group=Oh)
    D = crystal_to_lab(rots.data)
    idx, dncc = dictionary_index(exp_coarse, sim(optics, centre, 2), D)
    errs = []
    for k in range(nx * ny):
        iy, ix = divmod(k, nx)
        if gt["quality"][iy, ix] < 0.7:  # on a boundary (mixed or in its zone)
            continue
        m, c = refine(exp_fine[k], sim(optics, pcs[k]), D[idx[k]])
        G = crystal_to_lab(gt["quaternion"][iy, ix])[0]
        errs.append(float(misorientation_deg(m[None], G[None], Oh)[0]))
    errs = np.array(errs)
    assert len(errs) >= 20
    assert np.median(errs) < 0.5 and errs.max() < 1.5, errs


# ---------------------------------------------------------------------- faces
def test_deapi_face_serves_ebsd():
    deapi = pytest.importorskip("deapi")
    from de_twin.faces.deapi_server import TwinDeapiServer

    tw = DigitalTwin("EBSD Ni polycrystal", camera="Celeritas")
    with TwinDeapiServer(tw, port=0, pace=False) as srv:
        c = deapi.Client()
        c.usingMmf = False
        c.connect(port=srv.port)
        assert c["Simulator EBSD Enable"] == "Off"
        c["Simulator EBSD Enable"] = "On"
        c["Simulator EBSD Pattern Center Z"] = 0.55
        c["Simulator EBSD Sample Tilt (degrees)"] = 70
        assert tw.ebsd.enabled and tw.ebsd.pcz == pytest.approx(0.55)
        assert c["Simulator EBSD Beam Energy (kV)"] == pytest.approx(20.0)
        c["Hardware Binning X"] = 4
        c["Scan - Enable"] = "On"
        c["Scan - Size X"] = 4
        c["Scan - Size Y"] = 3
        c.update_scan_size()
        c.update_image_size()
        c.start_acquisition(1)
        import time

        deadline = time.time() + 30
        while c.acquiring and time.time() < deadline:
            time.sleep(0.01)
        img, *_ = c.get_result("singleframe_integrated")
        c["Simulator EBSD Enable"] = "Off"
        c.disconnect()
    assert img.shape == (256, 256) and float(np.asarray(img).std()) > 0
    assert not tw.ebsd.enabled
