"""Si indent: the residual strain / rotation field, its ground truth, the patterns it distorts and
test-only HR-EBSD recovering it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from de_twin import DigitalTwin
from de_twin.specimen.bulk import BULK_PRESETS, BulkSurface
from de_twin.specimen.indent import Indent, IndentField
from de_twin.specimen.materials import MaterialId
from de_twin.state import ScanRequest


def _twin(name="EBSD Si (001) Vickers indent", **geom):
    tw = DigitalTwin(name, camera="Celeritas")
    tw.configure_ebsd(beam_kv=20.0, **geom)
    return tw


def test_indent_field_physics():
    f = IndentField(Indent(size_um=5.0))
    a, b = 5.0, 9.0
    # deviatoric in-plane 1/r^2 decay along a direction away from cracks and corners
    th = math.radians(20.0)
    r = np.array([12.0, 24.0, 48.0])
    x, y = r * math.cos(th), r * math.sin(th)
    e = f.strain(x, y)
    assert np.allclose(e, np.transpose(e, (0, 2, 1)))
    assert np.allclose(e[:, 0, 2], 0) and np.allclose(e[:, 1, 2], 0)  # traction-free surface
    shear = np.abs(e[:, 0, 0] - e[:, 1, 1])
    assert shear[0] / shear[1] == pytest.approx(4.0, rel=0.15) and shear[1] / shear[2] == pytest.approx(
        4.0, rel=0.15
    )
    assert 1e-4 < shear[0] < 1e-2
    # rotation: ~tilt_deg at r = b, about the tangent, tipping the normal outwards
    w = f.rotation([b * 1.001], [0.0])[0]
    assert math.degrees(abs(w[1])) == pytest.approx(2.0, rel=0.1) and abs(w[0]) < 1e-6
    from scipy.spatial.transform import Rotation as _R

    n = _R.from_rotvec(w).apply([0.0, 0.0, -1.0])
    assert n[0] > 0  # outward normal tipped towards +x (away from the indent)
    # quality: impression ~0, plastic core low, cracks dark, far field perfect
    q = f.quality([0.0, 0.5 * (a + b), 1.5 * a, 1.5 * a, 40.0], [0.0, 0.0, 0.0, 0.3 * a, 40.0])
    assert q[0] < 0.05 and q[1] < 0.8 and q[2] < 0.1 and q[3] > q[2] and q[4] == pytest.approx(1.0)
    # GND: high in the plastic zone, tiny far away
    g = f.gnd([0.5 * (a + b), 40.0], [0.3, 40.0])
    assert g[0] > 1e14 and g[1] < 1e13
    assert IndentField(Indent(kind="berkovich")).p.corners == 3


def test_indent_ground_truth_and_patterns():
    from de_twin.crystal import crystal_to_lab

    tw = _twin()
    req = tw.request(
        hw_binning=(8, 8),
        frame_time_s=1e-3,
        total_frames=1,
        scan=ScanRequest(enabled=True, size=(21, 21), step_um=3.0, dwell_s=1e-3),
    )
    gt = tw.ground_truth(req, as_arrays=True)
    for k in ("deformation_gradient", "strain", "rotation", "gnd"):
        assert k in gt
    F, e, w = gt["deformation_gradient"], gt["strain"], gt["rotation"]
    assert F.shape == (21, 21, 3, 3) and np.allclose(e, np.transpose(e, (0, 1, 3, 2)), atol=1e-9)
    from scipy.spatial.transform import Rotation as _R

    R = _R.from_rotvec(w.reshape(-1, 3)).as_matrix()
    assert np.allclose(
        np.einsum("nij,njk->nik", R, np.eye(3) + e.reshape(-1, 3, 3)), F.reshape(-1, 3, 3), atol=1e-9
    )
    r = np.hypot(gt["x_um"], gt["y_um"])
    assert np.degrees(np.linalg.norm(w, axis=-1))[(r > 9) & (r < 12)].max() > 1.0
    assert np.abs(e[r > 25]).max() < 1e-3
    assert gt["band_contrast"][r < 4].max() < 0.2 * gt["band_contrast"][r > 25].max()
    # the local lattice orientation is the rotation applied to the wafer's
    s = tw.specimen.sample([0.0], [0.0])
    G = tw.specimen.orientations(s["hash"], s["phase"])[0]
    k = np.unravel_index(np.argmin(np.abs(r - 11)), r.shape)
    want = R.reshape(21, 21, 3, 3)[k] @ G
    assert np.allclose(crystal_to_lab(gt["quaternion"][k])[0], want, atol=1e-6)
    # the pattern at a strained point is the wafer's pattern through that F
    optics = tw.optics(req)
    rr = tw.renderer.ebsd
    idx = k[0] * 21 + k[1]
    p = rr.pattern(
        optics,
        G,
        gt["pattern_centre"][k],
        material_id=MaterialId.SILICON,
        F=F[k],
        quality=float(gt["quality"][k]),
    )
    assert np.allclose(p, tw.flux(req, idx), rtol=1e-4)
    berk = BulkSurface(BULK_PRESETS["EBSD Si (001) Berkovich indent"]).sample([0.0, 30.0], [0.0, 0.0])
    assert berk["quality"][0] < 0.05 and "F" in berk


@pytest.mark.timeout(300)
def test_hrebsd_recovers_strain_and_rotation():
    """Test-only HR-EBSD (remapping cross-correlation, 7x7 ROIs, full 1024^2 Celeritas resolution)
    on shot-noise-limited patterns (8000 e-/px: the sum of many frames, ~0.7 s at 10 nA)
    against a reference 57 µm from the indent: deviatoric strain to ~1e-4 and rotation to
    ~0.01 deg (RMS) of the ground truth relative to the reference."""
    from reconstruction.hrebsd import measure, truth_relative

    tw = _twin()
    req = tw.request(
        hw_binning=(1, 1),
        frame_time_s=1e-3,
        total_frames=1,
        scan=ScanRequest(enabled=True, size=(41, 41), step_um=2.0, dwell_s=1e-3),
    )
    gt = tw.ground_truth(req, as_arrays=True)
    s2d = tw.ebsd.sample_to_detector()
    H, W = tw.optics(req).extras["sensor_shape"]
    r = np.hypot(gt["x_um"], gt["y_um"]).ravel()
    F = gt["deformation_gradient"].reshape(-1, 3, 3)
    pcs = gt["pattern_centre"].reshape(-1, 3)
    q = gt["quality"].ravel()
    pts = np.random.default_rng(0).choice(np.flatnonzero((r > 12) & (r < 32) & (q > 0.9)), 6, replace=False)
    rng = np.random.default_rng(5)

    def pattern(k):
        f = tw.flux(req, int(k))
        return rng.poisson(f / f.mean() * 8000.0).astype(float)

    ref = pattern(0)
    es, ws, mags = [], [], []
    for k in pts:
        m = measure(
            ref,
            pattern(k),
            pcs[0],
            pcs[k],
            s2d,
            sensor_shape=(H, W),
            binning=1,
            size=160,
            n_side=7,
            sigma=40.0,
        )
        dt, rt = truth_relative(F[k], F[0])
        es.append((m.strain_dev - dt)[[0, 1, 0, 0, 1], [0, 1, 1, 2, 2]])
        ws.append(np.degrees(m.rotation - rt))
        mags.append(np.abs(dt).max())
    es, ws = np.array(es), np.array(ws)
    assert max(mags) > 5e-4  # the field is there to be measured
    assert np.sqrt(np.mean(es**2)) < 1.5e-4 and np.abs(es).max() < 5e-4, es
    assert np.sqrt(np.mean(ws**2)) < 0.01 and np.abs(ws).max() < 0.03, ws
