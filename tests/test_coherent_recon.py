"""Reconstructions on coherent twin data: tcBF / parallax and ptychography (tests/reconstruction helpers)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from coherent_helpers import stem_optics
from reconstruction import (epie, object_on_grid, phase_correlation, probe_from_aberrations,
                              probe_modes_on_grid, tcbf)
from de_twin.render import RenderConfig, Renderer
from de_twin.render.testing import Particle, SyntheticSpecimen
from de_twin.specimen.materials import MaterialId


def _particles(o, n, seed, rmin, rmax, lo=4, hi=28):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ix, iy = rng.uniform(lo, hi, 2)
        x, y = o.view.pixel_to_world(iy, ix)
        out.append(Particle(float(x), float(y), float(rng.uniform(rmin, rmax))))
    return out


# ------------------------------------------------------------------------ tcBF
@pytest.fixture(scope="module")
def tcbf_data():
    """Underfocused (-500 nm), astigmatic (condenser stig -> A1 = 80 + 50i nm), 30 deg scan
    rotation, 15 mrad, probe-corrected; carbon film + Au particles; 32 x 32 scan, 1 nm step,
    64 x 64 (2 x 2 binned) patterns just covering the BF disk."""
    o = stem_optics(scan=(32, 32), step_nm=1.0, conv_mrad=15.0, defocus_nm=-500.0, roi=128, binning=2,
                    det_k_over_alpha=1.3, probe_ab={"C3": 1000.0}, rotation_deg=30.0,
                    condenser_stig=(0.08, 0.05))
    spec = SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=10,
                             particles=_particles(o, 6, 1, 1.5, 3.0))
    r = Renderer(spec, RenderConfig(stem_model="coherent", coherent_max_modes=2))
    return o, r.datacube(o), r.ground_truth_ptychography(o)


def _gt_phase_at_scan(gt, smooth_nm=0.5):
    obj = gt["object"]
    ph = np.angle(obj * np.exp(-1j * np.angle(obj.mean())))
    ph = gaussian_filter(ph, smooth_nm / gt["dx_nm"])
    ny, nx = gt["scan_shape"]
    return map_coordinates(ph, gt["positions_px"].T, order=1).reshape(ny, nx)


def test_tcbf_fits_defocus_astigmatism_and_rotation(tcbf_data):
    o, dc, gt = tcbf_data
    s = gt["sampling"]
    res = tcbf(dc, recip_px_inv_nm=s.det_recip_px[0], scan_step_nm=gt["scan_step_nm"],
               wavelength_nm=o.wavelength_nm, rotation_hint_rad=math.radians(30.0))
    assert res.C1_nm == pytest.approx(gt["C1_nm"], rel=0.10)  # -500 nm, sign included
    assert res.C1_nm < 0
    assert abs(res.A1_nm - gt["A1_nm"]) < 0.15 * abs(gt["A1_nm"])  # magnitude and azimuth
    assert math.degrees(res.rotation_rad) == pytest.approx(30.0, abs=2.0)
    assert res.residual_px < 0.5
    # parallax: d_k = lambda C1 k (underfocus: pixel images move against k)
    k = res.k_inv_nm
    assert np.corrcoef(res.shifts_fit_px[:, 0] - res.shifts_fit_px[:, 0].mean(),
                       res.shifts_px[:, 0] - res.shifts_px[:, 0].mean())[0, 1] > 0.98
    assert res.aberrations["C1"].real == pytest.approx(res.C1_nm)
    assert len(k) > 50


def _counts(lam, seed):
    """Poisson counts of mean ``lam``, by inverse CDF on one uniform per pixel.

    ``rng.poisson`` draws a ``lam``-dependent number of uniforms per value, so a rounding-level
    change to the simulated patterns (another platform's libm, another kernel) reshuffles the
    whole realization; here it changes a count only where the uniform sits on a CDF step.
    """
    u = np.random.default_rng(seed).random(lam.shape)
    p = np.exp(-lam)
    cdf = p.copy()
    k = np.zeros(lam.shape)
    kmax = int(lam.max() + 10 * math.sqrt(lam.max()) + 20)
    for n in range(1, kmax):
        above = u > cdf
        if not above.any():
            break
        k += above
        p *= lam / n
        cdf += p
    return k


def test_tcbf_image_beats_single_pixel_and_plain_bf(tcbf_data):
    o, dc, gt = tcbf_data
    s = gt["sampling"]
    truth = _gt_phase_at_scan(gt)
    dose = 2e4  # electrons per pattern, counting noise
    lam = dc / dc.sum((2, 3)).mean() * dose

    def corr(a):
        return np.corrcoef(a.ravel(), truth.ravel())[0, 1]

    # one realization scores anywhere from ~0.2 to ~0.5: judge the median of several
    c_tcbf, c_single, c_bf = [], [], []
    for seed in range(5):
        noisy = _counts(lam, seed)
        res = tcbf(noisy, recip_px_inv_nm=s.det_recip_px[0], scan_step_nm=gt["scan_step_nm"],
                   wavelength_nm=o.wavelength_nm, rotation_hint_rad=math.radians(30.0), ctf_correct=True)
        assert res.C1_nm == pytest.approx(gt["C1_nm"], rel=0.10)
        cx, cy = res.center_px
        c_tcbf.append(corr(res.image))
        c_single.append(abs(corr(noisy[:, :, int(round(cy)), int(round(cx + 0.5 * res.radius_px))])))
        c_bf.append(abs(corr(res.bf_image)))
    c_tcbf = np.median(c_tcbf)
    assert c_tcbf > 0.35
    assert c_tcbf > np.median(c_single) + 0.2
    assert c_tcbf > np.median(c_bf) + 0.2


def test_tcbf_overfocus_sign():
    o = stem_optics(scan=(24, 24), step_nm=1.0, conv_mrad=15.0, defocus_nm=400.0, roi=128, binning=2,
                    det_k_over_alpha=1.3, probe_ab={"C3": 1000.0}, rotation_deg=-20.0)
    spec = SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=10,
                             particles=_particles(o, 5, 2, 1.5, 3.0, 3, 21))
    r = Renderer(spec, RenderConfig(stem_model="coherent", coherent_max_modes=1))
    dc = r.datacube(o)
    s = r.coherent_sampling(o)
    res = tcbf(dc, recip_px_inv_nm=s.det_recip_px[0], scan_step_nm=1.0, wavelength_nm=o.wavelength_nm,
               rotation_hint_rad=math.radians(-20.0))
    assert res.C1_nm == pytest.approx(400.0, rel=0.10)
    assert math.degrees(res.rotation_rad) == pytest.approx(-20.0, abs=2.0)
    assert abs(res.A1_nm) < 20.0


# ---------------------------------------------------------------- ptychography
@pytest.fixture(scope="module")
def ptycho_data():
    """32 x 32 scan, 0.25 nm step, 20 mrad probe-corrected, -10 nm defocus, 64 x 64 patterns to
    1.6 alpha; 5 nm amorphous carbon + a 3 nm Au particle. Default partial coherence."""
    o = stem_optics(scan=(32, 32), step_nm=0.25, conv_mrad=20.0, defocus_nm=-10.0, roi=64,
                    det_k_over_alpha=1.6, probe_ab={"C3": 1000.0})
    x, y = o.view.pixel_to_world(16, 16)
    spec = SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=5,
                             particles=[Particle(float(x), float(y), 1.5)])
    r = Renderer(spec, RenderConfig(stem_model="coherent"))
    dc = r.datacube(o)
    gt = r.ground_truth_ptychography(o)
    s = gt["sampling"]
    assert s.oversampling == 1  # the window fits: reconstruct at the detector sampling
    n = dc.shape[-1]
    dx = 1.0 / (n * s.det_recip_px[0])
    obj, pos = object_on_grid(gt, dx)
    mask = np.zeros(obj.shape, bool)
    lo, hi = pos.min(axis=0).astype(int) + 4, pos.max(axis=0).astype(int) - 4
    mask[lo[0]:hi[0], lo[1]:hi[1]] = True
    return o, dc.reshape(-1, n, n).astype(np.float64), gt, obj, pos, mask


def test_ptychography_known_probe_with_counting_noise(ptycho_data):
    o, data, gt, obj, pos, mask = ptycho_data
    s = gt["sampling"]
    n = data.shape[-1]
    probe = probe_from_aberrations((n, n), s.det_recip_px[0], o.wavelength_nm, o.convergence_mrad,
                                   gt["aberrations"])
    dose = 1e5  # electrons per pattern
    noisy = np.random.default_rng(3).poisson(data / data.sum(axis=(1, 2)).mean() * dose).astype(float)
    res = epie(noisy, pos, probe, object_shape=obj.shape, iterations=20, center=s.det_center_px, rpie=0.2)
    assert res.errors[-1] < res.errors[0]
    assert phase_correlation(res.object, obj, mask) > 0.8


def test_ptychography_mixed_state_recovers_the_phase(ptycho_data):
    o, data, gt, obj, pos, mask = ptycho_data
    s = gt["sampling"]
    n = data.shape[-1]
    modes, weights = probe_modes_on_grid(gt, (n, n))
    assert modes.shape[0] == len(gt["mode_weights"]) > 1  # partial coherence at the defaults
    res = epie(data, pos, modes, mode_weights=weights, object_shape=obj.shape, iterations=20,
               center=s.det_center_px, rpie=0.2)
    assert phase_correlation(res.object, obj, mask) > 0.85
