"""Coherent STEM probe: size, normalisation, sampling, aberration wiring, correctors."""

from __future__ import annotations

import math

import numpy as np
import pytest

from coherent_helpers import stem_optics
from de_twin.optics.aberrations import Aberrations
from de_twin.optics.state import probe_aberrations_of
from de_twin.render import RenderConfig, Renderer
from de_twin.render.coherent_setup import build_probe, plan_sampling
from de_twin.render.testing import SyntheticSpecimen
from de_twin.specimen.materials import MaterialId

COHERENT = RenderConfig(coherent_focal_spread=False, coherent_source_size=False)


def _probe(conv, ab, df=0.0, k=4.0, cfg=COHERENT, **kw):
    o = stem_optics(conv_mrad=conv, defocus_nm=df, probe_ab=ab, roi=128, det_k_over_alpha=k, **kw)
    p = build_probe(o, cfg)
    n = p.sampling.n
    y, x = np.indices((n, n)) - n // 2
    r = np.hypot(x, y) * p.sampling.dx_nm
    return o, p, r


def _fwhm(profile, dx):
    """Full width at half maximum of a 1-D peak, linearly interpolated."""
    half = profile.max() / 2
    i = np.flatnonzero(profile >= half)
    lo, hi = i[0], i[-1]
    left = lo - 1 + (half - profile[lo - 1]) / (profile[lo] - profile[lo - 1])
    right = hi + (profile[hi] - half) / (profile[hi] - profile[hi + 1])
    return (right - left) * dx


def test_focused_probe_is_diffraction_limited():
    o, p, _ = _probe(20.0, {"C3": 0.0})
    lam, a = o.wavelength_nm, o.convergence_mrad * 1e-3
    n = p.sampling.n
    fwhm = _fwhm(p.intensity[n // 2].astype(float), p.sampling.dx_nm)
    assert fwhm == pytest.approx(0.514 * lam / a, rel=0.08)  # Airy: 1.029 lambda / (2 alpha)
    assert p.weights.tolist() == [1.0] and p.modes.shape[0] == 1


def test_defocused_probe_size_is_two_df_alpha():
    df = -200.0
    o, p, r = _probe(20.0, {"C3": 0.0}, df=df, k=1.6)
    rms = math.sqrt(float((p.intensity * r ** 2).sum() / p.intensity.sum()))
    radius = abs(df) * o.convergence_mrad * 1e-3  # geometric disk, diameter 2 |df| alpha
    assert rms * math.sqrt(2.0) == pytest.approx(radius, rel=0.05)
    assert p.sampling.window_nm >= 2 * radius  # the window holds the defocused probe
    assert not p.sampling.truncated


def test_probe_normalisation_and_modes():
    o, p, _ = _probe(20.0, {"C3": 1000.0}, cfg=RenderConfig())
    assert float(p.intensity.sum()) == pytest.approx(1.0, rel=1e-4)
    for m in p.modes:
        assert float((np.abs(m) ** 2).sum()) == pytest.approx(1.0, rel=1e-4)
    assert p.weights.sum() == pytest.approx(1.0) and np.all(np.diff(p.weights) <= 1e-12)
    assert p.focal_samples == 3 and p.source_samples == 9  # both coherence terms active by default
    assert 0.5 < p.coherent_fraction < 1.0
    # vacuum pattern carries exactly the probe current
    r = Renderer(SyntheticSpecimen(film_material=MaterialId.VACUUM), RenderConfig(stem_model="coherent"))
    img = r.render(o, scan_point=(3, 4))
    assert img.shape == o.output_shape and img.dtype == np.float32
    assert img.sum() == pytest.approx(o.pattern_e_per_s, rel=1e-3)


def test_corrector_allows_large_convergence():
    """C3 ~ 0, small C5 (probe corrector): a compact probe at 28 mrad. Uncorrected C3 = 1.2 mm:
    fine at 8 mrad, blown up above ~10 mrad."""
    frac = {}
    for name, conv, ab in (("unc8", 8.0, {"C3": 1.2e6}), ("unc10", 10.0, {"C3": 1.2e6}),
                           ("unc15", 15.0, {"C3": 1.2e6}), ("unc28", 28.0, {"C3": 1.2e6}),
                           ("cor28", 28.0, {"C3": 0.0, "C5": 5e6})):
        o, p, r = _probe(conv, ab)
        lam, a = o.wavelength_nm, conv * 1e-3
        frac[name] = float(p.intensity[r < lam / a].sum())  # within the ~first Airy zero
    assert frac["cor28"] > 0.85
    assert frac["unc8"] > 0.65
    assert frac["unc15"] < 0.2 and frac["unc28"] < 0.05
    assert frac["unc10"] < frac["unc8"]
    # the pi/4 flat angle of each tableau agrees
    lam = 0.0025079
    assert Aberrations({"C3": 1.2e6}).flat_angle_mrad(lam) < 11.0
    assert Aberrations({"C3": 0.0, "C5": 5e6}).flat_angle_mrad(lam) > 25.0


def test_aberration_wiring_condenser_stig_and_column_sets():
    o = stem_optics(defocus_nm=-50.0, condenser_stig=(0.01, -0.02), probe_ab={"C3": 2000.0, "B2": 30 + 10j})
    ab = probe_aberrations_of(o)
    assert ab["C1"].real == pytest.approx(-50.0)
    assert ab["A1"] == pytest.approx(10.0 - 20.0j)  # 1000 nm per stigmator unit
    assert ab["C3"].real == pytest.approx(2000.0) and ab["B2"] == pytest.approx(30 + 10j)
    # the image side is separate: objective stig, uncorrected C3 by default
    assert o.image_aberrations["C3"].real == pytest.approx(1.2e6)
    assert o.image_aberrations["A1"] == 0
    # empty column set -> uncorrected C3 from the optics config
    o2 = stem_optics(defocus_nm=0.0)
    assert probe_aberrations_of(o2)["C3"].real == pytest.approx(1.2e6)


def test_sampling_oversamples_for_defocus_and_binning():
    small = plan_sampling(stem_optics(defocus_nm=-10.0, probe_ab={"C3": 0.0}), COHERENT)
    big = plan_sampling(stem_optics(defocus_nm=-1000.0, conv_mrad=15.0, det_k_over_alpha=1.3, roi=128,
                                    binning=2, probe_ab={"C3": 0.0}), COHERENT)
    assert small.oversampling == 1 and small.sub == (1, 1)
    assert big.oversampling > 1 and big.sub[0] == big.oversampling
    assert big.window_nm >= 2 * 1000.0 * 0.015  # holds the 30 nm defocused probe
    assert big.det_shape == (64, 64) and big.binning == (2, 2)
    # detector k range fits in the simulation grid with no wrap-around onto it
    kd = 64 * big.recip_px * 2 / 2
    assert big.n * big.dk / 2 >= kd


def test_mode_truncation_keeps_degenerate_groups_whole():
    """The 3 x 3 source sampling makes degenerate mode pairs. Cutting through a pair picks an
    arbitrary basis vector (LAPACK/BLAS-thread dependent), which made datacubes differ between
    machines. Truncation must keep the whole group, so the probe intensity stays symmetric."""
    o = stem_optics(scan=(8, 8), step_nm=1.0, conv_mrad=15.0, defocus_nm=-500.0, roi=128, binning=2,
                    det_k_over_alpha=1.3, probe_ab={"C3": 1000.0})
    cfg = RenderConfig(stem_model="coherent", coherent_max_modes=2)
    p = build_probe(o, cfg)
    w = np.asarray(p.weights)
    assert len(w) >= 2
    # no kept mode has an (almost) equal partner that was dropped: the kept set ends at a gap
    full = build_probe(o, RenderConfig(stem_model="coherent", coherent_max_modes=64, coherent_mode_power=1.0))
    ev = np.asarray(full.weights)
    k = len(w)
    if k < len(ev):
        assert abs(ev[k - 1] - ev[k]) > 1e-3 * ev[k - 1]
    # an isotropic source and a round (C1 + C3) probe give an x/y-symmetric intensity
    assert np.allclose(p.intensity, p.intensity.T, atol=1e-3 * p.intensity.max())
