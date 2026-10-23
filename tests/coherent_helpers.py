"""Shared set-up for the coherent 4D-STEM tests."""

from __future__ import annotations

from de_twin.optics import Calibration, OpticsConfig, derive_optics
from de_twin.render.testing import StubCamera
from de_twin.state import AcquisitionRequest, MicroscopeState, Roi, TemStem

CAM = StubCamera()  # DE16-like, 6.5 um pixels, 4096^2
LAM_200 = 0.0025079  # nm


def cl_for(recip_inv_nm: float) -> float:
    """Camera length (mm) giving ``recip_inv_nm`` per unbinned 6.5 um pixel at 200 kV."""
    return 6500.0 / (LAM_200 * recip_inv_nm) / 1e6


def stem_optics(*, scan=(32, 32), step_nm=0.25, conv_mrad=20.0, defocus_nm=-10.0, roi=64, binning=1,
                recip_inv_nm=None, det_k_over_alpha=1.6, probe_ab=None, rotation_deg=0.0,
                condenser_stig=(0.0, 0.0), spot=3, config=None):
    """STEM optics whose (binned) detector half-width covers ``det_k_over_alpha`` x alpha."""
    s = MicroscopeState()
    s.tem_stem = TemStem.STEM
    s.convergence_semi_angle_mrad = conv_mrad
    s.defocus_um = defocus_nm / 1000.0
    s.spot_size = spot
    s.condenser_stig.x, s.condenser_stig.y = condenser_stig
    if probe_ab is not None:
        s.probe_aberrations = dict(probe_ab)
    if recip_inv_nm is None:
        ka = conv_mrad * 1e-3 / LAM_200
        recip_inv_nm = det_k_over_alpha * ka / (roi / 2)
    s.camera_length_mm = cl_for(recip_inv_nm)
    r = AcquisitionRequest(hw_roi=Roi(2048 - roi // 2, 2048 - roi // 2, roi, roi), hw_binning=(binning, binning))
    r.scan.enabled = True
    r.scan.size = scan
    r.scan.step_um = step_nm / 1000.0
    r.scan.rotation_deg = rotation_deg
    return derive_optics(s, r, CAM, Calibration.default(), config or OpticsConfig())
