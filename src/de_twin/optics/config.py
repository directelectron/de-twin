"""Optics options: the column-side knobs of VirtualSpecimenProps plus the twin's
physical beam/aberration model.

Every C++ ``Simulator ...`` optics property has a snake_case field here (the
render-side ones live in :class:`de_twin.render.RenderConfig`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..state import ProbeMode, RenderMode


@dataclass
class OpticsConfig:
    # ---- scale fallbacks / overrides (C++: Fallback Pixel Size, Pixel Size Override, ...)
    fallback_pixel_nm: float = 1.0  # used when the calibration has no entry (sensor px)
    fallback_recip_inv_nm: float = 0.04  # 1/nm per unbinned detector pixel
    pixel_size_override_nm: float = 0.0  # >0 replaces the calibration outright (and the STEM step)
    geometric_diffraction: bool = True  # recip px from camera pixel / camera length (C++ SIMULATOR path)
    default_camera_length_mm: float = 250.0  # used when the column reports CL <= 0

    # ---- geometry (C++: Stage Offset X/Y/Z, Defocus Offset, Eucentric Height, Flip X/Y)
    stage_offset_um: tuple[float, float, float] = (0.0, 0.0, 0.0)
    defocus_offset_um: float = 0.0
    eucentric_height_um: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    render_mode_override: Optional[RenderMode] = None  # None = Auto
    #: The largest raster a TEM image is rendered at (pixels); a bigger frame is upsampled
    #: from it (`OpticsState.raster_downsample` says by how much). 0 renders at the
    #: frame's own sampling: what a CTF or MTF measurement needs, since an upsampled frame
    #: carries its contrast transfer only up to the raster's Nyquist. The default keeps a
    #: re-render (every stage move and focus step) well under a second on a 4096² camera.
    max_raster_pixels: int = 1024 * 1024

    # ---- illumination (C++: Illumination Semi-Angle TEM, SA Aperture, TEM Illuminated Area)
    # 0 = derive from the beam spread (brightness conservation, see beam.py);
    # >0 = fixed value (the C++ default was 0.1 mrad).
    illumination_semi_angle_mrad: float = 0.0
    sa_aperture_um: float = 10.0  # selected-area aperture projected to the specimen; 0 = none
    tem_illuminated_area_um: float = 0.0  # >0 forces the illuminated diameter; 0 = from Intensity
    stem_default_convergence_mrad: float = 10.0  # STEM alpha when the column reports 0
    probe_d0_nm: float = 0.0  # >0 overrides the d0(probe mode, spot) table

    # ---- physical dose model (see beam.py) --------------------------------
    dose_model: str = "physical"  # "physical" | "legacy" (C++ counts/s * (3/spot)^2)
    legacy_dose_rate_counts_per_s: float = 4000.0
    reference_current_pa: float = 600.0  # TEM-mode probe current at the reference spot
    reference_spot: int = 3
    spot_exponent: float = 2.0  # I ~ (ref/spot)^exponent (C++ law)
    probe_mode_current_factor: dict = field(default_factory=lambda: {
        ProbeMode.TEM: 1.0,
        ProbeMode.MICROPROBE: 1.0,
        ProbeMode.EDS: 0.5,
        ProbeMode.NANOPROBE: 0.05,
        ProbeMode.NBD: 0.02,
        ProbeMode.CBD: 0.1,
        ProbeMode.UNKNOWN: 1.0,
    })
    stem_current_factor: float = 0.05  # STEM with a TEM/unknown probe mode behaves like nanoprobe
    # Condenser aperture diameters (um) by 1-based index; current ~ (d/d_ref)^2.
    condenser_apertures_um: tuple[float, ...] = (150.0, 70.0, 50.0, 10.0)
    reference_condenser_index: int = 1
    # Illuminated diameter vs Intensity (C2): D = D_min * (D_max/D_min)**intensity
    illum_min_diameter_um: float = 0.1
    illum_max_diameter_um: float = 100.0
    # Brightness conservation: alpha_ill = ref_alpha * ref_D / D (when illumination_semi_angle_mrad == 0)
    illum_reference_alpha_mrad: float = 0.1
    illum_reference_diameter_um: float = 3.1623
    nanoprobe_illuminated_um: float = 0.20  # C++ kNanoprobeIlluminatedAreaUm (TEM-mode nanoprobe)
    lowmag_illumination_factor: float = 30.0  # LowMAG condenser setting spreads the beam this much more
    #: Intensity zoom (TFS "Intensity Zoom"): in TEM imaging the condenser follows the
    #: magnification, so the illuminated area keeps its size relative to the field of
    #: view and the dose per detector pixel stays put as the operator zooms. Intensity
    #: then sets the beam at `intensity_zoom_reference_mag` (and so at every mag). Off:
    #: the dose per pixel goes as 1/mag^2 and zooming out saturates the camera.
    intensity_zoom: bool = False
    intensity_zoom_reference_mag: float = 20000.0
    #: A real column's geometry (image rotation, true pixel size, image-shift matrices,
    #: magnification offsets, backlash): what SerialEM's calibrations measure. None: ideal.
    realism: Optional["ColumnRealism"] = None

    @classmethod
    def realistic(cls, seed: int = 0, **kw) -> "OpticsConfig":
        """A column as imperfect as a real one (:mod:`de_twin.optics.realism`), reproducibly."""
        from .realism import ColumnRealism

        return cls(realism=ColumnRealism(seed=seed), **kw)

    # ---- aberrations / coherence (twin additions) --------------------------
    cs_mm: float = 1.2
    cc_mm: float = 1.4
    energy_spread_ev: float = 0.7  # FWHM (Schottky FEG ~0.7, LaB6 ~1.5)
    objective_aperture_mrad: float = 0.0  # 0 = no objective aperture
    # Objective stigmator -> 2-fold astigmatism A1 (nm) = (stig - stig_zero) * nm_per_unit
    stig_nm_per_unit: float = 1000.0
    objective_stig_zero: tuple[float, float] = (0.0, 0.0)
    # Condenser stigmator -> STEM probe astigmatism A1 (nm), same linear law. 1 unit = 1 um
    # of A1: a coarse knob (a 20 mrad / 200 kV probe tolerates |A1| < lambda/(4 alpha^2) ~ 1.6 nm).
    condenser_stig_nm_per_unit: float = 1000.0
    condenser_stig_zero: tuple[float, float] = (0.0, 0.0)
    # STEM partial spatial coherence: FWHM (nm) of the demagnified source image at the
    # specimen at ``reference_spot``; it scales as reference_spot / spot (Schottky FEG:
    # probe current ~ d_source^2 ~ 1/spot^2). 0 = spatially fully coherent probe.
    stem_source_size_nm: float = 0.04
