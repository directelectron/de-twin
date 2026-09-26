"""Frozen per-acquisition optics: the renderer's view of the column.

``derive_optics`` (in :mod:`de_twin.optics.derive`) turns a
:class:`~de_twin.state.MicroscopeState` + :class:`~de_twin.state.AcquisitionRequest`
+ calibration into one of these. Renderers consume only this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..specimen.fieldmap import ViewWindow
from ..state import RenderMode
from .aberrations import Aberrations


@dataclass(frozen=True)
class OpticsState:
    render_mode: RenderMode

    # Beam
    ht_kv: float
    wavelength_nm: float
    beam_blanked: bool

    # Output geometry: the image the renderer must return, in *detector*
    # pixels of the requested hardware ROI before binning, shape (ny, nx).
    output_shape: tuple[int, int]

    # Real-space sampling of the specimen. For TEM this is the raster that is
    # rasterised (possibly coarser than output_shape; renderer upsamples).
    view: ViewWindow
    specimen_pixel_nm: float  # nm per *detector* pixel at the specimen (TEM imaging)

    # Reciprocal space (diffraction on the detector)
    recip_pixel_inv_nm: float  # 1/nm per detector pixel
    camera_length_mm: float
    diffraction_center_px: tuple[float, float]  # (x, y) on the output, includes diffraction shift

    # Illumination
    convergence_mrad: float  # semi-angle alpha
    illumination_mrad: float  # TEM parallel-beam illumination semi-angle (defocus blur)
    probe_diameter_nm: float  # STEM probe FWHM incl. defocus spread
    disk_radius_px: float  # CBED disk radius on the detector
    illuminated_diameter_um: float  # TEM/SAED illuminated area (inf = flood)
    sa_aperture_um: float  # selected-area aperture diameter projected to the specimen

    # Dose: electrons per *detector* pixel per second with the beam on.
    dose_e_per_px_s: float
    beam_current_pa: float

    # Focus / aberrations
    defocus_um: float  # effective: defocus + offsets + (z - eucentric)
    blur_sigma_px: float  # defocus blur on the rasterised view, raster pixels
    fresnel_gain: float
    fresnel_sign: float
    fresnel_sigma_px: float
    objective_stig: tuple[float, float] = (0.0, 0.0)
    beam_tilt_mrad: tuple[float, float] = (0.0, 0.0)
    # TEM imaging: where beam shift puts the illuminated disc, raster px from the centre
    beam_offset_px: tuple[float, float] = (0.0, 0.0)
    # Precession: cone half-angle (0 = off), frequency, descan; and, per frame, the
    # phase the sweep starts at and the arc it covers (2 pi or more: the whole cone).
    precession_mrad: float = 0.0
    precession_hz: float = 0.0
    precession_descan: bool = True
    precession_phase_rad: float = 0.0
    precession_arc_rad: float = 6.283185307179586

    # Specimen orientation
    alpha_rad: float = 0.0
    beta_rad: float = 0.0
    thickness_tilt_factor: float = 1.0  # 1/(cos a cos b)

    # STEM
    scan_step_um: float = 0.0
    scan_shape: tuple[int, int] = (0, 0)  # (ny, nx) scan points
    scan_rotation_rad: float = 0.0
    park_um: Optional[tuple[float, float]] = None  # parked probe position (world um)

    # Raster downsample: output pixels per raster pixel (TEM); 1 = no resample
    raster_downsample: int = 1

    # --- twin additions (all defaulted; filled by de_twin.optics.derive) ----
    # Wave-optical TEM imaging
    cs_mm: float = 1.2  # spherical aberration C3
    cc_mm: float = 1.4  # chromatic aberration (sets focal_spread_nm)
    focal_spread_nm: float = 0.0  # 1-sigma defocus spread (temporal coherence)
    astigmatism_nm: tuple[float, float] = (0.0, 0.0)  # A1 as (0 deg, 45 deg) components, nm
    objective_aperture_mrad: float = 0.0  # objective aperture semi-angle; 0 = none
    # Diffraction / STEM: electrons per second reaching the pattern with the beam on
    pattern_e_per_s: float = 0.0
    # Full axial aberration sets (CEOS notation, complex nm; de_twin.optics.aberrations),
    # *including* the operator's focus (C1 = defocus_um * 1000) and stigmator (A1):
    #   image = objective residual/corrected aberrations + C1 + objective stig  (TEM CTF)
    #   probe = probe-forming residual/corrected aberrations + C1 + condenser stig (STEM probe)
    # Empty = not derived (hand-built OpticsState); renderers then fall back to cs_mm /
    # defocus_um / astigmatism_nm (``image_aberrations_of`` / ``probe_aberrations_of``).
    # hash=False because Aberrations is mutable; equality still compares them.
    image_aberrations: Aberrations = field(default_factory=Aberrations, hash=False)
    probe_aberrations: Aberrations = field(default_factory=Aberrations, hash=False)
    aberrations_derived: bool = False  # True: the two sets above are authoritative (even if empty)
    # STEM partial spatial coherence: FWHM of the demagnified source at the specimen, nm
    source_size_nm: float = 0.0

    extras: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def resolution_warning(self) -> str:
        """Why a TEM frame does not carry detail at its own pixel size ("" when it does):
        it is upsampled from a coarser raster (`OpticsConfig.max_raster_pixels`), so its
        contrast transfer stops at the raster's Nyquist, not the frame's."""
        d = int(self.raster_downsample)
        if d <= 1:
            return ""
        return (f"TEM frame upsampled {d}x from a {self.view.shape[1]}x{self.view.shape[0]} raster: "
                f"no detail finer than {2 * d} frame pixels; OpticsConfig(max_raster_pixels=0) "
                f"renders at the frame's sampling")


def image_aberrations_of(optics: OpticsState) -> Aberrations:
    """TEM objective aberrations incl. focus and stigmation (fallback for hand-built optics)."""
    if optics.aberrations_derived or optics.image_aberrations:
        return optics.image_aberrations
    a1x, a1y = optics.astigmatism_nm
    return Aberrations({"C3": optics.cs_mm * 1e6, "C1": optics.defocus_um * 1000.0,
                        "A1": complex(a1x, a1y)})


def probe_aberrations_of(optics: OpticsState) -> Aberrations:
    """STEM probe aberrations incl. focus and condenser stigmation (fallback: C3 + C1)."""
    if optics.aberrations_derived or optics.probe_aberrations:
        return optics.probe_aberrations
    return Aberrations({"C3": optics.cs_mm * 1e6, "C1": optics.defocus_um * 1000.0})
