"""Render-side options (the renderer knobs of VirtualSpecimenProps plus the twin's
wave-optical TEM model)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderConfig:
    # ---- TEM imaging ------------------------------------------------------
    tem_model: str = "physical"  # "physical" (exit wave + CTF) | "legacy" (C++ port)
    diffraction_contrast_scale: float = 1.0  # scales the Bragg loss from the bright-field beam
    mip_phase: bool = True  # mean-inner-potential phase (gives Fresnel fringes)
    edge_taper_nm: float = 1.0  # Gaussian roughness of specimen edges applied to the MIP phase
    refraction_loss: bool = True  # electrons refracted beyond the imaging band by steep edges are lost
    phase_texture_scale: float = 1.0  # amorphous atomic-granularity phase (gives Thon rings)
    texture_bandlimit_nm: float = 0.07  # Gaussian sigma: atomic form-factor band limit
    amplitude_contrast: float = 0.07  # absorptive part of the amorphous texture (w)
    lattice_fringes: bool = True
    #: "kinematic": each resolved beam that reaches the image is a phase grating of its own
    #: kinematic strength, a = sqrt(2 f) for a Friedel pair carrying f of the electrons, grown
    #: with the thickness (heavy crystals give strong fringes, light ones weak), up to its
    #: MAX_FRINGE_BEAMS strongest pairs. "fixed": every crystal's fringes share a total phase
    #: `lattice_phase_rad` (the older model).
    lattice_fringe_model: str = "kinematic"
    #: The kinematic amplitude times this: what dynamical scattering (saturation in thick,
    #: heavy crystals), vibration, drift and the recording take from the fringes. Calibrated
    #: on Ted Pella's lattice images (646 Au [001], 675 Si <011>: fringe contrast 27 % and
    #: 34 %, twice the kinematic value).
    lattice_fringe_efficiency: float = 0.5
    lattice_phase_rad: float = 0.15  # total phase amplitude of resolved lattice fringes ("fixed")
    #: TEM imaging renders a view this fraction of the field larger on each side and
    #: serves a stage move that stays inside the margin by cropping it: the image moves
    #: rigidly with the specimen, so a joystick nudge costs a crop, not a render. 0 off.
    pan_margin: float = 0.15
    #: Pad by the objective's point spread instead (`renderer.transfer_spread_px`, x1.3,
    #: rounded up to one of a few guard sizes, ``min_margin_px`` to ``max_margin_px``): a
    #: smaller raster and FFT near focus and at low magnification, a wider guard at large
    #: defocus and high magnification. ``pan_margin`` is then unused.
    adaptive_margin: bool = True
    min_margin_px: int = 64
    max_margin_px: int = 256
    #: Padded rasters kept for cropping (a jump back to a recent place is a crop too).
    pan_cache_size: int = 6
    #: A new padded raster takes the specimen it shares with a cached one (same sampling, on
    #: the same world-fixed pixel lattice) from that one's field map when they overlap by at
    #: least ``reuse_min_overlap`` of its area, and rasterises only the rest.
    reuse_rasters: bool = True
    reuse_min_overlap: float = 0.25
    #: Threads a large raster is rasterised on, in row strips (1 = serial). Each strip's
    #: numba kernels get the matching share of numba's threads.
    raster_threads: int = 4
    #: Interactive use (dragging the stage, zooming): while the view keeps changing, a
    #: view the cache cannot crop is rendered at up to ``preview_side`` squared and upsampled
    #: (~35-70 ms rather than ~0.3-0.9 s); the first time the same view is asked for
    #: again — the next frame after you stop — it is rendered in full. Off by default,
    #: so a library caller always gets the full render.
    interactive: bool = False
    #: Live view: after each new view, render the view that repeating the last change would
    #: give (the next magnification rung, tilt, focus or stage step) in the background, so
    #: stepping through a series is a crop. One view at a time; the latest wins.
    prefetch: bool = False
    preview_side: int = 256
    #: How long (wall-clock seconds) the view must stay put before it counts as stopped
    #: and is rendered in full. Live view asks for several frames per drag step, so
    #: "asked for the same view twice" is not "stopped".
    settle_s: float = 0.3
    illumination_profile: bool = True  # draw the edge of the illuminated disk when it is in view
    texture_seed_salt: int = 0x7E57

    # ---- diffraction (de_twin.crystal library + DiffractionCache) -------
    diffraction_cache_mb: int = 256
    max_g_inv_nm: float = 25.0  # reciprocal-lattice extent (raise it for HOLZ reflections)
    film_halo: bool = True  # support-film halo under every crystalline pattern
    scaled_diffraction_saturation: bool = True  # per-material amorphous saturation thickness
    diffuse_scattering: bool = True  # (1-T) redistributed as a screened-Rutherford background
    beam_stop_radius_px: float = 0.0  # SAED beam stop, detector pixels; 0 = none

    # ---- STEM -----------------------------------------------------------
    probe_footprint_blend: bool = True
    add_descan: bool = False
    descan_ramp_scale: float = 1.0
    descan_ramp_px: tuple[float, float, float] = (6.0, 4.0, 4.0)  # C++ legacy x span, y span, diag
    intensity_jitter: bool = True  # C++ per-scan-point 0.9..1.1 fluctuation
    park_on_feature: bool = False  # C++ default was on; off keeps the probe on the optic axis

    # ---- coherent 4D-STEM (render/coherent.py) ----------------------------
    # "coherent": probe wavefunction x transmission function, |FFT|^2 (ptychography / tcBF work);
    # "kinematic": cached disk patterns (fast, no interference);
    # "auto": coherent when the simulation grid fits coherent_max_grid without cutting the probe
    #   window and the binned pattern is <= coherent_auto_max_pattern_px, or the scan step is
    #   below 2x the probe size (ptychographic overlap); else kinematic.
    stem_model: str = "auto"
    coherent_max_grid: int = 1024  # largest simulation grid side per pattern
    coherent_auto_max_pattern_px: int = 256 * 256
    coherent_probe_tail: float = 6.0  # window margin beyond the geometric probe radius, in lambda/alpha
    coherent_mode_power: float = 0.98  # partial coherence: keep probe modes up to this power fraction
    coherent_max_modes: int = 4
    coherent_focal_samples: int = 0  # 0 = auto (1/3/5 Gauss-Hermite samples of the focal spread)
    coherent_focal_spread: bool = True  # temporal partial coherence (OpticsState.focal_spread_nm)
    coherent_source_size: bool = True  # spatial partial coherence (OpticsState.source_size_nm)
    coherent_beam_tilt: bool = True  # beam tilt enters chi (coma etc.) and shifts the probe
    coherent_incoherent_scattering: bool = True  # Bragg disks + diffuse background added incoherently
    # t(r) band limit K_t, 1/nm: scattering to |g| <= K_t is coherent, beyond it incoherent. 0 = auto:
    # k_det + alpha/lambda (everything that can reach the detector is coherent)
    coherent_object_bandwidth_inv_nm: float = 0.0
    coherent_field_max_px: int = 8_000_000  # larger scan fields build transmission tiles per block (~1 GB peak at 8M)
    coherent_cache_mb: int = 64  # computed pattern blocks kept for frame-by-frame rendering
    coherent_read_ahead: int = 2  # live 4D-STEM: blocks of the scan computed ahead in the background

    # ---- caching --------------------------------------------------------
    fieldmap_cache_size: int = 4
    time_quantum_s: float = 0.1  # re-rasterise a time-dependent specimen at most this often
