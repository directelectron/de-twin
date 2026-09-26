"""``derive_optics``: MicroscopeState + AcquisitionRequest + calibration -> OpticsState.

A port of ``OpticsState::FromParams`` (virtual-specimen/.../OpticsState.cpp) with
the dose made physical. Differences from the C++ are listed in
``de_twin/render/README`` of the report and marked ``[twin]`` below.

Sign convention (C++ README 2.2, unchanged): the optic axis is fixed and the
specimen slides under it, so the world point on the axis is::

    view.x = sx * (-(stage.x + offset.x) + image_shift.x + beam_shift.x)
    view.y = sy * (-(stage.y + offset.y) + image_shift.y + beam_shift.y)

(sx/sy = -1 when Flip X/Y). The view moves *against* the stage, so image
content moves *with* the stage (+1 um of stage x moves content by
+1000/nm_per_px pixels); image/beam shift move the view with them, so content
moves the opposite way.
"""

from __future__ import annotations

import math

import numpy as np

from ..specimen.fieldmap import ViewWindow
from ..state import AcquisitionRequest, MicroscopeState, Projection, RenderMode, TemStem
from .beam import (beam_electrons_per_s, current_density_e_per_nm2_s, illuminated_diameter_um,
                   illumination_semi_angle_mrad, probe_d0_nm)
from .calibration import Calibration
from .config import OpticsConfig
from .aberrations import Aberrations, uncorrected
from .physics import electron_wavelength_nm, focal_spread_nm
from .state import OpticsState

#: The default of `OpticsConfig.max_raster_pixels`: the largest raster a TEM image is
#: rendered at; a bigger camera frame is upsampled from it. 1024² keeps a re-render (every
#: stage move and focus step) well under a second on a 4096² camera, at the cost of detail
#: finer than a quarter of its pixels. ``OpticsConfig(max_raster_pixels=0)`` renders at
#: the frame's own sampling.
MAX_RASTER_PIXELS = 1024 * 1024
MIN_RASTER_SIDE = 32
SAED_RASTER_SIDE = 256
EDGE_SOFTENING_SIGMA_PX = 0.5
FRESNEL_GAIN = 0.6
FRESNEL_SATURATION_UM = 5.0
FRESNEL_MIN_SIGMA_RASTER_PX = 1.0
FRESNEL_WIDTH_RATIO = 2.0
MIN_ALPHA_RAD = 1.0e-6
MAX_ALPHA_RAD = 0.1
MIN_DISK_RADIUS_PX = 0.75


def select_render_mode(state: MicroscopeState, request: AcquisitionRequest,
                       cfg: OpticsConfig) -> RenderMode:
    """C++ dispatch: STEM is tested before diffraction, then the override applies."""
    if request.scan.enabled:
        mode = RenderMode.STEM_4D
    elif state.tem_stem == TemStem.STEM:
        mode = RenderMode.STEM_PARKED
    elif state.projection == Projection.DIFFRACTION:
        mode = RenderMode.TEM_DIFFRACTION
    else:
        mode = RenderMode.TEM_IMAGING
    if cfg.render_mode_override is not None:
        mode = RenderMode(cfg.render_mode_override)
    return mode


def _output_geometry(request: AcquisitionRequest, camera) -> tuple[tuple[int, int], tuple[float, float]]:
    """(output_shape (h, w), ROI-centre offset from the sensor centre in pixels (dx, dy))."""
    sh, sw = (int(v) for v in camera.sensor_shape)
    roi = request.hw_roi
    if roi is None or roi.w <= 0 or roi.h <= 0:
        return (sh, sw), (0.0, 0.0)
    w = min(int(roi.w), sw)
    h = min(int(roi.h), sh)
    x = int(np.clip(roi.x, 0, sw - w))
    y = int(np.clip(roi.y, 0, sh - h))
    return (h, w), (x + w / 2.0 - sw / 2.0, y + h / 2.0 - sh / 2.0)


def _tem_imaging(state: MicroscopeState) -> bool:
    return state.tem_stem == TemStem.TEM and state.projection == Projection.IMAGING


def _shift_terms(state: MicroscopeState, cfg: OpticsConfig) -> tuple[float, float]:
    """Everything but the stage that moves the view centre (before the flips), um:
    image shift (through the column's image-shift matrix), beam shift, the displacement of a
    tilted specimen that is off eucentric height, and the magnification's image offset."""
    isx, isy = state.image_shift_um.x, state.image_shift_um.y
    r = cfg.realism
    if r is not None and _tem_imaging(state):
        isx, isy = (float(v) for v in r.is_matrix(state.mag_mode, state.magnification) @ (isx, isy))
    x = isx + state.beam_shift_um.x
    y = isy + state.beam_shift_um.y
    # a specimen dz above the eucentric plane puts the point dz tan(tilt) away (specimen
    # coordinates) on axis, which the foreshortened view images dz sin(tilt) across the axis
    dz = state.stage.z_um + cfg.stage_offset_um[2] - cfg.eucentric_height_um
    if dz:
        x += dz * math.tan(math.radians(state.stage.beta_deg))
        y += dz * math.tan(math.radians(state.stage.alpha_deg))
    if r is not None and _tem_imaging(state):
        ox, oy = r.mag_offset_um(state.mag_mode, state.magnification)
        x, y = x + ox, y + oy
    return x, y


def view_center_um(state: MicroscopeState, cfg: OpticsConfig) -> tuple[float, float]:
    sx = -1.0 if cfg.flip_x else 1.0
    sy = -1.0 if cfg.flip_y else 1.0
    ox, oy, _ = cfg.stage_offset_um
    err = getattr(state, "stage_error_um", None)
    ex, ey = (err.x, err.y) if err is not None else (0.0, 0.0)
    tx, ty = _shift_terms(state, cfg)
    cx = sx * (-(state.stage.x_um + ox + ex) + tx)
    cy = sy * (-(state.stage.y_um + oy + ey) + ty)
    return cx, cy


def stage_for_view_center(center_um: tuple[float, float], state: MicroscopeState,
                          cfg: OpticsConfig) -> tuple[float, float]:
    """The stage (x, y) µm that puts specimen point *center_um* on axis, with the column's
    current image and beam shifts — the inverse of :func:`view_center_um`."""
    sx = -1.0 if cfg.flip_x else 1.0
    sy = -1.0 if cfg.flip_y else 1.0
    ox, oy, _ = cfg.stage_offset_um
    err = getattr(state, "stage_error_um", None)
    ex, ey = (err.x, err.y) if err is not None else (0.0, 0.0)
    tx, ty = _shift_terms(state, cfg)
    x = -(center_um[0] / sx - tx) - ox - ex
    y = -(center_um[1] / sy - ty) - oy - ey
    return x, y


def derive_optics(state: MicroscopeState, request: AcquisitionRequest, camera,
                  calibration: Calibration | None = None,
                  cfg: OpticsConfig | None = None) -> OpticsState:
    calibration = calibration or Calibration.default()
    cfg = cfg or OpticsConfig()

    mode = select_render_mode(state, request, cfg)
    scanning = mode in (RenderMode.STEM_4D, RenderMode.STEM_PARKED)
    output_shape, roi_offset_px = _output_geometry(request, camera)
    oh, ow = output_shape

    # Beam energy
    ht_kv = state.ht_kv if state.ht_kv > 0 else 200.0
    lam = electron_wavelength_nm(ht_kv)
    mrad_per_inv_nm = 1000.0 * lam

    # Pixel size (unbinned detector pixels; the detector does the binning)
    if cfg.pixel_size_override_nm > 0:
        nm_per_px = float(cfg.pixel_size_override_nm)
    else:
        nm_per_px = calibration.specimen_pixel_nm(state, camera) if state.tem_stem == TemStem.TEM else 0.0
        if not nm_per_px > 0:
            nm_per_px = float(cfg.fallback_pixel_nm)
    # What the column's calibration says, and (realism) what the camera really sees
    nominal_nm_per_px = nm_per_px
    if cfg.realism is not None and _tem_imaging(state) and cfg.pixel_size_override_nm <= 0:
        nm_per_px = nm_per_px * cfg.realism.pixel_scale(state.mag_mode, state.magnification)

    # Reciprocal space
    cl_mm = state.camera_length_mm if state.camera_length_mm > 0 else cfg.default_camera_length_mm
    recip = 0.0
    if cfg.geometric_diffraction:
        recip = Calibration.geometric_recip_inv_nm(state, camera, cl_mm)
    if not recip > 0:
        recip = calibration.recip_pixel_inv_nm(state, camera)
    if not recip > 0:
        recip = float(cfg.fallback_recip_inv_nm)
    mrad_per_px = recip * mrad_per_inv_nm

    # STEM step (nm per scan point; no binning factor)
    scan = request.scan
    nx_scan, ny_scan = (int(scan.size[0]), int(scan.size[1])) if scanning else (0, 0)
    step_nm = 0.0
    if scanning:
        if scan.step_um > 0:
            step_nm = scan.step_um * 1000.0
        elif cfg.pixel_size_override_nm > 0:
            step_nm = float(cfg.pixel_size_override_nm)
        else:
            step_nm = calibration.stem_step_nm(state, (max(nx_scan, 1), max(ny_scan, 1)))
            if not step_nm > 0:
                step_nm = float(cfg.fallback_pixel_nm)

    # Tilt: alpha (about the lab x axis) compresses y, beta (about y) compresses x; the same
    # rotation S = Rx(alpha) Ry(beta) turns every grain (de_twin.crystal.orientation).
    alpha_rad = math.radians(state.stage.alpha_deg)
    beta_rad = math.radians(state.stage.beta_deg)
    cos_y = max(math.cos(alpha_rad), 0.1)
    cos_x = max(math.cos(beta_rad), 0.1)
    t_factor = 1.0 / (cos_x * cos_y)

    # Illumination and selected area
    d_ill_um = illuminated_diameter_um(state, cfg)
    sa_um = float(cfg.sa_aperture_um)
    d_sa_um = min(sa_um, d_ill_um) if sa_um > 0 else d_ill_um

    # Raster
    downsample = 1
    if scanning:
        ry, rx = max(1, ny_scan), max(1, nx_scan)
        raster_nm = step_nm
    elif mode == RenderMode.TEM_DIFFRACTION:
        ry = rx = SAED_RASTER_SIDE
        raster_nm = max(1e-6, d_sa_um) * 1000.0 / SAED_RASTER_SIDE
    else:
        d = 1
        cap = int(getattr(cfg, "max_raster_pixels", MAX_RASTER_PIXELS))
        while cap > 0 and (ow * oh) / float(d * d) > cap:
            d *= 2
        downsample = d
        rx = max(MIN_RASTER_SIDE, ow // d)
        ry = max(MIN_RASTER_SIDE, oh // d)
        raster_nm = nm_per_px * d

    # View
    cx, cy = view_center_um(state, cfg)
    if mode == RenderMode.TEM_IMAGING and (roi_offset_px[0] or roi_offset_px[1]):
        # [twin] an off-centre hardware ROI looks at an off-centre part of the field
        cx += roi_offset_px[0] * nm_per_px / 1000.0 / cos_x
        cy += roi_offset_px[1] * nm_per_px / 1000.0 / cos_y
    rotation = math.radians(scan.rotation_deg) if scanning else 0.0
    if cfg.realism is not None and mode == RenderMode.TEM_IMAGING and not scanning:
        rotation = cfg.realism.rotation_rad(state.mag_mode, state.magnification)
    view = ViewWindow(center_um=(cx, cy), pixel_um=raster_nm / 1000.0, shape=(int(ry), int(rx)),
                      rotation_rad=rotation, cos_alpha=cos_y, cos_beta=cos_x)

    # Focus
    ox, oy, oz = cfg.stage_offset_um
    defocus_um = (state.defocus_um + cfg.defocus_offset_um
                  + (state.stage.z_um + oz - cfg.eucentric_height_um))
    theta_ill_mrad = illumination_semi_angle_mrad(state, cfg)
    theta_ill = theta_ill_mrad / 1000.0
    # The convergence (and so the diffraction disk radius) is the column's in every mode:
    # parallel TEM/SAED ~0.01-0.1 mrad, NBD ~0.5-3, CBD ~3-15, STEM 5-30 mrad. (The C++
    # forced theta_ill in every diffraction mode, so NBD/CBD never produced disks.)
    if state.convergence_semi_angle_mrad > 0:
        alpha = state.convergence_semi_angle_mrad / 1000.0
    elif scanning:
        alpha = cfg.stem_default_convergence_mrad / 1000.0
    else:
        alpha = theta_ill
    alpha = float(np.clip(alpha, MIN_ALPHA_RAD, MAX_ALPHA_RAD))

    render_nm = nm_per_px if not scanning else step_nm
    sigma_geo = abs(defocus_um) * 1000.0 * theta_ill / max(render_nm, 1e-12)
    blur_sigma_render = max(sigma_geo, EDGE_SOFTENING_SIGMA_PX)
    sigma_raster = blur_sigma_render / max(1, downsample)
    fresnel_sign = 1.0 if defocus_um < 0 else (-1.0 if defocus_um > 0 else 0.0)
    fresnel_gain = FRESNEL_GAIN * min(1.0, abs(defocus_um) / FRESNEL_SATURATION_UM)
    fresnel_sigma = max(FRESNEL_WIDTH_RATIO * sigma_raster, FRESNEL_MIN_SIGMA_RASTER_PX)

    # Probe
    d0 = probe_d0_nm(state, cfg)
    probe_d = d0 + 2.0 * abs(defocus_um) * 1000.0 * alpha
    disk_r = alpha / (lam * recip) if recip > 0 else 0.0
    if disk_r > 0:
        disk_r = max(disk_r, MIN_DISK_RADIUS_PX)

    # Diffraction centre: the optic axis sits at the sensor centre (C++ w//2 convention)
    sh, sw = (int(v) for v in camera.sensor_shape)
    roi = request.hw_roi
    rx0 = int(roi.x) if roi is not None and roi.w > 0 else 0
    ry0 = int(roi.y) if roi is not None and roi.h > 0 else 0
    if roi is None or roi.w <= 0:
        axis = (ow // 2, oh // 2)
    else:
        axis = (sw // 2 - rx0, sh // 2 - ry0)
    ds = state.diffraction_shift_mrad
    bt = state.beam_tilt_mrad  # a tilted beam moves the whole pattern (no descan of tilt)
    center = (axis[0] + (ds.x + bt.x) / mrad_per_px, axis[1] + (ds.y + bt.y) / mrad_per_px)

    # Dose [twin]: physical electrons
    blanked = (state.beam_blanked or not state.ht_on or not state.column_valves_open
               or state.screen_position == 1 or not request.beam_reaches_detector)
    beam_e_s = beam_electrons_per_s(state, cfg)
    current_pa = beam_e_s * 1.602176634e-19 * 1e12
    if mode == RenderMode.TEM_IMAGING:
        pattern_e_s = 0.0
        dose = current_density_e_per_nm2_s(state, cfg) * nm_per_px * nm_per_px
    elif mode == RenderMode.TEM_DIFFRACTION:
        frac = min(1.0, (d_sa_um / d_ill_um) ** 2) if d_ill_um > 0 else 1.0
        pattern_e_s = beam_e_s * frac
        dose = pattern_e_s / float(ow * oh)
    else:
        pattern_e_s = beam_e_s
        dose = pattern_e_s / float(ow * oh)
    if cfg.dose_model == "legacy":
        spot = state.spot_size if state.spot_size > 0 else 3
        dose = cfg.legacy_dose_rate_counts_per_s * (3.0 / spot) ** 2
        pattern_e_s = dose * ow * oh if mode != RenderMode.TEM_IMAGING else 0.0

    # Park position (world um) for the parked probe / explicit park request
    park = None
    if scanning and scan.park:
        px, py = scan.park_position
        wx, wy = view.pixel_to_world(np.array([py]), np.array([px]))
        park = (float(wx[0]), float(wy[0]))

    # Aberrations [twin]
    stig = state.objective_stig
    a1 = ((stig.x - cfg.objective_stig_zero[0]) * cfg.stig_nm_per_unit,
          (stig.y - cfg.objective_stig_zero[1]) * cfg.stig_nm_per_unit)
    cstig = state.condenser_stig
    a1_probe = ((cstig.x - cfg.condenser_stig_zero[0]) * cfg.condenser_stig_nm_per_unit,
                (cstig.y - cfg.condenser_stig_zero[1]) * cfg.condenser_stig_nm_per_unit)
    # Residual/corrected aberrations from the column (corrector), else an uncorrected C3;
    # plus the operator's focus (C1) and the stigmator of each side (A1).
    c1_nm = defocus_um * 1000.0
    # a non-empty dict is the column's full residual set (zeros included: {"C3": 0} is corrected)
    img_d = getattr(state, "image_aberrations", None) or {}
    prb_d = getattr(state, "probe_aberrations", None) or {}
    image_ab = Aberrations(img_d) if img_d else uncorrected(cfg.cs_mm)
    image_ab = image_ab + {"C1": c1_nm, "A1": complex(a1[0], a1[1])}
    probe_ab = Aberrations(prb_d) if prb_d else uncorrected(cfg.cs_mm)
    probe_ab = probe_ab + {"C1": c1_nm, "A1": complex(a1_probe[0], a1_probe[1])}
    spot = state.spot_size if state.spot_size > 0 else cfg.reference_spot
    source_nm = cfg.stem_source_size_nm * cfg.reference_spot / spot
    bx, by = (request.hw_binning if request.hw_binning is not None else (1, 1))

    extras = {
        "selected_area_um": d_sa_um,
        "mrad_per_px": mrad_per_px,
        "axis_center_px": (float(axis[0]), float(axis[1])),
        "probe_d0_nm": d0,
        "probe_mode": int(state.probe_mode),
        "spot_size": int(state.spot_size),
        "blur_sigma_render_px": blur_sigma_render,
        "illumination_semi_angle_mrad": theta_ill_mrad,
        "stem_step_nm": step_nm,
        "probe_blend_offset_px": (0.5 * probe_d / step_nm) if step_nm > 0 else 0.0,
        "nominal_pixel_nm": float(nominal_nm_per_px),
        "render_nm_per_px": render_nm,
        "view_center_um": (cx, cy),
        "roi_offset_px": roi_offset_px,
        "scan_park": bool(scan.park),
        "park_position": tuple(scan.park_position),
        "exposure_s": (scan.dwell_s if scan.enabled else request.frame_time_s),
        "hw_binning": (max(1, int(bx)), max(1, int(by))),
    }

    return OpticsState(
        render_mode=mode,
        ht_kv=ht_kv,
        wavelength_nm=lam,
        beam_blanked=bool(blanked),
        output_shape=(int(oh), int(ow)),
        view=view,
        specimen_pixel_nm=float(nm_per_px),
        recip_pixel_inv_nm=float(recip),
        camera_length_mm=float(cl_mm),
        diffraction_center_px=(float(center[0]), float(center[1])),
        convergence_mrad=alpha * 1000.0,
        illumination_mrad=theta_ill_mrad,
        probe_diameter_nm=float(probe_d),
        disk_radius_px=float(disk_r),
        illuminated_diameter_um=float(d_ill_um),
        sa_aperture_um=sa_um,
        dose_e_per_px_s=float(dose),
        beam_current_pa=float(current_pa),
        defocus_um=float(defocus_um),
        blur_sigma_px=float(sigma_raster),
        fresnel_gain=float(fresnel_gain),
        fresnel_sign=float(fresnel_sign),
        fresnel_sigma_px=float(fresnel_sigma),
        objective_stig=(float(stig.x), float(stig.y)),
        beam_tilt_mrad=(float(state.beam_tilt_mrad.x), float(state.beam_tilt_mrad.y)),
        precession_mrad=float(state.precession_mrad) if state.precession_on else 0.0,
        precession_hz=float(state.precession_hz),
        precession_descan=bool(state.precession_descan),
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
        thickness_tilt_factor=t_factor,
        scan_step_um=step_nm / 1000.0,
        scan_shape=(int(ny_scan), int(nx_scan)) if scanning else (0, 0),
        scan_rotation_rad=rotation,
        park_um=park,
        raster_downsample=int(downsample),
        cs_mm=float(image_ab["C3"].real) * 1e-6,
        cc_mm=float(cfg.cc_mm),
        focal_spread_nm=float(focal_spread_nm(cfg.cc_mm, cfg.energy_spread_ev, ht_kv)),
        astigmatism_nm=(float(a1[0]), float(a1[1])),
        objective_aperture_mrad=float(cfg.objective_aperture_mrad),
        pattern_e_per_s=float(pattern_e_s),
        image_aberrations=image_ab,
        probe_aberrations=probe_ab,
        aberrations_derived=True,
        source_size_nm=float(source_nm),
        extras=extras,
    )


def track_live_view(optics: OpticsState, state: MicroscopeState,
                    cfg: OpticsConfig | None = None, calibration: Calibration | None = None,
                    request: AcquisitionRequest | None = None) -> OpticsState:
    """C++ ``TrackLiveView``: follow stage/shifts (and the STEM step) without
    touching anything else, with a 5 %-of-a-step deadband. Returns ``optics``
    unchanged when nothing moved."""
    import dataclasses

    cfg = cfg or OpticsConfig()
    scanning = optics.render_mode in (RenderMode.STEM_4D, RenderMode.STEM_PARKED)
    step_nm = optics.view.pixel_um * 1000.0
    if scanning:
        live = 0.0
        if cfg.pixel_size_override_nm > 0:
            live = cfg.pixel_size_override_nm
        elif request is not None and request.scan.step_um > 0:
            live = request.scan.step_um * 1000.0
        elif calibration is not None:
            ny, nx = optics.scan_shape
            live = calibration.stem_step_nm(state, (max(nx, 1), max(ny, 1)))
        if live > 0:
            step_nm = live
    cx, cy = view_center_um(state, cfg)
    ocx, ocy = optics.view.center_um
    dead = 0.05 * step_nm / 1000.0
    moved = abs(cx - ocx) > dead or abs(cy - ocy) > dead
    old_nm = optics.view.pixel_um * 1000.0
    scale_changed = scanning and abs(step_nm - old_nm) > 1e-4 * max(step_nm, old_nm)
    if not moved and not scale_changed:
        return optics
    view = dataclasses.replace(optics.view, center_um=(cx, cy),
                               pixel_um=(step_nm / 1000.0) if scale_changed else optics.view.pixel_um)
    extras = dict(optics.extras)
    if scale_changed:
        extras["stem_step_nm"] = step_nm
        extras["probe_blend_offset_px"] = 0.5 * optics.probe_diameter_nm / step_nm
    return dataclasses.replace(optics, view=view, extras=extras,
                               scan_step_um=(step_nm / 1000.0) if scanning else optics.scan_step_um)
