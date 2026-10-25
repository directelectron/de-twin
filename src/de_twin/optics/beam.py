"""Physical beam model: probe current, illuminated area, illumination angle, probe size.

This replaces the C++ "counts per second" dose with electrons.

Model (all knobs in :class:`~de_twin.optics.config.OpticsConfig`)
---------------------------------------------------------------
* **Beam current** ``I = I_ref * (spot_ref/spot)**p * f_mode * (d_CLA/d_ref)**2``.
  ``I_ref`` = 600 pA at spot 3 in TEM mode, ``p`` = 2 (the C++ ``1/spot^2`` law,
  so spot 1 : spot 5 = 25 : 1). ``f_mode`` scales for the probe mode (nanoprobe
  0.05, NBD 0.02, CBD 0.1, ...); in STEM a TEM/unknown probe mode uses
  ``stem_current_factor`` (0.05 -> 30 pA at spot 3). The condenser aperture
  scales the current with its area.
* **Illuminated diameter** (TEM/microprobe): ``D = D_min * (D_max/D_min)**intensity``
  with D_min = 0.1 um, D_max = 100 um, so intensity 0.5 -> 3.16 um (x30 in
  LowMAG, whose condenser setting floods the much larger field). Nanoprobe
  / NBD / CBD in TEM mode use the C++ ``0.2 um * spot/3`` spot.
* **Current density** ``J = I / (e * pi D^2/4)`` [e/nm^2/s] (uniform top hat);
  flux per detector pixel ``= J * p^2`` where p is nm of specimen per detector
  pixel. At spot 3, intensity 0.5 on a DE16 (6.5 um pixels, geometric
  calibration) that is ~50 e/px/s at 20 kx, ~22 at 30 kx and ~8 at 50 kx.
* **Illumination semi-angle** (Liouville / brightness conservation):
  ``alpha = alpha_ref * D_ref / D``; 0.1 mrad at 3.16 um, clamped to
  [0.005, 5] mrad. A focused beam is less coherent, which damps Thon rings.
"""

from __future__ import annotations

import math

import numpy as np

from ..state import MicroscopeState, ProbeMode, TemStem
from .config import OpticsConfig
from .physics import ELECTRON_CHARGE_C

_PROBE_FORMING = (ProbeMode.NANOPROBE, ProbeMode.NBD, ProbeMode.CBD)

# C++ NominalProbeDiameterNm, nm, spot 1..6
_D0_TABLE = {
    "nano": (0.5, 0.8, 1.3, 2.0, 3.2, 5.0),
    "cbd": (0.8, 1.3, 2.0, 3.2, 5.0, 8.0),
    "micro": (5.0, 8.0, 13.0, 20.0, 32.0, 50.0),
    "tem": (2.0, 3.2, 5.0, 8.0, 13.0, 20.0),
}


def _probe_mode(state: MicroscopeState) -> ProbeMode:
    try:
        return ProbeMode(int(state.probe_mode))
    except ValueError:
        return ProbeMode.UNKNOWN


def beam_current_pa(state: MicroscopeState, config: OpticsConfig | None = None) -> float:
    """Probe current in picoamperes for the column state (gun on)."""
    config = config or OpticsConfig()
    if not state.ht_on:
        return 0.0
    spot = int(state.spot_size) if state.spot_size and state.spot_size > 0 else config.reference_spot
    current = config.reference_current_pa * (config.reference_spot / spot) ** config.spot_exponent
    mode = _probe_mode(state)
    if state.tem_stem == TemStem.STEM and mode in (ProbeMode.TEM, ProbeMode.UNKNOWN):
        current *= config.stem_current_factor
    else:
        current *= float(config.probe_mode_current_factor.get(mode, 1.0))
    index = int(state.condenser_aperture_index)
    aps = config.condenser_apertures_um
    if aps and 1 <= index <= len(aps) and 1 <= config.reference_condenser_index <= len(aps):
        current *= (aps[index - 1] / aps[config.reference_condenser_index - 1]) ** 2
    return float(current)


def beam_electrons_per_s(state: MicroscopeState, config: OpticsConfig | None = None) -> float:
    return beam_current_pa(state, config) * 1e-12 / ELECTRON_CHARGE_C


def illuminated_diameter_um(state: MicroscopeState, config: OpticsConfig | None = None) -> float:
    """Diameter of the illuminated area on the specimen (TEM / SAED)."""
    config = config or OpticsConfig()
    if config.tem_illuminated_area_um > 0:
        return float(config.tem_illuminated_area_um)
    mode = _probe_mode(state)
    if mode in _PROBE_FORMING:
        spot = state.spot_size if state.spot_size > 0 else config.reference_spot
        return config.nanoprobe_illuminated_um * spot / config.reference_spot
    x = float(np.clip(state.intensity, 0.0, 1.0))
    r = getattr(config, "realism", None)
    if r is not None:
        # a real C2: the beam converges to a crossover and spreads again past it
        x0 = r.crossover(int(state.spot_size), int(state.probe_mode))
        span = max(x0, 1.0 - x0)
        d = math.hypot(config.illum_min_diameter_um, config.illum_max_diameter_um * (x - x0) / span)
    else:
        d = config.illum_min_diameter_um * (config.illum_max_diameter_um / config.illum_min_diameter_um) ** x
    zoom = _intensity_zoom(state, config)
    if zoom:
        # the condenser tracks the field of view (LowMAG included: no separate spread)
        factor = config.intensity_zoom_reference_mag / float(state.magnification)
        if not config.intensity_zoom:  # zooming out only
            factor = max(1.0, factor)
        return float(d * factor)
    mode = "".join(ch for ch in str(state.mag_mode).lower() if ch.isalnum())
    if mode.startswith("low") or mode == "lm":
        d *= config.lowmag_illumination_factor
    return float(d)


def _intensity_zoom(state: MicroscopeState, config: OpticsConfig) -> bool:
    """Intensity zoom applies to TEM imaging with a known magnification."""
    if not (getattr(config, "intensity_zoom", False) or getattr(config, "intensity_zoom_out", False)):
        return False
    if not state.magnification or state.magnification <= 0:
        return False
    if config.intensity_zoom_reference_mag <= 0:
        return False
    try:
        from ..state import Projection, TemStem

        return state.tem_stem == TemStem.TEM and state.projection == Projection.IMAGING
    except Exception:
        return False


def illumination_semi_angle_mrad(state: MicroscopeState, config: OpticsConfig | None = None) -> float:
    config = config or OpticsConfig()
    if config.illumination_semi_angle_mrad > 0:
        return float(config.illumination_semi_angle_mrad)
    d = illuminated_diameter_um(state, config)
    a = config.illum_reference_alpha_mrad * config.illum_reference_diameter_um / max(d, 1e-6)
    return float(np.clip(a, 0.005, 5.0))


def probe_d0_nm(state: MicroscopeState, config: OpticsConfig | None = None) -> float:
    """C++ NominalProbeDiameterNm(probeMode, spot), or the override."""
    config = config or OpticsConfig()
    if config.probe_d0_nm > 0:
        return float(config.probe_d0_nm)
    mode = _probe_mode(state)
    if mode in (ProbeMode.NANOPROBE, ProbeMode.NBD):
        column = _D0_TABLE["nano"]
    elif mode == ProbeMode.CBD:
        column = _D0_TABLE["cbd"]
    elif mode == ProbeMode.MICROPROBE:
        column = _D0_TABLE["micro"]
    else:
        column = _D0_TABLE["tem"]
    s = int(state.spot_size)
    index = 2 if s <= 0 else (5 if s >= 6 else s - 1)
    return column[index]


def current_density_e_per_nm2_s(state: MicroscopeState, config: OpticsConfig | None = None) -> float:
    """Uniform current density inside the illuminated disk (TEM)."""
    d_nm = illuminated_diameter_um(state, config) * 1000.0
    return beam_electrons_per_s(state, config) / (math.pi * 0.25 * d_nm * d_nm)
