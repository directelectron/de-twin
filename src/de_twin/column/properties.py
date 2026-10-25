"""The column's property tables: how each DE-TEM-Channel property name reads and writes a Column."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..state import Vec2
from . import ladders as L
from .errors import ColumnRefused

if TYPE_CHECKING:
    from .column import Column
from .vendor import max_spot_size, probe_mode_to_raw, screen_to_raw, submode_to_raw

#: Beam tilt: DE-TEM-Channel's range is the normalised [-1, 1]; one unit is this many mrad.
BEAM_TILT_MRAD_PER_UNIT = 10.0
#: Diffraction shift: normalised +/-1 per axis; one unit is this many mrad.
DIFF_SHIFT_MRAD_PER_UNIT = 10.0

STAGE_AXES = ("x", "y", "z", "a", "b")


def _method(name: str):
    """A setter that calls the Column method ``name`` (the tables are built before Column)."""
    return lambda column, value: getattr(column, name)(value)


def _pair(value: Any, name: str) -> tuple[float, float]:
    try:
        if isinstance(value, dict):
            x, y = value["x"], value["y"]
        elif isinstance(value, Vec2):
            x, y = value.x, value.y
        else:
            x, y = value
        x, y = float(x), float(y)
    except (TypeError, ValueError, KeyError):
        raise ColumnRefused(f"{name} needs exactly two numeric coordinates") from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ColumnRefused(f"{name} coordinates must be finite")
    return x, y


# ---------------------------------------------------------------------------
# Property tables
# ---------------------------------------------------------------------------


def _v2(v: Vec2, scale: float = 1.0) -> tuple[float, float]:
    return (v.x / scale, v.y / scale)


def _set_vec(attr: str, scale: float = 1.0):
    def setter(col: Column, value) -> None:
        x, y = _pair(value, attr)
        setattr(col._s, attr, Vec2(x * scale, y * scale))

    return setter


def _set_low_dose(c: Column, value) -> None:
    on = value.strip().lower() in ("1", "true", "on", "yes") if isinstance(value, str) else bool(value)
    ld = c.low_dose
    if on == ld.enabled:
        return
    if on:
        if not ld.areas:
            ld.seed_from(c, L.LOWMAG_MAGS + L.MAG1_MAGS)
        else:  # what was set while low dose was off becomes Record
            ld.area = "Record"
            ld.store(c)
            ld.apply(c, "Record", SETTERS)
        ld.enabled = True
    else:
        ld.store(c)
        ld.apply(c, "Record", SETTERS)
        ld.enabled = False


def _set_low_dose_area(c: Column, value) -> None:
    from .lowdose import normalise_area

    ld = c.low_dose
    if not ld.enabled:
        raise ColumnRefused("low dose is off (set LowDose first)")
    name = normalise_area(value)
    if name is None:
        raise ColumnRefused(f"unknown low-dose area {value!r} (Search, View, Focus, Record)")
    if name == ld.area:
        return
    ld.store(c)
    ld.apply(c, name, SETTERS)


def _set_low_dose_areas(c: Column, value) -> None:
    from .lowdose import LowDoseArea, area_from_state, normalise_area

    ld = c.low_dose
    value = dict(value)
    if not value:
        return
    # validate everything before changing anything
    updates = {}
    for key, params in value.items():
        name = normalise_area(key)
        if name is None:
            raise ColumnRefused(f"unknown low-dose area {key!r}")
        clean = {}
        for f, v in dict(params).items():
            if f not in LowDoseArea.__dataclass_fields__:
                raise ColumnRefused(f"unknown low-dose area setting {f!r}")
            try:
                if f in ("image_shift", "beam_shift"):
                    pair = tuple(float(x) for x in v)
                    if len(pair) != 2:
                        raise ValueError("needs (x, y)")
                    clean[f] = pair
                elif f == "spot_size":
                    clean[f] = int(v)
                else:
                    clean[f] = float(v)
            except (TypeError, ValueError) as e:
                raise ColumnRefused(f"low-dose {name} {f}: {v!r} ({e})") from None
            vals = clean[f] if isinstance(clean[f], tuple) else (clean[f],)
            if not all(math.isfinite(x) for x in vals):
                raise ColumnRefused(f"low-dose {name} {f} must be finite")
        updates[name] = clean
    if not ld.areas:
        ld.seed_from(c, L.LOWMAG_MAGS + L.MAG1_MAGS)
    if ld.enabled and ld.area in updates:  # keep what was set live in the area being edited
        ld.store(c)
    for name, clean in updates.items():
        a = ld.areas.get(name) or area_from_state(c._s)
        for f, v in clean.items():
            setattr(a, f, v)
        ld.areas[name] = a
        if ld.enabled and name == ld.area:
            ld.apply(c, name, SETTERS)


def _set_flag(attr: str):
    def setter(col: Column, value) -> None:
        v = value.strip().lower() in ("1", "true", "on", "yes") if isinstance(value, str) else bool(value)
        setattr(col._s, attr, v)

    return setter


def _set_float(attr: str, lo: float, hi: float):
    def setter(col: Column, value) -> None:
        v = float(value)
        if not lo <= v <= hi:
            raise ColumnRefused(f"{attr} must be in [{lo:g}, {hi:g}], got {v:g}")
        setattr(col._s, attr, v)

    return setter


def _set_extra_pair(key: str):
    def setter(col: Column, value) -> None:
        col._extra[key] = _pair(value, key)

    return setter


def _set_extra_float(key: str):
    def setter(col: Column, value) -> None:
        col._extra[key] = float(value)

    return setter


def _set_axis(i: int):
    def setter(col: Column, value) -> None:
        col._s_stage({STAGE_AXES[i]: float(value)})

    return setter


def _get_axis(i: int):
    def getter(col: Column) -> float:
        col._update_stage()
        return float(col._stage_now[i])

    return getter


GETTERS: dict[str, Callable[[Column], Any]] = {
    "Magnification": lambda c: float(c._s.magnification),
    "MagString": lambda c: format(float(c._s.magnification), ".6g"),
    "MagnificationIndex": lambda c: c._mag_index(),
    "CameraLength": lambda c: float(c._s.camera_length_mm) / 10.0,
    "Defocus": lambda c: float(c._s.defocus_um),
    "Focus": lambda c: int(c._s.defocus_um),
    "SpotSize": lambda c: int(c._s.spot_size),
    "ProbeMode": lambda c: probe_mode_to_raw(c._s.instrument_type, c._s.probe_mode),
    "ConvergenceAngle": lambda c: float(c._convergence()),
    "AlphaSelector": lambda c: int(c._s.alpha_selector),
    "CondenserApertureIndex": lambda c: int(c._extra["ApertureSizes"][L.APERTURE_KIND_CLA]),
    "Intensity": lambda c: float(c._s.intensity),
    "ImageShift": lambda c: _v2(c._s.image_shift_um),
    "BeamShift": lambda c: _v2(c._s.beam_shift_um),
    "BeamTilt": lambda c: _v2(c._s.beam_tilt_mrad, BEAM_TILT_MRAD_PER_UNIT),
    "DiffractionShift": lambda c: _v2(c._s.diffraction_shift_mrad, DIFF_SHIFT_MRAD_PER_UNIT),
    "Precession": lambda c: bool(c._s.precession_on),
    "LowDose": lambda c: bool(c.low_dose.enabled),
    "LowDoseArea": lambda c: c.low_dose.area if c.low_dose.enabled else "",
    "LowDoseAreas": lambda c: {k: a.as_dict() for k, a in c.low_dose.areas.items()},
    "PrecessionAngle": lambda c: float(c._s.precession_mrad),
    "PrecessionFrequency": lambda c: float(c._s.precession_hz),
    "PrecessionDescan": lambda c: bool(c._s.precession_descan),
    "ObjectiveStig": lambda c: _v2(c._s.objective_stig),
    "CondenserStig": lambda c: _v2(c._s.condenser_stig),
    "StagePosition": lambda c: c._stage_dict(),
    "StageX": _get_axis(0),
    "StageY": _get_axis(1),
    "StageZ": _get_axis(2),
    "StageA": _get_axis(3),
    "StageB": _get_axis(4),
    "TemStemMode": lambda c: int(c._s.tem_stem),
    "ProjectionMode": lambda c: 2 if c._fm == L.FM_DIFF else 1,
    "SubMode": lambda c: submode_to_raw(c._s.instrument_type, c._fm),
    "FunctionMode": lambda c: submode_to_raw(c._s.instrument_type, c._fm),
    "MagMode": lambda c: L.FUNCTION_MODE_NAMES[c._fm],
    "BeamBlank": lambda c: int(bool(c._s.beam_blanked)),
    "ScreenPosition": lambda c: screen_to_raw(c._s.instrument_type, c._s.screen_position),
    "ColumnValvesOpen": lambda c: int(bool(c._s.column_valves_open)),
    "HighTension": lambda c: float(c._s.ht_kv) * 1000.0,
    "HT": lambda c: float(c._s.ht_kv) * 1000.0,
    "MaxHighTension": lambda c: float(c._extra["MaxHighTension"]),
    "HTState": lambda c: int(bool(c._s.ht_on)),
    "Filament": lambda c: float(c._extra["Filament"]),
    "InstrumentType": lambda c: str(c._s.instrument_type),
    "OperationStatus": lambda c: c.op_state(),
    "StageMode": lambda c: int(c._extra["StageMode"]),
    "StageHolder": lambda c: int(c._extra["StageHolder"]),
    "ScanRotation": lambda c: float(c._extra["ScanRotation"]),
    "DiffractionFocus": lambda c: float(c._extra["DiffractionFocus"]),
    "GunShift": lambda c: tuple(c._extra["GunShift"]),
    "GunTilt": lambda c: tuple(c._extra["GunTilt"]),
    "ApertureKind": lambda c: int(c._extra["ApertureKind"]),
    "VacuumStatus": lambda c: float(c._extra["VacuumStatus"]),
    "VacuumGauges": lambda c: float(c._extra["VacuumGauges"]),
    "ScreenCurrent": lambda c: float(c._extra["ScreenCurrent"]),
    "ExposureTime": lambda c: float(c._extra["ExposureTime"]),
}


def _set_spot(c: Column, v) -> None:
    c._s.spot_size = min(max(int(v), 1), max_spot_size(c._s.instrument_type))


def _set_alpha(c: Column, v) -> None:
    c._s.alpha_selector = min(max(int(v), 0), 7)


def _set_blank(c: Column, v) -> None:
    c._s.beam_blanked = bool(int(v))


def _set_valves(c: Column, v) -> None:
    c._s.column_valves_open = bool(int(v))


def _set_ht_state(c: Column, v) -> None:
    c._s.ht_on = bool(int(v))


def _set_stage_mode(c: Column, v) -> None:
    c._extra["StageMode"] = min(max(int(v), 0), 1)


def _set_cla(c: Column, v) -> None:
    c._s_aperture_size((L.APERTURE_KIND_CLA, int(v)))


def _set_defocus(c: Column, v) -> None:
    c._s.defocus_um = float(v)


def _set_intensity(c: Column, v) -> None:
    c._s.intensity = float(v)


def _set_vendor(c: Column, v) -> None:
    c._s.instrument_type = str(v)


SETTERS: dict[str, Callable[[Column, Any], None]] = {
    "Magnification": _method("_s_mag"),
    "MagnificationIndex": _method("_s_mag_index"),
    "CameraLength": _method("_s_cl_cm"),
    "Defocus": _set_defocus,
    "SpotSize": _set_spot,
    "ProbeMode": _method("_s_probe"),
    "ConvergenceAngle": _method("_s_convergence"),
    "AlphaSelector": _set_alpha,
    "CondenserApertureIndex": _set_cla,
    "Intensity": _set_intensity,
    "ImageShift": _set_vec("image_shift_um"),
    "BeamShift": _set_vec("beam_shift_um"),
    "BeamTilt": _set_vec("beam_tilt_mrad", BEAM_TILT_MRAD_PER_UNIT),
    "DiffractionShift": _set_vec("diffraction_shift_mrad", DIFF_SHIFT_MRAD_PER_UNIT),
    "Precession": _set_flag("precession_on"),
    "LowDose": lambda c, v: _set_low_dose(c, v),
    "LowDoseArea": lambda c, v: _set_low_dose_area(c, v),
    "LowDoseAreas": lambda c, v: _set_low_dose_areas(c, v),
    "PrecessionAngle": _set_float("precession_mrad", 0.0, 100.0),
    "PrecessionFrequency": _set_float("precession_hz", 0.0, 1.0e5),
    "PrecessionDescan": _set_flag("precession_descan"),
    "ObjectiveStig": _set_vec("objective_stig"),
    "CondenserStig": _set_vec("condenser_stig"),
    "StagePosition": _method("_s_stage"),
    "StageX": _set_axis(0),
    "StageY": _set_axis(1),
    "StageZ": _set_axis(2),
    "StageA": _set_axis(3),
    "StageB": _set_axis(4),
    "TemStemMode": _method("_s_tem_stem"),
    "ProjectionMode": _method("_s_projection"),
    "SubMode": _method("_s_submode"),
    "FunctionMode": _method("_s_submode"),
    "BeamBlank": _set_blank,
    "ScreenPosition": _method("_s_screen"),
    "ColumnValvesOpen": _set_valves,
    "HighTension": _method("_s_ht"),
    "HT": _method("_s_ht"),
    "HTState": _set_ht_state,
    "Filament": _set_extra_float("Filament"),
    "InstrumentType": _set_vendor,
    "StageMode": _set_stage_mode,
    "ScanRotation": _set_extra_float("ScanRotation"),
    "DiffractionFocus": _set_extra_float("DiffractionFocus"),
    "GunShift": _set_extra_pair("GunShift"),
    "GunTilt": _set_extra_pair("GunTilt"),
    "ApertureKind": _method("_s_aperture_kind"),
    "ApertureSize": _method("_s_aperture_size"),
}
