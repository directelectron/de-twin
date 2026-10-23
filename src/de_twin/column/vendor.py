"""Vendor-specific raw values of the DE-TEM-Channel properties (JEOL vs. TFS)."""

from __future__ import annotations

from ..state import ProbeMode
from . import ladders as L
from .errors import ColumnRefused

# ---------------------------------------------------------------------------
# Vendor enumerations
# ---------------------------------------------------------------------------


def is_jeol(instrument_type: str) -> bool:
    return "JEOL" in str(instrument_type or "").upper()


def is_tfs(instrument_type: str) -> bool:
    v = str(instrument_type or "").upper()
    return any(t in v for t in ("FEI", "TFS", "THERMO"))


_JEOL_RAW = {
    ProbeMode.TEM: 0,
    ProbeMode.EDS: 1,
    ProbeMode.NBD: 2,
    ProbeMode.CBD: 3,
    ProbeMode.MICROPROBE: 0,
    ProbeMode.NANOPROBE: 2,
    ProbeMode.UNKNOWN: 0,
}
_FEI_RAW = {
    ProbeMode.NANOPROBE: 0,
    ProbeMode.MICROPROBE: 1,
    ProbeMode.TEM: 1,
    ProbeMode.EDS: 0,
    ProbeMode.NBD: 0,
    ProbeMode.CBD: 0,
    ProbeMode.UNKNOWN: 1,
}


def probe_mode_to_raw(instrument_type: str, mode: ProbeMode) -> int:
    """Vendor-normalised probe mode -> the raw value a vendor's channel reports."""
    table = _JEOL_RAW if is_jeol(instrument_type) or not is_tfs(instrument_type) else _FEI_RAW
    return table[ProbeMode(mode)]


def probe_mode_from_raw(instrument_type: str, raw: int) -> ProbeMode:
    """DE-Server ``NormalizeProbeMode`` (Camera.cpp)."""
    raw = int(raw)
    if is_jeol(instrument_type):
        return {0: ProbeMode.TEM, 1: ProbeMode.EDS, 2: ProbeMode.NBD, 3: ProbeMode.CBD}.get(
            raw, ProbeMode.UNKNOWN
        )
    return {0: ProbeMode.NANOPROBE, 1: ProbeMode.MICROPROBE}.get(raw, ProbeMode.UNKNOWN)


def jeol_row(mode: ProbeMode) -> int:
    """Row of the alpha/convergence table for a normalised probe mode."""
    return _JEOL_RAW[ProbeMode(mode)]


def screen_to_raw(instrument_type: str, position: int) -> int:
    """0 up / 1 down -> the vendor's enumeration (TFS: 2 up, 3 down)."""
    if is_tfs(instrument_type):
        return 3 if position else 2
    return 1 if position else 0


def screen_from_raw(instrument_type: str, raw: int) -> int:
    raw = int(raw)
    if is_tfs(instrument_type) or raw in (2, 3):
        # TFS: 1 = unknown/moving (treated as not-up: the beam may be intercepted)
        return 0 if raw in (0, 2) else 1
    if raw not in (0, 1):
        raise ColumnRefused(f"invalid screen position {raw} (0 = up, 1 = down)")
    return raw


def submode_to_raw(instrument_type: str, function_mode: int) -> int:
    if is_tfs(instrument_type):
        return L.FEI_SUBMODE_FROM_FM[function_mode]
    return int(function_mode)


def submode_from_raw(instrument_type: str, raw: int) -> int:
    raw = int(raw)
    if is_tfs(instrument_type):
        if raw not in L.FM_FROM_FEI_SUBMODE:
            raise ColumnRefused(f"unknown TFS projection submode {raw}")
        return L.FM_FROM_FEI_SUBMODE[raw]
    if not 0 <= raw <= 4:
        raise ColumnRefused(f"unknown JEOL function mode {raw} (0..4)")
    return raw


def function_mode_from_name(name: str) -> int:
    key = str(name or "").strip().upper()
    for i, n in enumerate(L.FUNCTION_MODE_NAMES):
        if n.upper() == key:
            return i
    aliases = {
        "LOWMAG": L.FM_LOWMAG,
        "LM": L.FM_LOWMAG,
        "SMAG": L.FM_SAMAG,
        "SA": L.FM_SAMAG,
        "M": L.FM_MAG1,
        "MH": L.FM_MAG2,
        "D": L.FM_DIFF,
        "LAD": L.FM_DIFF,
        "DIFFRACTION": L.FM_DIFF,
        "SADIFF": L.FM_DIFF,
    }
    return aliases.get(key, L.FM_MAG1)
