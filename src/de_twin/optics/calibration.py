"""Magnification / camera-length calibration (DE-Server ``mag.yaml`` semantics).

DE-Server's ``Camera::ParseEmMetaData`` looks the reported magnification up in
the project (``Mode``/``SubMode``/``TemStemMode``) of ``configurations/mag.yaml``:

* TEM ``Mags``: nm per *unbinned* detector pixel (the table is camera specific);
  the processed-frame value is ``table * hw_binning * sw_binning * TEM_Scale_Factor(HT)``.
* TEM ``CamLength(cm)``: 1/nm per unbinned pixel (x binning x TEM_Diff_Scale_Factor).
* STEM ``Mags``: nm per volt of scan; the step is ``table * (Vmax - Vmin) / max(nx, ny)
  * STEM_Virt_Scale_Factor``.
* STEM ``CamLength(cm)``: mrad per unbinned pixel (x STEM_Scale_Factor).

The twin always works in *unbinned* detector pixels (the detector bins), so the
binning factor never appears here. :meth:`Calibration.default` builds
JEOL-like ladders computed geometrically from the camera's pixel pitch, so the
twin works without any file.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ..column.ladders import (DEFAULT_CAMERA_LENGTHS_MM, DEFAULT_LOWMAG_MAGS, DEFAULT_SAMAG_MAGS, DEFAULT_STEM_MAGS,
                              MAG1_MAGS, Ladder)
from ..state import MicroscopeState
from .physics import electron_wavelength_nm

def _alphanumeric_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# mag_mode string (column) -> normalised project-name candidates
_MODE_ALIASES = {
    "mag1": ("mag1", "mag", "m", "sa"),
    "mag": ("mag1", "mag", "m", "sa"),
    "mag2": ("mag2", "mag1", "mag"),
    "lowmag": ("lowmag", "lm", "low"),
    "lm": ("lm", "lowmag", "low"),
    "samag": ("samag", "sa", "mag1"),
    "sa": ("sa", "samag", "mag1"),
    "smag": ("smag", "mag1"),
    "diff": ("diff", "diffmag", "d"),
    "diffmag": ("diffmag", "diff", "d"),
}


@dataclass
class MagProject:
    name: str
    mode: int
    sub_mode: int
    tem_stem: int  # 0 TEM, 1 STEM
    mags: dict = field(default_factory=dict)  # magnification -> value (see module doc)
    cam_lengths_cm: dict = field(default_factory=dict)  # CL [cm] -> value, or None = geometric

    @property
    def is_diffraction(self) -> bool:
        return bool(self.cam_lengths_cm) and not self.mags


_MISSING = object()


def _lookup(table: dict, key: float, rel_tol: float):
    """Exact key, else nearest key within ``rel_tol`` (DE-Server LookupNearest).

    Returns ``_MISSING`` when nothing matches (a stored ``None`` means "geometric").
    """
    if not table or key is None or key <= 0:
        return _MISSING
    if key in table:
        return table[key]
    best = min(table, key=lambda k: abs(math.log(k / key)) if k > 0 else math.inf)
    if best > 0 and abs(best - key) <= rel_tol * max(abs(key), 1e-12):
        return table[best]
    return _MISSING


class Calibration:
    """Magnification -> nm/px and camera length -> 1/nm/px, per mag mode."""

    def __init__(self, projects: list[MagProject], *, reference_pixel_um: Optional[float] = None,
                 ht_scale: Optional[dict] = None, tolerance: float = 0.005,
                 stem_scan_span_v: float = 1.0, stem_fov_nm_at_1x: float = 1.28e8,
                 name: str = "custom"):
        self.projects = list(projects)
        # Pixel pitch (um) the tables were measured with; None = use values as-is.
        self.reference_pixel_um = reference_pixel_um
        # {ht_kv: {"tem":, "tem_diff":, "stem":, "stem_virt":}}
        self.ht_scale = ht_scale or {}
        self.tolerance = tolerance
        self.stem_scan_span_v = stem_scan_span_v
        self.stem_fov_nm_at_1x = stem_fov_nm_at_1x  # geometric STEM: FOV = this / mag
        self.name = name

    # ------------------------------------------------------------------ io
    @classmethod
    def from_mag_yaml(cls, path, *, reference_pixel_um: Optional[float] = None,
                      ht_table=None, tolerance: float = 0.005) -> "Calibration":
        """Load a DE-Server ``mag.yaml`` (tabs tolerated) and optionally an ``ht_table_*.yaml``."""
        text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\t", "    ")
        data = yaml.safe_load(text) or []
        projects = []
        for p in data:
            if not isinstance(p, dict) or "Project" not in p:
                continue
            mags = {float(k): float(v) for k, v in (p.get("Mags") or {}).items()}
            cl = {float(k): float(v) for k, v in (p.get("CamLength(cm)") or {}).items()}
            if p.get("CamLengthInMM"):
                cl = {k / 10.0: v for k, v in cl.items()}
            projects.append(MagProject(str(p["Project"]), int(p.get("Mode", -1)),
                                       int(p.get("SubMode", -1)), int(p.get("TemStemMode", 0)),
                                       mags, cl))
        ht_scale = cls._load_ht_table(ht_table) if ht_table else None
        return cls(projects, reference_pixel_um=reference_pixel_um, ht_scale=ht_scale,
                   tolerance=tolerance, name=Path(path).name)

    @staticmethod
    def _load_ht_table(path) -> dict:
        text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\t", "    ")
        data = yaml.safe_load(text) or {}
        table = data.get("HT", data)
        out = {}
        for volts, row in (table or {}).items():
            if not isinstance(row, dict):
                continue
            out[round(float(volts) / 1000.0, 3)] = {
                "tem": float(row.get("TEM_Scale_Factor", 1.0)),
                "tem_diff": float(row.get("TEM_Diff_Scale_Factor", 1.0)),
                "stem": float(row.get("STEM_Scale_Factor", 1.0)),
                "stem_virt": float(row.get("STEM_Virt_Scale_Factor", 1.0)),
            }
        return out

    @classmethod
    def default(cls, post_magnification: float = 1.0) -> "Calibration":
        """Built-in JEOL-like ladders, computed geometrically.

        Tables are nm per 1-um pixel (``1000 / (mag * post_magnification)``) and
        are scaled by the camera's pixel pitch, so 20 kx on a 6.5 um pixel is
        0.325 nm/px. Camera lengths are geometric (``None`` entries).
        """
        def mags(ladder):
            return {float(m): 1000.0 / (m * post_magnification) for m in ladder}

        cl = {m / 10.0: None for m in DEFAULT_CAMERA_LENGTHS_MM}
        projects = [
            MagProject("MAG1", 1, 0, 0, mags(MAG1_MAGS)),
            MagProject("MAG2", 1, 1, 0, mags(MAG1_MAGS)),
            MagProject("LowMAG", 1, 2, 0, mags(DEFAULT_LOWMAG_MAGS)),
            MagProject("SAMAG", 1, 3, 0, mags(DEFAULT_SAMAG_MAGS)),
            MagProject("DIFF", 2, 4, 0, {}, dict(cl)),
            MagProject("STEM", 1, 0, 1, {float(m): None for m in DEFAULT_STEM_MAGS}, dict(cl)),
        ]
        return cls(projects, reference_pixel_um=1.0, name="default")

    # ------------------------------------------------------------ projects
    def _candidates(self, tem_stem: int, want_cl: bool):
        out = []
        for p in self.projects:
            if p.tem_stem != tem_stem:
                continue
            if want_cl and p.cam_lengths_cm:
                out.append(p)
            elif not want_cl and p.mags:
                out.append(p)
        return out

    def project_for(self, state: MicroscopeState, want_cl: bool = False) -> Optional[MagProject]:
        tem_stem = int(state.tem_stem)
        cands = self._candidates(tem_stem, want_cl)
        if not cands:
            return None
        mode = _alphanumeric_key(state.mag_mode)
        aliases = _MODE_ALIASES.get(mode, (mode,))
        mag = float(state.magnification)
        for alias in aliases:
            for p in cands:
                if _alphanumeric_key(p.name) == alias:
                    if want_cl or _lookup(p.mags, mag, self.tolerance) is not _MISSING:
                        return p
        # Diffraction: the TEM Diff project regardless of the imaging mode name
        if want_cl:
            if tem_stem == 0:
                diff = [p for p in cands if p.is_diffraction] or cands
                return diff[0]
            return cands[0]
        # Imaging: the first project whose ladder holds this magnification
        for p in cands:
            if _lookup(p.mags, float(state.magnification), self.tolerance) is not _MISSING:
                return p
        return cands[0]

    def _ht(self, state: MicroscopeState, key: str) -> float:
        if not self.ht_scale:
            return 1.0
        row = self.ht_scale.get(round(float(state.ht_kv), 3))
        return float(row.get(key, 1.0)) if row else 1.0

    def _pixel_scale(self, camera) -> float:
        if self.reference_pixel_um and camera is not None and getattr(camera, "pixel_um", 0) > 0:
            return float(camera.pixel_um) / float(self.reference_pixel_um)
        return 1.0

    # ------------------------------------------------------------- lookups
    def specimen_pixel_nm(self, state: MicroscopeState, camera) -> float:
        """nm of specimen per *unbinned* detector pixel (TEM imaging); 0.0 = not calibrated."""
        p = self.project_for(state, want_cl=False) if int(state.tem_stem) == 0 else None
        if p is None:
            return 0.0
        v = _lookup(p.mags, float(state.magnification), self.tolerance)
        if v is _MISSING or v is None:
            return 0.0
        return float(v) * self._pixel_scale(camera) * self._ht(state, "tem")

    def recip_pixel_inv_nm(self, state: MicroscopeState, camera) -> float:
        """1/nm per unbinned detector pixel in diffraction; 0.0 = not calibrated.

        Geometric entries use ``mrad/px = pixel_um / CL_mm`` and ``1/nm = mrad / (1000 lambda)``.
        """
        cl_mm = float(state.camera_length_mm)
        lam = electron_wavelength_nm(state.ht_kv)
        p = self.project_for(state, want_cl=True)
        if p is None or cl_mm <= 0:
            return 0.0
        v = _lookup(p.cam_lengths_cm, cl_mm / 10.0, self.tolerance)
        if v is _MISSING or v is None:
            return self.geometric_recip_inv_nm(state, camera)
        if p.tem_stem == 1:  # STEM table is mrad/px
            return float(v) * self._pixel_scale(camera) * self._ht(state, "stem") / (1000.0 * lam)
        return float(v) * self._pixel_scale(camera) * self._ht(state, "tem_diff")

    @staticmethod
    def geometric_recip_inv_nm(state: MicroscopeState, camera, camera_length_mm: float = 0.0) -> float:
        cl = camera_length_mm if camera_length_mm > 0 else float(state.camera_length_mm)
        pix = float(getattr(camera, "pixel_um", 0.0) or 0.0)
        if cl <= 0 or pix <= 0:
            return 0.0
        return (pix / cl) / (1000.0 * electron_wavelength_nm(state.ht_kv))

    def stem_step_nm(self, state: MicroscopeState, scan_size: tuple[int, int]) -> float:
        """nm per scan point for a (nx, ny) scan; 0.0 = not calibrated."""
        n = max(1, max(int(scan_size[0]), int(scan_size[1])))
        cands = self._candidates(1, want_cl=False)
        if not cands:
            return 0.0
        mode = _alphanumeric_key(state.mag_mode)
        mag = float(state.magnification)
        proj = next((p for p in cands if _alphanumeric_key(p.name) in _MODE_ALIASES.get(mode, (mode,))), None)
        proj = proj or next((p for p in cands
                             if _lookup(p.mags, mag, self.tolerance) is not _MISSING), None)
        if proj is None:
            return 0.0
        v = _lookup(proj.mags, mag, self.tolerance)
        if v is _MISSING:
            return 0.0
        if v is None:  # geometric ladder
            return self.stem_fov_nm_at_1x / mag / n
        return float(v) * self.stem_scan_span_v / n * self._ht(state, "stem_virt")

    # -------------------------------------------------------------- ladders
    def _mode_project(self, mode: Optional[str], tem_stem: int = 0) -> Optional[MagProject]:
        cands = self._candidates(tem_stem, want_cl=False)
        if not cands:
            return None
        if mode is None:
            return cands[0]
        m = _alphanumeric_key(mode)
        for alias in _MODE_ALIASES.get(m, (m,)):
            for p in cands:
                if _alphanumeric_key(p.name) == alias:
                    return p
        return None

    def mag_modes(self, tem_stem: int = 0) -> list[str]:
        return [p.name for p in self._candidates(tem_stem, want_cl=False)]

    def mag_ladder(self, mode: Optional[str] = "MAG1", tem_stem: int = 0) -> Ladder:
        p = self._mode_project(mode, tem_stem)
        return Ladder(sorted(p.mags)) if p else Ladder()

    def cl_ladder(self, tem_stem: int = 0) -> Ladder:
        """Camera lengths in **mm** (the twin's unit; mag.yaml stores cm)."""
        cands = self._candidates(tem_stem, want_cl=True)
        diff = [p for p in cands if p.is_diffraction] or cands
        if not diff:
            return Ladder()
        return Ladder(sorted(round(k * 10.0, 6) for k in diff[0].cam_lengths_cm))

    def snap_mag(self, mode: Optional[str], mag: float, tem_stem: int = 0) -> float:
        return self.mag_ladder(mode, tem_stem).snap_logarithmic(mag)

    def snap_cl(self, cl_mm: float, tem_stem: int = 0) -> float:
        return self.cl_ladder(tem_stem).snap_logarithmic(cl_mm)

    def __repr__(self) -> str:
        return f"Calibration({self.name!r}, projects={[p.name for p in self.projects]})"
