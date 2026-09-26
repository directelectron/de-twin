"""SerialEM calibrations to and from a column definition.

A twin's column (`OpticsConfig.realism`) can be written out as the calibrations SerialEM would
measure on it, and a column can be defined from a real microscope's SerialEM calibrations, so
the twin behaves like that microscope. Fitting or refining a definition from acquired data is
not done here (that is Ground Crew's / de_autopilot's job); this module only converts.

Files (formats from SerialEM's ``ParameterIO.cpp``):

``SerialEMcalibrations.txt``
    ``ImageShiftMatrix`` (count, then ``magInd camera xpx xpy ypx ypy mag``);
    ``StageToCameraMatrix magInd camera xpx xpy ypx ypy focus mag``; ``ImageShiftOffsets``
    (count, then ``magInd gif isx isy``): the image shift that re-centres each magnification;
    ``BeamShiftCalibration magInd xpx xpy ypx ypy alpha probe retain mag``: beam shift per
    image shift (read only: see below); ``CrossoverIntensity spot micro [nano]``;
    ``HighFocusMagCal spot probe defocus intensity scale rotation crossover aperture magInd``;
    ``FocusCalibration magInd camera slopeX slopeY beamTilt nPoints direction probe alpha
    mag`` then ``defocus dx dy`` lines, per unit of beam tilt.
``SerialEMproperties.txt``
    in a camera's ``CameraProperties n`` ... ``EndCameraProperties`` block:
    ``RotationAndPixel magInd deltaRotation rotation pixel_nm`` (999 = undefined).

Conventions (SerialEM's help, "Image Rotation", and ``ShiftManager.cpp``). SerialEM's camera
coordinates are right-handed: x right, y UP (the twin's raster has y down). Its specimen
coordinates are minus its stage coordinates, and ``SpecimenToCamera = R(rotation) / pixel``;
so ``StageToCamera = -R(rotation) / pixel`` (image rotation = ``atan2(-ypx, -xpx)``) and
``IStoCamera = SpecimenToCamera @ (specimen shift per image-shift unit)``. The twin's stage y
runs opposite to SerialEM's; with that, SerialEM's image rotation is the twin view's
``rotation_rad`` and every matrix has a positive determinant, as on a real scope. A matrix
``[[xpx, xpy], [ypx, ypy]]`` maps (x, y) to (xpx x + xpy y, ypx x + ypy y).

In the twin, image shift never moves the beam off the imaged area (the illumination follows
it, as on a scope that couples them), so the IS-to-BS calibration SerialEM would measure is
zero and none is written; a real file's is read and gives the beam-shift matrix.

Magnification indices: SerialEM numbers the scope's magnifications from 1. The twin's own are
its imaging ladder (LowMAG then MAG1) numbered from 1 (`twin_mag_table`); a real scope's table
can be passed as ``{index: magnification}`` with its low-mag ceiling as ``lm_max_mag``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from .realism import ColumnRealism

#: The beam tilt (mrad) and defocus points of the focus calibration written out.
FOCUS_CAL_TILT_MRAD = 3.0
FOCUS_CAL_DEFOCUS_UM = (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
#: The defocus values of the high-focus magnification calibration written out.
HIGH_FOCUS_DEFOCUS_UM = (-50.0, -100.0, -200.0, -300.0)
#: SerialEM's "undefined" for rotations and (old files) pixel sizes.
UNDEFINED = 999.0

_T_CAM = np.array([[1.0, 0.0], [0.0, -1.0]])  # twin raster (y down) -> SerialEM camera (y up)
_T_STAGE = np.array([[1.0, 0.0], [0.0, -1.0]])  # SerialEM specimen <-> twin world


def twin_mag_table() -> dict[int, float]:
    """The twin column's imaging magnifications, numbered from 1 as SerialEM numbers a scope's."""
    from ..column import ladders as L

    return {i + 1: float(m) for i, m in enumerate(L.LOWMAG_MAGS + L.MAG1_MAGS)}


def _twin_lm_max() -> float:
    from ..column import ladders as L

    return float(L.LOWMAG_MAGS[-1])


def _mode_for(mag: float, lm_max: Optional[float] = None) -> str:
    return "LowMAG" if mag <= (lm_max if lm_max is not None else _twin_lm_max()) else "MAG1"


def _is_lm(mode) -> bool:
    m = "".join(ch for ch in str(mode).lower() if ch.isalnum())
    return m.startswith("low") or m == "lm"


def _rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _probe(twin_probe: int) -> int:
    """SerialEM's probe index (0 nanoprobe, 1 microprobe) of a twin ProbeMode."""
    return 0 if int(twin_probe) == 0 else 1


# ------------------------------------------------------------------ the column definition
class ColumnDefinition:
    """A column defined by tables (from calibrations) rather than drawn from a seed: the same
    interface as :class:`~de_twin.optics.realism.ColumnRealism`, so it goes in
    ``OpticsConfig(realism=...)``.

    Per magnification: the image rotation (rad), the true pixel size (nm, for the camera it was
    calibrated on) against the twin's nominal one, the image-shift matrix (specimen um per
    unit) and the magnification's image offset (um). One beam-shift matrix, crossovers per
    (spot, SerialEM probe), and the high-defocus scale / rotation per micrometre. What the
    tables do not give is ideal. A magnification between entries takes the nearest (in log
    magnification) of its own range (LM or not), where it has one. Compared and hashed by
    identity (it holds mutable tables).
    """

    def __init__(self, lm_max_mag: Optional[float] = None):
        self.lm_max_mag = lm_max_mag if lm_max_mag is not None else _twin_lm_max()
        self.rotation: dict = {}  # mag -> rad
        self.pixel_nm: dict = {}  # mag -> true pixel (nm)
        self.nominal_pixel_nm: dict = {}  # mag -> the twin's nominal (nm)
        self.is_matrices: dict = {}  # mag -> 2x2
        self.mag_offsets: dict = {}  # mag -> (x, y) um
        self.bs: Optional[np.ndarray] = None
        self.crossovers: dict = {}  # (spot, SerialEM probe) -> intensity
        self.crossover_intensity = ColumnRealism.crossover_intensity
        self.backlash_um = 0.0
        self.is_coma_mrad_per_um = 0.0
        self.is_astig_nm_per_um = 0.0
        self.hd_scale_per_um = 0.0
        self.hd_rotation_deg_per_um = 0.0

    def _near(self, table: dict, mode, mag: float):
        if not table:
            return None
        lm = _is_lm(mode) if mode else float(mag) <= self.lm_max_mag
        same = [m for m in table if (m <= self.lm_max_mag) == lm] or list(table)
        k = min(same, key=lambda m: abs(math.log(max(m, 1e-9)) - math.log(max(float(mag), 1e-9))))
        return table[k]

    def rotation_rad(self, mag_mode, mag: float) -> float:
        v = self._near(self.rotation, mag_mode, mag)
        return 0.0 if v is None else float(v)

    def pixel_scale(self, mag_mode, mag: float) -> float:
        p = self._near(self.pixel_nm, mag_mode, mag)
        n = self._near(self.nominal_pixel_nm, mag_mode, mag)
        return 1.0 if not p or not n else float(p) / float(n)

    def is_matrix(self, mag_mode, mag: float) -> np.ndarray:
        v = self._near(self.is_matrices, mag_mode, mag)
        return np.eye(2) if v is None else np.asarray(v, float)

    def mag_offset_um(self, mag_mode, mag: float) -> tuple[float, float]:
        v = self._near(self.mag_offsets, mag_mode, mag)
        return (0.0, 0.0) if v is None else (float(v[0]), float(v[1]))

    def bs_matrix(self) -> np.ndarray:
        return np.eye(2) if self.bs is None else np.asarray(self.bs, float)

    def crossover(self, spot: int, probe_mode: int = 0) -> float:
        key = (int(spot), _probe(probe_mode))
        if key in self.crossovers:
            return float(self.crossovers[key])
        same = [v for (s, p), v in self.crossovers.items() if p == key[1]]
        if same or self.crossovers:
            return float(np.mean(same or list(self.crossovers.values())))
        return float(self.crossover_intensity)

    def is_tilt_mrad(self, isx_um: float, isy_um: float) -> tuple[float, float]:
        k = self.is_coma_mrad_per_um
        return k * isx_um, k * isy_um

    def is_astig_nm(self, isx_um: float, isy_um: float) -> complex:
        return self.is_astig_nm_per_um * complex(isx_um, isy_um)

    def defocus_scale(self, defocus_um: float) -> float:
        return 1.0 + self.hd_scale_per_um * abs(float(defocus_um))

    def defocus_rotation_rad(self, defocus_um: float) -> float:
        return math.radians(self.hd_rotation_deg_per_um * float(defocus_um))

    def truth(self, mag_mode, mag: float) -> dict:
        return {
            "image_rotation_deg": math.degrees(self.rotation_rad(mag_mode, mag)),
            "pixel_scale": self.pixel_scale(mag_mode, mag),
            "is_matrix_um_per_unit": self.is_matrix(mag_mode, mag).tolist(),
            "mag_offset_um": self.mag_offset_um(mag_mode, mag),
            "backlash_um": self.backlash_um,
            "bs_matrix_um_per_unit": self.bs_matrix().tolist(),
        }


# ------------------------------------------------------------------ twin -> SerialEM
def _nominal_pixel_nm(mag: float, camera, calibration=None, lm_max: Optional[float] = None) -> float:
    from ..state import MicroscopeState
    from .calibration import Calibration

    cal = calibration or Calibration.default()
    s = MicroscopeState(magnification=float(mag), mag_mode=_mode_for(mag, lm_max))
    return float(cal.specimen_pixel_nm(s, camera))


def _flip_matrix(flips) -> np.ndarray:
    fx, fy = flips
    return np.diag([-1.0 if fx else 1.0, -1.0 if fy else 1.0])


def _specimen_to_camera(theta: float, p_um: float, flips=(False, False)) -> np.ndarray:
    """SerialEM's SpecimenToCamera (camera px per specimen um) of a twin view rotated by
    *theta*: the twin maps world w to raster F R(-theta) w / p; SerialEM's specimen is T w and
    its camera T_cam raster."""
    return _T_CAM @ _flip_matrix(flips) @ _rot(-theta) @ _T_STAGE / p_um


def _camera_mats(realism, mag: float, nominal_nm: float, flips=(False, False), lm_max=None):
    """(StageToCamera, IStoCamera, rotation rad, true pixel nm) of one magnification."""
    mode = _mode_for(mag, lm_max)
    theta = realism.rotation_rad(mode, mag)
    p_um = nominal_nm * realism.pixel_scale(mode, mag) / 1000.0
    s2c = _specimen_to_camera(theta, p_um, flips)
    stage = -s2c  # specimen = -stage
    is_ = s2c @ _T_STAGE @ realism.is_matrix(mode, mag)  # IS moves the centred specimen point by T M u
    return stage, is_, theta, p_um * 1000.0


def _serialem_rotation_deg(stage: np.ndarray) -> float:
    """SerialEM's image rotation from a stage matrix: the mean of its X- and Y-axis estimates."""
    ax = math.degrees(math.atan2(-stage[1, 0], -stage[0, 0]))
    ay = math.degrees(math.atan2(-stage[1, 1], -stage[0, 1])) - 90.0
    d = (ay - ax + 180.0) % 360.0 - 180.0
    return (ax + 0.5 * d + 180.0) % 360.0 - 180.0


def to_serialem(realism, camera, calibrations_path, properties_path=None, *,
                mag_table: Optional[dict] = None, lm_max_mag: Optional[float] = None,
                calibration=None, spots=(1, 2, 3, 4, 5), reference_mag: float = 20000.0,
                camera_index: int = 0, flips=(False, False)) -> None:
    """Write the calibrations SerialEM would measure on a column (and, with
    *properties_path*, the camera's ``RotationAndPixel`` block). *flips* are the twin's
    ``OpticsConfig.flip_x / flip_y``."""
    from ..detector import camera as camera_by_name
    from ..column.column import BEAM_TILT_MRAD_PER_UNIT
    from ..state import ProbeMode

    cam = camera_by_name(camera) if isinstance(camera, str) else camera
    mags = mag_table or twin_mag_table()
    lines = ["SerialEMCalibrations"]
    rows, stage_rows, offs, rot_rows = [], [], [], []
    for ind, mag in sorted(mags.items()):
        nominal = _nominal_pixel_nm(mag, cam, calibration, lm_max_mag)
        stage, is_, theta, p_nm = _camera_mats(realism, mag, nominal, flips, lm_max_mag)
        rows.append(f"{ind} {camera_index} {is_[0, 0]:.12g} {is_[0, 1]:.12g} {is_[1, 0]:.12g} {is_[1, 1]:.12g}   {int(round(mag))}")
        stage_rows.append(f"StageToCameraMatrix {ind} {camera_index} {stage[0, 0]:.12g} {stage[0, 1]:.12g} "
                          f"{stage[1, 0]:.12g} {stage[1, 1]:.12g}   0.000000   {int(round(mag))}")
        mode = _mode_for(mag, lm_max_mag)
        o = np.asarray(realism.mag_offset_um(mode, mag))
        if np.any(o):
            isx, isy = -np.linalg.solve(realism.is_matrix(mode, mag), o)
            offs.append(f"{ind} 0 {isx:.12g} {isy:.12g}")
        rot_rows.append(f"RotationAndPixel {ind} {UNDEFINED:.0f} {_serialem_rotation_deg(stage):.12g} {p_nm:.12g}")
    lines.append(f"ImageShiftMatrix {len(rows)}")
    lines += rows
    lines += stage_rows
    if offs:
        lines.append(f"ImageShiftOffsets {len(offs)}")
        lines += offs
    micro, nano = int(ProbeMode.TEM), int(ProbeMode.NANOPROBE)
    for spot in spots:
        lines.append(f"CrossoverIntensity {spot} {realism.crossover(spot, micro):.12g} {realism.crossover(spot, nano):.12g}")
    # high-focus magnification: image scale and rotation relative to focus (a mag cal: index 0)
    for df in HIGH_FOCUS_DEFOCUS_UM:
        scale = 1.0 / realism.defocus_scale(df)
        rot = -math.degrees(realism.defocus_rotation_rad(df))
        lines.append(f"HighFocusMagCal 3 1 {df:.6f} 0.500000 {scale:.12g} {rot:.12g} "
                     f"{realism.crossover(3, micro):.12g} 0 0")
    # autofocus: image displacement (SerialEM camera px) between +/- beam tilt, per unit tilt
    ref = min(mags, key=lambda i: abs(mags[i] - reference_mag))
    mag = mags[ref]
    stage, is_, theta, p_nm = _camera_mats(realism, mag, _nominal_pixel_nm(mag, cam, calibration, lm_max_mag),
                                           flips, lm_max_mag)
    bt_units = FOCUS_CAL_TILT_MRAD / BEAM_TILT_MRAD_PER_UNIT
    # a tilt along world x displaces the image by 2 df tau along it: in SerialEM camera px
    d = _T_CAM @ _flip_matrix(flips) @ _rot(-theta) @ np.array([1.0, 0.0])
    per_um = 2.0 * 1000.0 * FOCUS_CAL_TILT_MRAD * 1e-3 / p_nm * d / bt_units
    pts = [(df, *(df * per_um)) for df in FOCUS_CAL_DEFOCUS_UM]
    lines.append(f"FocusCalibration {ref} {camera_index} {per_um[0]:.12g} {per_um[1]:.12g} "
                 f"{bt_units:.2f} {len(pts)} 0 1 -999   {int(round(mag))}")
    lines += [f"{a:.6f} {b:.12g} {c:.12g}" for a, b, c in pts]
    Path(calibrations_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if properties_path is not None:
        Path(properties_path).write_text(
            f"CameraProperties {camera_index}\n" + "\n".join(rot_rows) + "\nEndCameraProperties\n",
            encoding="utf-8")


# ------------------------------------------------------------------ SerialEM -> column
def _floats(parts):
    return [float(v) for v in parts]


def _rotation_and_pixel(path, camera_index: int) -> dict:
    """``RotationAndPixel`` of one camera: {magInd: (rotation deg or None, pixel nm or None)},
    from its ``CameraProperties`` block (or the whole file if it has no blocks)."""
    out, block, blocks = {}, None, False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "CameraProperties":
            blocks = True
            block = int(p[1]) if len(p) > 1 else None
            continue
        if p[0] == "EndCameraProperties":
            block = None
            continue
        if p[0] != "RotationAndPixel" or len(p) < 5:
            continue
        if blocks and block != camera_index:
            continue
        rot, pix = float(p[3]), float(p[4])
        out[int(p[1])] = (None if abs(rot) > 900 else rot,
                          None if pix <= 0 or 998.9 < abs(pix) < 999.1 else pix)
    return out


def from_serialem(calibrations_path, properties_path=None, *, camera, mag_table: Optional[dict] = None,
                  lm_max_mag: Optional[float] = None, calibration=None, camera_index: int = 0,
                  flips=(False, False)) -> ColumnDefinition:
    """A column definition from a microscope's SerialEM calibrations (and, if given, the
    camera's ``RotationAndPixel`` lines). Stage calibrations give each magnification's
    rotation and true pixel size; ``RotationAndPixel`` values override them where defined."""
    from ..detector import camera as camera_by_name

    cam = camera_by_name(camera) if isinstance(camera, str) else camera
    mags = mag_table or twin_mag_table()
    is_cam, stage_cam, offsets, bs_rows, cross, hf = {}, {}, {}, [], {}, []
    lines = Path(calibrations_path).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts:
            continue
        key = parts[0]
        try:
            if key == "ImageShiftMatrix":
                for _ in range(int(parts[1])):
                    p = lines[i].split()
                    i += 1
                    if int(p[1]) == camera_index:
                        is_cam[int(p[0])] = np.array(_floats(p[2:6])).reshape(2, 2)
            elif key == "StageToCameraMatrix" and int(parts[2]) == camera_index:
                stage_cam[int(parts[1])] = np.array(_floats(parts[3:7])).reshape(2, 2)
            elif key == "ImageShiftOffsets":
                for _ in range(int(parts[1])):
                    p = lines[i].split()
                    i += 1
                    if int(p[1]) == 0:
                        offsets[int(p[0])] = np.array(_floats(p[2:4]))
            elif key == "BeamShiftCalibration" and len(parts) >= 6:
                bs_rows.append((int(parts[1]), np.array(_floats(parts[2:6])).reshape(2, 2)))
            elif key == "CrossoverIntensity" and len(parts) >= 3:
                spot = int(parts[1])
                vals = _floats(parts[2:4])
                if vals[0]:
                    cross[(spot, 1)] = vals[0]
                if len(vals) > 1 and vals[1]:
                    cross[(spot, 0)] = vals[1]
            elif key == "HighFocusMagCal":
                hf.append(_floats(parts[3:7]))  # defocus, intensity, scale, rotation
            elif key == "FocusCalibration":
                i += int(parts[6])
            elif key == "STEMfocusVersusZ":
                i += int(parts[1])
        except (ValueError, IndexError):
            continue  # a form this reader does not know: SerialEM's older lines, say
    rot_pix = _rotation_and_pixel(properties_path, camera_index) if properties_path is not None else {}

    d = ColumnDefinition(lm_max_mag)
    for ind, mag in mags.items():
        d.nominal_pixel_nm[mag] = _nominal_pixel_nm(mag, cam, calibration, lm_max_mag)
        if ind in stage_cam:
            s2c = -stage_cam[ind]
            p_um = 1.0 / math.sqrt(abs(np.linalg.det(s2c)))
            # s2c p = T_cam F R(-theta) T_stage  ->  R(-theta)
            r = np.linalg.inv(_T_CAM @ _flip_matrix(flips)) @ (s2c * p_um) @ np.linalg.inv(_T_STAGE)
            d.rotation[mag] = -math.atan2(r[1, 0], r[0, 0])
            d.pixel_nm[mag] = p_um * 1000.0
        if ind in rot_pix:
            rot_deg, p_nm = rot_pix[ind]
            if rot_deg is not None and not flips[0] and not flips[1]:
                d.rotation[mag] = math.radians(rot_deg)
            if p_nm is not None:
                d.pixel_nm[mag] = p_nm
        if ind in is_cam and mag in d.rotation and mag in d.pixel_nm:
            s2c = _specimen_to_camera(d.rotation[mag], d.pixel_nm[mag] / 1000.0, flips)
            d.is_matrices[mag] = np.linalg.inv(_T_STAGE) @ np.linalg.solve(s2c, is_cam[ind])
        if ind in offsets and mag in d.is_matrices:
            d.mag_offsets[mag] = tuple(-(d.is_matrices[mag] @ offsets[ind]))
    # beam shift per unit: B = M (IS->BS)^-1, averaged over magnifications that have both
    est = []
    for ind, m in bs_rows:
        if ind in mags and mags[ind] in d.is_matrices and abs(np.linalg.det(m)) > 1e-12:
            est.append(d.is_matrices[mags[ind]] @ np.linalg.inv(m))
    if est:
        d.bs = np.mean(est, axis=0)
    d.crossovers = cross
    if hf:
        hf = np.array(hf)
        dfs, scales, rots = hf[:, 0], hf[:, 2], hf[:, 3]
        # scale = 1 / (1 + k |df|), rotation = -r df
        k = np.linalg.lstsq(np.abs(dfs)[:, None], 1.0 / scales - 1.0, rcond=None)[0][0]
        r = np.linalg.lstsq(dfs[:, None], -rots, rcond=None)[0][0]
        d.hd_scale_per_um = float(k)
        d.hd_rotation_deg_per_um = float(r)
    return d
