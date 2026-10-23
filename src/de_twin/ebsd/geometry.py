"""EBSD geometry: sample tilt, detector tilt, pattern centre and how they move with the beam.

Frames
------
All frames are right-handed.

* **Sample frame** (the frame every EBSD orientation is reported in): the de-twin specimen
  frame, ``x_s`` = scan x (right in the scan / orientation map), ``y_s`` = scan y (down in
  the map), ``z_s`` = ``x_s x y_s`` pointing *into* the sample. The surface's outward normal
  is ``-z_s``. In the SEM image the EBSD detector is at the *top* (towards ``-y_s``), and the
  tilt axis is ``x_s``.
* **Detector frame**: ``x_d`` along the pattern's columns (right), ``y_d`` along its rows
  (down), ``z_d`` from the source point towards the screen, normal to it. The pattern is the
  screen *as seen from the sample*: its top is towards the pole piece and the bright
  forward-scatter region is at the bottom.

The sample is tilted by ``sample_tilt_deg`` (sigma, default 70) about ``x_s`` towards the
detector; the screen's normal is tilted down towards the sample by ``detector_tilt_deg``
(theta). With ``D = sigma - theta`` a sample-frame vector ``v`` is ``S2D @ v`` in the
detector frame::

    S2D = [[1,       0,       0],
           [0, -sin D,  cos D],
           [0, -cos D, -sin D]]

Orientations follow the rest of de-twin (orix): a grain's rotation ``R`` (unit quaternion,
``a >= 0``) has crystal->sample matrix ``G = R.to_matrix().T``, so a crystal direction ``c``
appears in the sample frame as ``G @ c`` and on the detector along ``S2D @ G @ c``.

Pattern centre (Bruker convention)
----------------------------------
``(pcx, pcy, pcz)``: the point on the screen nearest the source, ``pcx`` in units of the
sensor *width* from its left edge, ``pcy`` in units of the sensor *height* from its top
edge, and the source-screen distance ``pcz`` in units of the sensor height. They refer to
the whole (unbinned) sensor, so binning and a hardware ROI leave them unchanged.
:func:`convert_pc` gives the EDAX/TSL, Oxford and EMsoft forms.

The pattern centre given in :class:`EbsdGeometry` is the one with the beam on the optic axis
(the scan centre) at ``calibration_wd_mm``. Moving the beam by ``(dx, dy)`` on the sample
moves the source point by ``S2D @ (dx, dy, 0)`` in the detector frame, so::

    pcx += dx / (W p)
    pcy -= sin(D) dy / (H p)
    pcz += cos(D) dy / (H p)

(W, H sensor pixels, p the pixel pitch), and a working distance ``dWD`` larger than the
calibration moves the source down by ``dWD``: ``pcy += cos(theta) dWD / (H p)``,
``pcz -= sin(theta) dWD / (H p)``. That is how a real system's pattern centre follows the
beam across a scan.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

PC_CONVENTION = "bruker"


def _opt(default, lo=None, hi=None, doc=""):
    return dataclasses.field(default=default, metadata={"lo": lo, "hi": hi, "doc": doc})


@dataclass(frozen=True)
class EbsdGeometry:
    """An SEM with an EBSD detector: everything about the experiment that is not the column's
    beam energy (``MicroscopeState.ht_kv``) or the scan (``AcquisitionRequest.scan``).

    Set it as ``OpticsConfig.ebsd`` (``DigitalTwin.configure_ebsd`` does) to put the twin in
    EBSD mode. Frozen and made of plain numbers, so a GUI can persist it with
    :meth:`to_dict` / :meth:`from_dict` and change it with :meth:`replace`.
    """

    enabled: bool = _opt(True, doc="EBSD mode on (every acquisition renders EBSD patterns).")
    sample_tilt_deg: float = _opt(70.0, 0.0, 89.0, "Sample tilt about x_s towards the detector.")
    detector_tilt_deg: float = _opt(0.0, -30.0, 30.0, "Screen normal tilted down towards the sample.")
    working_distance_mm: float = _opt(
        15.0, 1.0, 50.0, "Working distance (sample height below the pole piece)."
    )
    calibration_wd_mm: float = _opt(15.0, 1.0, 50.0, "Working distance the pattern centre was calibrated at.")
    pcx: float = _opt(0.5, -1.0, 2.0, "Pattern centre x / sensor width, from the left (Bruker).")
    pcy: float = _opt(0.3, -1.0, 2.0, "Pattern centre y / sensor height, from the top (Bruker).")
    pcz: float = _opt(0.6, 0.05, 5.0, "Source-screen distance / sensor height (Bruker).")
    probe_current_na: float = _opt(
        0.1,
        0.0,
        1000.0,
        "SEM probe current, nA (an integrating DE camera holds ~1 electron per pixel per frame at 20 kV).",
    )
    step_um: float = _opt(
        0.5, 1e-4, 1e4, "Scan step on the (tilt-corrected) sample surface when the request gives none."
    )
    band_contrast: float = _opt(
        0.06, 0.0, 1.0, "Kikuchi band modulation relative to the background for a perfect surface."
    )
    energy_spread_frac: float = _opt(
        0.08,
        0.0,
        0.5,
        "FWHM of the backscattered energies that form "
        "the bands, over the beam energy (blurs the band edges).",
    )
    angular_blur_deg: float = _opt(
        0.1, 0.0, 2.0, "Angular blur of the screen (scintillator / charge spread), sigma."
    )
    master_resolution_deg: float = _opt(0.15, 0.03, 1.0, "Pixel of the cached Lambert master pattern.")
    background: bool = _opt(True, doc="Angle-dependent backscatter background (off: isotropic, bands only).")
    boundary_blend: bool = _opt(
        True, doc="Mix the neighbouring grain's pattern within the interaction volume near grain boundaries."
    )
    interaction_scale: float = _opt(1.0, 0.0, 10.0, "Scales the interaction-volume footprint.")
    supersample: float = _opt(
        1,
        1,
        4,
        "Samples per binned pixel along each axis (binning integrates the band edges; 1 = the pixel centre).",
    )

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[dict] = None) -> "EbsdGeometry":
        return cls().replace(**(values or {}))

    def replace(self, **kw) -> "EbsdGeometry":
        """A copy with ``kw`` changed (numbers clamped to their documented range)."""
        meta = {f.name: f for f in fields(self)}
        out = {}
        for k, v in kw.items():
            if k not in meta:
                raise KeyError(f"unknown EBSD geometry field {k!r}")
            f = meta[k]
            if isinstance(f.default, bool):
                v = bool(v)
            else:
                v = float(v)
                lo, hi = f.metadata.get("lo"), f.metadata.get("hi")
                v = max(lo, v) if lo is not None else v
                v = min(hi, v) if hi is not None else v
            out[k] = v
        return dataclasses.replace(self, **out)

    @staticmethod
    def describe() -> dict:
        """Name -> {default, lo, hi, doc} for every field (for building a GUI)."""
        return {f.name: {"default": f.default, **f.metadata} for f in fields(EbsdGeometry)}

    # ------------------------------------------------------------------ geometry
    @property
    def delta_rad(self) -> float:
        return math.radians(self.sample_tilt_deg - self.detector_tilt_deg)

    def sample_to_detector(self) -> np.ndarray:
        """``S2D``: sample-frame vector -> detector-frame vector (see the module docstring)."""
        d = self.delta_rad
        s, c = math.sin(d), math.cos(d)
        return np.array([[1.0, 0.0, 0.0], [0.0, -s, c], [0.0, -c, -s]])

    def surface_normal_d(self) -> np.ndarray:
        """The surface's outward normal in the detector frame."""
        return self.sample_to_detector() @ np.array([0.0, 0.0, -1.0])

    def beam_direction_d(self) -> np.ndarray:
        """The incident beam's direction of travel (straight down) in the detector frame."""
        t = math.radians(self.detector_tilt_deg)
        return np.array([0.0, math.cos(t), math.sin(t)])

    def forward_direction_d(self) -> np.ndarray:
        """The specular (forward-scatter) direction of the beam off the tilted surface."""
        b = self.beam_direction_d()
        n = self.surface_normal_d()
        f = b - 2.0 * float(b @ n) * n
        return f / np.linalg.norm(f)

    def pattern_centre(
        self, dx_um=0.0, dy_um=0.0, sensor_shape=(1024, 1024), pixel_um: float = 15.0
    ) -> np.ndarray:
        """Bruker ``(pcx, pcy, pcz)`` with the beam ``(dx_um, dy_um)`` from the scan centre on the
        sample (sample frame), at this geometry's working distance. Arrays broadcast; the
        result has a trailing axis of 3."""
        H, W = (float(v) for v in sensor_shape)
        p = float(pixel_um)
        dx = np.asarray(dx_um, np.float64)
        dy = np.asarray(dy_um, np.float64)
        d = self.delta_rad
        t = math.radians(self.detector_tilt_deg)
        dwd = (self.working_distance_mm - self.calibration_wd_mm) * 1000.0
        sx = dx
        sy = -math.sin(d) * dy + math.cos(t) * dwd
        sz = -math.cos(d) * dy + math.sin(t) * dwd
        pcx = self.pcx + sx / (W * p)
        pcy = self.pcy + sy / (H * p)
        pcz = self.pcz - sz / (H * p)
        return np.stack(np.broadcast_arrays(pcx, pcy, pcz), axis=-1)


def pc_pixels(pc, sensor_shape) -> tuple[float, float, float]:
    """Bruker pattern centre -> (x_pc, y_pc, DD) in sensor pixels from the top-left corner."""
    H, W = (float(v) for v in sensor_shape)
    return float(pc[0]) * W, float(pc[1]) * H, float(pc[2]) * H


def convert_pc(pc, sensor_shape, to: str = "edax", pixel_um: float = 15.0, binning: int = 1) -> np.ndarray:
    """Convert a Bruker pattern centre (..., 3) to another vendor's convention.

    * ``"edax"`` / ``"tsl"`` and ``"oxford"``: ``(x*, y*, z*)`` with ``x* = pcx``,
      ``y* = (1 - pcy) H / W`` (from the *bottom*, in units of the width) and
      ``z* = pcz H / W``.
    * ``"emsoft"`` (v5, as kikuchipy converts it): ``(xpc, ypc, L)`` with
      ``xpc = W (0.5 - pcx)``, ``ypc = H (0.5 - pcy)`` in (binned) pixels from the screen centre
      and ``L = H pcz p`` in micrometres.
    * ``"bruker"``: unchanged.
    """
    pc = np.asarray(pc, np.float64)
    H, W = (float(v) for v in sensor_shape)
    x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
    to = to.lower()
    if to == "bruker":
        out = (x, y, z)
    elif to in ("edax", "tsl", "oxford"):
        out = (x, (1.0 - y) * H / W, z * H / W)
    elif to == "emsoft":
        b = max(1, int(binning))
        out = (W / b * (0.5 - x), H / b * (0.5 - y), H * z * float(pixel_um))
    else:
        raise ValueError(f"unknown pattern-centre convention {to!r}")
    return np.stack(np.broadcast_arrays(*out), axis=-1)


def detector_position(geometry: EbsdGeometry, matrix, direction, pc, sensor_shape):
    """Where the crystal direction ``direction`` (Cartesian, crystal frame) of a grain with
    crystal->sample matrix ``matrix`` meets the screen: ``(x, y)`` in sensor pixels from the
    top-left corner (pixel ``j`` spans ``[j, j + 1)``), or ``None`` when it points away from
    the screen. ``pc``: Bruker pattern centre."""
    v = geometry.sample_to_detector() @ np.asarray(matrix, np.float64) @ np.asarray(direction, np.float64)
    if v[2] <= 1e-12:
        return None
    x0, y0, dd = pc_pixels(pc, sensor_shape)
    return (x0 + dd * v[0] / v[2], y0 + dd * v[1] / v[2])


# ---------------------------------------------------------------------- physics helpers
def backscatter_yield(z: float, tilt_deg: float) -> float:
    """Backscatter coefficient of a flat sample of atomic number ``z`` at ``tilt_deg`` from
    normal incidence (Arnal et al.: ``eta = (1 + cos tilt)^(-9 / sqrt Z)``)."""
    return float((1.0 + math.cos(math.radians(tilt_deg))) ** (-9.0 / math.sqrt(max(z, 1.0))))


def kanaya_okayama_range_um(z: float, density_g_cm3: float, kv: float) -> float:
    """Kanaya-Okayama electron range (µm), with the atomic weight taken as ``2.1 Z``."""
    a = 2.1 * max(z, 1.0)
    return 0.0276 * a * float(kv) ** 1.67 / (max(z, 1.0) ** 0.89 * max(density_g_cm3, 0.1))


#: The EBSD signal's lateral footprint (radius across the tilt axis) as a fraction of the
#: Kanaya-Okayama range: ~30 nm for Ni and ~80 nm for Al at 20 kV, as measured.
EBSD_FOOTPRINT_FRACTION = 0.02


def footprint_um(
    z: float, density_g_cm3: float, kv: float, tilt_deg: float, scale: float = 1.0
) -> tuple[float, float]:
    """(rx, ry) µm: semi-axes of the area the Kikuchi signal comes from, along x_s (the tilt
    axis) and y_s (stretched by 1 / cos tilt on the tilted surface)."""
    rx = EBSD_FOOTPRINT_FRACTION * kanaya_okayama_range_um(z, density_g_cm3, kv) * float(scale)
    return rx, rx / max(math.cos(math.radians(tilt_deg)), 0.05)


def background_parameters(kv: float, tilt_deg: float) -> tuple[float, float]:
    """(lobe fraction, lobe concentration) of the backscatter angular distribution: a cosine
    (Lambert) part about the surface normal plus a von Mises-Fisher lobe about the specular
    direction, more of it and sharper at high tilt and beam energy (heuristic fit to the
    Monte-Carlo distributions of tilted bulk samples)."""
    s = math.sin(math.radians(tilt_deg))
    return 0.65 * s * s, 3.0 + 0.2 * float(kv)


def boundary_mix(distance_um, normal_xy, rx: float, ry: float):
    """Fraction of an elliptical footprint (semi-axes rx, ry along x_s, y_s) on the far side of
    a straight boundary ``distance_um`` from its centre with unit normal ``normal_xy``
    (..., 2): the circle-segment area fraction after the affine map of the ellipse."""
    n = np.asarray(normal_xy, np.float64)
    h = np.hypot(rx * n[..., 0], ry * n[..., 1])
    t = np.clip(np.asarray(distance_um, np.float64) / np.maximum(h, 1e-12), 0.0, 1.0)
    return (np.arccos(t) - t * np.sqrt(1.0 - t * t)) / math.pi
