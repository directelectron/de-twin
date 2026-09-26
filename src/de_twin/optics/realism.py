"""A realistic column's geometry: what SerialEM's calibrations exist to measure.

The ideal twin (the default) has none of this: the image never rotates, the pixel size is the
nominal one, image shift moves the specimen one micrometre per unit along the camera's axes,
changing magnification keeps the same point on axis, and the stage lands where it is told.
A real column does none of those, and every one of them is a calibration:

========================  =====================================================  =======================
effect                    model                                                  SerialEM calibration
========================  =====================================================  =======================
image rotation            per imaging mode a base angle, per magnification a     Image & Stage Shift,
                          few degrees of jitter (LowMAG and MAG1 differ by a     stage calibration
                          large angle)
true pixel size           nominal x (1 + a few % per magnification)              Pixel Size
image-shift matrix        per magnification a 2x2 of scale, skew and rotation   Image Shift
                          mapping image-shift units to specimen micrometres
magnification offsets     per magnification a shift of the image centre          Mag IS offsets
                          (a fraction of its field of view)
stage backlash            the stage stops short by half the backlash, against   stage backlash
                          the direction it came from                              correction
========================  =====================================================  =======================

Every value is a deterministic function of ``seed`` and (imaging mode, magnification), so a
twin reproduces its column; :meth:`ColumnRealism.truth` reports them for closed-loop tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _u(seed: int, *parts) -> float:
    """A uniform in [0, 1) from the seed and any non-negative integers (numpy's SeedSequence:
    stable across versions and platforms)."""
    words = [0x5E4A, int(seed) & 0xFFFFFFFF] + [int(p) & 0xFFFFFFFF for p in parts]
    return float(np.random.default_rng(np.random.SeedSequence(words)).random())


def _n(seed: int, *parts) -> float:
    """A standard normal from the seed and any integers (Box-Muller on two uniforms)."""
    a = max(_u(seed, *parts, 1), 1e-12)
    b = _u(seed, *parts, 2)
    return math.sqrt(-2.0 * math.log(a)) * math.cos(2.0 * math.pi * b)


_MODES = {"lowmag": 1, "lm": 1, "mag1": 2, "m": 2, "mag2": 3, "mh": 3, "samag": 4, "sa": 4}


def _mode_id(mag_mode) -> int:
    return _MODES.get("".join(ch for ch in str(mag_mode).lower() if ch.isalnum()), 2)


@dataclass(frozen=True)
class ColumnRealism:
    """The imperfect geometry of a real column (see the module docstring)."""

    seed: int = 0
    #: Spread of the per-magnification image rotation around its mode's base angle, degrees.
    rotation_jitter_deg: float = 2.0
    #: The base angle of each imaging mode is drawn over the full circle.
    image_rotation: bool = True
    #: Spread of true / nominal pixel size, per magnification (fraction).
    pixel_scale_sigma: float = 0.02
    #: Image-shift matrix: spread of each axis's scale, of the skew, and of its rotation (deg).
    is_scale_sigma: float = 0.05
    is_skew_sigma: float = 0.02
    is_rotation_sigma_deg: float = 3.0
    #: Image-centre offset on a magnification change, as a fraction of that mag's field width.
    mag_offset_fraction: float = 0.02
    #: The stage stops short by half of this against its approach direction, micrometres.
    backlash_um: float = 0.3

    def _key(self, mag_mode, mag: float) -> tuple[int, int]:
        return _mode_id(mag_mode), int(round(float(mag)))

    def rotation_rad(self, mag_mode, mag: float) -> float:
        """How far the image is rotated on the camera, radians (world -> camera)."""
        if not self.image_rotation:
            return 0.0
        mode, m = self._key(mag_mode, mag)
        base = 2.0 * math.pi * _u(self.seed, 11, mode)
        return base + math.radians(self.rotation_jitter_deg) * _n(self.seed, 12, mode, m)

    def pixel_scale(self, mag_mode, mag: float) -> float:
        """True pixel size / nominal pixel size."""
        mode, m = self._key(mag_mode, mag)
        return 1.0 + self.pixel_scale_sigma * _n(self.seed, 21, mode, m)

    def is_matrix(self, mag_mode, mag: float) -> np.ndarray:
        """2x2: specimen micrometres (world frame) per image-shift unit."""
        mode, m = self._key(mag_mode, mag)
        sx = 1.0 + self.is_scale_sigma * _n(self.seed, 31, mode, m)
        sy = 1.0 + self.is_scale_sigma * _n(self.seed, 32, mode, m)
        k = self.is_skew_sigma * _n(self.seed, 33, mode, m)
        r = math.radians(self.is_rotation_sigma_deg) * _n(self.seed, 34, mode, m)
        c, s = math.cos(r), math.sin(r)
        return np.array([[c, -s], [s, c]]) @ np.array([[sx, k], [0.0, sy]])

    def mag_offset_um(self, mag_mode, mag: float) -> tuple[float, float]:
        """Where this magnification puts the image centre, relative to the ideal, micrometres:
        `mag_offset_fraction` of a nominal field of 1e5 / mag um (5 um at 20 kx)."""
        mode, m = self._key(mag_mode, mag)
        f = self.mag_offset_fraction * 1.0e5 / max(float(mag), 1.0)
        return f * _n(self.seed, 41, mode, m), f * _n(self.seed, 42, mode, m)

    def truth(self, mag_mode, mag: float) -> dict:
        """Every value at one magnification, for tests that calibrate against the twin."""
        return {
            "image_rotation_deg": math.degrees(self.rotation_rad(mag_mode, mag)),
            "pixel_scale": self.pixel_scale(mag_mode, mag),
            "is_matrix_um_per_unit": self.is_matrix(mag_mode, mag).tolist(),
            "mag_offset_um": self.mag_offset_um(mag_mode, mag),
            "backlash_um": self.backlash_um,
        }
