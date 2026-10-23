"""Orientation conventions, stage tilt and grain textures.

Conventions (identical to diffsims/orix, verified in ``tests/test_crystal.py``):

* Lab frame = detector frame: ``x`` = detector column (+right), ``y`` = detector row
  (+down), ``z`` completes a right-handed set; the Ewald sphere is diffsims', centred at
  ``(0, 0, 1/lambda)``, so the beam travels along ``-z``.
* A grain orientation is an orix ``Rotation`` ``R`` stored as a unit quaternion
  ``(a, b, c, d)``. It is exactly the ``rotation`` argument of
  ``SimulationGenerator.calculate_diffraction2d``: a crystal-frame vector ``v`` appears in
  the lab frame as ``~R * v``, i.e. the crystal->lab matrix is ``R.to_matrix().T``.
  The identity puts the crystal ``[001]`` along ``z`` (the [001] zone-axis pattern).
* Stage tilt: ``alpha`` (goniometer) is a rotation about the lab ``x`` axis and foreshortens
  ``y`` (``ViewWindow.cos_alpha``); ``beta`` (holder) is about ``y`` and foreshortens ``x``.
  The holder rides on the goniometer, so ``S = Rx(alpha) @ Ry(beta)`` and the effective
  crystal->lab matrix of a grain is ``S @ R.to_matrix().T`` (``effective_matrices``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as _R


def quat_to_matrix(q) -> np.ndarray:
    """(N, 4) orix quaternions (a, b, c, d) -> (N, 3, 3) matrices ``M`` with ``R * v == M @ v``."""
    q = np.asarray(q, np.float64).reshape(-1, 4)
    return _R.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()  # scipy is scalar-last


def crystal_to_lab(q) -> np.ndarray:
    """Crystal->lab matrices (N, 3, 3) of orix/diffsims rotations ``q`` (N, 4)."""
    return quat_to_matrix(q).transpose(0, 2, 1)


def rotation_from_crystal_to_lab(m) -> np.ndarray:
    """Quaternions (N, 4), ``a >= 0``, of the diffsims rotation whose crystal->lab matrix is ``m``."""
    q = _R.from_matrix(np.asarray(m, np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)).as_quat()[
        :, [3, 0, 1, 2]
    ]
    return np.where(q[:, :1] < 0, -q, q)


def rot_z(t: float) -> np.ndarray:
    return _R.from_euler("z", t).as_matrix()


def stage_matrix(alpha_rad: float, beta_rad: float) -> np.ndarray:
    """Specimen -> lab rotation of a double-tilt stage (``Rx(alpha) @ Ry(beta)``)."""
    return _R.from_euler("XY", [alpha_rad, beta_rad]).as_matrix()


def effective_matrices(matrices: np.ndarray, alpha_rad: float = 0.0, beta_rad: float = 0.0) -> np.ndarray:
    """Crystal->lab matrices including the stage tilt."""
    if alpha_rad == 0.0 and beta_rad == 0.0:
        return matrices
    return np.einsum("ij,njk->nik", stage_matrix(alpha_rad, beta_rad), matrices)


def align(u, v) -> np.ndarray:
    """A rotation matrix taking direction ``u`` onto ``v`` (the smallest one)."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    ax, c = np.cross(u, v), float(np.dot(u, v))
    if np.linalg.norm(ax) < 1e-12:
        ax = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0]) if c < 0 else np.zeros(3)
        return _R.from_rotvec(ax / max(np.linalg.norm(ax), 1e-300) * (math.pi if c < 0 else 0.0)).as_matrix()
    return _R.from_rotvec(ax / np.linalg.norm(ax) * math.atan2(np.linalg.norm(ax), c)).as_matrix()


def _small_rotation(rng, sigma_deg: float) -> np.ndarray:
    v = rng.normal(size=3)
    return _R.from_rotvec(v / np.linalg.norm(v) * math.radians(sigma_deg) * abs(rng.normal())).as_matrix()


@dataclass(frozen=True)
class Texture:
    """How grain orientations are drawn.

    * ``random``: uniform on SO(3) (nanoparticles, untextured polycrystals);
    * ``fibre``: crystal direction ``axis`` along the specimen normal (``z``), random rotation
      about it, Gaussian spread ``spread_deg`` (deposited thin films: ``<111>``);
    * ``single``: zone axis ``axis`` along the beam with a small misorientation
      ``spread_deg`` (single-crystal matrices, e.g. a Si [110] lamella);
    * ``oriented``: as ``single``, and the crystal direction ``inplane`` (perpendicular to
      ``axis``) at ``angle_deg`` in the specimen plane from +x (crystals whose shape fixes
      their orientation: MoO3 laths, graphite shells, an epitaxial multilayer).
    """

    kind: str = "random"
    axis: tuple = (1, 1, 1)
    spread_deg: float = 0.0
    inplane: tuple = ()
    angle_deg: float = 0.0

    def draw(self, rng, crystal_basis=np.eye(3)) -> np.ndarray:
        """Crystal->specimen matrix. ``crystal_basis`` maps [uvw] to Cartesian (direct lattice)."""
        if self.kind == "random":
            return _R.random(random_state=rng).as_matrix()
        base = align(crystal_basis @ np.asarray(self.axis, float), [0.0, 0.0, 1.0])
        if self.kind == "fibre":
            return _small_rotation(rng, self.spread_deg) @ rot_z(rng.uniform(0, 2 * math.pi)) @ base
        if self.kind == "single":
            return _small_rotation(rng, self.spread_deg) @ base
        if self.kind == "oriented":
            v = base @ (crystal_basis @ np.asarray(self.inplane, float))
            turn = math.radians(self.angle_deg) - math.atan2(v[1], v[0])
            return _small_rotation(rng, self.spread_deg) @ rot_z(turn) @ base
        raise ValueError(f"unknown texture {self.kind!r}")


RANDOM = Texture()
FIBRE_111 = Texture("fibre", (1, 1, 1), 3.0)
SI_110_LAMELLA = Texture("single", (1, 1, 0), 0.6)
