"""The residual deformation around a Vickers or Berkovich indent in a brittle crystal (Si), at
the surface the EBSD signal comes from (the top ~20 nm).

Regions, by distance ``r`` from the indent centre (``a``: the contact half-diagonal / the
Berkovich circumradius; ``b = plastic_zone_ratio * a``):

* **Impression** (inside the contact polygon): the faces (Vickers 22 deg, Berkovich 24.7 deg
  from the surface) are tilted lattice that has been transformed to amorphous / high-pressure
  phases on unloading: quality ~0 (no pattern), the face tilt as a rotation.
* **Plastic zone** (``r < b``): dislocation-rich; the elastic field saturates at its value at
  ``b``, a smooth random lattice rotation grows towards the impression (up to
  ``plastic_rotation_deg``) and the pattern quality falls to ``core_quality``.
* **Elastic field** (``r > b``): the surface field of a point load (Boussinesq) left behind as a
  residual, Yoffe-blister-like field: in-plane, traction-free (``e13 = e23 = 0``)
  - a deviatoric part ``e_rr = -A (b/r)^2``, ``e_tt = +A (b/r)^2`` (cubic anisotropy and the
    indenter's corners add a ``1 + anisotropy cos(n (theta - theta0))`` modulation, n = 4 / 3),
  - a dilatational blister part ``e_rr = e_tt = D (b/r)^3`` with ``e33 = -2 nu/(1 - nu) D (b/r)^3``
    (plane stress),
  - the surface tilt of the residual uplift, a rotation about the tangent ``theta_hat`` of
    ``W (b/r)^2``, plus a small twist about the normal ``T (b/r)^2 sin(n (theta - theta0))``.
* **Cracks**: radial cracks from each corner of the impression along the diagonals (``<110>``
  for a (001) wafer with its diagonals along ``<110>``), to ``crack_length_ratio * a``: a dark
  line (quality ``crack_quality`` over ``crack_width_um``) and a mode-I crack-tip strain
  ``K_I / sqrt(2 pi rho)`` across the crack plane beyond the tip. **Lateral cracks**: a chipped,
  lifted sector between two radial cracks (``lateral_chips`` of them, seeded): an extra uplift
  tilt of ``chip_tilt_deg`` and quality x ``chip_quality``.

Everything is in the sample frame (x_s right, y_s down, z_s into the sample). The deformation
gradient is ``F = R(omega) (I + e)``: it maps the reference lattice to the deformed one
(``u = F u0``). The GND density proxy is the norm of the Nye tensor of the lattice rotation
(from its in-plane gradient) over the Burgers vector, 1/m^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..hashing import SeedKind, rng_for

BURGERS_SI_NM = 0.384  # a/2 <110>


@dataclass(frozen=True)
class Indent:
    """One indent (lengths µm, angles deg); see the module docstring."""

    center_um: tuple = (0.0, 0.0)
    kind: str = "vickers"  # "vickers" | "berkovich"
    size_um: float = 5.0  # Vickers half-diagonal / Berkovich circumradius (a)
    rotation_deg: float = 0.0  # direction of the first corner from +x
    plastic_zone_ratio: float = 1.8
    strain_amplitude: float = 2.5e-3  # A: deviatoric strain at r = b
    blister_strain: float = 0.8e-3  # D: dilatational strain at r = b
    anisotropy: float = 0.3
    tilt_deg: float = 2.0  # W: surface tilt at r = b
    twist_deg: float = 0.3  # T
    plastic_rotation_deg: float = 3.0
    core_quality: float = 0.25
    crack_length_ratio: float = 2.5
    crack_width_um: float = 0.15
    crack_quality: float = 0.05
    crack_tip_strain: float = 1.5e-3  # mode-I opening strain 0.5 µm ahead of a tip
    lateral_chips: int = 1
    chip_tilt_deg: float = 1.0
    chip_quality: float = 0.6
    poisson: float = 0.22
    seed: int = 1

    @property
    def corners(self) -> int:
        return 3 if self.kind.lower() == "berkovich" else 4

    @property
    def face_angle_deg(self) -> float:
        return 24.7 if self.kind.lower() == "berkovich" else 22.0


def _rotvec_to_matrix(w: np.ndarray) -> np.ndarray:
    """Rodrigues: (n, 3) rotation vectors (rad) -> (n, 3, 3)."""
    th = np.linalg.norm(w, axis=1)
    k = w / np.maximum(th, 1e-300)[:, None]
    K = np.zeros((len(w), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


class IndentField:
    """Evaluates an :class:`Indent`'s strain, rotation, quality and GND at surface points."""

    def __init__(self, indent: Indent):
        self.p = indent
        rng = rng_for(indent.seed, SeedKind.STRUCTURE, 0x1D3)
        nm = 12
        ang = rng.uniform(0, 2 * math.pi, (3, nm))
        kmag = 2 * math.pi / rng.uniform(0.6, 2.0, (3, nm))  # wavelengths 0.6-2 µm
        self._k = np.stack([kmag * np.cos(ang), kmag * np.sin(ang)], -1)  # (3, nm, 2)
        self._phi = rng.uniform(0, 2 * math.pi, (3, nm))
        self._amp = rng.normal(size=(3, nm)) / math.sqrt(nm)
        n = indent.corners
        gaps = rng.choice(n, size=min(max(indent.lateral_chips, 0), n), replace=False)
        self._chips = [math.radians(indent.rotation_deg) + (g + 0.5) * 2 * math.pi / n for g in gaps]

    # -------------------------------------------------------------- pieces
    def _polar(self, x, y):
        cx, cy = self.p.center_um
        dx, dy = np.asarray(x, np.float64) - cx, np.asarray(y, np.float64) - cy
        return dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)

    def inside_impression(self, x, y) -> np.ndarray:
        dx, dy, r, th = self._polar(x, y)
        p = self.p
        n = p.corners
        t0 = math.radians(p.rotation_deg)
        # polygon with corners at radius a, angles t0 + 2 pi k / n: inside if every edge's
        # support distance a cos(pi / n) exceeds the projection on its outward normal
        inside = np.ones(np.shape(dx), bool)
        for k in range(n):
            nrm = t0 + (k + 0.5) * 2 * math.pi / n
            inside &= dx * math.cos(nrm) + dy * math.sin(nrm) <= p.size_um * math.cos(math.pi / n)
        return inside

    def rotation(self, x, y) -> np.ndarray:
        """(n, 3) lattice rotation vectors (rad, sample frame)."""
        p = self.p
        dx, dy, r, th = self._polar(x, y)
        a, b = p.size_um, p.plastic_zone_ratio * p.size_um
        n, t0 = p.corners, math.radians(p.rotation_deg)
        s2 = (b / np.maximum(r, b)) ** 2
        tilt = math.radians(p.tilt_deg) * s2
        for c in self._chips:  # lifted lateral-crack chip: extra uplift tilt within the crack length
            d = np.angle(np.exp(1j * (th - c)))
            sector = np.clip(1 - np.abs(d) / (math.pi / n), 0, 1) ** 0.5
            ring = (r > a) & (r < p.crack_length_ratio * a)
            tilt = tilt + math.radians(p.chip_tilt_deg) * sector * ring * np.sin(
                np.pi * np.clip((r - a) / ((p.crack_length_ratio - 1) * a), 0, 1)
            )
        w = np.zeros((np.size(dx), 3))
        # uplift sloping down away from the indent: the normal tips outwards, a rotation about the
        # tangent t = z x r_hat ... with z into the sample, the outward normal is -z; tipping it
        # towards +r_hat is a rotation about (-z) x r_hat = (sin th, -cos th, 0)
        w[:, 0] = (tilt * np.sin(th)).ravel()
        w[:, 1] = (-tilt * np.cos(th)).ravel()
        w[:, 2] = (math.radians(p.twist_deg) * s2 * np.sin(n * (th - t0))).ravel()
        # plastic zone: smooth random rotations growing towards the impression
        grow = np.clip((b - r) / (b - a), 0, 1).ravel() ** 1.5 * math.radians(p.plastic_rotation_deg)
        if np.any(grow > 0):
            X, Y = dx.ravel(), dy.ravel()
            for i in range(3):
                ph = X[:, None] * self._k[i, :, 0] + Y[:, None] * self._k[i, :, 1] + self._phi[i]
                w[:, i] += grow * (np.sin(ph) @ self._amp[i])
        # the impression's faces: tilted by the face angle towards the centre
        inside = self.inside_impression(x, y).ravel()
        if np.any(inside):
            fa = math.radians(p.face_angle_deg)
            k = np.floor(((th.ravel() - t0) % (2 * math.pi)) / (2 * math.pi / n))
            nrm = t0 + (k + 0.5) * 2 * math.pi / n  # the face's outward direction
            w[inside, 0] = -fa * np.sin(nrm[inside])
            w[inside, 1] = fa * np.cos(nrm[inside])
            w[inside, 2] = 0.0
        return w

    def strain(self, x, y) -> np.ndarray:
        """(n, 3, 3) small strain (sample frame, symmetric)."""
        p = self.p
        dx, dy, r, th = self._polar(x, y)
        dx, dy, r, th = dx.ravel(), dy.ravel(), r.ravel(), th.ravel()
        a, b = p.size_um, p.plastic_zone_ratio * p.size_um
        n, t0 = p.corners, math.radians(p.rotation_deg)
        rs = np.maximum(r, b)
        s2, s3 = (b / rs) ** 2, (b / rs) ** 3
        mod = 1.0 + p.anisotropy * np.cos(n * (th - t0))
        err = -p.strain_amplitude * s2 * mod + p.blister_strain * s3
        ett = p.strain_amplitude * s2 * mod + p.blister_strain * s3
        c, s = np.cos(th), np.sin(th)
        e = np.zeros((len(r), 3, 3))
        e[:, 0, 0] = err * c * c + ett * s * s
        e[:, 1, 1] = err * s * s + ett * c * c
        e[:, 0, 1] = e[:, 1, 0] = (err - ett) * s * c
        # radial cracks: mode-I opening strain across each crack plane ahead of its tip
        L = p.crack_length_ratio * a
        for k in range(n):
            ca = t0 + 2 * math.pi * k / n
            u = np.array([math.cos(ca), math.sin(ca)])
            v = np.array([-u[1], u[0]])
            along = dx * u[0] + dy * u[1] - L
            across = dx * v[0] + dy * v[1]
            rho = np.hypot(along, across)
            phi = np.arctan2(across, along)
            amp = p.crack_tip_strain * np.sqrt(0.5 / np.maximum(rho, 0.05)) * np.exp(-rho / (0.5 * a))
            f = np.cos(phi / 2) * (1 + np.sin(phi / 2) * np.sin(1.5 * phi))
            env = np.where(rho > 0.05, amp * f, 0.0)
            # tensile strain along v (normal to the crack plane), Poisson contraction along u
            e[:, :2, :2] += env[:, None, None] * (np.outer(v, v) - p.poisson * np.outer(u, u))[None]
        tr = e[:, 0, 0] + e[:, 1, 1]
        e[:, 2, 2] = -p.poisson / (1 - p.poisson) * tr  # plane stress at the free surface
        return e

    def quality(self, x, y) -> np.ndarray:
        p = self.p
        dx, dy, r, th = self._polar(x, y)
        dx, dy, r, th = dx.ravel(), dy.ravel(), r.ravel(), th.ravel()
        a, b = p.size_um, p.plastic_zone_ratio * p.size_um
        n, t0 = p.corners, math.radians(p.rotation_deg)
        t = np.clip((b - r) / (b - a), 0, 1)
        q = 1.0 - (1.0 - p.core_quality) * t**0.7
        L = p.crack_length_ratio * a
        for k in range(n):
            ca = t0 + 2 * math.pi * k / n
            along = dx * math.cos(ca) + dy * math.sin(ca)
            across = -dx * math.sin(ca) + dy * math.cos(ca)
            on = (along > 0) & (along < L)
            d = np.where(on, np.abs(across), np.hypot(np.maximum(along - L, 0), across))
            q = q * (1 - (1 - p.crack_quality) * np.exp(-0.5 * (d / p.crack_width_um) ** 2))
        for c in self._chips:
            dth = np.angle(np.exp(1j * (th - c)))
            chip = (np.abs(dth) < math.pi / n) & (r > a) & (r < L)
            q = np.where(chip, q * p.chip_quality, q)
        q = np.where(self.inside_impression(x, y).ravel(), 0.02, q)
        return np.clip(q, 0.01, 1.0)

    # -------------------------------------------------------------- outputs
    def deformation_gradient(self, x, y) -> np.ndarray:
        """(n, 3, 3) F = R(omega) (I + e)."""
        e = self.strain(x, y)
        return np.einsum("nij,njk->nik", _rotvec_to_matrix(self.rotation(x, y)), np.eye(3) + e)

    def gnd(self, x, y, h_um: float = 0.05) -> np.ndarray:
        """GND density proxy, 1/m^2: |Nye tensor| / b from the in-plane gradient of the lattice
        rotation (central differences, ``h_um``); out-of-plane gradients are not observable."""
        x = np.asarray(x, np.float64).ravel()
        y = np.asarray(y, np.float64).ravel()
        kx = (self.rotation(x + h_um, y) - self.rotation(x - h_um, y)) / (2 * h_um)  # rad / µm
        ky = (self.rotation(x, y + h_um) - self.rotation(x, y - h_um)) / (2 * h_um)
        kappa = np.zeros((len(x), 3, 3))
        kappa[:, :, 0], kappa[:, :, 1] = kx, ky  # kappa_ij = d omega_i / d x_j
        alpha = np.transpose(kappa, (0, 2, 1)) - np.trace(kappa, axis1=1, axis2=2)[:, None, None] * np.eye(3)
        return np.linalg.norm(alpha, axis=(1, 2)) * 1e6 / (BURGERS_SI_NM * 1e-9)
