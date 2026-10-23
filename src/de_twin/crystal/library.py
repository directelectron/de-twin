"""Reciprocal-lattice library of one phase and exact kinematic excitation.

:class:`CrystalLibrary` computes, once per phase, every allowed reciprocal lattice
vector out to ``max_g_inv_nm`` (crystal Cartesian frame, 1/nm) with its kinematic
``|F|^2`` (diffsims, Lobato electron scattering factors, Debye-Waller factor
``exp(-B g^2 / 2)``, g in 1/A - the same convention as diffsims). From that,
:meth:`CrystalLibrary.excite` evaluates a batch of orientations at once:

* lab-frame vectors ``g_lab = M g`` (``M`` crystal->lab, see :mod:`.orientation`);
* the exact Ewald-sphere excitation error, as in diffsims:
  ``s = (1/lambda - sqrt(1/lambda^2 - gx^2 - gy^2)) - gz``;
* the kinematic two-beam intensity ``I_g = (pi t / xi_g)^2 <sinc^2(pi t s)>`` with the
  extinction distance ``xi_g = pi V_c / (lambda gamma |F_g|)``; ``<>`` averages the
  rocking curve over the illumination cone (``s +- g alpha``) analytically, so convergent
  beams excite a band of orientations and parallel beams reduce to the plain ``sinc^2``.

Reflections further than ``S_WINDOW_INV_NM`` from the Ewald sphere are dropped with a
smooth taper (their kinematic tails are negligible and the taper keeps every intensity a
continuous function of orientation). :meth:`CrystalLibrary.powder` gives the
orientation average (``<I_g> = t / (2 g xi_g^2) pi^2``) for Debye-Scherrer rings.
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import sici

S_WINDOW_INV_NM = 1.0  # excitation errors beyond this are dropped ...
S_TAPER_INV_NM = 0.5  # ... after a cos^2 taper starting here
MAX_G_INV_NM = 25.0
MIN_F2_FRACTION = 1e-8
CHUNK = 256


def relativistic_gamma(ht_kv: float) -> float:
    return 1.0 + ht_kv / 510.99895


def _sinc2_antiderivative(x):
    """F(x) = Si(2x) - sin^2(x)/x, so that F' = sinc^2 = sin^2(x)/x^2 (odd, F(0) = 0)."""
    x = np.asarray(x, np.float64)
    si, _ = sici(2.0 * x)
    safe = np.where(np.abs(x) < 1e-12, 1.0, x)
    return si - np.where(np.abs(x) < 1e-12, 0.0, np.sin(safe) ** 2 / safe)


def rocking_curve(x, a):
    """Mean of sinc^2 over [x - a, x + a] (a >= 0), where sinc(x) = sin(x)/x."""
    x = np.asarray(x, np.float64)
    a = np.asarray(a, np.float64)
    small = a < 1e-3
    safe_x = np.where(np.abs(x) < 1e-9, 1e-9, x)
    plain = (np.sin(safe_x) / safe_x) ** 2
    a_safe = np.where(small, 1.0, a)
    avg = (_sinc2_antiderivative(x + a_safe) - _sinc2_antiderivative(x - a_safe)) / (2.0 * a_safe)
    return np.where(small, plain, avg)


@dataclass
class Excitation:
    """Excited reflections of a batch of N orientations (flattened, K entries)."""

    owner: np.ndarray  # (K,) orientation index
    index: np.ndarray  # (K,) reflection index into the library
    gx: np.ndarray  # (K,) 1/nm, detector x
    gy: np.ndarray  # (K,) 1/nm, detector y
    s: np.ndarray  # (K,) excitation error, 1/nm
    intensity: np.ndarray  # (K,) kinematic fraction (pi t / xi_g)^2 <sinc^2>
    total: np.ndarray  # (N,) sum of intensity per orientation


class CrystalLibrary:
    """Reciprocal lattice + kinematic ``|F|^2`` of one phase, computed once."""

    def __init__(
        self,
        phase,
        *,
        max_g_inv_nm: float = MAX_G_INV_NM,
        debye_waller_a2: float = 0.0,
        scattering_params: str = "lobato",
        cache_dir: str | None = None,
    ):
        self.phase = phase
        self.max_g_inv_nm = float(max_g_inv_nm)
        self.debye_waller_a2 = float(debye_waller_a2)
        self.scattering_params = scattering_params
        self.volume_a3 = float(phase.structure.lattice.volume)
        data = self._load(cache_dir)
        if data is None:
            with warnings.catch_warnings():  # diffpy.structure 3.x deprecations inside diffsims
                warnings.simplefilter("ignore", DeprecationWarning)
                data = self._compute()
            self._save(cache_dir, data)
        self.g, self.hkl, self.f2 = data
        self.g_len = np.linalg.norm(self.g, axis=1)
        self._rings = None

    # ---------------------------------------------------------------- build
    def _compute(self):
        from diffsims.crystallography import ReciprocalLatticeVector

        rlv = ReciprocalLatticeVector.from_min_dspacing(self.phase, min_dspacing=10.0 / self.max_g_inv_nm)
        try:
            rlv = rlv[rlv.allowed]
        except NotImplementedError:  # primitive hexagonal (graphite): the structure factor decides
            pass
        rlv.sanitise_phase()
        rlv.calculate_structure_factor(self.scattering_params)
        g_a = rlv.gspacing  # 1/A
        f2 = np.abs(np.asarray(rlv.structure_factor)) ** 2 * np.exp(-0.5 * self.debye_waller_a2 * g_a**2)
        keep = f2 > MIN_F2_FRACTION * f2.max()
        return (
            np.asarray(rlv.data, np.float64)[keep] * 10.0,
            np.rint(rlv.hkl[keep]).astype(np.int32),
            f2[keep].astype(np.float64),
        )

    def _cache_path(self, cache_dir):
        cache_dir = cache_dir or os.environ.get("DE_TWIN_CRYSTAL_CACHE")
        if not cache_dir:
            return None
        s = self.phase.structure
        key = repr(
            (
                str(s.lattice),
                [(a.element, tuple(np.round(a.xyz, 6))) for a in s],
                self.phase.space_group.number if self.phase.space_group else 0,
                self.max_g_inv_nm,
                self.debye_waller_a2,
                self.scattering_params,
            )
        )
        return Path(cache_dir) / f"{self.phase.name}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"

    def _load(self, cache_dir):
        p = self._cache_path(cache_dir)
        if p is None or not p.exists():
            return None
        with np.load(p) as z:
            return z["g"], z["hkl"], z["f2"]

    def _save(self, cache_dir, data):
        p = self._cache_path(cache_dir)
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, g=data[0], hkl=data[1], f2=data[2])

    # -------------------------------------------------------------- physics
    def inv_xi(self, ht_kv: float, wavelength_nm: float) -> np.ndarray:
        """pi / xi_g in 1/nm (xi_g: kinematic extinction distance)."""
        lam_a = wavelength_nm * 10.0
        return 10.0 * lam_a * relativistic_gamma(ht_kv) * np.sqrt(self.f2) / self.volume_a3

    def extinction_distance_nm(self, hkl, ht_kv: float, wavelength_nm: float) -> float:
        i = int(np.flatnonzero((self.hkl == np.asarray(hkl)).all(axis=1))[0])
        return float(math.pi / self.inv_xi(ht_kv, wavelength_nm)[i])

    def geometry(self, matrices, wavelength_nm: float, convergence_mrad: float = 0.0):
        """(owner, index, gx, gy, s) of the reflections near the Ewald sphere for each
        crystal->lab matrix (N, 3, 3) - the orientation-only part of :meth:`excite`,
        sorted by owner."""
        m = np.asarray(matrices, np.float64).reshape(-1, 3, 3)
        inv_lam = 1.0 / wavelength_nm
        s_max = S_WINDOW_INV_NM + self.g_len * (convergence_mrad * 1e-3)
        parts = []
        for c0 in range(0, len(m), CHUNK):
            lab_vector = np.einsum("nij,gj->ngi", m[c0 : c0 + CHUNK], self.g)
            r2 = lab_vector[..., 0] ** 2 + lab_vector[..., 1] ** 2
            s = inv_lam - np.sqrt(np.maximum(inv_lam * inv_lam - r2, 0.0)) - lab_vector[..., 2]
            own, idx = np.nonzero(np.abs(s) < s_max)
            parts.append((own + c0, idx, lab_vector[own, idx, 0], lab_vector[own, idx, 1], s[own, idx]))
        if not parts:
            e = np.zeros(0)
            return e.astype(np.int64), e.astype(np.int64), e, e, e
        return tuple(np.concatenate(v) for v in zip(*parts))

    def intensity_at(
        self, idx, s, thickness_nm, ht_kv: float, wavelength_nm: float, convergence_mrad: float = 0.0
    ) -> np.ndarray:
        """Kinematic fraction ``(pi t / xi_g)^2 <sinc^2>`` of reflections ``idx`` at excitation
        errors ``s`` and thicknesses ``thickness_nm`` (all (K,)): the thickness-dependent part
        of :meth:`excite`, element by element."""
        k = self.inv_xi(ht_kv, wavelength_nm)
        alpha = convergence_mrad * 1e-3
        tt = np.asarray(thickness_nm, np.float64)
        ga = self.g_len[idx] * alpha
        excess = np.clip((np.abs(s) - S_TAPER_INV_NM - ga) / (S_WINDOW_INV_NM - S_TAPER_INV_NM), 0.0, 1.0)
        taper = np.cos(0.5 * math.pi * excess) ** 2
        return (k[idx] * tt) ** 2 * rocking_curve(math.pi * tt * s, math.pi * tt * ga) * taper

    def excite(
        self,
        matrices,
        wavelength_nm: float,
        thickness_nm,
        ht_kv: float,
        convergence_mrad: float = 0.0,
        min_intensity: float = 0.0,
    ) -> Excitation:
        """Excited reflections of each crystal->lab matrix (N, 3, 3) at thickness (N,) nm.

        ``thickness_nm`` may also be (N, T): every orientation at T thicknesses; ``intensity``
        is then (K, T) and ``total`` (N, T) (``min_intensity`` must be 0)."""
        m = np.asarray(matrices, np.float64).reshape(-1, 3, 3)
        n = len(m)
        t_in = np.asarray(thickness_nm, np.float64)
        t = np.broadcast_to(t_in, (n,) + t_in.shape[1:]) if t_in.ndim == 2 else np.broadcast_to(t_in, (n,))
        k = self.inv_xi(ht_kv, wavelength_nm)
        own, idx, gx, gy, s = self.geometry(m, wavelength_nm, convergence_mrad)
        alpha = convergence_mrad * 1e-3
        tt = t[own]
        column = (lambda v: v[:, None]) if t.ndim == 2 else (lambda v: v)
        x = math.pi * tt * column(s)
        a = math.pi * tt * column(self.g_len[idx] * alpha)
        excess = np.clip(
            (np.abs(s) - S_TAPER_INV_NM - self.g_len[idx] * alpha) / (S_WINDOW_INV_NM - S_TAPER_INV_NM),
            0.0,
            1.0,
        )
        taper = np.cos(0.5 * math.pi * excess) ** 2
        inten = (column(k[idx]) * tt) ** 2 * rocking_curve(x, a) * column(taper)
        if t.ndim == 2:
            total = np.stack(
                [np.bincount(own, weights=inten[:, j], minlength=n) for j in range(t.shape[1])], 1
            )
            return Excitation(own, idx, gx, gy, s, inten, total)
        total = np.bincount(own, weights=inten, minlength=n)
        if min_intensity > 0:
            keep = inten >= min_intensity
            own, idx, gx, gy, s, inten = own[keep], idx[keep], gx[keep], gy[keep], s[keep], inten[keep]
        return Excitation(own, idx, gx, gy, s, inten, total)

    # --------------------------------------------------------------- powder
    def rings(self, ht_kv: float, wavelength_nm: float) -> tuple[np.ndarray, np.ndarray]:
        """(ring radius g [1/nm], orientation-averaged intensity per nm of thickness) per family."""
        key = (round(ht_kv, 6), round(wavelength_nm, 12))
        if self._rings is None or self._rings[0] != key:
            k = self.inv_xi(ht_kv, wavelength_nm)
            per = k * k / (2.0 * self.g_len)
            r = np.round(self.g_len, 4)
            uniq, inv = np.unique(r, return_inverse=True)
            self._rings = (key, uniq, np.bincount(inv, weights=per))
        return self._rings[1], self._rings[2]

    # ------------------------------------------------- orientation mapping
    def template_library(
        self,
        resolution_deg: float = 1.0,
        ht_kv: float = 200.0,
        max_excitation_error: float = 0.05,
        **generator_options,
    ):
        """A diffsims ``Simulation2D`` over the reduced fundamental zone (orix
        ``get_sample_reduced_fundamental``), i.e. an orientation-mapping template library
        in the twin's own convention (pyxem ``get_orientation``)."""
        from diffsims.generators.simulation_generator import SimulationGenerator
        from orix.sampling import get_sample_reduced_fundamental

        rot = get_sample_reduced_fundamental(resolution_deg, point_group=self.phase.point_group)
        gen = SimulationGenerator(ht_kv, **generator_options)
        return gen.calculate_diffraction2d(
            self.phase,
            rotation=rot,
            reciprocal_radius=self.max_g_inv_nm / 10.0,
            max_excitation_error=max_excitation_error,
            with_direct_beam=False,
        )


@functools.cache
def library_for(material_id: int, max_g_inv_nm: float = MAX_G_INV_NM) -> CrystalLibrary | None:
    """Cached library of a material (``None`` for amorphous materials)."""
    from ..specimen.materials import material
    from .phases import phase_for

    phase = phase_for(int(material_id))
    if phase is None:
        return None
    return CrystalLibrary(
        phase, max_g_inv_nm=max_g_inv_nm, debye_waller_a2=material(int(material_id)).debye_waller_a2
    )
