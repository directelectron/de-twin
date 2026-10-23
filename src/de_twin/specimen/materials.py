"""Material table.

Materials are addressed by a small integer id stored in the field map's
``material_id`` layer. Id 0 is vacuum. Crystalline materials carry a
:class:`Crystal` (space group + lattice + asymmetric unit, or a CIF path); the
diffraction physics lives in :mod:`de_twin.crystal`, which turns it into an orix
``Phase``. :func:`register_material` appends user materials (see
:func:`de_twin.crystal.register_cif`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class MaterialId(IntEnum):
    VACUUM = 0
    AMORPHOUS_CARBON = 1
    VITREOUS_ICE = 2
    SILICON_NITRIDE = 3
    COPPER = 4
    GOLD = 5
    ALUMINUM = 6
    IRON = 7
    SILICON = 8
    PLATINUM = 9
    PROTEIN = 10
    MOLYBDENUM_TRIOXIDE = 11
    NICKEL = 12
    AUSTENITE = 13  # face-centred cubic (gamma) iron, the austenite of steels; IRON is BCC ferrite


N_MATERIALS = len(MaterialId)  # built-in materials (see n_materials() for registered ones)
GRAINS_PER_MATERIAL = 512


@dataclass(frozen=True)
class Crystal:
    """Crystal structure: ``space_group`` + ``lattice`` (a, b, c nm; alpha, beta, gamma deg) +
    asymmetric-unit ``atoms`` ``(element, x, y, z)``; or a CIF file (``cif``) instead."""

    space_group: int = 0
    lattice: tuple = ()
    atoms: tuple = ()
    cif: Optional[str] = None

    @classmethod
    def cubic(cls, element: str, space_group: int, a_nm: float) -> "Crystal":
        return cls(space_group, (a_nm, a_nm, a_nm, 90.0, 90.0, 90.0), ((element, 0.0, 0.0, 0.0),))


@dataclass(frozen=True)
class Material:
    id: int
    name: str
    density_g_cm3: float
    absorption_length_200kv_nm: float  # mass-thickness mean free path at 200 kV (0 -> transparent)
    z_eff: float  # atomic number for the screened-Rutherford diffuse background
    mean_inner_potential_v: float
    scatterer_density_nm3: float  # atoms / nm^3 (amorphous phase-texture strength)
    crystal: Optional[Crystal] = None
    debye_waller_a2: float = 0.0  # B in Angstrom^2
    # Amorphous halo rings (g in 1/nm, weight). For crystalline materials these
    # describe the glassy/amorphous state used by in-situ crystallisation.
    halos: tuple[tuple[float, float], ...] = field(default=())

    @property
    def crystalline(self) -> bool:
        return self.crystal is not None

    @property
    def amorphous(self) -> bool:
        return self.crystal is None and self.id != MaterialId.VACUUM

    def absorption_length_nm(self, ht_kv: float) -> float:
        """Mean free path scaled from 200 kV with the Malis relativistic factor."""
        if self.absorption_length_200kv_nm <= 0:
            return np.inf
        return self.absorption_length_200kv_nm * malis_factor(200.0) / malis_factor(ht_kv)


def malis_factor(kv: float) -> float:
    """(1 + E/1022) / (1 + E/511)^2 with E in keV. Mean free path scales as 1/F."""
    return (1.0 + kv / 1022.0) / (1.0 + kv / 511.0) ** 2


def _glassy(a: float) -> tuple[tuple[float, float], ...]:
    return ((np.sqrt(3.0) / a, 0.65), (2.0 / a, 0.25))


def _metal(mid, name, el, sg, a, rho, lam, z, v0, n, b):
    return Material(mid, name, rho, lam, z, v0, n, Crystal.cubic(el, sg, a), b, _glassy(a))


M = MaterialId
MATERIALS: list[Material] = [
    Material(M.VACUUM, "Vacuum", 0.0, 0.0, 1.0, 0.0, 1.0),
    Material(M.AMORPHOUS_CARBON, "Amorphous carbon", 2.0, 150.0, 6.0, 9.0, 100.0,
             halos=((3.00, 0.60), (5.20, 0.25))),
    Material(M.VITREOUS_ICE, "Vitreous ice", 0.92, 300.0, 7.2, 4.5, 33.0, halos=((2.75, 0.65), (4.50, 0.20))),
    Material(M.SILICON_NITRIDE, "Silicon nitride", 3.17, 120.0, 10.0, 13.0, 95.0,
             halos=((3.20, 0.60), (5.40, 0.25))),
    _metal(M.COPPER, "Copper", "Cu", 225, 0.3615, 8.96, 65.0, 29.0, 20.0, 85.0, 0.55),
    _metal(M.GOLD, "Gold", "Au", 225, 0.4078, 19.3, 25.0, 79.0, 29.0, 59.0, 0.55),
    _metal(M.ALUMINUM, "Aluminum", "Al", 225, 0.4050, 2.70, 130.0, 13.0, 12.4, 60.0, 0.85),
    _metal(M.IRON, "Iron", "Fe", 229, 0.2866, 7.87, 60.0, 26.0, 20.0, 85.0, 0.35),
    _metal(M.SILICON, "Silicon", "Si", 227, 0.5431, 2.33, 145.0, 14.0, 12.1, 50.0, 0.45),
    _metal(M.PLATINUM, "Platinum", "Pt", 225, 0.3924, 21.45, 26.0, 78.0, 30.0, 66.0, 0.40),
    Material(M.PROTEIN, "Protein", 1.35, 400.0, 6.5, 7.5, 70.0, halos=((2.5, 0.6), (4.8, 0.2))),
    # MoO3's mean free path is scaled from aluminium's by the Z^(4/3)-weighted atom density.
    # Pnma setting (a is the layer-stacking axis; b the long axis of the laths)
    Material(M.MOLYBDENUM_TRIOXIDE, "Molybdenum trioxide", 4.69, 63.0, 30.0, 17.0, 78.5,
             Crystal(62, (1.3856, 0.3697, 0.3963, 90.0, 90.0, 90.0),
                     (("Mo", 0.1016, 0.25, 0.0867), ("O", 0.0866, 0.25, 0.5212),
                      ("O", 0.2214, 0.25, -0.0373), ("O", 0.4351, 0.25, 0.4994))), 0.5,
             halos=((3.0, 0.6), (5.0, 0.25))),
    # Bulk metals for EBSD (de_twin.specimen.bulk). Mean free paths scaled from copper's by density.
    _metal(M.NICKEL, "Nickel", "Ni", 225, 0.3524, 8.91, 65.0, 28.0, 21.0, 91.4, 0.37),
    _metal(M.AUSTENITE, "Austenite", "Fe", 225, 0.3591, 8.01, 72.0, 26.0, 20.0, 86.4, 0.45),
]
del M


def material(mid: int) -> Material:
    return MATERIALS[int(mid)]


def n_materials() -> int:
    return len(MATERIALS)


def register_material(**kw) -> Material:
    """Append a material (its ``id`` is assigned) and return it. Register before building a
    :class:`~de_twin.specimen.Specimen` so its grain table covers the new id."""
    if len(MATERIALS) >= 255:
        raise ValueError("material ids are uint8")
    m = Material(id=len(MATERIALS), **kw)
    MATERIALS.append(m)
    return m


def material_array(attr: str) -> np.ndarray:
    """Per-material float32 lookup array of one attribute (e.g. ``"z_eff"``)."""
    return np.array([getattr(m, attr) for m in MATERIALS], np.float32)


def absorption_lengths_nm(ht_kv: float) -> np.ndarray:
    """Per-material mean free path lookup array (inf for vacuum)."""
    return np.array([m.absorption_length_nm(ht_kv) for m in MATERIALS], dtype=np.float32)
