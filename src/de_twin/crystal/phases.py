"""orix ``Phase`` objects (with a diffpy ``Structure``) for the crystalline materials.

A phase is built from the material's :class:`~de_twin.specimen.materials.Crystal`
(space group + lattice + asymmetric unit, or a CIF) and *expanded to the full unit
cell* (``ReciprocalLatticeVector.sanitise_phase``), so structure factors and
``diffsims.SimulationGenerator`` see every atom. Amorphous materials have no phase;
they keep their halo description in the material table.
"""

from __future__ import annotations

import functools
import warnings

from ..specimen.materials import Material, material, register_material


def _expand(phase):
    from diffsims.crystallography import ReciprocalLatticeVector

    rlv = ReciprocalLatticeVector(phase, hkl=[[1, 0, 0]])
    rlv.sanitise_phase()
    return rlv.phase


def build_phase(m: Material):
    """Full-unit-cell orix Phase for a crystalline material (``None`` if amorphous)."""
    c = m.crystal
    if c is None:
        return None
    from diffpy.structure import Atom, Lattice, Structure
    from orix.crystal_map import Phase

    if c.cif:
        phase = Phase.from_cif(c.cif)
        phase.name = m.name
    else:
        a, b, cc, al, be, ga = c.lattice
        atoms = [Atom(el, [x, y, z]) for el, x, y, z in c.atoms]
        s = Structure(atoms=atoms, lattice=Lattice(a * 10, b * 10, cc * 10, al, be, ga))  # Angstrom
        phase = Phase(m.name, space_group=int(c.space_group), structure=s)
    return _expand(phase)


@functools.cache
def phase_for(material_id: int):
    """Cached full-unit-cell orix Phase of ``material_id`` (``None`` for amorphous/vacuum)."""
    with warnings.catch_warnings():  # diffpy.structure 3.x camelCase deprecations inside orix/diffsims
        warnings.simplefilter("ignore", DeprecationWarning)
        return build_phase(material(int(material_id)))


def register_cif(
    path: str, name: str | None = None, *, like: int = 5, debye_waller_a2: float = 0.5, **overrides
) -> int:
    """Register the phase in a CIF file as a new crystalline material and return its id.

    Non-diffraction properties (absorption length, mean inner potential, ...) are copied
    from the material ``like`` (default gold) unless given in ``overrides``.
    """
    from orix.crystal_map import Phase

    from ..specimen.materials import Crystal

    base = material(like)
    phase = Phase.from_cif(str(path))
    lat = phase.structure.lattice
    a = lat.a / 10.0
    material_fields = {
        "name": name or phase.name or str(path),
        "density_g_cm3": base.density_g_cm3,
        "absorption_length_200kv_nm": base.absorption_length_200kv_nm,
        "z_eff": base.z_eff,
        "mean_inner_potential_v": base.mean_inner_potential_v,
        "scatterer_density_nm3": base.scatterer_density_nm3,
        "crystal": Crystal(cif=str(path)),
        "debye_waller_a2": debye_waller_a2,
        "halos": ((1.0 / max(a, 1e-3) * 1.7, 0.65), (1.0 / max(a, 1e-3) * 2.0, 0.25)),
    }
    material_fields.update(overrides)
    return int(register_material(**material_fields).id)
