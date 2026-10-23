"""de_twin.crystal: the diffsims/orix library, rotation conventions, stage tilt, textures,
CIF registration and orientation ground truth."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from orix.quaternion import Rotation
from orix.vector import Vector3d

from de_twin.crystal import (
    crystal_to_lab,
    effective_matrices,
    library_for,
    phase_for,
    quat_to_matrix,
    rotation_from_crystal_to_lab,
    stage_matrix,
)
from de_twin.render.diffraction import bragg_fraction, bucket_patterns
from de_twin.render.testing import Particle, SyntheticSpecimen, on_zone_grains
from de_twin.specimen import Specimen, from_name
from de_twin.specimen.fieldmap import GrainTable
from de_twin.specimen.materials import GRAINS_PER_MATERIAL, MATERIALS, MaterialId

LAM200 = 0.0025079
AU = MaterialId.GOLD
OPTICS = SimpleNamespace(alpha_rad=0.0, beta_rad=0.0, wavelength_nm=LAM200, ht_kv=200.0, convergence_mrad=0.0)


# ---------------------------------------------------------- conventions
def test_quaternion_matrix_convention_matches_orix():
    r = Rotation.random(20)
    assert np.allclose(quat_to_matrix(r.data), r.to_matrix())
    v = np.array([0.3, -1.2, 0.7])
    lab = (~r * Vector3d(v)).data  # diffsims: a crystal vector appears in the lab as ~R * v
    assert np.allclose(crystal_to_lab(r.data) @ v, lab)
    q = rotation_from_crystal_to_lab(crystal_to_lab(r.data))
    assert np.allclose(np.abs(np.sum(q * r.data, axis=1)), 1.0)  # same rotation (up to sign)


def test_stage_axes_match_the_view_foreshortening():
    s = stage_matrix(math.radians(30), 0.0)  # alpha: about x -> y foreshortened by cos(alpha)
    assert np.allclose(s @ [1, 0, 0], [1, 0, 0])
    assert np.isclose((s @ [0, 1, 0])[1], math.cos(math.radians(30)))
    s = stage_matrix(0.0, math.radians(20))  # beta: about y -> x foreshortened by cos(beta)
    assert np.allclose(s @ [0, 1, 0], [0, 1, 0])
    assert np.isclose((s @ [1, 0, 0])[0], math.cos(math.radians(20)))


@pytest.mark.parametrize("mid", [MaterialId.GOLD, MaterialId.SILICON, MaterialId.IRON])
def test_library_matches_diffsims(mid):
    """Same spot positions, hkl and (up to the kinematic prefactor) intensities as diffsims'
    SimulationGenerator for the same rotation, with a sinc^2 shape factor."""
    from diffsims.generators.simulation_generator import SimulationGenerator

    lib, phase = library_for(mid), phase_for(mid)
    t = 25.0
    gen = SimulationGenerator(200, shape_factor_model=lambda s, w: np.sinc(t * 10 * s) ** 2,
                              minimum_intensity=1e-12)
    q = np.random.default_rng(int(mid)).normal(size=(3, 4))
    for qi in q / np.linalg.norm(q, axis=1, keepdims=True):
        rot = Rotation(qi)
        sim = gen.calculate_diffraction2d(phase, rotation=rot, reciprocal_radius=2.0, max_excitation_error=0.05,
                                          with_direct_beam=False,
                                          debye_waller_factors={phase.structure[0].element:
                                                                MATERIALS[mid].debye_waller_a2})
        c = sim.coordinates
        xy = c.data[:, :2] * 10.0  # 1/A -> 1/nm
        ex = lib.excite(crystal_to_lab(rot.data), gen.wavelength / 10.0, t, 200.0)
        assert len(xy) > 3
        d = np.hypot(ex.gx[None, :] - xy[:, 0, None], ex.gy[None, :] - xy[:, 1, None])
        j = d.argmin(axis=1)
        assert d[np.arange(len(j)), j].max() < 1e-9
        # hkl: invert the lab vector back to the crystal frame (cubic: hkl = a g_crystal)
        a = phase.structure.lattice.a / 10.0
        g_cryst = crystal_to_lab(rot.data)[0].T @ (c.data * 10.0).T
        assert np.array_equal(np.rint(a * g_cryst.T).astype(int), lib.hkl[ex.index[j]])
        ratio = ex.intensity[j] / c.intensity
        assert np.ptp(ratio) < 1e-6 * ratio.mean()


def test_identity_is_the_001_zone_axis():
    ex = library_for(AU).excite(np.eye(3)[None], LAM200, 10.0, 200.0)
    strong = ex.intensity > 1e-3 * ex.intensity.max()
    assert np.all(library_for(AU).hkl[ex.index[strong], 2] == 0)  # only the ZOLZ (hk0) is excited


def test_extinction_distance_is_sensible():
    xi = library_for(AU).extinction_distance_nm((1, 1, 1), 200.0, LAM200)
    assert 15.0 < xi < 30.0  # Au(111) at 200 kV: ~18-25 nm depending on the scattering factors


# -------------------------------------------------------------- tilt
def test_tilted_grain_equals_pre_rotated_grain():
    """Pattern for (grain R, stage tilt T) == pattern for (grain T.R, no tilt)."""
    gid = AU * GRAINS_PER_MATERIAL + 11
    grains = GrainTable.generate(42)
    tilt = SimpleNamespace(**{**vars(OPTICS), "alpha_rad": math.radians(7.0), "beta_rad": math.radians(-4.0)})
    a = bucket_patterns([AU], [gid], [18.0], [1.0], grains, tilt)[0]
    m = stage_matrix(tilt.alpha_rad, tilt.beta_rad) @ grains.matrices[gid]
    q = grains.quaternions.copy()
    q[gid] = rotation_from_crystal_to_lab(m)[0]
    b = bucket_patterns([AU], [gid], [18.0], [1.0], GrainTable(q, grains.nucleation_u), OPTICS)[0]
    assert np.allclose(a.spots, b.spots, atol=1e-9) and np.allclose(a.rings, b.rings)
    assert len(a.spots) > 1


def test_stage_tilt_moves_a_grain_continuously_through_bragg_conditions():
    gid = AU * GRAINS_PER_MATERIAL + 1
    grains = on_zone_grains(gid, (0, 1, 1), in_plane_rad=0.3)
    lib = library_for(AU)

    def bragg(alpha_deg):
        m = effective_matrices(grains.matrices[[gid]], math.radians(alpha_deg), 0.0)
        return float(bragg_fraction(lib.excite(m, LAM200, 15.0, 200.0, 0.05).total[0]))

    coarse = np.array([bragg(a) for a in np.arange(0.0, 10.01, 0.5)])
    assert np.ptp(coarse) > 0.3  # on zone -> off zone -> through other Bragg conditions
    fine = np.array([bragg(a) for a in np.arange(0.0, 10.001, 0.02)])
    assert np.abs(np.diff(fine)).max() < 0.1 * np.ptp(fine)  # no jumps
    assert np.allclose(fine[::25], coarse)


# ------------------------------------------------------------ textures
def test_default_grain_orientations_are_uniform():
    m = GrainTable.generate(7).matrices[AU * 512:(AU + 1) * 512]
    c = np.abs((m @ [0.0, 0.0, 1.0])[:, 2])  # |cos| of [001] with the beam: uniform on [0, 1]
    assert abs(c.mean() - 0.5) < 0.05 and abs((c ** 2).mean() - 1 / 3) < 0.05


def test_thin_film_has_a_111_fibre_texture():
    s = Specimen(from_name("Au thin film 20 nm"))
    m = s.grains.matrices[AU * 512:(AU + 1) * 512]
    tilt = np.degrees(np.arccos(np.abs((m @ (np.ones(3) / math.sqrt(3)))[:, 2])))
    assert np.median(tilt) < 5.0 and tilt.max() < 20.0
    azimuth = np.arctan2(*(m @ [1.0, 0.0, 0.0])[:, :2].T)
    assert np.ptp(azimuth) > 5.0  # random rotation about the fibre axis
    # nanoparticles stay random
    p = Specimen(from_name("Dense Au on holey C")).grains.matrices[AU * 512:(AU + 1) * 512]
    assert np.median(np.degrees(np.arccos(np.abs((p @ (np.ones(3) / math.sqrt(3)))[:, 2])))) > 20.0


# ------------------------------------------------------------- phases
def test_phases_and_cif_registration(tmp_path):
    from de_twin.crystal import library as library_mod
    from de_twin.crystal import phases

    au = phase_for(AU)
    assert len(au.structure) == 4 and au.space_group.number == 225  # full unit cell
    assert len(phase_for(MaterialId.SILICON).structure) == 8
    assert phase_for(MaterialId.AMORPHOUS_CARBON) is None
    cif = tmp_path / "cu.cif"
    phase_for(MaterialId.COPPER).structure.write(str(cif), "cif")
    n = len(MATERIALS)
    try:
        mid = phases.register_cif(str(cif), "Copper (CIF)", like=MaterialId.COPPER)
        assert mid == n and MATERIALS[mid].crystalline
        lib, ref = library_for(mid), library_for(MaterialId.COPPER)
        assert lib.g.shape == ref.g.shape
        assert np.allclose(np.sort(lib.g_len), np.sort(ref.g_len))
    finally:
        del MATERIALS[n:]
        phases.phase_for.cache_clear()
        library_mod.library_for.cache_clear()


def test_template_library_for_orientation_mapping():
    sim = library_for(AU).template_library(resolution_deg=8.0)
    assert sim.rotations.size > 5
    assert len(sim.coordinates) == sim.rotations.size


# ------------------------------------------------------- ground truth
def test_ground_truth_crystal_map_includes_stage_tilt():
    from de_twin.clock import ManualClock
    from de_twin.twin import DigitalTwin

    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM, particles=[Particle(0.0, 0.0, 12.0, grain=3)])
    twin = DigitalTwin(spec, camera="DE16", clock=ManualClock())
    twin.column.set("TemStemMode", 1)
    req = twin.request()
    req.scan.enabled = True
    req.scan.size = (16, 16)
    req.scan.step_um = 0.002
    gid = AU * GRAINS_PER_MATERIAL + 3
    for alpha in (0.0, 12.0):
        twin.column.move_stage(alpha=alpha)
        xmap = twin.ground_truth(req)
        assert xmap.shape == (16, 16)
        au = xmap.phase_id == AU
        assert au.any() and (~au).any()
        assert set(np.unique(xmap.prop["grain_id"][au])) == {gid}
        expect = rotation_from_crystal_to_lab(effective_matrices(spec.grains.matrices[[gid]],
                                                                 math.radians(alpha), 0.0))
        got = xmap[au].rotations.data
        assert np.allclose(np.abs(got @ expect[0]), 1.0)
    twin.close()
