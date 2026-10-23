"""User EBSD master patterns: EMsoft / kikuchipy HDF5 and NumPy input, round trips against the
kinematic master, hemispheres, energy selection and validation."""

from __future__ import annotations

import numpy as np
import pytest

from de_twin import DigitalTwin
from de_twin.ebsd import kernels, master_for
from de_twin.ebsd.io import from_array, read_emsoft, write_emsoft
from de_twin.ebsd.master import lambert_sample
from de_twin.specimen.materials import MaterialId
from de_twin.state import ScanRequest

KV = 20.0
NI_LATTICE = (0.3524, 0.3524, 0.3524, 90.0, 90.0, 90.0)


def _twin():
    twin = DigitalTwin("EBSD Ni polycrystal", camera="Celeritas")
    twin.configure_ebsd(beam_kv=KV)
    request = twin.request(
        hw_binning=(4, 4),
        frame_time_s=1e-3,
        total_frames=1,
        scan=ScanRequest(enabled=True, size=(3, 2), step_um=4.0, dwell_s=1e-3),
    )
    return twin, request


def _raws(twin, request):
    """The noiseless projected master values at every scan point (fresh plan each call)."""
    o = twin.optics(request)
    r = twin.renderer.ebsd
    return np.stack([r.raw_pattern(twin.specimen, o, ix, iy) for iy in range(2) for ix in range(3)])


def _corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _same(reference, got, corr=0.998, resid=0.07):
    """Every pattern pair correlates above ``corr`` and differs, after an affine fit (a user
    master is restandardised), by less than ``resid`` of the pattern's std. The residual is the
    smoothing of resampling the 0.15 deg Lambert grid into the cube map."""
    for a, b in zip(reference, got):
        A = np.vstack([b.ravel(), np.ones(b.size)]).T
        coef, *_ = np.linalg.lstsq(A, a.ravel(), rcond=None)
        assert _corr(a, b) > corr
        assert np.std(a.ravel() - A @ coef) < resid * np.std(a)
    return True


def _kinematic_north(twin):
    return master_for(MaterialId.NICKEL, KV, twin.ebsd).values("north").astype(np.float64)


def test_emsoft_round_trip_reproduces_the_kinematic_patterns(tmp_path):
    twin, request = _twin()
    reference = _raws(twin, request)
    path = tmp_path / "Ni-master-20kV.h5"
    write_emsoft(path, _kinematic_north(twin), energies=[KV], space_group=225, lattice=NI_LATTICE)
    master = twin.set_master_pattern("Nickel", path)
    assert (
        master.kv == KV and master.south is None and master.cube.shape[0] == 3
    )  # recognised as centrosymmetric
    assert master.symmetry_error < 0.05 and master.nbytes < 3e6
    assert _same(reference, _raws(twin, request))
    # forced through both hemispheres (six cube faces): the same patterns
    mp2 = twin.set_master_pattern("Nickel", path, centrosymmetric=False)
    assert mp2.south is not None and mp2.cube.shape[0] == 6 and mp2.nbytes < 6e6
    assert _same(reference, _raws(twin, request))
    assert twin.master_patterns()[0]["hemispheres"] == 2
    twin.clear_master_patterns()
    assert np.allclose(_raws(twin, request), reference)


def test_southern_hemisphere_is_read_where_emsoft_puts_it():
    """A non-centrosymmetric master (a bright spot on the southern hemisphere only): the kernel's
    pattern equals the numpy reference read at Lambert(x, y, |z|) of the southern array."""
    twin, request = _twin()
    north = _kinematic_north(twin)
    n = north.shape[0]
    south = north[::-1, ::-1].copy()
    spot = np.array([0.3, -0.2, -0.93])
    spot /= np.linalg.norm(spot)
    d = kernels.lambert_directions(n) * np.array([1.0, 1.0, -1.0])  # southern directions of each node
    south += 4.0 * np.exp(-(1.0 - d @ spot) / 0.002)  # ~4 deg spot
    master = twin.set_master_pattern(
        "Nickel", np.stack([north, south]), hemisphere="both", kv=KV, validate=False
    )
    assert master.south is not None
    o = twin.optics(request)
    r = twin.renderer.ebsd
    g = twin.ebsd
    # an orientation that puts the spot direction at the pattern centre
    from de_twin.crystal.orientation import align

    pc = g.pattern_centre()
    G = align(spot, g.sample_to_detector().T @ np.array([0.0, 0.0, 1.0]))
    image = r.pattern(o, G, pc, material_id=MaterialId.NICKEL, raw=True)
    h, w = image.shape
    H, W = o.extras["sensor_shape"]
    ys, xs = np.mgrid[0:h, 0:w]
    rd = np.stack([(xs + 0.5) * 4 - pc[0] * W, (ys + 0.5) * 4 - pc[1] * H, np.full((h, w), pc[2] * H)], -1)
    c = rd @ (g.sample_to_detector() @ G)  # crystal directions: G^T S2D^T r
    want = master.value(c)
    r0, c0 = int(pc[1] * H / 4), int(pc[0] * W / 4)  # the pattern centre on the 4x-binned grid
    centre = image[r0 - 4 : r0 + 4, c0 - 4 : c0 + 4]
    assert centre.mean() > image.mean() + 2.0  # the spot is there
    assert np.abs(image - want).max() < 0.1 * (want.max() - want.min())
    # and the centrosymmetric fold would not have it
    assert np.abs(want - lambert_sample(master.values("north"), None, c)).max() > 1.0


def test_energy_selection_and_numsets(tmp_path):
    import h5py

    n = 33
    path = tmp_path / "multi.h5"
    slices = np.stack([np.full((n, n), v, np.float32) for v in (1.0, 2.0, 4.0)])  # 10, 15, 20 kV
    with h5py.File(path, "w") as f:
        g = f.create_group("EMData/EBSDmaster")
        g["mLPNH"] = np.stack([slices, slices])  # numset = 2: summed
        g["EkeVs"] = np.array([10.0, 15.0, 20.0], np.float32)
        f["EMData/MCOpenCL/accum_e"] = np.array([1, 1, 2], np.int32).reshape(3, 1, 1)
    assert read_emsoft(path, energy_kv=15.0)["north"][0, 0] == pytest.approx(4.0)
    assert read_emsoft(path, energy_kv=17.5)["north"][0, 0] == pytest.approx(6.0)  # interpolated
    weighted = read_emsoft(path)
    assert weighted["north"][0, 0] == pytest.approx(2 * (1 + 2 + 2 * 4) / 4) and weighted["kv"] == 20.0
    assert weighted["south"] is None
    a = from_array(slices, energies=[10, 15, 20], energy_kv=20)
    assert a["north"][0, 0] == 4.0 and a["kv"] == 20.0


def test_kikuchipy_layout(tmp_path):
    import h5py

    twin, request = _twin()
    reference = _raws(twin, request)
    north = _kinematic_north(twin)
    path = tmp_path / "ni_mp.hspy"
    with h5py.File(path, "w") as f:
        e = f.create_group("Experiments/ni_mp")
        e["data"] = np.stack([north, north[::-1, ::-1]])[:, None].astype(
            np.float32
        )  # (hemisphere, energy, y, x)
        for i, (name, off) in enumerate(
            (("hemisphere", 0.0), ("energy", KV), ("height", 0.0), ("width", 0.0))
        ):
            ax = e.create_group(f"axis-{i}")
            ax.attrs["name"], ax.attrs["offset"], ax.attrs["scale"] = name, off, 1.0
        md = e.create_group("metadata/Simulation/EBSD_master_pattern")
        md.attrs["projection"], md.attrs["hemisphere"] = "lambert", "both"
    master = twin.set_master_pattern(MaterialId.NICKEL, path)
    assert master.kv == KV and master.south is None
    assert _same(reference, _raws(twin, request))


def test_stereographic_array():
    twin, request = _twin()
    reference = _raws(twin, request)
    north = _kinematic_north(twin)
    m = 801
    c = np.linspace(-1, 1, m)
    X, Y = np.meshgrid(c, c)
    r2 = X**2 + Y**2
    d = np.stack([2 * X, 2 * Y, 1 - r2], -1) / (1 + r2)[..., None]  # inverse stereographic, z >= 0
    st = lambert_sample(north, None, d)
    twin.set_master_pattern("Nickel", st, projection="stereographic", kv=KV)
    assert _same(reference, _raws(twin, request), corr=0.99, resid=0.13)


def test_validation(tmp_path):
    twin, _ = _twin()
    north = _kinematic_north(twin)
    with pytest.raises(ValueError, match="invariant"):
        twin.set_master_pattern("Nickel", np.roll(north, north.shape[0] // 7, axis=1), kv=KV)
    path = tmp_path / "bcc.h5"
    write_emsoft(path, north, energies=[KV], space_group=229, lattice=NI_LATTICE)
    with pytest.raises(ValueError, match="space group"):
        twin.set_master_pattern("Nickel", path)
    with pytest.raises(KeyError):
        twin.set_master_pattern("Unobtainium", north, kv=KV)
    # a master for another energy is not used at this one
    twin.set_master_pattern("Nickel", north, kv=30.0)
    r = twin.renderer.ebsd
    assert r.master(MaterialId.NICKEL, KV, twin.ebsd).source == "kinematic"
    assert r.master(MaterialId.NICKEL, 30.0, twin.ebsd).source == "array"


def test_deapi_master_pattern_file(tmp_path):
    deapi = pytest.importorskip("deapi")
    from de_twin.faces.deapi_server import TwinDeapiServer

    twin, _ = _twin()
    path = tmp_path / "ni.h5"
    write_emsoft(path, _kinematic_north(twin), energies=[KV], space_group=225, lattice=NI_LATTICE)
    with TwinDeapiServer(twin, port=0, pace=False) as srv:
        c = deapi.Client()
        c.usingMmf = False
        c.connect(port=srv.port)
        c["Simulator EBSD Master Pattern Phase"] = "Nickel"
        c["Simulator EBSD Master Pattern File"] = str(path)
        status = c["Simulator EBSD Master Pattern Status"]
        c["Simulator EBSD Master Pattern File"] = str(tmp_path / "missing.h5")
        bad = c["Simulator EBSD Master Pattern Status"]
        c.disconnect()
    assert status.startswith("Nickel") and "20 kV" in status
    assert bad.startswith("error")
    assert twin.master_patterns()[0]["source"] == str(path)
