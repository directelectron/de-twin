import numpy as np
import pytest

from de_twin.specimen import PATTERNS, PRESETS, MaterialId, Specimen, SpecimenConfig, ViewWindow, from_name
from de_twin.specimen.fieldmap import LAYER_DESCAN, LAYER_STRAIN
from de_twin.specimen.materials import GRAINS_PER_MATERIAL, N_MATERIALS

ALL_LAYERS = frozenset({LAYER_DESCAN, LAYER_STRAIN})


def _feature_view(s: Specimen, pixel_um: float, n: int = 256, **kw) -> ViewWindow:
    f = s.features()[0]
    return ViewWindow(center_um=f.center_um, pixel_um=pixel_um, shape=(n, n), **kw)


def _check_fieldmap(field_map, s):
    assert field_map.material_id.dtype == np.uint8 and field_map.material_id.max() < N_MATERIALS
    assert field_map.thickness_nm.dtype == np.float32
    assert np.all(np.isfinite(field_map.thickness_nm)) and field_map.thickness_nm.min() >= 0
    g = field_map.grain_id
    assert g.dtype == np.int32 and g.min() >= -1 and g.max() < N_MATERIALS * GRAINS_PER_MATERIAL
    # empty pixels are vacuum
    assert np.all(field_map.thickness_nm[field_map.material_id == 0] < 1.0)


def test_same_seed_identical_different_seed_differs():
    config = from_name("Dense Au on holey C", seed=11)
    a, b = Specimen(config), Specimen(config)
    v = _feature_view(a, 0.004, 256)
    fa, fb = a.rasterize(v, ALL_LAYERS), b.rasterize(v, ALL_LAYERS)
    assert np.array_equal(fa.material_id, fb.material_id)
    assert np.array_equal(fa.thickness_nm, fb.thickness_nm)
    assert np.array_equal(fa.grain_id, fb.grain_id)
    # re-rasterising (resident cache hit) reproduces byte for byte
    assert np.array_equal(a.rasterize(v).thickness_nm, fa.thickness_nm)
    c = Specimen(from_name("Dense Au on holey C", seed=12))
    fc = c.rasterize(v)
    assert not np.array_equal(fa.thickness_nm, fc.thickness_nm)


@pytest.mark.parametrize("name", ["Dense Au on holey C", "Au thin film 20 nm", "Apoferritin in ice"])
def test_shifted_view_equals_shifted_raster(name):
    s = Specimen(from_name(name))
    px = 0.004
    v0 = _feature_view(s, px, 256)
    k_r, k_c = 23, -41
    v1 = ViewWindow(
        center_um=(v0.center_um[0] + k_c * px, v0.center_um[1] + k_r * px), pixel_um=px, shape=v0.shape
    )
    a, b = s.rasterize(v0), s.rasterize(v1)
    # pixel (i, j) of view 1 is pixel (i + k_r, j + k_c) of view 0
    A = a.thickness_nm[k_r:, :k_c]
    B = b.thickness_nm[:-k_r, -k_c:]
    same = np.isclose(A, B, atol=0.05)
    assert same.mean() > 0.995
    assert (a.material_id[k_r:, :k_c] == b.material_id[:-k_r, -k_c:]).mean() > 0.995


def test_rotation_and_tilt_views_render():
    s = Specimen(from_name("Dense Au on holey C"))
    v = _feature_view(s, 0.004, 128, rotation_rad=0.7, flip_x=True, cos_alpha=0.8)
    _check_fieldmap(s.rasterize(v, ALL_LAYERS), s)


@pytest.mark.parametrize("name", list(PATTERNS) + list(PRESETS))
def test_every_pattern_and_preset_renders(name):
    s = Specimen(from_name(name))
    ov = s.overview((128, 128))
    # bars / rim / FIB body are opaque metal; the authored chip is SiN membrane + Pt heater
    assert ov.shape == (128, 128) and ov.max() > (100 if s.config.holder == "insitu_heating_chip" else 1000)
    for px in (0.5, 0.01, 0.002):
        field_map = s.rasterize(_feature_view(s, px, 128), ALL_LAYERS)
        _check_fieldmap(field_map, s)
        assert field_map.descan is not None and field_map.strain is not None


EXPECTED_MATERIAL = {
    "Dense Au on holey C": MaterialId.GOLD,
    # 8 nm blobs never out-thicken 60 nm of ice, so (as in the C++ claim rule) the pixels stay
    # labelled ice and the proteins are pure thickness contrast; see the dedicated test below.
    "Apoferritin in ice": MaterialId.VITREOUS_ICE,
    "Negative stain on carbon": MaterialId.PLATINUM,
    "Au thin film 20 nm": MaterialId.GOLD,
    "Al thin film 100 nm": MaterialId.ALUMINUM,
    "Steel lamella with carbides": MaterialId.IRON,
    "PN junction lamella": MaterialId.SILICON,
    "Droplet crystallization": MaterialId.GOLD,
}


@pytest.mark.parametrize("name,mat", list(EXPECTED_MATERIAL.items()))
def test_feature_views_contain_the_specimen(name, mat):
    s = Specimen(from_name(name))
    field_map = s.rasterize(_feature_view(s, 0.002, 256))
    assert np.count_nonzero(field_map.material_id == mat) > 20
    t = field_map.thickness_nm[field_map.material_id != 0]
    assert 1.0 < float(np.median(t)) < 2000.0  # electron-transparent where the specimen is


def test_mesh_grid_statistics():
    s = Specimen(SpecimenConfig(seed=3, holder="mesh_grid", preparation="nanoparticles", film="continuous"))
    ov = s.rasterize(ViewWindow(center_um=(0.0, 0.0), pixel_um=3.2, shape=(1024, 1024)))
    cu = ov.material_id == MaterialId.COPPER
    disk = np.hypot(*np.meshgrid((np.arange(1024) - 511.5) * 3.2, (np.arange(1024) - 511.5) * 3.2)) < 1300
    # 200 mesh: bar fraction 1 - (90/125)^2 ~ 0.48 inside the rim
    assert 0.35 < cu[disk].mean() < 0.62
    # torn fragments double it; aggregate particle rects add a few nm at bar edges
    assert np.isclose(ov.thickness_nm[cu], 20000.0, atol=50.0).mean() > 0.99
    holes = disk & ~cu
    film = np.median(ov.thickness_nm[holes & (ov.material_id == MaterialId.AMORPHOUS_CARBON)])
    assert 12.0 < film < 20.0  # continuous 12 nm + aggregated particle mass


def test_optional_layers_from_structures():
    s = Specimen(from_name("PN junction lamella"))
    field_map = s.rasterize(_feature_view(s, 0.02, 256), ALL_LAYERS)
    si = field_map.material_id == MaterialId.SILICON
    assert np.ptp(field_map.descan[1][si]) > 3.0  # the 4 px junction step in y
    s = Specimen(from_name("Strained inclusion lamella"))
    field_map = s.rasterize(_feature_view(s, 0.02, 256), ALL_LAYERS)
    assert field_map.strain[0].max() == pytest.approx(1.02, abs=1e-3)
    assert field_map.strain[2].min() == pytest.approx(0.99, abs=1e-3)
    # every lamella matrix grain is a single crystal close to a low-index zone ([110] for Si)
    g = field_map.grain_id
    ok = g >= 0
    assert ok.any()
    si = np.unique(g[ok & (field_map.material_id == MaterialId.SILICON)])
    assert len(si)
    zone = s.grains.matrices[si] @ (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))
    assert np.all(np.degrees(np.arccos(np.abs(zone[:, 2]))) < 3.0)
    assert "descan" in Specimen(from_name("PN junction lamella")).required_layers()


def test_fib_posts_have_distinct_kinds_and_features():
    s = Specimen(from_name("FIB Liftout"))
    feats = s.features()
    assert len(feats) == 4
    assert len({f.kind for f in feats}) == 4
    assert feats[0].label.startswith("post 0 lamella 0")
    near = s.nearest_feature(*feats[2].center_um)
    assert near.post_index == 2


def test_nearest_cluster_feature_on_mesh():
    s = Specimen(from_name("Dense Au on holey C"))
    f = s.nearest_feature(10.0, -20.0)
    assert f is not None and f.label.endswith("cluster") and f.weight > 0
    chip = Specimen(from_name("Droplet crystallization"))
    assert all(ft.label.startswith("window") for ft in chip.features())
    assert len(chip.features()) == 10


def test_bounds_and_overview_extent():
    s = Specimen(from_name("Dense Au on holey C"))
    xmin, _ymin, xmax, _ymax = s.bounds_um()
    assert xmin == pytest.approx(-1525.0) and xmax == pytest.approx(1525.0)
    ov = s.overview((64, 64))
    assert ov[0, 0] == 0 and ov[32, 32] >= 0


def test_protein_blobs_are_thickness_contrast_in_ice():
    s = Specimen(from_name("Apoferritin in ice"))
    field_map = s.rasterize(_feature_view(s, 0.001, 256))
    ice = field_map.thickness_nm[field_map.material_id == MaterialId.VITREOUS_ICE]
    assert ice.size > 0.9 * field_map.thickness_nm.size
    # blobs of 0.7 * 12 nm peak projected thickness on ~60 nm of ice (plus edge thickening)
    assert 3.0 < np.percentile(ice, 99) - np.percentile(ice, 5) < 25.0
    stain = Specimen(from_name("Negative stain on carbon"))
    field_map = stain.rasterize(_feature_view(stain, 0.001, 256))
    assert (
        np.count_nonzero(field_map.material_id == MaterialId.PROTEIN) > 100
    )  # blobs out-thicken 12 nm carbon


@pytest.mark.parametrize(
    "name",
    [
        "Dense Au on holey C",
        "Al thin film 100 nm",
        "FIB Liftout",
        "Droplet crystallization",
        "PN junction lamella",
    ],
)
def test_rotated_view_matches_unrotated_statistics(name):
    s = Specimen(from_name(name))
    f = s.features()[0]
    a = s.rasterize(ViewWindow(center_um=f.center_um, pixel_um=0.05, shape=(256, 256)))
    b = s.rasterize(
        ViewWindow(center_um=f.center_um, pixel_um=0.05, shape=(256, 256), rotation_rad=np.pi / 2)
    )
    # a quarter turn samples the same world patch: same material make-up and mean thickness
    ha = np.bincount(a.material_id.ravel(), minlength=16) / a.material_id.size
    hb = np.bincount(b.material_id.ravel(), minlength=16) / b.material_id.size
    assert np.abs(ha - hb).max() < 0.02
    assert (
        abs(float(a.thickness_nm.mean()) - float(b.thickness_nm.mean()))
        <= 0.02 * float(a.thickness_nm.mean()) + 0.5
    )
