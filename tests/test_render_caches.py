"""Caches of the live-view path (stage moves, magnification steps) give the image a fresh
render gives."""

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.crystal import effective_matrices, library_for
from de_twin.render import tem, tem_contrast
from de_twin.render.diffraction import THICKNESS_BIN_NM, bragg_fraction, thickness_bin_for
from de_twin.specimen.materials import GRAINS_PER_MATERIAL
from de_twin.twin import DigitalTwin


def _dense_loss(field_map, optics, grains, crystallinity, config):
    """The Bragg loss per pixel from a dense (every grain x every thickness bin) excitation."""
    t = field_map.thickness_nm.astype(np.float32) * np.float32(optics.thickness_tilt_factor)
    gid = field_map.grain_id.astype(np.int64)
    gid = np.where(
        (gid >= 0) & (gid < len(grains)) & (gid // GRAINS_PER_MATERIAL == field_map.material_id), gid, -1
    )
    loss = np.zeros(gid.shape, np.float32)
    rows, cols = np.nonzero(gid >= 0)
    g_u, g_inv = np.unique(gid[rows, cols], return_inverse=True)
    tb_u, t_inv = np.unique(thickness_bin_for(t[rows, cols]), return_inverse=True)
    cr = np.asarray(crystallinity(g_u), float)
    lam = optics.wavelength_nm
    g_obj = optics.objective_aperture_mrad / (1000.0 * lam) if optics.objective_aperture_mrad > 0 else 0.0
    table = np.zeros((len(g_u), len(tb_u)))
    for mid in np.unique(g_u // GRAINS_PER_MATERIAL):
        sel = np.flatnonzero(g_u // GRAINS_PER_MATERIAL == mid)
        lib = library_for(int(mid), config.max_g_inv_nm)
        if lib is None:
            continue
        m = effective_matrices(grains.matrices[g_u[sel]], optics.alpha_rad, optics.beta_rad)
        ex = lib.excite(
            m,
            lam,
            np.broadcast_to(tb_u * THICKNESS_BIN_NM, (len(sel), len(tb_u))),
            optics.ht_kv,
            optics.convergence_mrad,
        )
        out = np.hypot(ex.gx, ex.gy) > g_obj
        p_out = np.stack(
            [
                np.bincount(ex.owner, weights=ex.intensity[:, j] * out, minlength=len(sel))
                for j in range(len(tb_u))
            ],
            1,
        )
        table[sel] = cr[sel, None] * bragg_fraction(ex.total) / np.maximum(ex.total, 1e-300) * p_out
    loss[rows, cols] = config.diffraction_contrast_scale * table[g_inv, t_inv]
    return loss


GRATING = "Ted Pella 607 - 2160 l/mm grating replica (waffle)"


@pytest.mark.parametrize("name", ["Dense Au on holey C", GRATING])
def test_the_bragg_memo_is_the_dense_excitation_in_any_order(name):
    twin = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=1)
    request = twin.request()
    r = twin.renderer
    tem_contrast._BRAGG_MEMOS.clear()
    views = []
    for magnification in (25000.0, 20000.0, 300000.0):
        twin.column.set("Magnification", magnification)
        o = twin.optics(request)
        views.append((o, r.field_map(o)[0]))
    first = [
        tem_contrast.bragg_contrast(field_map, o, r.grains, r.crystallinity, r.config)
        for o, field_map in views
    ]
    assert tem_contrast._BRAGG_MEMOS, "the excitations are kept"
    tem_contrast._BRAGG_MEMOS.clear()
    again = [
        tem_contrast.bragg_contrast(field_map, o, r.grains, r.crystallinity, r.config)
        for o, field_map in reversed(views)
    ][::-1]
    for (o, field_map), a, b in zip(views, first, again):
        np.testing.assert_array_equal(a.loss, b.loss)
        assert a.fringes.keys() == b.fringes.keys()
        for k in a.fringes:
            for x, y in zip(a.fringes[k], b.fringes[k]):
                np.testing.assert_array_equal(x, y)
        np.testing.assert_allclose(
            a.loss, _dense_loss(field_map, o, r.grains, r.crystallinity, r.config), rtol=1e-6, atol=1e-9
        )


def test_a_stage_tilt_is_a_new_excitation():
    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=1)
    request = twin.request()
    r = twin.renderer
    o = twin.optics(request)
    field_map = r.field_map(o)[0]
    a = tem_contrast.bragg_contrast(field_map, o, r.grains, r.crystallinity, r.config)
    twin.column.move_stage(alpha=4.0)
    o2 = twin.optics(request)
    b = tem_contrast.bragg_contrast(field_map, o2, r.grains, r.crystallinity, r.config)
    assert not np.array_equal(a.loss, b.loss)
    np.testing.assert_allclose(
        b.loss, _dense_loss(field_map, o2, r.grains, r.crystallinity, r.config), rtol=1e-6, atol=1e-9
    )


# ------------------------------------------------------------------ raster reuse


@pytest.fixture
def make_render_twin(make_twin):
    """A twin whose ``RenderConfig`` takes the keyword options."""
    from de_twin.render.config import RenderConfig

    def build(name="Dense Au on holey C", **render_options):
        return make_twin(name, seed=3, render_config=RenderConfig(**render_options))

    return build


def _move(twin, dx_um, dy_um=0.0):
    s = twin.column.state().stage
    twin.column.move_stage(x=s.x_um + dx_um, y=s.y_um + dy_um)


@pytest.mark.parametrize("name", ["Dense Au on holey C", GRATING, "Negative stain on carbon"])
@pytest.mark.parametrize("dx,dy", [(0.3, 0.0), (0.0, -0.4), (-0.35, 0.3)])
def test_a_reused_field_map_is_the_fresh_raster(name, dx, dy, make_render_twin):
    twin = make_render_twin(name)
    request = twin.request()
    twin.flux(request)
    n = twin.optics(request).view.shape[0]
    px = twin.optics(request).view.pixel_um
    _move(twin, dx * n * px, dy * n * px)
    before = twin.renderer.rasters_reused
    twin.flux(request)
    assert twin.renderer.rasters_reused == before + 1, "the move reused the last field map"
    field_map = next(reversed(twin.renderer._fieldmaps.values()))[0]
    _same_raster(field_map, twin.specimen.rasterize(field_map.view))


def _same_raster(field_map, ref):
    """The same pixels, up to the rounding of world coordinates (a pixel centre a few ulp
    from an edge may land on the other side of it) and the painter's order where two
    particles overlap: the scene paints particles window by window, and which one claims
    an overlap pixel can depend on how a view's windows were grouped (below 1e-3 of the
    pixels, grain ids only, in the densest scenes)."""
    n = field_map.material_id.size
    assert np.count_nonzero(field_map.material_id != ref.material_id) <= 1e-4 * n
    assert np.count_nonzero(field_map.grain_id != ref.grain_id) <= 1e-3 * n
    assert (
        np.count_nonzero(~np.isclose(field_map.thickness_nm, ref.thickness_nm, rtol=1e-5, atol=1e-2))
        <= 1e-4 * n
    )
    assert (field_map.under_thickness_nm is None) == (
        ref.under_thickness_nm is None
    ) or not ref.under_thickness_nm.any()
    if ref.under_thickness_nm is not None and field_map.under_thickness_nm is not None:
        assert (
            np.count_nonzero(
                ~np.isclose(field_map.under_thickness_nm, ref.under_thickness_nm, rtol=1e-5, atol=1e-2)
            )
            <= 2e-5 * n
        )


@pytest.mark.parametrize("rotation", [0.0, 0.5 * np.pi])
def test_a_turned_flipped_tilted_view_reuses_on_its_own_lattice(rotation, make_render_twin):
    import dataclasses

    from de_twin.render import Renderer

    twin = make_render_twin()
    o = twin.optics(twin.request())
    v = dataclasses.replace(o.view, rotation_rad=rotation, flip_x=True, cos_alpha=0.9)
    r = Renderer(twin.specimen, twin.renderer.config)
    a = r._padded_optics(dataclasses.replace(o, view=v))[0]
    r._panned_field_map(a, 0.0)
    step = 0.25 * v.shape[1] * v.pixel_um
    moved = dataclasses.replace(v, center_um=(v.center_um[0] + step, v.center_um[1] - 0.1))
    b = r._padded_optics(dataclasses.replace(o, view=moved))[0]
    field_map, _ = r._panned_field_map(b, 0.0)
    assert r.rasters_reused == 1
    _same_raster(field_map, twin.specimen.rasterize(field_map.view))


def test_a_moved_view_is_the_image_without_reuse(make_render_twin):
    a, b = make_render_twin(), make_render_twin(reuse_rasters=False)
    request = a.request()
    for twin in (a, b):
        twin.flux(request)
        n = twin.optics(request).view.shape[0]
        _move(
            twin, 0.3 * n * twin.optics(request).view.pixel_um, -0.2 * n * twin.optics(request).view.pixel_um
        )
    img_a, img_b = a.flux(request), b.flux(request)
    assert a.renderer.rasters_reused == 1 and b.renderer.rasters_reused == 0
    close = np.isclose(img_a, img_b, rtol=1e-3, atol=1e-3 * float(img_b.mean()))
    assert close.mean() > 0.999
    assert img_a.mean() == pytest.approx(img_b.mean(), rel=1e-4)


def test_a_view_is_the_same_image_however_it_was_reached(make_render_twin):
    """World-locked and deterministic: a view reached by a drag (crops of lattice-centred
    rasters) is the fresh render of that view, away from the transfer's periodic edges."""
    a, b = make_render_twin(), make_render_twin()
    request = a.request()
    a.flux(request)
    n = a.optics(request).view.shape[0]
    px = a.optics(request).view.pixel_um
    for step in (0.07, 0.3, 0.07):
        _move(a, step * n * px, 0.5 * step * n * px)
        a.flux(request)
    _move(b, 0.44 * n * px, 0.22 * n * px)
    img_a, img_b = a.flux(request), b.flux(request)
    inner = (slice(60, -60), slice(60, -60))
    assert np.corrcoef(img_a[inner].ravel(), img_b[inner].ravel())[0, 1] > 0.995
    assert img_a[inner].mean() == pytest.approx(img_b[inner].mean(), rel=0.01)


def test_a_magnification_step_renders_the_new_view(make_render_twin):
    twin, ref = make_render_twin(), make_render_twin()
    request = twin.request()
    twin.flux(request)
    built = twin.renderer.rasters_built
    for t in (twin, ref):
        t.column.set("Magnification", 25000.0)
    img = twin.flux(request)
    assert twin.renderer.rasters_built == built + 1 and twin.renderer.rasters_reused == 0
    np.testing.assert_array_equal(img, ref.flux(request))


def test_a_jump_back_is_a_crop(make_render_twin):
    twin = make_render_twin()
    request = twin.request()
    home = twin.flux(request)
    _move(twin, 30.0, 20.0)
    twin.flux(request)
    _move(twin, -30.0, -20.0)
    built = twin.renderer.rasters_built
    np.testing.assert_array_equal(twin.flux(request), home)
    assert twin.renderer.rasters_built == built


# ------------------------------------------------------------------ interactive
def test_a_magnification_step_previews_then_settles_to_the_full_render(make_render_twin):
    import time

    twin = make_render_twin(interactive=True, settle_s=0.05)
    ref = make_render_twin()
    request = twin.request()
    twin.flux(request)
    for t in (twin, ref):
        t.column.set("Magnification", 25000.0)
    moving = twin.flux(request)
    assert twin.renderer.previews_rendered == 1
    full = ref.flux(request)
    assert moving.shape == full.shape
    assert moving.mean() == pytest.approx(full.mean(), rel=0.05)
    time.sleep(0.06)
    np.testing.assert_array_equal(twin.flux(request), full)
    assert twin.renderer.previews_rendered == 1


def test_previews_do_not_push_out_the_field_maps_a_move_reuses(make_render_twin):
    import time

    twin = make_render_twin(interactive=True, settle_s=0.05)
    request = twin.request()
    twin.flux(request)
    n = twin.optics(request).view.shape[0]
    px = twin.optics(request).view.pixel_um
    for _ in range(3):
        _move(twin, 0.3 * n * px)
        twin.flux(request)  # a preview
        time.sleep(0.06)
        twin.flux(request)  # the full render, from the last full field map plus a strip
    assert twin.renderer.rasters_reused >= 3


def test_serve_interactive_flag():
    import argparse

    from de_twin.cli import _build_twin

    args = argparse.Namespace(
        specimen="Dense Au on holey C",
        camera="DESim",
        seed=0,
        holder="none",
        time_scale=1.0,
        mirror_temchannel=None,
        corrector="none",
        interactive=True,
    )
    assert _build_twin(args).renderer.config.interactive
    args.interactive = False
    assert not _build_twin(args).renderer.config.interactive


@pytest.mark.parametrize("dx_px,dy_px", [(7.4, -3.3), (0.5, 0.25), (-120.7, 60.45)])
def test_a_crop_is_where_a_fresh_render_puts_the_view(dx_px, dy_px, make_render_twin):
    """Inside the margin a move is a crop, interpolated to a fraction of a pixel: the image
    sits where a fresh render of the view puts it (not on the nearest raster pixel)."""
    from test_realism_calibrations import _shift_px

    # the fixed 15 % pan margin, so the larger moves stay inside it
    twin, ref = make_render_twin(adaptive_margin=False), make_render_twin(adaptive_margin=False)
    request = twin.request()
    twin.flux(request)
    built = twin.renderer.rasters_built
    px = twin.optics(request).view.pixel_um
    for t in (twin, ref):
        _move(t, dx_px * px, dy_px * px)
    crop = twin.flux(request)
    assert twin.renderer.rasters_built == built, "a crop"
    fresh = ref.flux(request)
    inner = (slice(64, -64), slice(64, -64))
    # (the whitened phase correlation weighs the pixel-scale texture, which a sub-pixel shift
    # of the exit wave and one of the image move a little differently)
    assert np.abs(_shift_px(fresh[inner].astype(float), crop[inner].astype(float))).max() <= 0.2
    assert np.corrcoef(crop[inner].ravel(), fresh[inner].ravel())[0, 1] > 0.998


# ------------------------------------------------------------------ strips
@pytest.mark.parametrize("name", ["Dense Au on holey C", GRATING, "Negative stain on carbon"])
def test_a_raster_in_parallel_strips_is_the_serial_raster(name, monkeypatch, make_render_twin):
    from de_twin.specimen.model import Specimen, _numba_threadsafe

    # strips are only used where NumPy work remains; force them here
    monkeypatch.setattr(Specimen, "_numpy_work", lambda self, view, span: True)
    twin = make_render_twin(name)
    request = twin.request()
    twin.flux(request)  # runs the numba kernels (the threading layer is known after)
    v = twin.renderer._padded_optics(twin.optics(request))[0].view
    serial = twin.specimen.rasterize(v)
    strips = twin.specimen.rasterize(v, threads=4)
    if _numba_threadsafe():
        assert twin.specimen.last_stats.get("strips") == 4
    _same_raster(strips, serial)


def test_a_piece_of_a_view_is_drawn_as_the_whole_view():
    """The scene populates a placement area or draws its aggregate by how large it is in
    the view; a piece of a view (a strip, the part a stage move brings in) must decide as
    the whole view does."""
    from de_twin.render.renderer import _sub_view
    from de_twin.specimen import Specimen, ViewWindow, from_name

    s = Specimen(from_name("Dense Au on holey C"))
    v = ViewWindow(center_um=s.features()[0].center_um, pixel_um=0.1, shape=(1344, 1344))
    whole = s.rasterize(v)
    piece = s.rasterize(_sub_view(v, 544, 800, 544, 800), span_px=1344)
    alone = s.rasterize(_sub_view(v, 544, 800, 544, 800))
    np.testing.assert_array_equal(piece.material_id, whole.material_id[544:800, 544:800])
    assert not np.array_equal(alone.material_id, piece.material_id), "a small view alone decides otherwise"


def test_a_rotated_view_is_rasterised_whole(make_render_twin):
    import dataclasses

    from de_twin.render import Renderer

    twin = make_render_twin()
    o = twin.optics(twin.request())
    v = dataclasses.replace(o.view, rotation_rad=0.7)
    r = Renderer(twin.specimen, twin.renderer.config)
    r._panned_field_map(r._padded_optics(dataclasses.replace(o, view=v))[0], 0.0)
    step = 0.25 * v.shape[1] * v.pixel_um
    moved = dataclasses.replace(v, center_um=(v.center_um[0] + step, v.center_um[1]))
    field_map, _ = r._panned_field_map(r._padded_optics(dataclasses.replace(o, view=moved))[0], 0.0)
    assert r.rasters_reused == 0 and "strips" not in twin.specimen.last_stats
    ref = twin.specimen.rasterize(field_map.view)
    np.testing.assert_array_equal(field_map.grain_id, ref.grain_id)


@pytest.mark.parametrize("name", ["Dense Au on holey C", GRATING])
def test_the_bragg_table_is_the_exact_excitation(name):
    """Every (grain, thickness bin) entry of a material's table matches the exact (scipy sici)
    evaluation to 1e-6 relative."""
    twin = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=1)
    r = twin.renderer
    o = twin.optics(twin.request())
    field_map = r.field_map(o)[0]
    g = field_map.grain_id[field_map.grain_id >= 0]
    mid = int(np.bincount(g // GRAINS_PER_MATERIAL).argmax())
    lib = library_for(mid, r.config.max_g_inv_nm)
    memo = tem_contrast._BraggMemo(r.grains)
    memo._g_obj = 0.0
    table = memo.table(lib, r.grains, mid, o)
    rng = np.random.default_rng(0)
    gids = mid * GRAINS_PER_MATERIAL + rng.integers(0, GRAINS_PER_MATERIAL, 400)
    tbs = rng.integers(0, table.shape[1], 400)
    exact = memo.values(lib, r.grains, gids, np.arange(400), tbs, o)
    np.testing.assert_allclose(table[gids - mid * GRAINS_PER_MATERIAL, tbs], exact, rtol=1e-6, atol=1e-12)


def test_a_tilt_series_keeps_its_tables(make_render_twin):
    twin = make_render_twin()
    request = twin.request()
    twin.flux(request)
    tem_contrast._BRAGG_MEMOS.clear()
    for a in (0.5, 1.0):
        twin.column.move_stage(alpha=a)
        twin.flux(request)
    kept = dict(tem_contrast._BRAGG_MEMOS)
    assert len(kept) == 2
    twin.column.move_stage(alpha=0.5)
    twin.renderer.invalidate()  # not a crop: the frame is rendered again, the table is reused
    twin.flux(request)
    assert dict(tem_contrast._BRAGG_MEMOS) == kept, "a return to a tilt reuses its table"


# ------------------------------------------------------------------ fused kernels
@pytest.mark.parametrize(
    "name,mag,defocus",
    [
        ("Dense Au on holey C", 20000.0, -1.0),
        ("Apoferritin in ice", 50000.0, -2.0),
        ("Negative stain on carbon", 100000.0, -0.5),
        (GRATING, 150000.0, -0.3),
    ],
)
def test_the_fused_kernels_are_the_numpy_reference(name, mag, defocus, monkeypatch):
    """The numba exit wave, Bragg lookup, transfer function and |psi|^2 against the NumPy
    reference: float rounding only."""
    from de_twin.render import fastpath
    from de_twin.render.tem import TransferCache

    if not fastpath.AVAILABLE:
        pytest.skip("no numba")
    twin = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=2)
    twin.column.set("Magnification", mag)
    twin.column.set_defocus_um(defocus)
    twin.column.set("ObjectiveStig", (0.3, -0.2))  # A1: the transfer function's astigmatic terms
    request = twin.request()
    r = twin.renderer
    po, sh = r._padded_optics(twin.optics(request))
    field_map, tok = r.field_map(po)
    out = []
    for fast in (False, True):
        monkeypatch.setattr(tem._fp, "AVAILABLE", fast)
        tem_contrast._BRAGG_MEMOS.clear()
        out.append(
            tem.render_tem_raster(
                field_map, po, r.grains, r.crystallinity, r.config, r.seed, TransferCache(), tok, sh
            )
        )
    ref, fast = out
    d = np.abs(fast - ref) / ref.mean()
    assert d.max() < 1e-4 and np.sqrt((d * d).mean()) < 1e-5


# ------------------------------------------------------------------ prefetch
def test_a_series_is_prefetched_and_is_the_exact_render(make_render_twin):
    """With prefetch on, the view that repeating the last change gives is rendered in the
    background; asking for it is a crop, and the image is the one rendered on demand."""
    live, ref = make_render_twin(prefetch=True), make_render_twin()
    request = live.request()
    for t in (live, ref):
        t.flux(request)
    for magnification in (25000.0, 30000.0, 40000.0):
        for t in (live, ref):
            t.column.set("Magnification", magnification)
        img = live.flux(request)
        live._pf_future.result()
        np.testing.assert_allclose(img, ref.flux(request), rtol=1e-5, atol=1e-5)
    assert live.renderer.prefetched >= 2
    built = live.renderer.rasters_built
    live.column.set("Magnification", 50000.0)  # the next rung: prefetched
    live.flux(request)
    assert live.renderer.rasters_built == built


def test_prefetch_is_off_by_default(make_render_twin):
    twin = make_render_twin()
    request = twin.request()
    twin.flux(request)
    twin.column.set("Magnification", 25000.0)
    twin.flux(request)
    assert getattr(twin, "_pf_future", None) is None


def test_strips_only_where_numpy_work_remains(make_render_twin):
    """A specimen drawn entirely by numba kernels is rasterised whole (its kernels are
    parallel); one with NumPy structures in strips. Decided per view, so a view is always
    rasterised the same way."""
    from de_twin.specimen import raster_nb

    if not raster_nb.AVAILABLE:
        pytest.skip("no numba")
    au = make_render_twin()
    v = au.renderer._padded_optics(au.optics(au.request()))[0].view
    assert not au.specimen._numpy_work(v, max(v.shape))
    film = make_render_twin("Au thin film 20 nm")
    v = film.renderer._padded_optics(film.optics(film.request()))[0].view
    assert film.specimen._numpy_work(v, max(v.shape))


# ------------------------------------------------------------------ fast imaging (FFTs, guard band)
def test_the_transposed_ffts_are_scipys():
    from scipy import fft as sfft

    from de_twin.render import fastpath

    rng = np.random.default_rng(0)
    a = (rng.random((200, 330)) + 1j * rng.random((200, 330))).astype(np.complex64)
    spec = fastpath.fft2_t(a.copy())
    ref = sfft.fft2(a)
    np.testing.assert_allclose(spec, ref.T, rtol=0, atol=1e-5 * np.abs(ref).max())
    back = fastpath.ifft2_t(spec.copy())
    np.testing.assert_allclose(back, a, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "name,mag,defocus",
    [
        ("Dense Au on holey C", 25000.0, -1.0),
        ("Apoferritin in ice", 100000.0, -3.0),
        (GRATING, 3000.0, -20.0),
    ],
)
def test_the_fast_imaging_is_the_double_precision_image(name, mag, defocus):
    """float32 kernels and transposed single-precision FFTs against the NumPy exit wave with
    complex128 FFTs on the same raster: 1e-5 RMS."""

    from scipy import fft as sfft

    from de_twin.render import fastpath
    from de_twin.render.tem import TransferCache

    twin = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=2)
    twin.column.set("Magnification", mag)
    twin.column.set_defocus_um(defocus)
    r = twin.renderer
    po, sh = r._padded_optics(twin.optics(twin.request()))
    field_map, tok = r.field_map(po)
    fast = tem.render_tem_raster(
        field_map, po, r.grains, r.crystallinity, r.config, r.seed, TransferCache(), tok, sh
    )
    was = fastpath.AVAILABLE
    fastpath.AVAILABLE = False
    try:
        psi = tem.transmission_function(field_map, po, r.grains, r.crystallinity, r.config, r.seed)
        H = tem.transfer_function(po.view.shape, po.view.pixel_um * 1000.0, po)
    finally:
        fastpath.AVAILABLE = was
    spec = sfft.fft2(psi.astype(np.complex128))
    if H is not None:
        spec *= H
    ramps = tem._shift_ramps(po.view.shape, sh)
    if ramps is not None:
        spec *= ramps[0].astype(np.complex128)[:, None] * ramps[1].astype(np.complex128)[None, :]
    ref = np.abs(sfft.ifft2(spec)) ** 2
    d = (fast - ref) / ref.mean()
    assert np.sqrt((d * d).mean()) < 1e-5 and np.abs(d).max() < 1e-3


@pytest.mark.parametrize("mag,defocus", [(25000.0, -1.0), (100000.0, -0.3), (100000.0, -3.0)])
def test_the_guard_band_keeps_the_fft_wrap_out(mag, defocus):
    """One exit wave on a large raster, propagated whole and cropped to the adaptive guard
    band: the view's image changes by < 1e-3 RMS (the fixed 15 % margin: 1e-2 at 100k,
    -3 um)."""
    import dataclasses

    from scipy import fft as sfft

    from de_twin.render.renderer import _guard_px

    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=2)
    twin.column.set("Magnification", mag)
    twin.column.set_defocus_um(defocus)
    r = twin.renderer
    o = twin.optics(twin.request())
    big = 1024 + 2 * 768
    po = dataclasses.replace(o, view=dataclasses.replace(o.view, shape=(big, big)))
    psi = tem.transmission_function(
        twin.specimen.rasterize(po.view), po, r.grains, r.crystallinity, r.config, r.seed
    ).astype(np.complex128)

    def image(p):
        n = p.shape[0]
        opt = dataclasses.replace(o, view=dataclasses.replace(o.view, shape=(n, n)))
        H = tem.transfer_function((n, n), o.view.pixel_um * 1000.0, opt)
        s = sfft.fft2(p)
        return np.abs(sfft.ifft2(s * H if H is not None else s)) ** 2

    c = big // 2
    ref = image(psi)[c - 512 : c + 512, c - 512 : c + 512]
    g = _guard_px(o, r.config.min_margin_px, r.config.max_margin_px)
    n = 1024 + 2 * g
    img = image(psi[c - n // 2 : c + n // 2, c - n // 2 : c + n // 2])[g : g + 1024, g : g + 1024]
    d = (img - ref) / ref.mean()
    assert np.sqrt((d * d).mean()) < 1e-3
