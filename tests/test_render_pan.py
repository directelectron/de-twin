"""Panning: a stage move inside the render margin is a crop, not a render — and the same image."""

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.render.config import RenderConfig
from de_twin.twin import DigitalTwin


def _pair(name="Dense Au on holey C"):
    # the fixed pan margin (a 15 % guard on every side), not the adaptive guard band
    pan = DigitalTwin(
        name, camera="DESim", clock=ManualClock(), seed=3, render_config=RenderConfig(adaptive_margin=False)
    )
    ref = DigitalTwin(
        name, camera="DESim", clock=ManualClock(), seed=3, render_config=RenderConfig(pan_margin=0.0)
    )
    return pan, ref


def _move(tws, dx_um, dy_um=0.0):
    for twin in tws:
        s = twin.column.state().stage
        twin.column.move_stage(x=s.x_um + dx_um, y=s.y_um + dy_um)


@pytest.mark.parametrize("dx_px,dy_px", [(7, 0), (0, -30), (90, 60)])
def test_a_panned_view_is_the_full_render(dx_px, dy_px):
    pan, ref = _pair()
    request = pan.request()
    pan.flux(request)
    px = pan.optics(request).specimen_pixel_nm / 1000.0
    _move((pan, ref), dx_px * px, dy_px * px)
    a, b = pan.flux(request), ref.flux(request)
    inner = (slice(40, -40), slice(40, -40))  # away from the transfer's periodic edges
    assert np.corrcoef(a[inner].ravel(), b[inner].ravel())[0, 1] > 0.99
    assert a.mean() == pytest.approx(b.mean(), rel=0.02)


def test_a_nudge_is_a_crop_and_a_jump_renders():
    pan, _ = _pair()
    request = pan.request()
    pan.flux(request)
    built = pan.renderer.rasters_built
    px = pan.optics(request).specimen_pixel_nm / 1000.0
    _move((pan,), 50 * px)
    pan.flux(request)
    assert pan.renderer.rasters_built == built, "inside the margin: no new raster"
    _move((pan,), 3000 * px)
    pan.flux(request)
    assert pan.renderer.rasters_built == built + 1, "beyond it: rendered again"


def test_the_frames_of_an_exposure_share_one_array():
    pan, _ = _pair()
    request = pan.request()
    a, b = pan.flux(request), pan.flux(request)
    assert a is b and not a.flags.writeable


def test_an_exposure_renders_once_on_a_running_clock():
    """The clock advances every frame; a static specimen must not re-finish its image."""
    from de_twin.clock import ManualClock as MC

    clock = MC()
    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=clock, seed=3)
    request = twin.request(frame_time_s=0.025, total_frames=10)
    list(twin.frames(twin.request(frame_time_s=0.025, total_frames=1)))
    before = twin.renderer.frames_from_cache
    list(twin.frames(request))
    assert twin.renderer.frames_from_cache - before >= 9


def test_interactive_previews_while_moving_and_renders_in_full_once_settled():
    twin = DigitalTwin(
        "Dense Au on holey C",
        camera="DESim",
        clock=ManualClock(),
        seed=3,
        render_config=RenderConfig(interactive=True, settle_s=0.05),
    )
    ref = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=3)
    request = twin.request()
    twin.flux(request)
    assert getattr(twin.renderer, "previews_rendered", 0) == 0, "the first view is in full"
    px = twin.optics(request).specimen_pixel_nm / 1000.0
    _move((twin, ref), 3000 * px)  # beyond the margin: not a crop
    moving = twin.flux(request)
    assert twin.renderer.previews_rendered == 1
    full = ref.flux(request)
    assert moving.shape == full.shape
    assert moving.mean() == pytest.approx(full.mean(), rel=0.05)
    import time

    time.sleep(0.06)
    settled = twin.flux(request)
    assert twin.renderer.previews_rendered == 1, "settled: the full render, not a preview"
    np.testing.assert_allclose(settled, full, rtol=1e-5, atol=1e-6)


def test_interactive_is_off_by_default():
    twin, _ = _pair()
    request = twin.request()
    twin.flux(request)
    px = twin.optics(request).specimen_pixel_nm / 1000.0
    _move((twin,), 3000 * px)
    twin.flux(request)
    assert getattr(twin.renderer, "previews_rendered", 0) == 0
