"""Re-render cost of the live-view operations: a magnification step and stage moves.

    PYTHONPATH="src;tests" python benchmarks/mag_stage.py [--cameras DESim DE16] [--profile]

For each preset and camera it times, on a warm twin, the first frame's flux after
(a) a magnification step (2000x -> 2500x, 20000x -> 25000x), (b) a small stage move inside
the pan margin and one just outside it, (c) a large stage jump. The flux is split into
specimen rasterisation, the TEM raster render (exit wave + CTF), the finish (illumination
disc + upsampling) and the detector (counting/noise) exposure of one frame.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from collections import defaultdict

from de_twin.clock import ManualClock
from de_twin.render import tem
from de_twin.render.config import RenderConfig
from de_twin.specimen.model import Specimen
from de_twin.twin import DigitalTwin

PRESETS = ("Dense Au on holey C", "Ted Pella 607 - 2160 l/mm grating replica (waffle)")

T = defaultdict(float)
COLUMNS = ("total", "raster", "tem", "finish", "detector", "settled")


def _timed(name, fn):
    def wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name] += time.perf_counter() - t0

    return wrap


_orig = {"raster": Specimen.rasterize, "tem": tem.render_tem_raster, "finish": tem.finish_tem}
Specimen.rasterize = _timed("raster", _orig["raster"])
tem.render_tem_raster = _timed("tem", _orig["tem"])
tem.finish_tem = _timed("finish", _orig["finish"])


def _settle(tw, req):
    """The full render of the current view (interactive mode: after it settles)."""
    tw.flux(req)
    if tw.renderer.config.interactive:
        time.sleep(tw.renderer.config.settle_s + 0.02)
        tw.flux(req)


def _frame(tw, req):
    T.clear()
    reused, built = getattr(tw.renderer, "rasters_reused", 0), tw.renderer.rasters_built
    t0 = time.perf_counter()
    flux = tw.flux(req)
    t1 = time.perf_counter()
    tw.detector.expose(flux, req.frame_time_s, req, 0, ht_kv=tw.column.state().ht_kv)
    t2 = time.perf_counter()
    # the TEM raster time includes neither the rasterisation (done before) nor the finish
    out = {
        "total": t1 - t0,
        "raster": T["raster"],
        "tem": T["tem"],
        "finish": T["finish"],
        "detector": t2 - t1,
        "settled": float("nan"),
    }
    if tw.renderer.config.interactive:  # the full render once the view has settled
        time.sleep(tw.renderer.config.settle_s + 0.02)
        t0 = time.perf_counter()
        tw.flux(req)
        out["settled"] = time.perf_counter() - t0
    if getattr(tw.renderer, "rasters_reused", 0) > reused:
        out["how"] = "reuse"
    else:
        out["how"] = "crop" if tw.renderer.rasters_built == built else "full"
    return out


def _move(tw, dx_um, dy_um=0.0):
    s = tw.column.state().stage
    tw.column.move_stage(x=s.x_um + dx_um, y=s.y_um + dy_um)


def scenarios(tw, req):
    col = tw.column
    home = col.state().stage
    for m0, m1 in ((2000.0, 2500.0), (20000.0, 25000.0)):
        col.move_stage(x=home.x_um, y=home.y_um)
        col.set("Magnification", m0)
        _settle(tw, req)
        col.set("Magnification", m1)
        yield f"mag {m0:g}->{m1:g}", _frame(tw, req)
    col.move_stage(x=home.x_um, y=home.y_um)
    col.set("Magnification", 20000.0)
    _settle(tw, req)
    opt = tw.optics(req)
    px_um = opt.specimen_pixel_nm / 1000.0 * max(1, opt.raster_downsample)
    ny = opt.view.shape[0]
    _move(tw, 0.05 * ny * px_um)
    yield "move 5% (in margin)", _frame(tw, req)
    _move(tw, 0.25 * ny * px_um)
    yield "move 25% (out of margin)", _frame(tw, req)
    _move(tw, 0.3 * ny * px_um, 0.3 * ny * px_um)
    yield "move 30%,30% (drag step)", _frame(tw, req)
    _move(tw, 40.0, 25.0)
    yield "jump 40,25 um", _frame(tw, req)
    col.move_stage(x=home.x_um, y=home.y_um)
    yield "jump back home", _frame(tw, req)
    # a tilt series (alpha steps of 0.5 deg) and a return to a previous tilt
    for a in (0.5, 1.0, 1.5):
        col.move_stage(alpha=a)
        yield f"tilt alpha {a:g} deg", _frame(tw, req)
    col.move_stage(alpha=0.5)
    yield "tilt back to 0.5 deg", _frame(tw, req)
    col.move_stage(alpha=0.0)
    _settle(tw, req)
    # a defocus series and a return to a previous defocus
    for df in (-0.5, -1.0, -1.5):
        col.set_defocus_um(df)
        yield f"defocus {df:g} um", _frame(tw, req)
    col.set_defocus_um(-1.0)
    yield "defocus back to -1 um", _frame(tw, req)
    col.set_defocus_um(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", nargs="+", default=["DESim", "DE16"])
    ap.add_argument("--presets", nargs="+", default=list(PRESETS))
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()
    config = RenderConfig(interactive=args.interactive)
    prof = cProfile.Profile() if args.profile else None
    print(
        f"{'preset':28s} {'camera':6s} {'scenario':26s} {'total':>7s} {'raster':>7s} {'tem':>7s} "
        f"{'finish':>7s} {'detect':>7s} {'settled':>7s}  (ms)  raster"
    )
    for preset in args.presets:
        for cam in args.cameras:
            tw = DigitalTwin(preset, camera=cam, clock=ManualClock(), seed=0, render_config=config)
            req = tw.request()
            f = tw.flux(req)  # warm: imports, numba, populated areas
            _settle(tw, req)
            tw.detector.expose(f, req.frame_time_s, req, 0, ht_kv=tw.column.state().ht_kv)
            if prof:
                prof.enable()
            for name, r in scenarios(tw, req):
                cols = " ".join(f"{1000 * r[k]:7.0f}" for k in COLUMNS)
                print(f"{preset[:28]:28s} {cam:6s} {name:26s} {cols}  {r['how']}", flush=True)
            if prof:
                prof.disable()
    if prof:
        pstats.Stats(prof).sort_stats("cumulative").print_stats(45)


if __name__ == "__main__":
    main()
