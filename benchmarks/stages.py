"""Per-stage time of a new view: magnification step, stage move, defocus step, tilt step.

    PYTHONPATH="src;tests" python benchmarks/stages.py [--camera DESim] [--presets ...]

Each case is timed on a warm twin (the scenario's previous view rendered and settled): the
flux of the first frame of the new view split into specimen rasterisation, the Bragg
lookup, the texture noise, the rest of the exit wave, the forward FFT, the transfer
function (CTF), the inverse FFT + |psi|^2, the finish (illumination disc + upsampling), the
detector exposure of one frame, and what is left (crops, bookkeeping).
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import de_twin.render.util as tem_util
from de_twin.clock import ManualClock
from de_twin.render import tem
from de_twin.specimen.model import Specimen
from de_twin.twin import DigitalTwin

PRESETS = (
    "Dense Au on holey C",
    "Ted Pella 607 - 2160 l/mm grating replica (waffle)",
    "Ted Pella 619 - Diffraction standard, evaporated aluminum",
    "Apoferritin in ice",
)
T = defaultdict(float)
_depth = defaultdict(int)


def _wrap(owner, name, label):
    fn = getattr(owner, name)

    def w(*a, **k):
        _depth[label] += 1
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _depth[label] -= 1
            if _depth[label] == 0:
                T[label] += time.perf_counter() - t0

    setattr(owner, name, w)


_wrap(Specimen, "rasterize", "raster")
_wrap(tem, "bragg_contrast", "bragg")
_wrap(tem, "texture_noise", "noise")
_wrap(tem, "transmission_function", "exitwave")
_wrap(tem, "transfer_function", "ctf")
_wrap(tem, "render_physical", "tem")
_wrap(tem, "finish_tem", "finish")
_ffts = {"fft2": 0}
_fft2 = tem.sfft.fft2


class _Sfft:  # the forward FFT of the exit wave, timed on its own
    def __getattr__(self, k):
        return getattr(_real, k)

    def fft2(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return _real.fft2(*a, **k)
        finally:
            T["fft"] += time.perf_counter() - t0


_real = tem.sfft
tem.sfft = _Sfft()
if tem._fp.AVAILABLE:  # the transposed FFTs of the fast path
    for _name, _label in (("fft2_t", "fft"), ("ifft2_t", "ifft_only")):
        _f = getattr(tem._fp, _name)

        def _w(*a, _f=_f, _label=_label, **k):
            t0 = time.perf_counter()
            try:
                return _f(*a, **k)
            finally:
                T[_label] += time.perf_counter() - t0

        setattr(tem._fp, _name, _w)

COLS = ("total", "raster", "bragg", "noise", "exit", "fft", "ctf", "ifft", "finish", "detector", "other")


def _frame(tw, req):
    T.clear()
    t0 = time.perf_counter()
    flux = tw.flux(req)
    t1 = time.perf_counter()
    tw.detector.expose(flux, req.frame_time_s, req, 0, ht_kv=tw.column.state().ht_kv)
    t2 = time.perf_counter()
    total = t1 - t0
    exit_rest = T["exitwave"] - T["bragg"] - T["noise"]
    ifft = T["tem"] - T["exitwave"] - T["fft"] - T["ctf"]  # inverse FFT, |psi|^2, the product
    r = {
        "total": total + (t2 - t1),
        "raster": T["raster"],
        "bragg": T["bragg"],
        "noise": T["noise"],
        "exit": exit_rest,
        "fft": T["fft"],
        "ctf": T["ctf"],
        "ifft": ifft,
        "finish": T["finish"],
        "detector": t2 - t1,
    }
    r["other"] = total - (T["raster"] + T["tem"] + T["finish"])
    return r


def cases(tw, req):
    col = tw.column
    col.set("Magnification", 20000.0)
    tw.flux(req)
    col.set("Magnification", 25000.0)
    yield "mag 20k->25k", _frame(tw, req)
    col.set("Magnification", 20000.0)
    tw.flux(req)
    o = tw.optics(req)
    s = col.state().stage
    col.move_stage(x=s.x_um + 0.3 * o.view.shape[1] * o.view.pixel_um)
    yield "move 30% of the field", _frame(tw, req)
    col.set_defocus_um(-1.0)
    yield "defocus 0 -> -1 um", _frame(tw, req)
    col.set_defocus_um(0.0)
    tw.flux(req)
    col.move_stage(alpha=1.0)
    yield "tilt alpha 0 -> 1 deg", _frame(tw, req)
    col.move_stage(alpha=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="DESim")
    ap.add_argument("--presets", nargs="+", default=list(PRESETS))
    args = ap.parse_args()
    print(f"{'preset':24s} {'case':22s} " + " ".join(f"{c:>7s}" for c in COLS) + "  (ms)")
    for preset in args.presets:
        warm = DigitalTwin(preset, camera=args.camera, clock=ManualClock(), seed=0)
        req = warm.request()
        f = warm.flux(req)
        warm.detector.expose(f, req.frame_time_s, req, 0, ht_kv=warm.column.state().ht_kv)
        list(cases(warm, req))  # numba compiled / loaded, imports done
        tem._BRAGG_MEMOS.clear()  # process-wide caches: start the measured twin cold
        tem_util._NOISE_TILES.clear()
        tw = DigitalTwin(preset, camera=args.camera, clock=ManualClock(), seed=0)
        f = tw.flux(req)
        tw.detector.expose(f, req.frame_time_s, req, 0, ht_kv=tw.column.state().ht_kv)  # its maps
        for name, r in cases(tw, req):
            print(
                f"{preset[:24]:24s} {name:22s} " + " ".join(f"{1000 * r[c]:7.0f}" for c in COLS), flush=True
            )


if __name__ == "__main__":
    main()
