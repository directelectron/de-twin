"""EBSD camera frame rate: DESim 1024^2 binned 2x2 to 512^2, 100 nA, 20 kV, 0.125 ms frames,
low gain, "EBSD duplex steel", 256 x 256 scan.

    PYTHONPATH=src python -u benchmarks/ebsd_frames.py [--frames 20000] [--skip-general]

Reports the general path's stages (render_ebsd_frame's projection + Poisson means, the
detector's deposit and analogue passes, the copy into a slot) and frame rate, then the fused
path's: one frame on one core, frames a second through ``twin.frames``, and through the
shared-memory face into a private mapping, raster and steered, with a consumer that copies
each frame out (as DE-Server does) and with one that only advances the read pointer.
Every run is warm (kernels compiled, a first batch discarded).
"""

from __future__ import annotations

import argparse
import os
import struct
import threading
import time
import uuid

import numpy as np

from de_twin import DigitalTwin
from de_twin.clock import ManualClock
from de_twin.detector import fast
from de_twin.state import ScanRequest

DWELL = 0.125e-3
KV = 20.0


def twin() -> DigitalTwin:
    tw = DigitalTwin("EBSD duplex steel", camera="DESim", seed=1, clock=ManualClock(),
                     detector_overrides={"gain_mode": "low"})
    tw.configure_ebsd(beam_kv=KV, probe_current_na=100.0)
    return tw


def request(tw, frames=0, size=(256, 256)):
    return tw.request(hw_binning=(2, 2), frame_time_s=DWELL, total_frames=frames,
                      scan=ScanRequest(enabled=True, size=size, step_um=0.5, dwell_s=DWELL))


def per_call(fn, n=200, reps=5) -> float:
    """Median seconds per call of fn(i)."""
    fn(0)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for i in range(n):
            fn(i)
        out.append((time.perf_counter() - t0) / n)
    return float(np.median(out))


def frames_rate(it, n) -> float:
    for _ in range(min(500, n // 10)):
        next(it)
    t0 = time.perf_counter()
    for _ in range(n):
        next(it)
    return n / (time.perf_counter() - t0)


def general(tw, n):
    os.environ["DE_TWIN_FUSED_EBSD"] = "0"
    try:
        req = request(tw)
        optics = tw.optics(req)
        ren, det = tw.renderer, tw.detector
        t_nb = DWELL * 4
        points = [(i % 256, (i * 7 // 256) % 256) for i in range(1000)]
        proj = per_call(lambda i: ren.render_ebsd_frame(optics, i, points[i % 1000], 0.0, t_nb))
        flux, means = ren.render_ebsd_frame(optics, 0, (100, 100), 0.0, t_nb)
        lam, p0 = means[0].copy(), means[1].copy()
        S = np.empty_like(lam)
        m = det.model
        k = 1.0 / max(m.adu_cv, 1e-3) ** 2
        lut, table, adu = fast.gamma_lut(k), fast.normal_table(), det._adu_at(KV)
        dep = per_call(lambda i: fast.deposit(lam, p0, adu, k, np.uint64(fast.frame_key(1, i, 1)), S, table, 1234 * i,
                                              lut))
        det._build_maps()
        gain, off = det._binned_maps(det.roi(req), 2, 2)
        raw = np.empty((512, 512), np.uint16)
        hot = np.zeros((512, 512), np.float32)
        ana = per_call(lambda i: fast.analog_out(S, True, m.charge_spread_at(KV) / 2, 0.25, gain, off, 0.0,
                                                 m.read_noise_adu / 2, 1.0, table, 777 * i, hot, float(m.max_value),
                                                 raw))
        exp = per_call(lambda i: det.expose(flux, DWELL, req, i, ht_kv=KV, means=means))
        slot = np.empty(512 * 512, np.uint16)
        cp = per_call(lambda i: np.copyto(slot, raw.reshape(-1)), n=1000)
        fps = frames_rate(tw.frames(req), n)
    finally:
        del os.environ["DE_TWIN_FUSED_EBSD"]
    print("general path, per frame (warm, numba threads: all)")
    print(f"  projection + Poisson means   {proj * 1e3:7.3f} ms")
    print(f"  detector deposit             {dep * 1e3:7.3f} ms")
    print(f"  detector analogue            {ana * 1e3:7.3f} ms")
    print(f"  Detector.expose (all of it)  {exp * 1e3:7.3f} ms")
    print(f"  copy into a slot             {cp * 1e3:7.3f} ms")
    print(f"  twin.frames                  {1e3 / fps:7.3f} ms  = {fps:6.0f} frames/s")


def fused(tw, n):
    from de_twin.workers import pool

    req = request(tw)
    src = tw.ebsd_frame_source(req)
    local, out = {}, np.empty(src.shape, np.uint16)
    one = per_call(lambda i: src.render_one(local, out, (i * 37) % 65536, 1, i), n=100, reps=3)
    per_job = tw.FRAMES_PER_JOB
    workers = pool()
    in_flight = workers.threads + workers.threads // 2  # as twin.frames keeps them
    ring = [np.empty((per_job, *src.shape), np.uint16) for _ in range(in_flight + 1)]

    def run(jobs):
        from collections import deque

        pending = deque()
        for i in range(jobs):
            if len(pending) == in_flight:
                pending.popleft().wait()
            ks = [(i * per_job * 37 + b) % 65536 for b in range(per_job)]
            pending.append(src.submit(workers, ring[i % len(ring)], ks, 1, range(i * per_job, (i + 1) * per_job)))
        for job in pending:
            job.wait()

    run(in_flight)
    jobs = max(n // per_job, 4 * in_flight)
    t0 = time.perf_counter()
    run(jobs)
    raw_rate = jobs * per_job / (time.perf_counter() - t0)
    fps = frames_rate(tw.frames(req), n)
    print(f"fused path ({workers.threads} worker threads)")
    print(f"  one frame on one core        {one * 1e3:7.3f} ms")
    print(f"  workers alone (no consumer)  {raw_rate:7.0f} frames/s")
    print(f"  twin.frames                  {fps:7.0f} frames/s")


CONSUMER = r"""
import json, sys, time
import numpy as np
from de_twin.transport.shm import FrameConsumer

name, n, copy = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
consumer = FrameConsumer(name, max_frame_bytes=512 * 512 * 2)
consumer.begin(frame_shape=(512, 512), binning=(2, 2), sensor_shape=(1024, 1024), frame_time_s=%r,
               total_frames=0, scan_size=(256, 256))
h = consumer.hdr
buf = np.empty(512 * 512, np.uint16)
got, t0, base = 0, None, 0
while got < n:
    if copy:  # as DE-Server: each frame copied out of its slot
        consumer.read(timeout=600, out=buf)
        got += 1
    else:  # only the read pointer moved on
        rc, wc = h.read_count, h.write_count
        if wc > rc:
            h.read_count = wc
            got += wc - rc
    if t0 is None and got >= 500:
        t0, base = time.perf_counter(), got
rate = (got - base) / (time.perf_counter() - t0)
consumer.end()
print(json.dumps({"rate": rate}))
sys.stdout.flush()
time.sleep(1.0)  # (the producer sees the end before the mapping goes)
consumer.close()
""" % DWELL


def face(tw, n, *, steered, copy):
    """Frames published through ShmFace to a reader in another process, as DE-Server is."""
    import json
    import subprocess
    import sys

    from de_twin.faces.shm_face import ShmFace
    from de_twin.optics.scan_coils import point_volts
    from de_twin.transport import scan_output as S

    tag = f"{os.getpid()}_{uuid.uuid4().hex[:6]}"
    scan_name, frame_name = f"DE_ScanOutputBench_{tag}", f"DE_ExternalFramesBench_{tag}"
    writer = S.ScanOutputWriter(scan_name) if steered else None
    if steered:
        g = S.ScanGeometry(size=(256, 256), voltage_h=(-1.0, 1.0), voltage_v=(-1.0, 1.0), rotation_deg=0.0,
                           flip=(False, False), points_per_frame=1.0, total_points=256 * 256)
        writer.begin(g)
        scan = ScanRequest(enabled=True, size=g.size)
        writer.write(np.array([point_volts(scan, i % 256, i // 256) for i in range(256 * 256)]))  # a raster
    reader = subprocess.Popen([sys.executable, "-c", CONSUMER, frame_name, str(n), "1" if copy else "0"],
                              stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    f = ShmFace(tw, name=frame_name, scan_output=scan_name, pace=False, warm_up=False).start()
    try:
        out, _ = reader.communicate(timeout=3600)
    finally:
        f.stop()
        if writer is not None:
            writer.close()
    rate = json.loads(out.strip().splitlines()[-1])["rate"]
    kind = "steered" if steered else "raster "
    how = "copies each frame out" if copy else "only advances the pointer"
    print(f"  ShmFace, {kind}, reader {how:26s} {rate:7.0f} frames/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20000)
    ap.add_argument("--skip-general", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    tw = twin()
    print(f"logical processors: {os.cpu_count()}")
    if not args.skip_general:
        general(tw, min(args.frames, 4000))
    for _ in range(args.repeat):  # the machine is shared: repeat, and read the spread
        fused(tw, args.frames)
        for steered in (False, True):
            for copy in (True, False):
                face(tw, args.frames, steered=steered, copy=copy)


if __name__ == "__main__":
    main()
