"""Shared-memory face: the twin as DE-Server's external frame source.

DE-Server (test pattern "External Frame Source (Shared Memory)") creates the shared
memory and publishes each acquisition's request. This face turns the request into an
``AcquisitionRequest``, renders frames through the twin and publishes them, paced to the
frame time and throttled by DE-Server's reads.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from ..transport import shm_layout as L

log = logging.getLogger(__name__)

#: DE-Server gives up on the producer when a frame is ~1 s late (GrabberSim: two frame
#: times + 1000 ms) and stops the acquisition. A render can take longer than that — the
#: first view of a new magnification, the full render after a drag — so when no new
#: frame is ready after this long, the last one is published again.
KEEPALIVE_S = 0.5
#: A pool nobody has asked for frames for in this long is closed (its thread stops).
POOL_IDLE_S = 5.0
_DONE = object()


class ShmFace:
    """Serve DE-Server's requests from ``twin``.

    ``warm_up`` renders one frame before attaching: a fresh twin spends seconds on its
    first frame (imports, GPU start-up, building the specimen), while DE-Server waits about
    one frame time plus a second before it gives up on the acquisition. ``ready`` is set
    once the face is warm and serving.

    Throughput (the corner cut): ``reuse=N`` publishes each rendered frame up to N times,
    so a 4096² camera can stream at hundreds of frames per second while the detector
    physics renders tens; ``threads`` caps the cores the physics takes. Every frame is a
    correct single frame of the current view; what recycling costs is independence, so a
    sum of many frames is noisier than N-fold as many would be. ``reuse=1`` (the default)
    publishes every frame once.
    """

    def __init__(self, twin, name: str = L.DEFAULT_NAME, pace: bool = True, warm_up: bool = True,
                 *, reuse: int = 1, pool_size: int = 16, threads: Optional[int] = None):
        self.twin = twin
        self.reuse = max(1, int(reuse))
        self.pool_size = int(pool_size)
        self.threads = threads
        self.frames_reused = 0
        self._pool = None
        self._pool_key = None
        self._pool_used = 0.0
        self._last_frame = None  # the last frame published, for keep-alives across requests
        self.name = name
        self.pace = pace
        self.warm_up = warm_up
        self.ready = threading.Event()
        self.producer = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames_published = 0
        self.keepalives = 0
        self.requests_served = 0

    def start(self) -> "ShmFace":
        self._thread = threading.Thread(target=self._run, name="de-twin-shm", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._close_pool()
        if self.producer is not None:
            self.producer.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---------------------------------------------------------------- loop
    def _attach(self) -> bool:
        from ..transport.shm import FrameProducer

        try:
            self.producer = FrameProducer(self.name)
            log.info("attached to shared memory %s", self.name)
            return True
        except FileNotFoundError:
            return False

    def _warm_up(self) -> None:
        try:
            req = self.twin.request(frame_time_s=0.01, total_frames=1)
            next(iter(self.twin.frames(req)))
        except Exception:  # a failed warm-up only costs the first request its speed
            log.exception("warm-up render failed")

    def _run(self) -> None:
        if self.warm_up:
            self._warm_up()
        self.ready.set()
        pending = None
        while not self._stop.is_set():
            if self.producer is None:
                if not self._attach():
                    self._stop.wait(0.5)  # DE-Server creates the mapping at its first acquisition
                continue
            got = pending or self.producer.poll_request(timeout=0.05)
            pending = None
            if got is None:
                if self._pool is not None and _now() - self._pool_used > POOL_IDLE_S:
                    self._close_pool()
                continue
            self.requests_served += 1
            try:
                pending = self._serve(*got)
            except Exception:  # keep serving later requests
                log.exception("failed serving request %d", got[0])

    def _serve(self, request_id: int, request):
        """Publish frames for one request; returns a newer request if one pre-empts it.

        The twin renders on a thread of its own into a small pool (`_FramePool`), so a
        slow render does not starve DE-Server: while it runs, the last frame is
        republished every `KEEPALIVE_S`. With ``reuse > 1`` the pool also recycles: each
        rendered frame may be published up to ``reuse`` times, fresh frames first, so
        the stream runs at the frame rate DE-Server asks for rather than at the rate the
        detector physics can manage.
        """
        import time

        p = self.producer
        h = p.hdr
        shape = (h.frame_height, h.frame_width)
        dtype = np.uint8 if h.bytes_per_pixel == 1 else np.uint16
        recycle = self.reuse > 1
        pool = self._pool_for(request, recycle)
        frame_time = max(float(request.frame_time_s), 1e-6)
        t0 = time.monotonic()
        k = 0
        try:
            while True:
                if self._stop.is_set() or p.request_changed(request_id):
                    return p.poll_request()
                got = pool.take(KEEPALIVE_S)
                if got is _DONE:
                    return None
                if isinstance(got, BaseException):
                    raise got
                if got is None:  # nothing rendered for this request yet
                    last = self._last_frame
                    if last is None or last[0].shape != shape or last[0].dtype != dtype:
                        continue
                    got = (last[1], False)
                (raw, meta), fresh = got
                if not fresh and not recycle:
                    self.keepalives += 1
                while not p.wait_slot_free(timeout=0.05):
                    if self._stop.is_set() or p.request_changed(request_id):
                        return p.poll_request()
                out = _fit(raw, shape, dtype)
                p.publish(out, request_id=request_id, frame_index=k,
                          flags=L.FLAG_BLANKED if meta.blanked else 0)
                self._last_frame = (out, (raw, meta))
                self._pool_used = _now()
                self.frames_published += 1
                self.frames_reused += 0 if fresh else 1
                k += 1
                if request.total_frames > 0 and k >= request.total_frames:
                    return None
                if recycle and self.pace:
                    delay = t0 + k * frame_time - time.monotonic()
                    if delay > 0:
                        self._stop.wait(delay)
        finally:
            self._pool_used = _now()
            if self._pool_key is None:  # a one-off pool (scan, pinned seed)
                pool.close()
                if self._pool is pool:
                    self._pool = None

    def _pool_for(self, request, recycle: bool) -> "_FramePool":
        """The frame pool for *request*: the running one when the request asks for the
        same frames as the last (DE-MC's live view is a string of short acquisitions,
        one request each), so its rendered frames, and the keep-alive, carry over;
        otherwise a new one. A scan or a pinned seed depends on the frame index, so it
        always starts afresh."""
        import dataclasses

        if request.scan.enabled or request.seed is not None:
            key = None
        else:
            key = dataclasses.replace(request, total_frames=0, acquisition_index=0)
        pool = self._pool
        if key is not None and pool is not None and self._pool_key == key and pool.alive:
            return pool
        self._close_pool()
        live = key if key is not None else request
        pool = _FramePool(self.twin, live, size=max(2, self.pool_size if recycle else 2),
                          reuse=self.reuse, pace=self.pace and not recycle, threads=self.threads)
        self._pool, self._pool_key = pool, key
        return pool

    def _close_pool(self) -> None:
        pool, self._pool, self._pool_key = self._pool, None, None
        if pool is not None:
            pool.close()

def _now() -> float:
    import time

    return time.monotonic()


def _fit(frame: np.ndarray, shape: tuple[int, int], dtype) -> np.ndarray:
    """Crop or zero-pad to exactly the hw_frame DE-Server asked for."""
    if frame.shape == shape and frame.dtype == dtype:
        return frame
    out = np.zeros(shape, dtype)
    h, w = min(shape[0], frame.shape[0]), min(shape[1], frame.shape[1])
    out[:h, :w] = np.clip(frame[:h, :w], 0, np.iinfo(dtype).max)
    return out



class _FramePool:
    """The twin's frames for one request, rendered on a thread of their own.

    A ring of up to *size* frames of the CURRENT view: when the microscope state of a new
    frame differs from the last one's, the older frames are dropped, so a recycled frame
    never shows a view the column has left. `take` hands out the oldest frame not yet
    published, else (recycling) the least-used one still under *reuse* publications, else
    waits; after *keepalive* seconds with nothing new it hands back the last frame
    anyway. *threads* caps the numba threads the detector uses (leave cores for the
    consumer's own processing).
    """

    def __init__(self, twin, request, *, size: int, reuse: int, pace: bool,
                 threads: Optional[int] = None):
        self.size = int(size)
        self.reuse = max(1, int(reuse))
        self._cond = threading.Condition()
        self._ring: list = []  # [item, uses]
        self._last = None
        self._end = None  # _DONE or an exception
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._render, args=(twin, request, pace, threads),
                                        name="de-twin-shm-render", daemon=True)
        self._thread.start()

    def _render(self, twin, request, pace, threads) -> None:
        if threads:
            try:
                import numba

                numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
            except Exception:  # noqa: BLE001 - no numba: nothing to cap
                pass
        state = None
        try:
            for item in twin.frames(request, pace=pace, stop=self._stop):
                with self._cond:
                    if item[1].microscope != state:
                        state = item[1].microscope
                        self._ring = []  # a new view: nothing older may be shown
                    while not self._stop.is_set() and sum(1 for e in self._ring if e[1] == 0) >= self.size:
                        self._cond.wait(0.05)
                    if self._stop.is_set():
                        return
                    self._ring.append([item, 0])
                    if len(self._ring) > self.size:  # drop the most-used frame
                        worst = max(range(len(self._ring) - 1), key=lambda i: self._ring[i][1])
                        del self._ring[worst]
                    self._cond.notify_all()
            end = _DONE
        except Exception as e:  # noqa: BLE001 - raised on the publishing thread
            end = e
        with self._cond:
            self._end = end
            self._cond.notify_all()

    def take(self, keepalive: float):
        """``((raw, meta), fresh)``, ``_DONE``, an exception, or None (nothing yet)."""
        import time

        deadline = time.monotonic() + keepalive
        with self._cond:
            while True:
                fresh = [e for e in self._ring if e[1] == 0]
                if fresh:
                    e = fresh[0]
                    e[1] += 1
                    self._cond.notify_all()
                    self._last = e[0]
                    return e[0], True
                if self._end is not None:
                    return self._end
                if self.reuse > 1:
                    usable = [e for e in self._ring if e[1] < self.reuse]
                    if usable:
                        e = min(usable, key=lambda e: e[1])
                        e[1] += 1
                        self._last = e[0]
                        return e[0], False
                left = deadline - time.monotonic()
                if left <= 0:
                    return (self._last, False) if self._last is not None else None
                self._cond.wait(left)

    @property
    def alive(self) -> bool:
        with self._cond:
            return self._end is None and not self._stop.is_set()

    def close(self) -> None:
        self._stop.set()  # the renderer stops at its next frame; the twin's lock orders them
        with self._cond:
            self._cond.notify_all()
