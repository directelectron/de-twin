"""Shared-memory face: the twin as DE-Server's external frame source.

DE-Server (test pattern "External Frame Source (Shared Memory)") creates the shared
memory and publishes each acquisition's request. This face turns the request into an
``AcquisitionRequest``, renders frames through the twin and publishes them, paced to the
frame time and throttled by DE-Server's reads.

In a scan, DE-Server's scan simulator also shares the coil voltages it outputs
(:mod:`de_twin.transport.scan_output`), as the scan card drives a microscope's coils. Each
frame is then rendered where those voltages put the beam, so whatever steers the scan (a
pixel offset, an XY array, drift correction) moves the beam with the latency DE-Server's
output buffer gives it on hardware. Without them the twin rasters the grid by frame index.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections import OrderedDict
from typing import Optional

import numpy as np

from ..optics.scan_coils import point_from_volts
from ..transport import scan_output as S
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
    publishes every frame once. In a scan steered by DE-Server's voltages, the frames are
    kept per scan point instead: a probe held on one point is rendered ``pool_size``
    times and then picked from.
    """

    def __init__(self, twin, name: str = L.DEFAULT_NAME, pace: bool = True, warm_up: bool = True,
                 *, reuse: int = 1, pool_size: int = 16, threads: Optional[int] = None,
                 scan_output: str = S.DEFAULT_NAME):
        self.twin = twin
        self.scan_output_name = scan_output
        self._scan_output: Optional[S.ScanOutputReader] = None
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
        if self._scan_output is not None:
            self._scan_output.close()

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
        if request.scan.enabled:
            coils = self._coils()
            if coils is not None:
                return self._serve_steered(request_id, request, coils)
            if self.reuse == 1 and self._fused():
                feed = _RasterFeed.make(self.twin, request)
                if feed is not None and self._fits(feed):
                    return self._serve_fused(request_id, request, feed)
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
                while not p.wait_slot_free(timeout=0.05):
                    if self._stop.is_set() or p.request_changed(request_id):
                        return p.poll_request()
                # as many frames as DE-Server has room for, in one go
                count = p.free_slots() if recycle else 1
                if request.total_frames > 0:
                    count = min(count, request.total_frames - k)
                got = pool.take_many(count, KEEPALIVE_S)
                if got is _DONE:
                    return None
                if isinstance(got, BaseException):
                    raise got
                if got is None:  # nothing rendered for this request yet
                    last = self._last_frame
                    if last is None or last[0].shape != shape or last[0].dtype != dtype:
                        continue
                    got = [(last[1], False)]
                for (raw, meta), fresh in got:
                    if not fresh and not recycle:
                        self.keepalives += 1
                    out = _fit(raw, shape, dtype)
                    p.publish(out, request_id=request_id, frame_index=k,
                              flags=L.FLAG_BLANKED if meta.blanked else 0)
                    self.frames_reused += 0 if fresh else 1
                    k += 1
                self._last_frame = (out, (raw, meta))
                self._pool_used = _now()
                self.frames_published += len(got)
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

    def _coils(self) -> Optional[S.ScanOutputReader]:
        """DE-Server's scan output, once its scan simulator has made one."""
        if self._scan_output is None:
            try:
                self._scan_output = S.ScanOutputReader(self.scan_output_name)
            except (FileNotFoundError, RuntimeError):
                return None
        return self._scan_output

    def _serve_steered(self, request_id: int, request, coils: S.ScanOutputReader):
        """`_serve` for a scan whose beam DE-Server's coil voltages place: each frame is
        rendered at the grid point its scan point's voltages put the beam on, through a
        library per point (`_PointLibrary`). Rendering happens on this thread, between
        publications, so nothing competes with publishing for the interpreter."""
        p = self.producer
        h = p.hdr
        shape = (h.frame_height, h.frame_width)
        dtype = np.uint8 if h.bytes_per_pixel == 1 else np.uint16
        library = _PointLibrary(self.twin, request, coils, size=self.pool_size if self.reuse > 1 else 0)
        if self.reuse == 1 and self._fused():
            feed = _SteeredFeed(self.twin, library)
            if feed.source is not None and self._fits(feed):
                try:
                    return self._serve_fused(request_id, request, feed)
                finally:
                    library.close()
        frame_time = max(float(request.frame_time_s), 1e-6)
        t0 = time.monotonic()
        k = 0
        try:
            while True:
                if self._stop.is_set() or p.request_changed(request_id):
                    return p.poll_request()
                while not p.wait_slot_free(timeout=0.05):
                    if self._stop.is_set() or p.request_changed(request_id):
                        return p.poll_request()
                count = p.free_slots()
                if request.total_frames > 0:
                    count = min(count, request.total_frames - k)
                if count <= 0:
                    continue
                for (raw, meta), fresh in library.frames(k, count):
                    out = _fit(raw, shape, dtype)
                    p.publish(out, request_id=request_id, frame_index=k,
                              flags=L.FLAG_BLANKED if meta.blanked else 0)
                    self.frames_reused += 0 if fresh else 1
                    k += 1
                self._last_frame = (out, (raw, meta))
                self.frames_published += count
                if request.total_frames > 0 and k >= request.total_frames:
                    return None
                if self.pace:
                    delay = t0 + k * frame_time - time.monotonic()
                    if delay > 0:
                        self._stop.wait(delay)
        finally:
            library.close()

    def _fused(self) -> bool:
        """Whether the twin can make frames with the fused EBSD kernel."""
        return callable(getattr(self.twin, "ebsd_frame_source", None))

    def _fits(self, feed) -> bool:
        """Whether *feed*'s frames are the frames DE-Server asked for (shape and type)."""
        h = self.producer.hdr
        return h.bytes_per_pixel == 2 and tuple(feed.source.shape) == (h.frame_height, h.frame_width)

    #: Frames a worker makes per job (see `DigitalTwin.FRAMES_PER_JOB`).
    FRAMES_PER_JOB = 4

    def _serve_fused(self, request_id: int, request, feed):
        """`_serve` for frames made by the fused EBSD kernel (`DigitalTwin.ebsd_frame_source`):
        a few frames a job for the worker threads (:mod:`de_twin.workers`), submitted ahead
        into a ring of frame buffers, and published in order as DE-Server frees its slots.
        DE-Server's ring holds a few frames, too few to keep every core busy, so the frames
        are made in buffers of the twin's and copied into the slots (~40 us a frame).

        A steered scan's frames are made only once DE-Server has written the voltages that
        place their beam (`_SteeredFeed.settled`), as far ahead as it writes them."""
        from collections import deque

        from ..workers import pool

        p = self.producer
        h = p.hdr
        shape = (h.frame_height, h.frame_width)
        workers = pool()
        per_job = self.FRAMES_PER_JOB
        jobs_ahead = workers.threads + workers.threads // 2
        ring = [np.empty((per_job, *shape), np.uint16) for _ in range(jobs_ahead + 1)]
        frame_time = max(float(request.frame_time_s), 1e-6)
        total = request.total_frames
        pending: deque = deque()  # (first frame, frames, slot, batch)
        t0 = time.monotonic()
        k = nxt = 0
        n_jobs = 0
        last = None

        def drain():
            for *_, batch in pending:
                batch.wait_done(10.0)
            pending.clear()

        try:
            while True:
                if self._stop.is_set() or p.request_changed(request_id):
                    return p.poll_request()
                if feed.changed():  # a new scan, or the column moved: frames from k on afresh
                    drain()
                    nxt = k
                limit = 1 << 62 if total <= 0 else total
                limit = min(limit, max(k + 1, feed.settled()))
                while len(pending) < jobs_ahead and nxt < limit:
                    count = min(per_job, limit - nxt)
                    slot = n_jobs % len(ring)
                    ks, serials, seed = [], [], 0
                    for j, point in enumerate(feed.points(nxt, count), nxt):
                        ks.append(feed.flat(point, j))
                        serial, seed = self.twin.noise_serial(request, j)
                        serials.append(serial)
                    pending.append((nxt, count, slot, feed.source.submit(workers, ring[slot], ks, seed, serials)))
                    nxt += count
                    n_jobs += 1
                first, count, slot, batch = pending[0]
                if not batch.wait_done(KEEPALIVE_S):  # a slow start (the kernels compiling)
                    if last is not None and p.free_slots() > 0:
                        p.publish(last, request_id=request_id, frame_index=k)
                        self.keepalives += 1
                    continue
                batch.wait()
                pending.popleft()
                for b in range(k - first, count):  # (after a restart, the frames before k are done)
                    while not p.wait_slot_free(timeout=0.05):
                        if self._stop.is_set() or p.request_changed(request_id):
                            return p.poll_request()
                    last = ring[slot][b]
                    p.publish(last, request_id=request_id, frame_index=k)
                    k += 1
                    self.frames_published += 1
                    if total > 0 and k >= total:
                        return None
                    if self.pace:
                        delay = t0 + k * frame_time - time.monotonic()
                        if delay > 0:
                            self._stop.wait(delay)
        finally:
            drain()
            if last is not None:  # the keep-alive frame for what comes next
                frame = last.copy()
                self._last_frame = (frame, (frame, _BEAM_ON))

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
        # A scan's frames depend on their index, so only a fixed view is re-rendered from
        # the start when the column moves (diffraction and STEM freeze the optics for the
        # whole of a `frames` call, so a running one would never see the move).
        watch = not request.scan.enabled
        try:
            restart = True
            while restart:
                restart = False
                for item in twin.frames(request, pace=pace, stop=self._stop):
                    with self._cond:
                        if item[1].microscope != state:
                            state = item[1].microscope
                            self._ring = []  # a new view: nothing older may be shown
                        # A ring full of frames that may still be published is a library:
                        # stop rendering, which also stops taking the interpreter from the
                        # thread that publishes them, until the view changes.
                        while not self._stop.is_set() and self._publishable() >= self.size:
                            self._cond.wait(0.05)
                            if watch and twin.column.state() != state:
                                restart, self._ring = True, []
                                break
                        if self._stop.is_set():
                            return
                        if restart:
                            break
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

    def _publishable(self) -> int:
        return sum(1 for e in self._ring if e[1] < self.reuse)

    def take(self, keepalive: float):
        """``((raw, meta), fresh)``, ``_DONE``, an exception, or None (nothing yet)."""
        got = self.take_many(1, keepalive)
        return got[0] if isinstance(got, list) else got

    def take_many(self, count: int, keepalive: float):
        """Up to *count* frames as a list of ``((raw, meta), fresh)``, in the order
        :meth:`take` would hand them out, or what `take` returns when there are none. Waits
        only for the first: one lock and one pass over the ring for the lot is what lets a
        recycled stream reach tens of thousands of frames a second."""
        import time

        deadline = time.monotonic() + keepalive
        with self._cond:
            while True:
                taken = [e for e in self._ring if e[1] == 0][:count]
                for e in taken:
                    e[1] += 1
                out = [(e[0], True) for e in taken]
                if taken:
                    self._cond.notify_all()
                elif self._end is not None:
                    return self._end
                if self.reuse > 1 and len(out) < count and self._end is None:
                    usable = sorted((e for e in self._ring if e[1] < self.reuse), key=lambda e: e[1])
                    while usable and len(out) < count:  # least used first, round and round
                        for e in usable:
                            e[1] += 1
                            out.append((e[0], False))
                            if len(out) == count:
                                break
                        usable = [e for e in usable if e[1] < self.reuse]
                if out:
                    self._last = out[-1][0]
                    return out
                left = deadline - time.monotonic()
                if left <= 0:
                    return [(self._last, False)] if self._last is not None else None
                self._cond.wait(left)

    @property
    def alive(self) -> bool:
        with self._cond:
            return self._end is None and not self._stop.is_set()

    def close(self) -> None:
        self._stop.set()  # the renderer stops at its next frame; the twin's lock orders them
        with self._cond:
            self._cond.notify_all()


#: A scan's frame library holds at most this many bytes; the points used longest ago go.
LIBRARY_BYTES = 512 * 1024 * 1024
#: How often a library checks that the column has not moved under it.
VIEW_CHECK_S = 0.05
#: Frames whose beam positions a library works out at once.
POINTS_AHEAD = 1024


class _BeamOn:
    blanked = False


_BEAM_ON = _BeamOn()


class _RasterFeed:
    """A scan rastered by frame index (no coil voltages), for `ShmFace._serve_fused`."""

    def __init__(self, twin, request, source, optics):
        self.twin, self.request, self.source, self.optics = twin, request, source, optics

    @classmethod
    def make(cls, twin, request):
        state = twin.column.state()
        optics = twin.optics(request, state)
        source = twin.ebsd_frame_source(request, state=state, optics=optics)
        return None if source is None else cls(twin, request, source, optics)

    def changed(self) -> bool:
        return False  # a scan freezes the optics for its acquisition, as `DigitalTwin.frames`

    def settled(self) -> int:
        return 1 << 62

    def points(self, first: int, count: int) -> list:
        return [self.twin.scan_point(self.request, j) for j in range(first, first + count)]

    def flat(self, point, index: int) -> int:
        return self.twin.renderer.ebsd_flat_point(self.optics, point, index)


class _SteeredFeed:
    """A scan whose beam DE-Server's coil voltages place (`_PointLibrary`'s points), for
    `ShmFace._serve_fused`. A new scan or a move of the column makes a new source."""

    def __init__(self, twin, library: "_PointLibrary"):
        self.twin = twin
        self.library = library
        self.source = None
        library.check()
        self._remake()

    def _remake(self) -> None:
        lib = self.library
        request = dataclasses.replace(lib._request, total_frames=0, scan=lib.scan)
        self.optics = self.twin.optics(request, lib._state)
        self.source = self.twin.ebsd_frame_source(request, state=lib._state, optics=self.optics)

    def changed(self) -> bool:
        if self.library.check():
            self._remake()
            return True
        return False

    def settled(self) -> int:
        """Frames whose beam position DE-Server has written (the rest hold the last one)."""
        lib = self.library
        return int((lib.coils.written - 1) / lib.geometry.points_per_frame) + 1

    def points(self, first: int, count: int) -> list:
        return self.library.points(first, count)

    def flat(self, point, index: int) -> int:
        return self.twin.renderer.ebsd_flat_point(self.optics, point, index)


class _Wanted:
    """A scan's points as ``twin.frames`` reads them: whichever point is rendered next."""

    point = (0, 0)

    def __len__(self) -> int:
        return 1 << 62

    def __getitem__(self, index) -> tuple[int, int]:
        return self.point


class _PointLibrary:
    """Frames of one scan, rendered where DE-Server's coil voltages put the beam.

    Frame k covers scan point ``k * points_per_frame``; that point's voltages, through the
    inverse of the coil transform, give the grid point the beam is on (rounded, and kept
    on the grid: the twin renders the grid's field only). Voltages not written yet are
    the coils still driven at the last ones written, as a real scan card holds its output.

    With *size*, each grid point keeps up to *size* frames: a probe held on one point is
    rendered *size* times and then picked from. A move of the column, or a new scan,
    starts the library again.
    """

    def __init__(self, twin, request, coils: S.ScanOutputReader, *, size: int):
        self.twin = twin
        self.coils = coils
        self.size = int(size)
        self._request = request
        self._wanted = _Wanted()
        self._frames: "OrderedDict[tuple[int, int], list]" = OrderedDict()  # point -> [items, next]
        self._bytes = 0
        self._render = None
        self._scan_id = None
        self._state = None
        self._checked = 0.0
        self._stop = threading.Event()

    def _restart(self) -> None:
        g = self.coils.geometry()
        self._scan_id = self.coils.scan_id
        self.geometry = g
        self.scan = dataclasses.replace(self._request.scan, size=g.size, voltage_h=g.voltage_h,
                                        voltage_v=g.voltage_v, rotation_deg=g.rotation_deg, flip=g.flip,
                                        pixel_offset=(0.0, 0.0), points=self._wanted)
        if self._render is not None:
            self._render.close()
        # the coil transform is affine, so its inverse is too: (x, y) = (H, V) @ matrix + origin
        origin = np.array(point_from_volts(self.scan, 0.0, 0.0))
        self._matrix = np.array([point_from_volts(self.scan, 1.0, 0.0), point_from_volts(self.scan, 0.0, 1.0)]) - origin
        self._origin = origin
        self._state = self.twin.column.state()
        self._render = self.twin.frames(dataclasses.replace(self._request, total_frames=0, scan=self.scan),
                                        stop=self._stop)
        self._frames.clear()
        self._bytes = 0
        self._ahead = (0, [])

    def points(self, first_frame: int, count: int) -> list:
        """The grid points, a list of (x, y), the beam is on for frames *first_frame*
        onwards. DE-Server frees a slot or two at a time, so the points of frames whose
        voltages are already written are worked out POINTS_AHEAD at a time and kept."""
        first, ahead = self._ahead
        if first <= first_frame and first_frame + count <= first + len(ahead):
            return ahead[first_frame - first:first_frame - first + count]
        # keep only frames whose voltages are written: the others hold the last ones for now
        last_written = int((self.coils.written - 1) / self.geometry.points_per_frame)
        settled = max(0, min(POINTS_AHEAD, last_written - first_frame + 1))
        points = [tuple(p) for p in self._points(first_frame, max(count, settled)).tolist()]
        self._ahead = (first_frame, points[:settled])
        return points[:count]

    def _points(self, first_frame: int, count: int) -> np.ndarray:
        nx, ny = self.geometry.size
        index = (np.arange(first_frame, first_frame + count) * self.geometry.points_per_frame).astype(np.int64)
        volts = self.coils.volts_many(index)
        unset = np.isnan(volts[:, 0])
        xy = np.rint(volts @ self._matrix + self._origin)
        xy[unset] = np.stack([index[unset] % nx, (index[unset] // nx) % ny], axis=1)  # nothing output yet
        return np.clip(xy, 0, [nx - 1, ny - 1]).astype(np.int64)

    def check(self) -> bool:
        """Start again on a new scan, or when the column has moved; True when it did."""
        now = time.monotonic()
        if self._render is None or self.coils.scan_id != self._scan_id:
            self._restart()
            return True
        if now - self._checked > VIEW_CHECK_S:
            self._checked = now
            if self.twin.column.state() != self._state:
                self._restart()
                return True
        return False

    def frames(self, first_frame: int, count: int) -> list:
        """``((raw, meta), fresh)`` for each of *count* frames from *first_frame* on."""
        self.check()
        return [self._frame(point) for point in self.points(first_frame, count)]

    def _frame(self, point: tuple[int, int]):
        entry = self._frames.get(point)
        if entry is not None and len(entry[0]) >= self.size:
            self._frames.move_to_end(point)
            item = entry[0][entry[1] % len(entry[0])]
            entry[1] += 1
            return item, False
        self._wanted.point = point
        item = next(self._render)
        if self.size > 0:
            if entry is None:
                entry = self._frames[point] = [[], 0]
            entry[0].append(item)
            self._frames.move_to_end(point)
            self._bytes += item[0].nbytes
            while self._bytes > LIBRARY_BYTES and len(self._frames) > 1:
                _, (items, _) = self._frames.popitem(last=False)
                self._bytes -= sum(raw.nbytes for raw, _ in items)
        return item, True

    def close(self) -> None:
        self._stop.set()
        if self._render is not None:
            self._render.close()
