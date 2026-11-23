"""One deapi acquisition: frames rendered on a background thread, and image conversion for the wire."""

from __future__ import annotations

import contextlib
import itertools
import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..state import AcquisitionRequest, ExposureMode, FrameChunk
from .deapi_protocol import GPU_BUFFER_FRAMES, GPU_PUBLISH_S, PIXEL_DTYPES
from .deapi_request import (
    _CoilTimeline,
    camera_in_scan,
    park_position,
    parked,
    queued_scan_points,
    scan_coil_settings,
)

if TYPE_CHECKING:
    from .deapi_server import TwinFakeServer

log = logging.getLogger(__name__)


def _resample(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to (h, w), preserving dtype."""
    if image.shape == shape:
        return image
    ys = (np.arange(shape[0]) * image.shape[0] / shape[0]).astype(int)
    xs = (np.arange(shape[1]) * image.shape[1] / shape[1]).astype(int)
    return image[ys][:, xs]


def _gpu_image_dtype():
    from .gpu_frames import IMAGE

    return IMAGE


def _chunks(twin, request: AcquisitionRequest, *, pace: bool, stop: threading.Event):
    """*twin*'s frames for *request* as :class:`FrameChunk` s: its own when it makes them
    (`DigitalTwin.frame_chunks`), else its frames one at a time."""
    make = getattr(twin, "frame_chunks", None)
    if callable(make):
        yield from make(request, pace=pace, stop=stop)
        return
    for raw, meta in twin.frames(request, pace=pace, stop=stop):
        yield FrameChunk.of_frame(raw, meta)


def _sum(frames: np.ndarray) -> np.ndarray:
    """The sum of a stack of frames, without a float per pixel per frame for whole numbers
    (a chunk of 16-bit frames cannot overflow 64 bits)."""
    if frames.dtype.kind in "ui":
        return np.add.reduce(frames, axis=0, dtype=np.uint64)
    return np.add.reduce(frames, axis=0, dtype=np.float64)


def _convert(image: np.ndarray, pixel_format: int) -> np.ndarray:
    dtype = PIXEL_DTYPES[pixel_format]
    if dtype is np.float32:
        return np.ascontiguousarray(image, dtype=np.float32)
    info = np.iinfo(dtype)
    return np.ascontiguousarray(
        np.clip(np.rint(image) if image.dtype.kind == "f" else image, info.min, info.max).astype(dtype)
    )


class _Acquisition:
    """One start_acquisition: frames rendered by a background thread."""

    def __init__(self, server: TwinFakeServer, request: AcquisitionRequest, n_acq: int, movie: bool):
        self.request = request
        # 0 acquisitions = repeat until stopped (live view), as DE-Server does
        self.n_acq = max(0, int(n_acq))
        self.continuous = self.n_acq == 0
        self.movie = movie
        self.stop = threading.Event()
        self.cond = threading.Condition()
        self.last: np.ndarray | None = None
        self.last_meta = None
        self.sum: np.ndarray | None = None
        self.count = 0
        self.index = 0  # acquisition index within n_acq
        self.movie_frames: deque = deque()
        self.movie_next = 0
        self.error: str | None = None
        self.refs = None
        self.done = False
        self.started = time.time()
        self.gpu_images: list = []
        self.gpu_info: list = []
        self.gpu_batch_started = 0.0
        scan = request.scan
        #: A scan with ``Scan - Use DE Camera`` Off forms only the external (HAADF) image:
        #: the beam still dwells, but nothing of the camera's is kept.
        self.camera = not scan.enabled or scan.park or camera_in_scan(server.__getitem__)
        self.virtual = None
        if scan.enabled and not scan.park:
            nx, ny = scan.size
            self.virtual = np.zeros((5, ny, nx), np.float64)
        self._server = server
        self.thread = threading.Thread(target=self._run, name="TwinFakeServer-acq", daemon=True)

    def start(self) -> _Acquisition:
        self.thread.start()
        return self

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def _prepare_references(self) -> None:
        """Pick up (or measure) dark/gain references like DE-Server's RefFinder. A recording's
        frames were corrected before they were saved, and are served as they were."""
        self.refs = None
        processor = getattr(self._server.twin, "processor", None)
        if processor is None or self.request.exposure_mode in (ExposureMode.DARK, ExposureMode.GAIN):
            return
        if getattr(self._server.twin, "recording", None) is not None:
            return
        self.refs = processor.lookup(self.request)
        if getattr(self._server, "auto_references", False) and (self.refs is None or self.refs.gain is None):
            self.refs = processor.references(self.request)

    def _store_references(self) -> None:
        processor = getattr(self._server.twin, "processor", None)
        if processor is None or self.sum is None or self.stop.is_set():
            return
        if getattr(self._server.twin, "recording", None) is not None:
            return  # a recording's frames need none, and its sensor is not the camera's
        mean = (self.sum / self.count).astype(np.float32)
        mode = self.request.exposure_mode
        if mode == ExposureMode.DARK:
            processor.store_dark(self.request, mean, self.count)
        elif mode == ExposureMode.GAIN:
            try:
                processor.store_gain(self.request, mean, self.count)
            except RuntimeError as exc:
                log.warning("gain reference not stored: %s", exc)

    def _run(self) -> None:
        acquiring = getattr(self._server.twin, "acquiring", None)
        try:
            with acquiring(self.request) if callable(acquiring) else contextlib.nullcontext():
                self._acquire()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("twin acquisition failed:\n%s", traceback.format_exc())
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()

    def _acquire(self) -> None:
        srv = self._server
        self._prepare_references()
        for a in itertools.count() if self.continuous else range(self.n_acq):
            self.index = a
            if self.continuous and a:
                with self.cond:
                    self.sum = None  # the sum is per repetition, like one live buffer
                # a client steers the probe while it scans: take the coils' settings now
                if self.request.scan.enabled:
                    self.request = replace(
                        self.request,
                        scan=replace(self.request.scan, **scan_coil_settings(srv.__getitem__)),
                    )
            # DE-Server moves a parked beam the moment the position is written; here it
            # moves at the next acquisition (a parked probe runs one after another).
            if a and self.request.scan.park and parked(srv.__getitem__):
                self.request = replace(
                    self.request,
                    scan=replace(self.request.scan, park_position=park_position(srv.__getitem__)),
                )
            if self.request.scan.enabled:
                self._scan()
            else:
                for chunk in _chunks(srv.twin, self.request, pace=srv.pace, stop=self.stop):
                    self._add(chunk)
                    if self.stop.is_set():
                        break
            if self.stop.is_set():
                break
        self._share_gpu_frames()
        self._store_references()

    def _scan(self) -> None:
        """One scan acquisition, a repeat at a time, each repeat with the coil settings
        DE-Server would have written for it (see :class:`_CoilTimeline`)."""
        srv, request = self._server, self.request
        scan = request.scan
        repeats = max(1, scan.repeats)
        frames_per_repeat = request.total_frames // repeats
        points_per_repeat = max(1, frames_per_repeat // max(1, scan.frames_per_point))
        queued = queued_scan_points(srv.__getitem__, 1.0 / max(request.frame_time_s, 1e-9))
        timeline = _CoilTimeline({name: getattr(scan, name) for name in scan_coil_settings(srv.__getitem__)})
        frames = 0
        for repeat in range(repeats):
            settings = timeline.at(repeat * points_per_repeat - queued)
            this_repeat = replace(
                request, total_frames=frames_per_repeat, scan=replace(scan, repeats=1, **settings)
            )
            for chunk in _chunks(srv.twin, this_repeat, pace=srv.pace, stop=self.stop):
                self._add(chunk)
                frames += len(chunk)
                timeline.record(frames // max(1, scan.frames_per_point), scan_coil_settings(srv.__getitem__))
                if self.stop.is_set():
                    return

    def _share_gpu_frames(self) -> None:
        """Hand the frames gathered so far to a GPU frames client, if one is attached."""
        shared = self._server._shared_frames
        if not self.gpu_images or shared is None:
            self.gpu_images, self.gpu_info = [], []
            return
        images = np.stack(self.gpu_images).astype(np.float32, copy=False)
        info = np.zeros(len(images), _gpu_image_dtype())
        info["frame"], info["scanX"], info["scanY"] = np.array(self.gpu_info).T
        info["valid"] = 1
        self.gpu_images, self.gpu_info = [], []
        shared.allocate(GPU_BUFFER_FRAMES, *images.shape[1:])
        shared.publish(images, info, stop=self.stop)

    def _gather_gpu_frames(self, frames: np.ndarray, chunk: FrameChunk, first_count: int) -> None:
        shared = self._server._shared_frames
        if shared is None or not shared.attached:
            return
        for i, frame in enumerate(frames):
            if not self.gpu_images:
                self.gpu_batch_started = time.monotonic()
            point = chunk.point(i)
            self.gpu_images.append(frame)
            self.gpu_info.append((first_count + i, *(point if point is not None else (-1, -1))))
            if (
                len(self.gpu_images) >= GPU_BUFFER_FRAMES
                or time.monotonic() - self.gpu_batch_started >= GPU_PUBLISH_S
            ):
                self._share_gpu_frames()

    def _add(self, chunk: FrameChunk) -> None:
        """Take in a chunk of frames: the last one, the sum, the movie and the virtual
        detectors, a stack at a time (the virtual detectors are the masks, as a matrix,
        times the stack)."""
        frames = chunk.frames
        n = len(frames)
        if not n:
            return
        last_meta = chunk.meta(n - 1)
        if not self.camera:
            with self.cond:
                self.last_meta = last_meta
                self.count += n
                self.cond.notify_all()
            return
        if self.refs is not None:
            correct = self._server.twin.processor.correct
            frames = np.stack([correct(frame, self.refs) for frame in frames])
        vi_values = None
        if self.virtual is not None and chunk.points is not None:
            weights = self._server._mask_weights(frames.shape[1:])
            vi_values = frames.reshape(n, -1).astype(np.float32, copy=False) @ weights.T  # (n, 5)
        total = _sum(frames)
        # a chunk of one is the twin's own frame; from a larger one the last is copied, so the
        # chunk's buffer is free to be used again
        last = frames[-1] if n == 1 else frames[-1].copy()
        with self.cond:
            first_count = self.count
            self.last = last
            self.last_meta = last_meta
            self.sum = total.astype(np.float64) if self.sum is None else self.sum + total
            self.count += n
            if self.movie:
                self.movie_frames.extend(frames)
            if vi_values is not None:
                ix, iy = chunk.points[:, 0], chunk.points[:, 1]
                ny, nx = self.virtual.shape[1:]
                inside = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
                np.add.at(self.virtual, (slice(None), iy[inside], ix[inside]), vi_values[inside].T)
            self.cond.notify_all()
        self._gather_gpu_frames(frames, chunk, first_count)

    def wait_first(self, timeout: float) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: self.count > 0 or self.done, timeout)
