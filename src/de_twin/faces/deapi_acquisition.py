"""One deapi acquisition: frames rendered on a background thread, and image conversion for the wire."""

from __future__ import annotations

import itertools
import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING, Optional
import numpy as np
from ..state import AcquisitionRequest, ExposureMode
from .deapi_protocol import GPU_BUFFER_FRAMES, GPU_PUBLISH_S, PIXEL_DTYPES
from .deapi_request import _CoilTimeline, queued_scan_points, scan_coil_settings

if TYPE_CHECKING:
    from .deapi_server import TwinFakeServer

log = logging.getLogger(__name__)


def _resample(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to (h, w), preserving dtype."""
    if img.shape == shape:
        return img
    ys = (np.arange(shape[0]) * img.shape[0] / shape[0]).astype(int)
    xs = (np.arange(shape[1]) * img.shape[1] / shape[1]).astype(int)
    return img[ys][:, xs]


def _gpu_image_dtype():
    from .gpu_frames import IMAGE

    return IMAGE


def _convert(img: np.ndarray, pixel_format: int) -> np.ndarray:
    dtype = PIXEL_DTYPES[pixel_format]
    if dtype is np.float32:
        return np.ascontiguousarray(img, dtype=np.float32)
    info = np.iinfo(dtype)
    return np.ascontiguousarray(np.clip(np.rint(img) if img.dtype.kind == "f" else img,
                                        info.min, info.max).astype(dtype))


class _Acquisition:
    """One start_acquisition: frames rendered by a background thread."""

    def __init__(self, server: "TwinFakeServer", request: AcquisitionRequest, n_acq: int,
                 movie: bool):
        self.request = request
        # 0 acquisitions = repeat until stopped (live view), as DE-Server does
        self.n_acq = max(0, int(n_acq))
        self.continuous = self.n_acq == 0
        self.movie = movie
        self.stop = threading.Event()
        self.cond = threading.Condition()
        self.last: Optional[np.ndarray] = None
        self.last_meta = None
        self.sum: Optional[np.ndarray] = None
        self.count = 0
        self.index = 0  # acquisition index within n_acq
        self.movie_frames: deque = deque()
        self.movie_next = 0
        self.error: Optional[str] = None
        self.refs = None
        self.done = False
        self.started = time.time()
        self.gpu_images: list = []
        self.gpu_info: list = []
        self.gpu_batch_started = 0.0
        scan = request.scan
        self.virtual = None
        if scan.enabled:
            nx, ny = scan.size
            self.virtual = np.zeros((5, ny, nx), np.float64)
        self._server = server
        self.thread = threading.Thread(target=self._run, name="TwinFakeServer-acq", daemon=True)

    def start(self) -> "_Acquisition":
        self.thread.start()
        return self

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def _prepare_references(self) -> None:
        """Pick up (or measure) dark/gain references like DE-Server's RefFinder."""
        self.refs = None
        processor = getattr(self._server.twin, "processor", None)
        if processor is None or self.request.exposure_mode in (ExposureMode.DARK, ExposureMode.GAIN):
            return
        self.refs = processor.lookup(self.request)
        if getattr(self._server, "auto_references", False) and (self.refs is None or self.refs.gain is None):
            self.refs = processor.references(self.request)

    def _store_references(self) -> None:
        processor = getattr(self._server.twin, "processor", None)
        if processor is None or not self.count or self.stop.is_set():
            return
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
        srv = self._server
        try:
            self._prepare_references()
            for a in (itertools.count() if self.continuous else range(self.n_acq)):
                self.index = a
                if self.continuous and a:
                    with self.cond:
                        self.sum = None  # the sum is per repetition, like one live buffer
                    # a client steers the probe while it scans: take the coils' settings now
                    if self.request.scan.enabled:
                        self.request = replace(self.request, scan=replace(
                            self.request.scan, **scan_coil_settings(srv.__getitem__)))
                if self.request.scan.enabled:
                    self._scan()
                else:
                    for raw, meta in srv.twin.frames(self.request, pace=srv.pace, stop=self.stop):
                        self._add(raw, meta)
                        if self.stop.is_set():
                            break
                if self.stop.is_set():
                    break
            self._share_gpu_frames()
            self._store_references()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("twin acquisition failed:\n%s", traceback.format_exc())
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()

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
            this_repeat = replace(request, total_frames=frames_per_repeat,
                                  scan=replace(scan, repeats=1, **settings))
            for raw, meta in srv.twin.frames(this_repeat, pace=srv.pace, stop=self.stop):
                self._add(raw, meta)
                frames += 1
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

    def _gather_gpu_frame(self, frame: np.ndarray, meta) -> None:
        shared = self._server._shared_frames
        if shared is None or not shared.attached:
            return
        if not self.gpu_images:
            self.gpu_batch_started = time.monotonic()
        point = getattr(meta, "scan_point", None)
        self.gpu_images.append(frame)
        self.gpu_info.append((self.count - 1, *(point if point is not None else (-1, -1))))
        if len(self.gpu_images) >= GPU_BUFFER_FRAMES or time.monotonic() - self.gpu_batch_started >= GPU_PUBLISH_S:
            self._share_gpu_frames()

    def _add(self, raw: np.ndarray, meta) -> None:
        if self.refs is not None:
            raw = self._server.twin.processor.correct(raw, self.refs)
        masks = self._server._masks_for(raw.shape) if self.virtual is not None else None
        vi_vals = None
        if masks is not None and getattr(meta, "scan_point", None) is not None:
            f = raw.astype(np.float64, copy=False)
            vi_vals = []
            for k, (pos, neg, calc) in enumerate(masks):
                s = float(f[pos].sum())
                if calc == "difference":
                    s -= float(f[neg].sum())
                elif neg is not None:
                    s += float(f[neg].sum())
                vi_vals.append(s)
        with self.cond:
            self.last = raw
            self.last_meta = meta
            self.sum = raw.astype(np.float64) if self.sum is None else self.sum + raw
            self.count += 1
            if self.movie:
                self.movie_frames.append(raw)
            if vi_vals is not None:
                ix, iy = meta.scan_point
                ny, nx = self.virtual.shape[1:]
                if 0 <= iy < ny and 0 <= ix < nx:
                    self.virtual[:, iy, ix] += vi_vals
            self.cond.notify_all()
        self._gather_gpu_frame(raw, meta)

    def wait_first(self, timeout: float) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: self.count > 0 or self.done, timeout)
