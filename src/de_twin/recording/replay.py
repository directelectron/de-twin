"""A recording served as the twin's camera: where each frame is read from, and how.

**Where.** Every frame of a scan is taken where the scan coils put the beam: the scan point,
through the coil voltages its scan asks for (size, pixel offset, voltage range, rotation,
flips: :mod:`de_twin.optics.scan_coils`), lands on a point of the recorded field, which was
scanned at the default range at the recording's size. The frame recorded at that point is
served; off the recorded field, zeros. So a scan of the recording's size replays it point
for point, a smaller scan has a larger step and subsamples it (the same field), a region,
an XY array or an offset reads the recorded points under it, and a frame with no scan is
the field's middle point. Through DE-Server's shared memory the voltages are the ones
DE-Server output, so whatever steered its scan steers the replay.

**When.** Which recorded scan a frame comes from is :class:`~.clock.RecordedClock`'s.

**The frame.** The sensor is the recording's frame. A hardware ROI is cut from the middle
of it, and binning sums.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator

import numpy as np

from ..optics.scan_coils import points_from_volts, points_volts
from ..state import AcquisitionRequest, FrameChunk, ScanRequest
from .clock import RecordedClock
from .formats import Recording, Unreadable

#: How much acquisition time one chunk of frames covers: the frames of a live view still
#: arrive in good time, and a fast scan is read a few hundred frames at a time.
CHUNK_S = 0.05
#: The most frames, and bytes, in one chunk.
CHUNK_FRAMES = 1024
CHUNK_BYTES = 32 << 20
#: Chunk buffers kept for reuse.
RING = 4


def _free_refs() -> int:
    """References a ring entry nothing else holds has, as ``sys.getrefcount(ring[i])``
    counts them (calibrated: the count differs between Python versions)."""
    ring = [np.empty(1)]
    return sys.getrefcount(ring[0])


_FREE_REFS = _free_refs()


def chunk_frames(frame_time_s: float, frame_bytes: int) -> int:
    """Frames per chunk at *frame_time_s*, of frames of *frame_bytes*."""
    most = min(CHUNK_FRAMES, max(1, CHUNK_BYTES // max(1, int(frame_bytes))))
    return int(min(most, max(1, CHUNK_S / max(float(frame_time_s), 1e-9))))


class Replay:
    """*recording* as the twin's camera, on a :class:`RecordedClock` of its own."""

    def __init__(self, recording: Recording, *, clock: RecordedClock | None = None) -> None:
        if recording.passes > 1 and not recording.frame_time_s:
            raise Unreadable(f"{recording.name}: {recording.passes} scans, but nothing says its frame time")
        self.recording = recording
        self.clock = clock or RecordedClock(
            recording.frame_time_s or 0.0, recording.pass_frames, recording.passes
        )
        #: The scan the recording was taken with: its field, at the default voltage range.
        self.field_scan = ScanRequest(enabled=True, size=recording.field)
        self._ring: list[np.ndarray] = []
        self._points_cache: tuple[object, np.ndarray] | None = None
        self._lock = threading.Lock()
        #: Frames served so far.
        self.frames_served = 0

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.recording.frame_shape

    # ── where ──
    def scan_points(self, request: AcquisitionRequest, first: int, count: int) -> np.ndarray | None:
        """The scan point (x, y) of frames *first* .. *first* + *count* of *request*, an
        (n, 2) int array, as :meth:`DigitalTwin.scan_point` gives them; None without a scan."""
        scan = request.scan
        if not scan.enabled:
            return None
        if scan.park:
            return np.tile(np.asarray(scan.park_position, np.int64), (count, 1))
        point = (first + np.arange(count, dtype=np.int64)) // max(1, int(scan.frames_per_point))
        if scan.points:
            listed = self._listed(scan.points)
            return listed[point % len(listed)]
        nx, ny = (max(1, int(v)) for v in scan.size)
        point %= nx * ny
        return np.stack([point % nx, point // nx], axis=1)

    def _listed(self, points) -> np.ndarray:
        # kept with the list itself, so a scan's list is converted once, and a new list is
        # never taken for the old one
        cached = self._points_cache
        if cached is None or cached[0] is not points:
            cached = self._points_cache = (points, np.asarray(points, np.int64).reshape(-1, 2))
        return cached[1]

    def recorded_points(self, volts: np.ndarray | None, count: int) -> tuple[np.ndarray, np.ndarray]:
        """(columns, rows) of the recorded points the beam is on at *volts* ((n, 2) H, V);
        the field's middle point for every frame when *volts* is None (no scan)."""
        nx, ny = self.recording.field
        if volts is None:
            return np.full(count, nx // 2, np.int64), np.full(count, ny // 2, np.int64)
        xy = points_from_volts(self.field_scan, volts)
        # the recorded point whose cell holds the beam (a hair over, so that a scan point
        # half way between two recorded ones lands the same way however it was rounded)
        at = np.floor(xy + 0.5 + 1e-9)
        bad = ~np.isfinite(at).all(axis=1)
        at[bad] = -1
        return at[:, 0].astype(np.int64), at[:, 1].astype(np.int64)

    # ── reading ──
    def _buffer(self, count: int) -> np.ndarray:
        """A (count, h, w) array for frames, from a buffer nothing else holds any more."""
        ring = self._ring
        for i in range(len(ring)):
            # held by the ring and the call only: whoever had its frames has let them go
            if ring[i].shape[0] >= count and sys.getrefcount(ring[i]) <= _FREE_REFS:
                return ring[i][:count]
        frame_bytes = int(np.prod(self.frame_shape)) * self.recording.dtype.itemsize
        size = max(count, min(CHUNK_FRAMES, CHUNK_BYTES // frame_bytes))
        ring.append(np.empty((size, *self.frame_shape), self.recording.dtype))
        if len(ring) > RING:  # forget a free one; those in use are their holders'
            free = [i for i in range(len(ring) - 1) if sys.getrefcount(ring[i]) <= _FREE_REFS]
            del ring[free[0] if free else 0]
        return ring[-1][:count]

    def read(self, passes: np.ndarray, columns: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """The recorded frames at (*columns*, *rows*) of recorded scans *passes*, (n, h, w);
        zeros off the field."""
        numbers = self.recording.locate(passes, rows, columns)
        with self._lock:
            out = self._buffer(len(numbers))
            self.recording.files.read(numbers, out)
            self.frames_served += len(numbers)
        return out

    def finish(self, frames: np.ndarray, request: AcquisitionRequest) -> np.ndarray:
        """*frames* as the camera reads them out for *request*: the hardware ROI's size cut
        from the middle of the recorded frame, then binned by summing."""
        height, width = self.frame_shape
        roi = request.hw_roi
        h = height if roi is None or roi.h <= 0 else min(int(roi.h), height)
        w = width if roi is None or roi.w <= 0 else min(int(roi.w), width)
        bx, by = (max(1, int(v)) for v in (request.hw_binning or (1, 1)))
        h, w = (h // by) * by, (w // bx) * bx
        y0, x0 = (height - h) // 2, (width - w) // 2
        out = frames[:, y0 : y0 + h, x0 : x0 + w] if (h, w) != (height, width) else frames
        if (bx, by) != (1, 1):
            n = len(out)
            wide = np.uint32 if out.dtype.kind in "ui" else np.float32
            out = out.reshape(n, h // by, by, w // bx, bx).sum(axis=(2, 4), dtype=wide)
        return out

    def output_shape(self, request: AcquisitionRequest) -> tuple[int, int]:
        return self.finish(np.empty((0, *self.frame_shape), self.recording.dtype), request).shape[1:]

    # ── serving ──
    def chunk(
        self, request: AcquisitionRequest, first: int, count: int, *, volts=None, blanked: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Frames *first* .. *first* + *count* of *request*, finished, and their scan points.

        *volts*: the coil voltages of each frame when the caller has them (DE-Server's own);
        else those *request*'s scan gives. *blanked*: the beam does not reach the camera, so
        the frames are zeros (the recording's dark was taken off before it was saved). The
        recorded clock moves on by *count* frames either way."""
        points = self.scan_points(request, first, count)
        if volts is None and points is not None:
            volts = points_volts(request.scan, points)
        passes = self.clock.take(count)
        if blanked:
            with self._lock:
                frames = self._buffer(count)
            frames[:] = 0
        else:
            columns, rows = self.recorded_points(volts, count)
            frames = self.read(passes, columns, rows)
        return self.finish(frames, request), points

    def chunks(
        self, twin, request: AcquisitionRequest, *, pace: bool = False, stop=None
    ) -> Iterator[FrameChunk]:
        """*request*'s frames, a chunk at a time, for :meth:`DigitalTwin.frame_chunks`."""
        frame_time = max(float(request.frame_time_s), 1e-9)
        per_chunk = chunk_frames(frame_time, self.recording.files.frame_bytes)
        total = int(request.total_frames)
        state = twin.column.state()
        blanked = not twin.beam_on_detector(request, state)
        t_start = twin.clock.now()
        i = 0
        while total <= 0 or i < total:
            if stop is not None and stop.is_set():
                return
            count = per_chunk if total <= 0 else min(per_chunk, total - i)
            t = twin.clock.now()
            frames, points = self.chunk(request, i, count, blanked=blanked)
            if pace:
                twin.clock.sleep_until(t_start + (i + count) * frame_time)
            yield FrameChunk(
                frames=frames,
                first_index=i,
                points=points,
                time_s=t,
                exposure_s=frame_time,
                render_mode=self.recording.render_mode,
                blanked=blanked,
                microscope=state,
                holder=twin.holder_state(t),
            )
            i += count

    def state(self) -> dict:
        """What is loaded, where its clock is, and what has been read."""
        files = self.recording.files
        return {
            **self.recording.describe(),
            "clock": self.clock.state(),
            "frames_served": self.frames_served,
            "read_gb": round(files.bytes_read / 1e9, 3),
            "read_s": round(files.read_s, 2),
        }

    def close(self) -> None:
        self.recording.close()
        self._ring = []
