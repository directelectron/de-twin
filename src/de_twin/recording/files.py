"""A recording's frames on disk, read by plain file reads of contiguous runs.

A recorded movie can be far larger than memory (a crystallization series of 400 scans is
859 GB), and the replay touches it all, a scan at a time, for as long as it plays. A
memory map's pages count against the process however they are touched, so reading through
one grows the working set with the file; here every read is a ``readinto`` of a run of
neighbouring frames, straight into the caller's array or into one reused buffer, and the
process's memory stays flat however long it plays. What the operating system caches of the
file is its to drop.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass

import numpy as np

#: The most one reused read buffer holds.
SCRATCH_BYTES = 32 << 20
#: Wanted frames this many or fewer apart are read as one run (and the gap discarded):
#: one read of a few frames more is cheaper than another read.
MERGE_GAP = 4


@dataclass(frozen=True)
class FramePart:
    """One file's frames: *frames* of them from byte *offset*, the first being frame
    *first* of the recording."""

    path: str
    offset: int
    frames: int
    first: int


class FrameFiles:
    """Frames of one shape and type, back to back in one or more files (*parts*, in
    recording order). Thread-safe; reads are serialised."""

    def __init__(self, parts: list[FramePart], shape: tuple[int, int], dtype) -> None:
        if not parts:
            raise ValueError("a recording needs at least one file of frames")
        self.parts = list(parts)
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = np.dtype(dtype)
        self.frame_bytes = self.shape[0] * self.shape[1] * self.dtype.itemsize
        self.total = max(part.first + part.frames for part in self.parts)
        self._starts = np.array([part.first for part in self.parts], np.int64)
        self._handles: dict[str, object] = {}
        self._scratch: np.ndarray | None = None
        self._lock = threading.Lock()
        #: Bytes read, reads made and the seconds they took.
        self.bytes_read = 0
        self.reads = 0
        self.read_s = 0.0

    @property
    def scratch_frames(self) -> int:
        return max(1, SCRATCH_BYTES // self.frame_bytes)

    def read(self, numbers: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Fill *out* (n, h, w) with frames *numbers* (n,); a number outside the recording
        (-1 for none) gives zeros. Returns *out*."""
        numbers = np.asarray(numbers, np.int64).reshape(-1)
        if out.shape != (len(numbers), *self.shape) or not out.flags.c_contiguous:
            raise ValueError(f"out must be C-contiguous {(len(numbers), *self.shape)}, not {out.shape}")
        valid = (numbers >= 0) & (numbers < self.total)
        if not valid.all():
            out[~valid] = 0
        slots = np.flatnonzero(valid)
        if not len(slots):
            return out
        wanted = numbers[slots]
        order = np.argsort(wanted, kind="stable")
        wanted, slots = wanted[order], slots[order]
        with self._lock:
            for begin, end in self._runs(wanted):
                self._place(wanted[begin:end], slots[begin:end], out)
        return out

    def _runs(self, wanted: np.ndarray):
        """(begin, end) index ranges of sorted *wanted* that one read each covers: frames at
        most `MERGE_GAP` apart, in one file, no more than the scratch buffer holds."""
        limit = self.scratch_frames
        part_of = np.searchsorted(self._starts, wanted, side="right") - 1
        breaks = np.flatnonzero((np.diff(wanted) > MERGE_GAP) | (np.diff(part_of) != 0)) + 1
        edges = [0, *breaks.tolist(), len(wanted)]
        for begin, end in itertools.pairwise(edges):
            first = int(wanted[begin])
            while begin < end:  # a run longer than the buffer: split it
                stop = int(np.searchsorted(wanted, first + limit, side="left"))
                stop = min(max(stop, begin + 1), end)
                yield begin, stop
                begin = stop
                if begin < end:
                    first = int(wanted[begin])

    def _place(self, wanted: np.ndarray, slots: np.ndarray, out: np.ndarray) -> None:
        first, count = int(wanted[0]), int(wanted[-1] - wanted[0] + 1)
        in_order = count == len(wanted) and (len(slots) == 1 or bool(np.all(np.diff(slots) == 1)))
        if in_order:  # the frames land side by side, in order: read them straight there
            self._read_into(first, out[int(slots[0]) : int(slots[0]) + count])
            return
        if self._scratch is None:
            self._scratch = np.empty((self.scratch_frames, *self.shape), self.dtype)
        run = self._scratch[:count]
        self._read_into(first, run)
        out[slots] = run[wanted - first]

    def _read_into(self, first: int, dest: np.ndarray) -> None:
        """Frames *first* onwards into *dest*, a C-contiguous (count, h, w) array, within one
        file (`_runs` never spans two)."""
        part = self.parts[int(np.searchsorted(self._starts, first, side="right")) - 1]
        handle = self._handles.get(part.path)
        if handle is None:
            handle = self._handles[part.path] = open(part.path, "rb", buffering=0)  # noqa: SIM115
        began = time.perf_counter()
        view = memoryview(dest.reshape(-1).view(np.uint8))
        handle.seek(part.offset + (first - part.first) * self.frame_bytes)
        done = 0
        while done < len(view):
            got = handle.readinto(view[done:])
            if not got:
                raise OSError(f"{part.path}: ended {len(view) - done} bytes early")
            done += got
        self.read_s += time.perf_counter() - began
        self.bytes_read += done
        self.reads += 1

    def close(self) -> None:
        with self._lock:
            handles, self._handles = self._handles, {}
            self._scratch = None
        for handle in handles.values():
            try:
                handle.close()
            except OSError:
                pass
