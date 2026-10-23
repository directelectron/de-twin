"""Shared-memory frame source (protocol v2, see :mod:`de_twin.transport.shm_layout`).

* :class:`FrameProducer` is the twin's side: it opens the mapping DE-Server created,
  watches for acquisition requests and publishes frames.
* :class:`FrameConsumer` is a Python stand-in for DE-Server's ``ExternalFrameSource``
  (tests, or any reader that is not DE-Server): it creates the mapping, publishes a
  request and reads frames.
"""

from __future__ import annotations

import mmap
import os
import struct
import sys
import time

import numpy as np

from ..state import AcquisitionRequest, ExposureMode, Roi, ScanRequest
from . import shm_layout as L

_WINDOWS = sys.platform == "win32"
#: The hot path packs fields directly: a field at a time through ``L.View`` is ~1 us each,
#: a third of the budget for a frame at 30,000 frames a second.
_READ_COUNT = L.HEADER["read_count"].offset
_WRITE_COUNT = L.HEADER["write_count"].offset
#: A slot's fields after frame_number, in ``L.SLOT`` order: request_id .. flags.
_SLOT_FIELDS = "<QQIIII"


def mapping_exists(name: str) -> bool:
    if _WINDOWS:
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenFileMappingW.restype = ctypes.c_void_p
        h = k32.OpenFileMappingW(0x0004, False, name)  # FILE_MAP_READ
        if h:
            k32.CloseHandle(ctypes.c_void_p(h))
            return True
        return False
    return os.path.exists(_posix_path(name))


def _posix_path(name: str) -> str:
    return os.path.join("/dev/shm" if os.path.isdir("/dev/shm") else "/tmp", name)


def _map(name: str, size: int):
    """Open-or-create a named mapping of ``size`` bytes."""
    if _WINDOWS:
        return mmap.mmap(-1, size, tagname=name)
    path = _posix_path(name)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.fstat(fd).st_size < size:
            os.ftruncate(fd, size)
        return mmap.mmap(fd, size)
    finally:
        os.close(fd)


def _dtype(bytes_per_pixel: int):
    return np.uint8 if bytes_per_pixel == 1 else np.uint16


class _Mapping:
    def __init__(self, name: str, mm, slot_bytes: int, slot_count: int):
        self.name = name
        self.mm = mm
        self.slot_bytes = slot_bytes
        self.slot_count = slot_count
        self.hdr = L.View(mm, 0, L.HEADER)

    def slot(self, n: int) -> L.View:
        return L.View(self.mm, L.slot_offset(n, self.slot_bytes, self.slot_count), L.SLOT)

    def payload(self, n: int) -> int:
        return L.slot_offset(n, self.slot_bytes, self.slot_count) + L.SLOT_HEADER_BYTES

    def close(self) -> None:
        if self.mm is not None:
            self.mm.close()
            self.mm = None
            if not _WINDOWS and isinstance(self, FrameConsumer):
                try:
                    os.unlink(_posix_path(self.name))
                except OSError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------- producer
class FrameProducer(_Mapping):
    """The twin's end: open DE-Server's mapping and publish frames into it."""

    def __init__(self, name: str = L.DEFAULT_NAME, timeout: float = 0.0):
        deadline = time.monotonic() + timeout
        while not mapping_exists(name):
            if time.monotonic() >= deadline:
                raise FileNotFoundError(f"shared memory {name!r} does not exist yet "
                                        "(DE-Server creates it at the first acquisition)")
            time.sleep(0.05)
        head = _map(name, L.HEADER_BYTES)
        hdr = L.View(head, 0, L.HEADER)
        while hdr.magic != L.MAGIC:  # the creator is still initialising
            if time.monotonic() >= deadline + 1.0:
                raise RuntimeError(f"shared memory {name!r} was never initialised")
            time.sleep(0.01)
        if hdr.version != L.VERSION:
            raise RuntimeError(f"shared memory {name!r} is protocol v{hdr.version}, expected v{L.VERSION}")
        slot_bytes, slot_count = hdr.slot_bytes, hdr.slot_count
        head.close()
        super().__init__(name, _map(name, L.mapping_size(slot_bytes, slot_count)), slot_bytes, slot_count)
        # A request DE-Server is still acquiring for when we attach gets served; an old one does not.
        self._served = 0 if self.hdr.acquiring else self.hdr.request_id
        self._written = self.hdr.write_count  # only the producer writes it

    @property
    def max_frame_bytes(self) -> int:
        return self.slot_bytes - L.SLOT_HEADER_BYTES

    def acquiring(self) -> bool:
        return bool(self.hdr.acquiring)

    def poll_request(self, timeout: float = 0.0) -> tuple[int, AcquisitionRequest] | None:
        """A new acquisition request as ``(request_id, AcquisitionRequest)``, or None."""
        deadline = time.monotonic() + timeout
        while True:
            rid = self.hdr.request_id
            if rid != self._served and self.hdr.acquiring:
                self._served = rid
                return rid, request_from_header(self.hdr)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.002)

    def request_changed(self, request_id: int) -> bool:
        return self.hdr.request_id != request_id or not self.hdr.acquiring

    def free_slots(self) -> int:
        return self.slot_count - (self._written - struct.unpack_from("<Q", self.mm, _READ_COUNT)[0])

    def wait_slot_free(self, timeout: float) -> bool:
        # Yield instead of sleeping: at tens of thousands of frames a second the whole ring
        # drains in well under a millisecond, and Windows rounds any sleep up to about one.
        deadline = time.monotonic() + timeout
        while self.free_slots() <= 0:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0)
        return True

    def publish(self, frame: np.ndarray, *, request_id: int, frame_index: int, flags: int = 0) -> int:
        """Write ``frame`` into the next slot. Call :meth:`wait_slot_free` first."""
        frame = np.ascontiguousarray(frame)
        if frame.nbytes > self.max_frame_bytes:
            raise ValueError(f"frame of {frame.nbytes} bytes exceeds the slot ({self.max_frame_bytes} bytes)")
        n = self._written
        slot = L.slot_offset(n, self.slot_bytes, self.slot_count)
        struct.pack_into("<Q", self.mm, slot, 0)  # frame_number: not ready while being written
        dst = np.frombuffer(self.mm, np.uint8, frame.nbytes, slot + L.SLOT_HEADER_BYTES)
        np.copyto(dst, frame.reshape(-1).view(np.uint8))  # one copy, no bytes object
        del dst  # a live view would keep the mapping from closing
        height, width = frame.shape
        struct.pack_into(_SLOT_FIELDS, self.mm, slot + L.SLOT["request_id"].offset,
                         request_id, frame_index, width, height, frame.dtype.itemsize, flags)
        struct.pack_into("<Q", self.mm, slot, n + 1)  # frame_number: ready
        self._written = n + 1
        struct.pack_into("<Q", self.mm, _WRITE_COUNT, n + 1)
        return n


def request_from_header(h: L.View) -> AcquisitionRequest:
    roi = Roi(h.roi_x, h.roi_y, h.roi_width, h.roi_height) if h.roi_width and h.roi_height else None
    try:
        mode = ExposureMode(h.exposure_mode)
    except ValueError:
        mode = ExposureMode.NORMAL
    scan = ScanRequest()
    if h.scan_width and h.scan_height:
        scan = ScanRequest(enabled=True, size=(h.scan_width, h.scan_height),
                           frames_per_point=max(1, h.frames_per_scan_point),
                           dwell_s=h.frame_time_ns * 1e-9 * max(1, h.frames_per_scan_point))
    return AcquisitionRequest(
        exposure_mode=mode,
        frame_time_s=h.frame_time_ns * 1e-9,
        total_frames=h.total_frames,
        hw_roi=roi,
        hw_binning=(max(1, h.bin_x), max(1, h.bin_y)),
        bit_depth=8 if h.bytes_per_pixel == 1 else None,
        scan=scan,
    )


# --------------------------------------------------------------------- consumer
class FrameConsumer(_Mapping):
    """What DE-Server does, in Python: create the mapping, request an acquisition, read frames."""

    def __init__(self, name: str = L.DEFAULT_NAME, max_frame_bytes: int = 4096 * 4096 * 2,
                 slot_count: int = L.SLOT_COUNT):
        slot_bytes = L.slot_bytes_for(max_frame_bytes)
        mm = _map(name, L.mapping_size(slot_bytes, slot_count))
        super().__init__(name, mm, slot_bytes, slot_count)
        h = self.hdr
        if h.magic != L.MAGIC:
            mm[:L.HEADER_BYTES] = bytes(L.HEADER_BYTES)
            h.version, h.slot_count, h.slot_bytes = L.VERSION, slot_count, slot_bytes
            h.magic = L.MAGIC
        self.request_id = 0

    def begin(self, *, frame_shape: tuple[int, int], bytes_per_pixel: int = 2,
              sensor_shape: tuple[int, int] | None = None, exposure_mode: ExposureMode = ExposureMode.NORMAL,
              frame_time_s: float = 0.01, total_frames: int = 0, roi: Roi | None = None,
              binning: tuple[int, int] = (1, 1), scan_size: tuple[int, int] = (0, 0),
              frames_per_scan_point: int = 1) -> int:
        h = self.hdr
        fh, fw = frame_shape
        sh, sw = sensor_shape or (fh * binning[1], fw * binning[0])
        roi = roi or Roi(0, 0, sw, sh)
        h.exposure_mode = int(exposure_mode)
        h.frame_width, h.frame_height, h.bytes_per_pixel = fw, fh, bytes_per_pixel
        h.sensor_width, h.sensor_height = sw, sh
        h.roi_x, h.roi_y, h.roi_width, h.roi_height = roi.x, roi.y, roi.w, roi.h
        h.bin_x, h.bin_y = binning
        h.scan_width, h.scan_height = scan_size
        h.frames_per_scan_point = frames_per_scan_point
        h.frame_time_ns = round(frame_time_s * 1e9)
        h.total_frames = total_frames
        h.read_count = h.write_count
        h.acquiring = 1
        self.request_id = h.request_id + 1
        h.request_id = self.request_id
        return self.request_id

    def read(self, timeout: float = 1.0, out: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """The next frame of the current request, as ``(pixels, slot header)``. With *out*
        (of the frame's size and dtype) the pixels are copied into it, as DE-Server copies
        into its own buffers, instead of into a new array."""
        deadline = time.monotonic() + timeout
        h = self.hdr
        while True:
            n = h.read_count
            s = self.slot(n)
            if s.frame_number == n + 1:
                info = s.as_dict()
                current = info["request_id"] == self.request_id
                if current:
                    count = info["width"] * info["height"]
                    off = self.payload(n)
                    dtype = _dtype(info["bytes_per_pixel"])
                    src = np.frombuffer(self.mm, dtype=dtype, count=count, offset=off)
                    if out is not None and out.size == count and out.dtype == dtype:
                        np.copyto(out.reshape(-1), src)
                        px = out.reshape(info["height"], info["width"])
                    else:
                        px = src.copy().reshape(info["height"], info["width"])
                    del src
                h.read_count = n + 1
                if current:
                    return px, info
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("no frame from the producer")
            time.sleep(0.0005)

    def end(self) -> None:
        self.hdr.acquiring = 0
