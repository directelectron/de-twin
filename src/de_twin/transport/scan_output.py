"""The scan coil voltages DE-Server outputs, shared with the twin (layout version 1).

The C++ side is ``cpp/ScanOutput.h``, where DE-Server's scan simulator writes, block by
block and as far ahead of the beam as the scan card's FIFO would hold, the voltages it
would have sent to the card. ``tests/test_shm_layout.py`` checks the two stay identical.

Shared memory ``DE_ScanOutput`` (created by DE-Server)::

    ScanOutputHeader                 (HEADER_BYTES reserved)
    capacity points, a ring          (H, V) volts, two float64; point n at n % capacity

1. At each scan start DE-Server writes the scan geometry, sets ``written = output = 0``,
   then increments ``scan_id``.
2. It writes the voltages of point n, then sets ``written = n + 1``.
3. ``output`` counts the points the beam has reached. A camera frame covers
   ``points_per_frame`` points from ``frame index * points_per_frame`` on.

* :class:`ScanOutputReader` is the twin's side: where the beam is for each frame.
* :class:`ScanOutputWriter` is a Python stand-in for DE-Server (tests, or any scan
  source that is not DE-Server).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .shm import _map, mapping_exists
from .shm_layout import View, _fields

MAGIC = 0x4F534544  # "DESO"
VERSION = 1
HEADER_BYTES = 4096
CAPACITY = 1 << 20
POINT_BYTES = 16
DEFAULT_NAME = "DE_ScanOutput"

# ScanOutputHeader
HEADER = _fields([
    ("magic", 0, "I"),
    ("version", 4, "I"),
    ("capacity", 8, "Q"),
    ("scan_id", 16, "Q"),
    ("written", 24, "Q"),
    ("output", 32, "Q"),
    ("total_points", 40, "Q"),
    ("size_x", 48, "I"),
    ("size_y", 52, "I"),
    ("voltage_min_h", 56, "d"),
    ("voltage_max_h", 64, "d"),
    ("voltage_min_v", 72, "d"),
    ("voltage_max_v", 80, "d"),
    ("rotation_deg", 88, "d"),
    ("flip_horizontal", 96, "I"),
    ("flip_vertical", 100, "I"),
    ("points_per_frame", 104, "d"),
    ("output_hz", 112, "d"),
])
HEADER_STRUCT_BYTES = 120

_WRITTEN = HEADER["written"].offset
_SCAN_ID = HEADER["scan_id"].offset


def mapping_size(capacity: int = CAPACITY) -> int:
    return HEADER_BYTES + capacity * POINT_BYTES


@dataclass(frozen=True)
class ScanGeometry:
    """What one scan's voltages are laid out on: the full grid and the range it spans."""

    size: tuple[int, int]
    voltage_h: tuple[float, float]
    voltage_v: tuple[float, float]
    rotation_deg: float
    flip: tuple[bool, bool]
    points_per_frame: float
    total_points: int


class _Mapping:
    def __init__(self, mm, capacity: int):
        self.mm = mm
        self.capacity = capacity
        self.hdr = View(mm, 0, HEADER)
        self._points = np.frombuffer(mm, np.float64, 2 * capacity, HEADER_BYTES).reshape(capacity, 2)

    def close(self) -> None:
        if self.mm is not None:
            self._points = None  # a live view would keep the mapping from closing
            self.mm.close()
            self.mm = None


class ScanOutputReader(_Mapping):
    """The twin's end: open DE-Server's scan output, if there is one, and read it."""

    def __init__(self, name: str = DEFAULT_NAME):
        if not mapping_exists(name):
            raise FileNotFoundError(f"shared memory {name!r} does not exist "
                                    "(DE-Server's scan simulator creates it at its first scan)")
        head = _map(name, HEADER_BYTES)
        hdr = View(head, 0, HEADER)
        if hdr.magic != MAGIC or hdr.version != VERSION:
            magic, version = hdr.magic, hdr.version
            head.close()
            raise RuntimeError(f"shared memory {name!r} is not scan output v{VERSION} "
                               f"(magic {magic:#x}, version {version})")
        capacity = hdr.capacity
        head.close()
        super().__init__(_map(name, mapping_size(capacity)), capacity)

    @property
    def scan_id(self) -> int:
        return struct.unpack_from("<Q", self.mm, _SCAN_ID)[0]

    @property
    def written(self) -> int:
        return struct.unpack_from("<Q", self.mm, _WRITTEN)[0]

    def geometry(self) -> ScanGeometry:
        h = self.hdr
        return ScanGeometry(size=(h.size_x, h.size_y), voltage_h=(h.voltage_min_h, h.voltage_max_h),
                            voltage_v=(h.voltage_min_v, h.voltage_max_v), rotation_deg=h.rotation_deg,
                            flip=(bool(h.flip_horizontal), bool(h.flip_vertical)),
                            points_per_frame=h.points_per_frame or 1.0, total_points=h.total_points)

    def volts(self, point: int) -> tuple[float, float] | None:
        """The (H, V) voltages of *point*, or None when they have not been written yet (or
        were written so long ago that the ring has overwritten them)."""
        written = self.written
        if not 0 <= point < written or point < written - self.capacity:
            return None
        h, v = self._points[point % self.capacity]
        return float(h), float(v)

    def volts_many(self, points: np.ndarray) -> np.ndarray:
        """(H, V) for each of *points*, an (n, 2) array. A point not written yet gets the
        last voltages written, where the coils are left driven; with none written, or one
        the ring has overwritten, it is NaN."""
        points = np.asarray(points, np.int64)
        written = self.written
        out = np.full((len(points), 2), np.nan)
        if written == 0:
            return out
        held = np.minimum(points, written - 1)
        kept = (points >= 0) & (held >= written - self.capacity)
        out[kept] = self._points[held[kept] % self.capacity]
        return out

    def latest(self) -> tuple[int, tuple[float, float]] | None:
        """The last point written and its voltages: where the coils are left driven."""
        written = self.written
        if written == 0:
            return None
        h, v = self._points[(written - 1) % self.capacity]
        return written - 1, (float(h), float(v))


class ScanOutputWriter(_Mapping):
    """What DE-Server's scan simulator does, in Python: create the mapping and write voltages."""

    def __init__(self, name: str = DEFAULT_NAME, capacity: int = CAPACITY):
        super().__init__(_map(name, mapping_size(capacity)), capacity)
        self._pending = 0

    def begin(self, geometry: ScanGeometry, *, output_hz: float = 0.0) -> int:
        h = self.hdr
        h.magic, h.version, h.capacity = MAGIC, VERSION, self.capacity
        h.written = h.output = 0
        h.total_points = geometry.total_points
        h.size_x, h.size_y = geometry.size
        h.voltage_min_h, h.voltage_max_h = geometry.voltage_h
        h.voltage_min_v, h.voltage_max_v = geometry.voltage_v
        h.rotation_deg = geometry.rotation_deg
        h.flip_horizontal, h.flip_vertical = (int(f) for f in geometry.flip)
        h.points_per_frame = geometry.points_per_frame
        h.output_hz = output_hz
        self._pending = 0
        h.scan_id = h.scan_id + 1
        return h.scan_id

    def write(self, volts: np.ndarray) -> None:
        """Append points, an (n, 2) array of (H, V), and make them visible."""
        volts = np.asarray(volts, np.float64).reshape(-1, 2)
        for i in range(0, len(volts), self.capacity):
            chunk = volts[i:i + self.capacity]
            start = self._pending % self.capacity
            first = min(len(chunk), self.capacity - start)
            self._points[start:start + first] = chunk[:first]
            self._points[:len(chunk) - first] = chunk[first:]
            self._pending += len(chunk)
        struct.pack_into("<Q", self.mm, _WRITTEN, self._pending)

    def set_output(self, points: int) -> None:
        self.hdr.output = points
