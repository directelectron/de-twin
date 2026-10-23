"""Binary layout of the shared-memory frame source (protocol version 2).

The C++ side is ``cpp/ExternalFrameSource.h``: DE-Server's GrabberSim reads frames
through it when the test pattern "External Frame Source (Shared Memory)" is selected.
``tests/test_shm_layout.py`` checks the two stay identical.

Shared memory ``DE_ExternalFrames`` (created by DE-Server)::

    ExternalFrameHeader              (HEADER_BYTES reserved)
    SLOT_COUNT slots, each:          ExternalFrameSlot (64 bytes) + one frame of pixels

1. DE-Server creates the mapping. Before each acquisition it writes the request
   (frame size, exposure mode, frame time, ...) and increments ``request_id``.
2. The producer (the twin) writes frame ``n`` into slot ``n % slot_count``, sets
   ``slot.frame_number = n + 1`` last, then ``write_count = n + 1``. It never runs more
   than ``slot_count`` frames ahead of ``read_count``.
3. DE-Server copies frames in order and sets ``read_count = n + 1``, skipping frames made
   for an older request.

Pixels are unsigned, row-major, ``frame_width x frame_height`` (the hardware frame after
ROI and binning), ``bytes_per_pixel`` each.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = 0x57544544  # "DETW"
VERSION = 2
SLOT_COUNT = 8
HEADER_BYTES = 4096
SLOT_HEADER_BYTES = 64
DEFAULT_NAME = "DE_ExternalFrames"
TEST_PATTERN = "External Frame Source (Shared Memory)"

FLAG_BLANKED = 1 << 0  # the beam did not reach the detector (dark frame)


@dataclass(frozen=True)
class Field:
    name: str
    offset: int
    fmt: str  # struct format, little-endian

    @property
    def size(self) -> int:
        return struct.calcsize("<" + self.fmt)


def _fields(spec):
    return {n: Field(n, o, t) for n, o, t in spec}


# ExternalFrameHeader (request fields are ExternalFrameRequest, at offset 40)
HEADER = _fields(
    [
        ("magic", 0, "I"),
        ("version", 4, "I"),
        ("slot_count", 8, "I"),
        ("slot_bytes", 16, "Q"),
        ("request_id", 24, "Q"),
        ("acquiring", 32, "I"),
        # ExternalFrameRequest
        ("exposure_mode", 40, "I"),
        ("frame_width", 44, "I"),
        ("frame_height", 48, "I"),
        ("bytes_per_pixel", 52, "I"),
        ("sensor_width", 56, "I"),
        ("sensor_height", 60, "I"),
        ("roi_x", 64, "I"),
        ("roi_y", 68, "I"),
        ("roi_width", 72, "I"),
        ("roi_height", 76, "I"),
        ("bin_x", 80, "I"),
        ("bin_y", 84, "I"),
        ("scan_width", 88, "I"),
        ("scan_height", 92, "I"),
        ("frames_per_scan_point", 96, "I"),
        ("frame_time_ns", 104, "Q"),
        ("total_frames", 112, "Q"),
        # progress
        ("read_count", 120, "Q"),
        ("write_count", 128, "Q"),
    ]
)

# ExternalFrameSlot
SLOT = _fields(
    [
        ("frame_number", 0, "Q"),
        ("request_id", 8, "Q"),
        ("frame_index", 16, "Q"),
        ("width", 24, "I"),
        ("height", 28, "I"),
        ("bytes_per_pixel", 32, "I"),
        ("flags", 36, "I"),
    ]
)


def slot_offset(index: int, slot_bytes: int, slot_count: int = SLOT_COUNT) -> int:
    return HEADER_BYTES + (index % slot_count) * slot_bytes


def slot_bytes_for(max_frame_bytes: int) -> int:
    return (SLOT_HEADER_BYTES + max_frame_bytes + 63) // 64 * 64


def mapping_size(slot_bytes: int, slot_count: int = SLOT_COUNT) -> int:
    return HEADER_BYTES + slot_count * slot_bytes


class View:
    """Typed accessor over a region of a writable buffer (an mmap)."""

    def __init__(self, buf, base: int, fields: dict[str, Field]):
        object.__setattr__(self, "_buf", buf)
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name):
        f = self._fields.get(name)
        if f is None:
            raise AttributeError(name)
        return struct.unpack_from("<" + f.fmt, self._buf, self._base + f.offset)[0]

    def __setattr__(self, name, value):
        f = self._fields.get(name)
        if f is None:
            raise AttributeError(name)
        struct.pack_into("<" + f.fmt, self._buf, self._base + f.offset, value)

    def as_dict(self) -> dict:
        return {n: getattr(self, n) for n in self._fields}
