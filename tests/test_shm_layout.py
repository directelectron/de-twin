"""Protocol v2 layout: Python agrees with the compiled C++ header (cpp/ExternalFrameSource.h)."""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from de_twin.transport import shm_layout as L

EXE = Path(__file__).resolve().parents[1] / "cpp" / "build" / "test_consumer.exe"
HEADER = Path(__file__).resolve().parents[1] / "cpp" / "ExternalFrameSource.h"

# C++ name -> Python name
CPP_TO_PY = {
    "magic": "magic", "version": "version", "slotCount": "slot_count", "slotBytes": "slot_bytes",
    "requestId": "request_id", "acquiring": "acquiring", "readCount": "read_count", "writeCount": "write_count",
    "exposureMode": "exposure_mode", "frameWidth": "frame_width", "frameHeight": "frame_height",
    "bytesPerPixel": "bytes_per_pixel", "sensorWidth": "sensor_width", "sensorHeight": "sensor_height",
    "roiX": "roi_x", "roiY": "roi_y", "roiWidth": "roi_width", "roiHeight": "roi_height",
    "binX": "bin_x", "binY": "bin_y", "scanWidth": "scan_width", "scanHeight": "scan_height",
    "framesPerScanPoint": "frames_per_scan_point", "frameTimeNs": "frame_time_ns", "totalFrames": "total_frames",
    "frameNumber": "frame_number", "frameIndex": "frame_index", "width": "width", "height": "height",
    "flags": "flags",
}


def test_fields_do_not_overlap():
    for fields, total in ((L.HEADER, L.HEADER_BYTES), (L.SLOT, L.SLOT_HEADER_BYTES)):
        spans = sorted((f.offset, f.offset + f.size, f.name) for f in fields.values())
        for (a0, a1, an), (b0, b1, bn) in itertools.pairwise(spans):
            assert a1 <= b0, f"{an} overlaps {bn}"
        assert spans[-1][1] <= total


def test_constants_match_header():
    text = HEADER.read_text()
    assert f'kMappingName     = "{L.DEFAULT_NAME}"' in text
    assert f'kTestPattern     = "{L.TEST_PATTERN}"' in text
    assert "kMagic           = 0x57544544" in text and L.MAGIC == 0x57544544
    assert f"kVersion         = {L.VERSION}" in text
    assert f"kSlotCount       = {L.SLOT_COUNT}" in text
    assert f"kHeaderBytes     = {L.HEADER_BYTES}" in text
    assert f"kSlotHeaderBytes = {L.SLOT_HEADER_BYTES}" in text


@pytest.mark.skipif(sys.platform != "win32" or not EXE.exists(), reason="build cpp/test_consumer.exe with cpp\\build.bat")
def test_offsets_match_compiled_cpp():
    out = subprocess.run([str(EXE), "layout"], capture_output=True, text=True, check=True).stdout
    cpp = {}
    for line in out.splitlines():
        name, off = line.split()
        struct_name, field = name.split(".")
        cpp[(struct_name, field)] = int(off)
    request_base = cpp[("ExternalFrameHeader", "request")]
    for (struct_name, field), off in cpp.items():
        if field == "request" or struct_name == "ScanOutputHeader":  # test_scan_output.py
            continue
        py = CPP_TO_PY[field]
        if struct_name == "ExternalFrameSlot":
            assert L.SLOT[py].offset == off, field
        elif struct_name == "ExternalFrameRequest":
            assert L.HEADER[py].offset == request_base + off, field
        else:
            assert L.HEADER[py].offset == off, field
