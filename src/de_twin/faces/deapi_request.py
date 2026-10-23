"""Turning deapi properties into twin acquisition requests and scan-coil timelines."""

from __future__ import annotations

import bisect
from collections.abc import Callable
from typing import Any

from ..state import AcquisitionRequest, ExposureMode, Roi, ScanRequest
from .deapi_protocol import EXPOSURE_MODES, _to_bool

#: Read-only property naming the simulator behind the frames (``"de-twin <version>"``), so
#: a client can stamp provenance on what it saves. A real DE-Server has no such property.
SIMULATOR_SOURCE_PROPERTY = "Simulator Source"


def _super_resolution(get: Callable[[str], Any]) -> bool:
    """DE-Server's "Centroiding Mode" asks for super-resolution (counting cameras only)."""
    try:
        return "super" in str(get("Centroiding Mode") or "").lower()
    except Exception:
        return False


def model_counting(twin) -> bool:
    return bool(getattr(twin.detector.model, "hardware_counting", False))


#: DE-Server's scan coil properties, with DE-Server's defaults.
SCAN_COIL_PROPERTIES = {
    "Scan - Voltage H Min": -1.0, "Scan - Voltage H Max": 1.0,
    "Scan - Voltage V Min": -1.0, "Scan - Voltage V Max": 1.0,
    "Scan - Pixel Offset X": 0.0, "Scan - Pixel Offset Y": 0.0,
    "Scan - Rotation": 0.0,
}
SCAN_FLIP_PROPERTIES = ("Scan - Flip Horizontally", "Scan - Flip Vertically")
#: How much scan output DE-Server queues ahead of the beam, in milliseconds; 0 is its
#: default, two quarter-FIFO blocks (DEFAULT_QUEUED_POINTS).
OUTPUT_BUFFER_PROPERTY = "Scan - Output Buffer (milliseconds)"
DEFAULT_QUEUED_POINTS = 2 * (1048576 // 4)


def scan_coil_settings(get: Callable[[str], Any]) -> dict:
    """The ScanRequest fields the scan coils take from properties (de_twin.optics.scan_coils).

    These are what a client changes WHILE scanning (pan, zoom, drift correction, a moving
    probe), so a running acquisition re-reads them between repetitions."""
    def number(name):
        try:
            return float(get(name))
        except (KeyError, TypeError, ValueError):
            return SCAN_COIL_PROPERTIES[name]

    def flag(name):
        try:
            return _to_bool(get(name))
        except KeyError:
            return False

    return {
        "voltage_h": (number("Scan - Voltage H Min"), number("Scan - Voltage H Max")),
        "voltage_v": (number("Scan - Voltage V Min"), number("Scan - Voltage V Max")),
        "pixel_offset": (number("Scan - Pixel Offset X"), number("Scan - Pixel Offset Y")),
        "rotation_deg": number("Scan - Rotation"),
        "flip": tuple(flag(name) for name in SCAN_FLIP_PROPERTIES),
    }


def queued_scan_points(get: Callable[[str], Any], fps: float) -> int:
    """How many scan points DE-Server has written ahead of the beam, so how many points
    later a coil setting written now reaches it (ScanController's analog output: four
    blocks of at least 2 ms when ``Scan - Output Buffer (milliseconds)`` is set)."""
    try:
        ms = float(get(OUTPUT_BUFFER_PROPERTY))
    except (KeyError, TypeError, ValueError):
        ms = 0.0
    if ms <= 0:
        return DEFAULT_QUEUED_POINTS
    block = max(1, int(0.002 * fps), int(ms * 1e-3 * fps) // 4)
    return min(4 * block, DEFAULT_QUEUED_POINTS)


class _CoilTimeline:
    """The coil settings a client wrote during one acquisition, by the scan point the beam
    was on when it wrote them.

    DE-Server writes scan output ``queued`` points ahead of the beam and adds the offset
    (and the rest of the coil settings) as it writes, so repeat ``r`` of an N-point
    pattern scans with the settings as of point ``r * N - queued``: a change made now
    lands ``queued`` points later, on a repeat boundary.
    """

    def __init__(self, settings: dict):
        self._points = [0]
        self._settings = [settings]

    def record(self, point: int, settings: dict) -> None:
        if settings != self._settings[-1]:
            self._points.append(point)
            self._settings.append(settings)

    def at(self, point: int) -> dict:
        return self._settings[max(0, bisect.bisect_right(self._points, point) - 1)]


def request_from_properties(get: Callable[[str], Any], camera_model: str,
                            sensor_shape: tuple[int, int], *,
                            scan_points: list | None = None,
                            acquisition_index: int = 0) -> AcquisitionRequest:
    """Map deapi / DE-Server property values to an AcquisitionRequest.

    ``get(name)`` returns a property value (string or number).
    """
    h, w = sensor_shape
    fps = max(float(get("Frames Per Second")), 1e-6)
    frame_count = max(1, int(float(get("Frame Count"))))
    mode = EXPOSURE_MODES.get(str(get("Exposure Mode")).strip().lower(), ExposureMode.NORMAL)
    ox, oy = int(float(get("Hardware ROI Offset X"))), int(float(get("Hardware ROI Offset Y")))
    sx, sy = int(float(get("Hardware ROI Size X"))), int(float(get("Hardware ROI Size Y")))
    roi = None if (ox, oy, sx, sy) == (0, 0, w, h) else Roi(ox, oy, sx, sy)
    bx = int(float(get("Hardware Binning X") or 1))
    by = int(float(get("Hardware Binning Y") or 1))
    scan = ScanRequest(enabled=False)
    if _to_bool(get("Scan - Enable")):
        nx, ny = int(float(get("Scan - Size X"))), int(float(get("Scan - Size Y")))
        fpp = max(1, int(float(get("Scan - Camera Frames Per Point") or 1)))
        repeats = max(1, int(float(get("Scan - Repeats") or 1)))
        sroi = None
        if _to_bool(get("Scan - ROI Enable")):
            sroi = Roi(int(float(get("Scan - ROI Offset X"))), int(float(get("Scan - ROI Offset Y"))),
                       int(float(get("Scan - ROI Size X"))), int(float(get("Scan - ROI Size Y"))))
        points = None
        if scan_points and str(get("Scan - Type")) in ("XY File", "XY Array"):
            points = [tuple(map(int, p)) for p in scan_points]
        elif sroi is not None:
            points = [(sroi.x + i, sroi.y + j) for j in range(sroi.h) for i in range(sroi.w)]
        scan = ScanRequest(enabled=True, size=(nx, ny), dwell_s=fpp / fps, frames_per_point=fpp,
                           roi=sroi, repeats=repeats, points=points, **scan_coil_settings(get))
        n_points = len(points) if points else nx * ny
        frame_count = n_points * fpp * repeats
    return AcquisitionRequest(
        camera_model=camera_model, exposure_mode=mode, frame_time_s=1.0 / fps,
        total_frames=frame_count,
        frames_per_buffer=max(1, int(float(get("Grabbing - Frames Per Buffer") or 1))),
        hw_roi=roi, hw_binning=(bx, by), scan=scan, acquisition_index=acquisition_index,
        super_resolution=_super_resolution(get))
