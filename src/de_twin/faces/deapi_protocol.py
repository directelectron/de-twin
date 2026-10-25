"""DE-Server's command ids, property names and unit conversions that the deapi face shares."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..state import ExposureMode

EXPOSURE_MODES = {
    "dark": ExposureMode.DARK,
    "trial": ExposureMode.TRIAL,
    "gain": ExposureMode.GAIN,
    "normal": ExposureMode.NORMAL,
    "raw": ExposureMode.RAW,
    "vacuum": ExposureMode.VACUUM,
}
PIXEL_DTYPES = {1: np.uint8, 5: np.uint16, 13: np.float32}

# command ids (FakeServer has some of these; the client's full list in client.py)
SET_HW_ROI = 16
SET_SCAN_SIZE = 27
SET_SCAN_ROI = 28
SET_SCAN_XY_ARRAY = 32
GET_PROPERTY_SPECIFICATIONS = 35
GET_GPU_FRAMES = 44

#: Frames per shared GPU buffer, and the longest a frame waits for its buffer to fill
#: before it is handed over anyway (a live view should not wait for 64 frames).
GPU_BUFFER_FRAMES = 64
GPU_PUBLISH_S = 0.02

TEMPERATURE_PROP = "Simulator Virtual Specimen Temperature (C)"

#: ``Simulator EBSD <label>`` deapi properties -> EbsdGeometry fields.
EBSD_PROPERTIES = {
    "enabled": "Enable",
    "sample_tilt_deg": "Sample Tilt (degrees)",
    "detector_tilt_deg": "Detector Tilt (degrees)",
    "working_distance_mm": "Working Distance (mm)",
    "pcx": "Pattern Center X",
    "pcy": "Pattern Center Y",
    "pcz": "Pattern Center Z",
    "probe_current_na": "Probe Current (nA)",
    "step_um": "Default Step (micrometers)",
    "band_contrast": "Band Contrast",
    "background": "Background",
    "boundary_blend": "Boundary Blend",
}

# DE-Server "Instrument Project Name" per (function mode, TEM/STEM) (mag.yaml projects)
_PROJECT_NAMES = {
    (0, 0): "MAG1",
    (1, 0): "MAG2",
    (2, 0): "Low Mag",
    (3, 0): "SMAG",
    (4, 0): "Diff Mag",
    (2, 1): "HR_STEM",
    (4, 1): "NBD_STEM",
}


def _property_key(name: str) -> str:
    return name.replace(" ", "_").lower().replace("(", "").replace(")", "")


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("on", "true", "1", "yes")


#: The exposure a fresh face starts with, in seconds (see `_frame_count`).
DEFAULT_EXPOSURE_S = 0.5


def _recv_exact(conn, n: int) -> bytes:
    buffer = b""
    while len(buffer) < n:
        chunk = conn.recv(n - len(buffer))
        if not chunk:
            raise ConnectionResetError(f"connection closed after {len(buffer)} of {n} bytes")
        buffer += chunk
    return buffer
