"""Scan coil voltages: where on the specimen each scan point puts the probe.

DE-Server drives the scan coils with a voltage range, ``Scan - Voltage H/V Min/Max``
(default -1 V to 1 V), and that range IS the scanned field: the full scan grid
(``Scan - Size X/Y``) is spread across it, and everything else a client does to the
scan is a position inside it.

* A smaller range zooms in about the range's midpoint; moving the range pans.
* ``Scan - Pixel Offset X/Y`` adds ``offset x volts per point`` to every output
  voltage, so it pans by a fraction of a scan point, in the coils' frame (scan rotation
  does not turn it). Drift correction and a live probe crosshair steer this.
* The offset moves the scan by at most half its own range each way, and never past the
  scan card's +-10 V output (``ScanController::GetAnalogOutputRange``); beyond that it
  is clamped.
* An ROI or an XY array addresses points of the same full grid.

This follows ``ScanPattern::_TransformCoordinates`` and
``ScanController::GetScanVoltageOffsets`` in DE-Server: a point is taken relative to the
grid centre, flipped, rotated, and scaled by the volts per point; the longer side of the
grid spans the whole range and the shorter one a centred part of it, so the points are
square.

One step is ``span / n``, as DE-Server reports the pixel size and converts the pixel
offset, so an offset of one point moves the probe exactly one step. (DE-Server's own
output puts the end points on the ends of the range, ``span / (n - 1)``: 1/n larger.)
The twin's magnification calibration gives the field at the default range, so a scan at
the default range covers exactly the specimen it did before voltages were modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: DE-Server's default ``Scan - Voltage H/V Min/Max``.
DEFAULT_VOLTAGE_RANGE = (-1.0, 1.0)
DEFAULT_VOLTAGE_SPAN = DEFAULT_VOLTAGE_RANGE[1] - DEFAULT_VOLTAGE_RANGE[0]
#: What DE-Server's scan card can output, in volts.
OUTPUT_LIMIT_VOLTS = 10.0


@dataclass(frozen=True)
class ScanCoils:
    """The voltage geometry of one scan.

    ``volts_per_point`` is the step between neighbouring scan points (horizontal; the
    vertical step matches it unless the H and V ranges differ, which the twin's square
    raster cannot show). ``center_volts`` is where the grid centre sits, offset included.
    """

    volts_per_point: float
    center_volts: tuple[float, float]
    rotation_deg: float
    flip: tuple[bool, bool]


def _actual_ranges(
    size: tuple[int, int], voltage_h: tuple[float, float], voltage_v: tuple[float, float]
) -> tuple[float, float]:
    """The H and V spans the grid really uses: the longer side gets its whole range."""
    nx, ny = max(1, int(size[0])), max(1, int(size[1]))
    span_h = float(voltage_h[1]) - float(voltage_h[0])
    span_v = float(voltage_v[1]) - float(voltage_v[0])
    if nx > ny:
        span_v *= ny / nx
    elif nx < ny:
        span_h *= nx / ny
    return span_h, span_v


def _clamp_shift(shift: float, middle: float, span: float) -> float:
    """The offset DE-Server can output: half the scan's span each way, inside the card's
    range."""
    half = 0.5 * abs(span)
    low = max(-half, -OUTPUT_LIMIT_VOLTS - middle + half)
    high = min(half, OUTPUT_LIMIT_VOLTS - middle - half)
    return min(max(shift, low), high)


def scan_coils(scan) -> ScanCoils:
    """The coil geometry a :class:`~de_twin.state.ScanRequest` asks for."""
    nx, ny = max(1, int(scan.size[0])), max(1, int(scan.size[1]))
    span_h, span_v = _actual_ranges((nx, ny), scan.voltage_h, scan.voltage_v)
    volts_per_point = span_h / nx if nx >= ny else span_v / ny
    middle_h = 0.5 * (scan.voltage_h[0] + scan.voltage_h[1])
    middle_v = 0.5 * (scan.voltage_v[0] + scan.voltage_v[1])
    shift_h = _clamp_shift(float(scan.pixel_offset[0]) * span_h / nx, middle_h, span_h)
    shift_v = _clamp_shift(float(scan.pixel_offset[1]) * span_v / ny, middle_v, span_v)
    center = (middle_h + shift_h, middle_v + shift_v)
    return ScanCoils(
        volts_per_point=volts_per_point,
        center_volts=center,
        rotation_deg=float(scan.rotation_deg),
        flip=(bool(scan.flip[0]), bool(scan.flip[1])),
    )


def point_volts(scan, x: float, y: float) -> tuple[float, float]:
    """The (H, V) coil voltages for full-grid point (x, y): DE-Server's forward transform,
    at the twin's step (module docstring)."""
    coils = scan_coils(scan)
    nx, ny = max(1, int(scan.size[0])), max(1, int(scan.size[1]))
    u, v = x - (nx - 1) / 2.0, y - (ny - 1) / 2.0
    if coils.flip[0]:
        u = -u
    if coils.flip[1]:
        v = -v
    a = math.radians(coils.rotation_deg)
    c, s = math.cos(a), math.sin(a)
    u, v = u * c - v * s, u * s + v * c
    return (
        coils.center_volts[0] + coils.volts_per_point * u,
        coils.center_volts[1] + coils.volts_per_point * v,
    )


def point_from_volts(scan, h: float, v: float) -> tuple[float, float]:
    """The full-grid point (x, y) the coils put the probe on at voltages (H, V): the
    inverse of :func:`point_volts`. Not rounded, nor kept inside the grid: an offset can
    put the probe off the grid's edge."""
    coils = scan_coils(scan)
    nx, ny = max(1, int(scan.size[0])), max(1, int(scan.size[1]))
    u = (h - coils.center_volts[0]) / coils.volts_per_point
    w = (v - coils.center_volts[1]) / coils.volts_per_point
    a = math.radians(coils.rotation_deg)
    c, s = math.cos(a), math.sin(a)
    u, w = u * c + w * s, -u * s + w * c
    if coils.flip[0]:
        u = -u
    if coils.flip[1]:
        w = -w
    return u + (nx - 1) / 2.0, w + (ny - 1) / 2.0


def nm_per_volt(default_step_nm: float, size: tuple[int, int]) -> float:
    """Specimen nm per coil volt, from the calibrated step of a scan of *size* at the
    default range (the field at the default range, over that range)."""
    n = max(1, int(size[0]), int(size[1]))
    return default_step_nm * n / DEFAULT_VOLTAGE_SPAN
