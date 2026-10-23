"""Large soft disks, drawn row by row over the detector only (numba).

`diffraction.stamp_disks` evaluates a disk's whole square stamp. That is right for the
small disks of TEM diffraction, but a STEM probe's disks can be larger than the detector
(a 550 px radius is a 1135² stamp, ~70 ms each and hundreds per pattern). Here each
detector row gets, per disk, a constant over the flat interior and the edge profile only
in the thin band where it changes.
"""

from __future__ import annotations

import math

import numpy as np

from ..parallel import numba_jit

try:
    import numba as nb

    AVAILABLE = True
except Exception:  # noqa: BLE001 - numba missing: stamping remains
    nb = None
    AVAILABLE = False

#: The edge band: the profile is 0.5 erfc((d - R) / (sqrt2 sigma)), within 3e-7 of 1 (0)
#: at BAND sigma inside (outside) the edge.
BAND = 5.0


_njit = numba_jit


@_njit(parallel=True)
def _draw(out, x, y, amp, radius, sigma, erfc_lut, lut_step):
    h, w = out.shape
    rin = radius - BAND * sigma
    rout = radius + BAND * sigma
    rin2 = rin * rin if rin > 0 else -1.0
    rout2 = rout * rout
    nlut = erfc_lut.shape[0]
    for r in nb.prange(h):
        orow = out[r]
        for k in range(x.shape[0]):
            dy = r - y[k]
            dy2 = dy * dy
            if dy2 >= rout2:
                continue
            a = amp[k]
            xk = x[k]
            ho = math.sqrt(rout2 - dy2)
            c0 = max(0, int(math.ceil(xk - ho)))
            c1 = min(w - 1, int(math.floor(xk + ho)))
            if c0 > c1:
                continue
            if dy2 < rin2:
                hi = math.sqrt(rin2 - dy2)
                i0 = max(c0, int(math.ceil(xk - hi)))
                i1 = min(c1, int(math.floor(xk + hi)))
            else:
                i0 = c1 + 1
                i1 = c1
            for c in range(c0, c1 + 1):
                if i0 <= c <= i1:
                    orow[c] += a
                    continue
                dx = c - xk
                t = (math.sqrt(dx * dx + dy2) - rin) / lut_step
                j = int(t)
                if j >= nlut - 1:
                    continue
                f = t - j
                orow[c] += a * (erfc_lut[j] * (1.0 - f) + erfc_lut[j + 1] * f)


_LUT_N = 1024


def draw_disks(out: np.ndarray, x, y, w, radius: float, sigma: float) -> None:
    """Add soft disks (a uniform disk convolved with a Gaussian PSF of *sigma*), each
    integrating to its weight over the infinite plane; what falls off *out* is lost."""
    from scipy.special import erfc

    x = np.ascontiguousarray(x, np.float64)
    y = np.ascontiguousarray(y, np.float64)
    # the plane integral of 0.5 erfc((d - R) / (sqrt2 s)) is pi (R^2 + s^2)
    amp = np.asarray(w, np.float64) / (math.pi * (radius * radius + sigma * sigma))
    step = 2.0 * BAND * sigma / (_LUT_N - 1)
    d = radius - BAND * sigma + step * np.arange(_LUT_N)
    lut = 0.5 * erfc((d - radius) / (math.sqrt(2.0) * sigma))
    lut[-1] = 0.0
    tmp = np.zeros(out.shape, np.float64)
    _draw(tmp, x, y, amp, float(radius), float(sigma), lut, step)
    out += tmp.astype(out.dtype)
