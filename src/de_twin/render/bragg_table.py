"""Bragg loss tables: every grain of a material at every thickness bin, in one pass.

For one stage tilt and beam (HT, convergence, objective aperture), the fraction of the
beam a grain Bragg-scatters outside the objective aperture depends only on the grain (its
orientation) and its projected thickness, which the renderer already bins
(`de_twin.render.diffraction.THICKNESS_BIN_NM`). So the loss of every pixel is a gather,
``table[grain, thickness bin]``, from a table of ``GRAINS_PER_MATERIAL x (MAX_THICKNESS_BIN
+ 1)`` values per crystalline material, built once per tilt and beam by a numba kernel
(:func:`loss_table`) and kept (see `de_twin.render.tem._BraggMemo`).

The kernel is the kinematic two-beam intensity of :meth:`de_twin.crystal.CrystalLibrary.excite`
with the rocking curve's ``sinc^2`` antiderivative ``F(x) = Si(2x) - sin^2(x)/x`` taken from a
cubic Hermite table (``F`` and ``F' = sinc^2`` exact at the nodes, spacing 1/256) for
``|x| < 64`` and the asymptotic series of ``Si`` beyond: ``F`` to ~1e-12, so a table entry
matches the exact (scipy ``sici``) evaluation to ~1e-9 relative (tested to 1e-6).
"""

from __future__ import annotations

import math

import numpy as np

from ..crystal.library import S_TAPER_INV_NM, S_WINDOW_INV_NM, _sinc2_antiderivative
from ..parallel import numba_jit

_X0 = 64.0  # |x| beyond this: the asymptotic series
_H = 1.0 / 256.0  # Hermite node spacing
_XS = np.arange(int(_X0 / _H) + 2) * _H
_FT = _sinc2_antiderivative(_XS).astype(np.float64)
with np.errstate(invalid="ignore", divide="ignore"):
    _DT = np.where(_XS > 0, (np.sin(_XS) / np.where(_XS > 0, _XS, 1.0)) ** 2, 1.0)

try:
    import numba as nb

    AVAILABLE = True
except Exception:
    AVAILABLE = False


def _F_py(x, FT, DT):
    ax = abs(x)
    if ax >= _X0:
        z = 2.0 * ax
        iz = 1.0 / z
        iz2 = iz * iz
        f = iz * (1.0 - 2.0 * iz2 * (1.0 - 12.0 * iz2 * (1.0 - 30.0 * iz2)))
        g = iz2 * (1.0 - 6.0 * iz2 * (1.0 - 20.0 * iz2 * (1.0 - 42.0 * iz2)))
        si = 0.5 * math.pi - f * math.cos(z) - g * math.sin(z)
        s = math.sin(ax)
        v = si - s * s / ax
    else:
        u = ax / _H
        i = int(u)
        i = min(i, FT.shape[0] - 2)
        t = u - i
        t2 = t * t
        t3 = t2 * t
        v = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * FT[i]
            + (t3 - 2.0 * t2 + t) * (DT[i] * _H)
            + (-2.0 * t3 + 3.0 * t2) * FT[i + 1]
            + (t3 - t2) * (DT[i + 1] * _H)
        )
    return v if x >= 0.0 else -v


def _table_py(starts, kx, s, ga, taper, outm, tvals, total, pout):
    n = starts.shape[0] - 1
    nt = tvals.shape[0]
    for p in _prange(n * nt):
        g = p // nt
        j = p - g * nt
        t = tvals[j]
        tot = 0.0
        po = 0.0
        for b in range(starts[g], starts[g + 1]):
            x = math.pi * t * s[b]
            a = math.pi * t * ga[b]
            if a < 1e-3:
                xs = x if abs(x) >= 1e-9 else 1e-9
                r = math.sin(xs) / xs
                r = r * r
            else:
                r = (_F(x + a, _FT, _DT) - _F(x - a, _FT, _DT)) / (2.0 * a)
            k = kx[b] * t
            v = k * k * r * taper[b]
            tot += v
            po += v * outm[b]
        total[g, j] = tot
        pout[g, j] = po


if AVAILABLE:
    _F = numba_jit(fastmath=False)(_F_py)
    _prange = nb.prange
    _table = numba_jit(fastmath=False, parallel=True)(_table_py)
else:  # pragma: no cover - the pure-Python loop is slow; the renderer then keeps its memo
    _F = _F_py
    _prange = range
    _table = _table_py


def loss_table(lib, geom: list, optics, g_obj: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """(``frac * p_out``, ``total``), each (len(geom), n_bins): for each grain's reflections
    ``geom[i] = (index, gx, gy, s, |g_xy|)`` at thickness bins 0..n_bins-1, the fraction of
    the beam lost outside ``g_obj`` (all of it when 0; see
    `de_twin.render.tem_contrast.bragg_contrast`) and the total kinematic intensity."""
    from .diffraction import THICKNESS_BIN_NM, bragg_fraction

    lens = np.array([len(g[0]) for g in geom], np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    empty = np.zeros(0)
    idx = np.concatenate([g[0] for g in geom]) if len(geom) else empty.astype(np.int64)
    s = np.concatenate([g[3] for g in geom]) if len(geom) else empty
    gxy = np.concatenate([g[4] for g in geom]) if len(geom) else empty
    alpha = optics.convergence_mrad * 1e-3
    kx = lib.inv_xi(optics.ht_kv, optics.wavelength_nm)[idx].astype(np.float64)
    ga = (lib.g_len[idx] * alpha).astype(np.float64)
    excess = np.clip((np.abs(s) - S_TAPER_INV_NM - ga) / (S_WINDOW_INV_NM - S_TAPER_INV_NM), 0.0, 1.0)
    taper = np.cos(0.5 * math.pi * excess) ** 2
    outm = (gxy > g_obj).astype(np.float64)
    tvals = np.arange(n_bins, dtype=np.float64) * THICKNESS_BIN_NM
    total = np.zeros((len(geom), n_bins))
    pout = np.zeros((len(geom), n_bins))
    _table(starts, kx, np.ascontiguousarray(s, np.float64), ga, taper, outm, tvals, total, pout)
    frac = bragg_fraction(total) / np.maximum(total, 1e-300)
    return frac * pout, total
