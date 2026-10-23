"""Fast spheres: an explicit set of spheres resting on a film, binned on a grid, so a pixel
tests only the few spheres near it (numba). Chords, top heights and the occlusion of a
light source (cast shadows with a penumbra) are exact per sphere.
"""

from __future__ import annotations

import math

import numpy as np

from ..hashing import splitmix64_kernel
from ..parallel import numba_jit

try:
    import numba as nb

    AVAILABLE = True
except Exception:  # noqa: BLE001
    nb = None
    AVAILABLE = False


_njit = numba_jit


_prange = nb.prange if AVAILABLE else range


@_njit()
def _bin_of(x, y, x0, y0, inv, nx, ny):
    i = int(math.floor((x - x0) * inv))
    j = int(math.floor((y - y0) * inv))
    if i < 0 or j < 0 or i >= nx or j >= ny:
        return -1
    return j * nx + i


@_njit()
def top_chord1(xk, yk, cx, cy, r, start, members, x0, y0, inv, nx, ny):
    """(top, chord) of `_top_chord_nb` at one point (um)."""
    b = _bin_of(xk, yk, x0, y0, inv, nx, ny)
    t = 0.0
    c = 0.0
    if b >= 0:
        for m in range(start[b], start[b + 1]):
            s = members[m]
            dx = xk - cx[s]
            dy = yk - cy[s]
            d2 = r[s] * r[s] - dx * dx - dy * dy
            if d2 > 0.0:
                half = math.sqrt(d2)
                c += 2.0 * half
                if r[s] + half > t:
                    t = r[s] + half
    return t, c


@_njit()
def occlusion1(xk, yk, zk, cx, cy, r, start, members, x0, y0, inv, nx, ny, dx, dy, dz, reach, step, pen):
    """`_occlusion_nb` at one point."""
    occ = 0.0
    last = -2
    s_ = 0.0
    while s_ <= reach:
        b = _bin_of(xk + dx / math.sqrt(dx * dx + dy * dy) * s_, yk + dy / math.sqrt(dx * dx + dy * dy) * s_,
                    x0, y0, inv, nx, ny)
        if b >= 0 and b != last:
            for m in range(start[b], start[b + 1]):
                s = members[m]
                px = cx[s] - xk
                py = cy[s] - yk
                pz = r[s] - zk
                t = px * dx + py * dy + pz * dz
                if t > 0.5 * r[s]:
                    D = math.sqrt(max(px * px + py * py + pz * pz - t * t, 0.0))
                    w = max(t * pen, 1e-9)
                    o = (r[s] - D) / w + 0.5
                    if o > occ:
                        occ = min(o, 1.0)
        last = b
        s_ += step
    return occ


@_njit(parallel=True)
def _top_chord_nb(x, y, cx, cy, r, start, members, x0, y0, inv, nx, ny, top, chord):
    for k in _prange(x.size):
        b = _bin_of(x[k], y[k], x0, y0, inv, nx, ny)
        t = 0.0
        c = 0.0
        if b >= 0:
            for m in range(start[b], start[b + 1]):
                s = members[m]
                dx = x[k] - cx[s]
                dy = y[k] - cy[s]
                d2 = r[s] * r[s] - dx * dx - dy * dy
                if d2 > 0.0:
                    half = math.sqrt(d2)
                    c += 2.0 * half
                    if r[s] + half > t:
                        t = r[s] + half
        top[k] = t
        chord[k] = c


@_njit(parallel=True)
def _occlusion_nb(x, y, z0, cx, cy, r, start, members, x0, y0, inv, nx, ny, dx, dy, dz, reach, step, pen, out):
    for k in _prange(x.size):
        occ = 0.0
        last = -2
        s_ = 0.0
        while s_ <= reach:
            b = _bin_of(x[k] + dx / math.sqrt(dx * dx + dy * dy) * s_, y[k] + dy / math.sqrt(dx * dx + dy * dy) * s_,
                        x0, y0, inv, nx, ny)
            if b >= 0 and b != last:
                for m in range(start[b], start[b + 1]):
                    s = members[m]
                    px = cx[s] - x[k]
                    py = cy[s] - y[k]
                    pz = r[s] - z0[k]
                    t = px * dx + py * dy + pz * dz
                    if t > 0.5 * r[s]:
                        D = math.sqrt(max(px * px + py * py + pz * pz - t * t, 0.0))
                        w = max(t * pen, 1e-9)
                        o = (r[s] - D) / w + 0.5
                        if o > occ:
                            occ = min(o, 1.0)
            last = b
            s_ += step
        out[k] = occ


class SphereSet:
    """Spheres (centres `cx, cy`, radii `r`, all um) binned for fast lookups over the region
    they were generated for (spheres from outside it must be included by the caller)."""

    def __init__(self, cx, cy, r):
        self.cx = np.ascontiguousarray(cx, np.float64)
        self.cy = np.ascontiguousarray(cy, np.float64)
        self.r = np.ascontiguousarray(r, np.float64)
        n = self.cx.size
        if n == 0:
            self.bin = 1.0
            self.x0 = self.y0 = 0.0
            self.nx = self.ny = 1
            self.start = np.zeros(2, np.int64)
            self.members = np.zeros(0, np.int64)
            return
        self.bin = max(2.0 * float(self.r.max()), 1e-6)
        rm = float(self.r.max())
        self.x0 = float(self.cx.min()) - rm
        self.y0 = float(self.cy.min()) - rm
        self.nx = int(math.floor((float(self.cx.max()) + rm - self.x0) / self.bin)) + 1
        self.ny = int(math.floor((float(self.cy.max()) + rm - self.y0) / self.bin)) + 1
        # every bin a sphere overlaps lists it
        lists = []
        for s in range(n):
            i0 = int(math.floor((self.cx[s] - self.r[s] - self.x0) / self.bin))
            i1 = int(math.floor((self.cx[s] + self.r[s] - self.x0) / self.bin))
            j0 = int(math.floor((self.cy[s] - self.r[s] - self.y0) / self.bin))
            j1 = int(math.floor((self.cy[s] + self.r[s] - self.y0) / self.bin))
            for j in range(max(j0, 0), min(j1, self.ny - 1) + 1):
                for i in range(max(i0, 0), min(i1, self.nx - 1) + 1):
                    lists.append((j * self.nx + i, s))
        lists = np.array(lists, np.int64).reshape(-1, 2)
        order = np.argsort(lists[:, 0], kind="stable")
        lists = lists[order]
        counts = np.bincount(lists[:, 0], minlength=self.nx * self.ny)
        self.start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.members = np.ascontiguousarray(lists[:, 1])

    def top_chord(self, x, y):
        """(top height, chord) um at (x, y) um: the highest sphere surface and the summed
        projected thickness through the spheres there."""
        shape = np.shape(x)
        x = np.ascontiguousarray(x, np.float64).ravel()
        y = np.ascontiguousarray(y, np.float64).ravel()
        top, chord = np.zeros(x.size), np.zeros(x.size)
        if self.cx.size and x.size:
            _top_chord_nb(x, y, self.cx, self.cy, self.r, self.start, self.members, self.x0, self.y0,
                          1.0 / self.bin, self.nx, self.ny, top, chord)
        return top.reshape(shape), chord.reshape(shape)

    def occlusion(self, x, y, z0, to_source, elev: float, penumbra: float):
        """How much of a source at elevation `elev` towards `to_source` (unit xy) the spheres
        hide from surface points (x, y, z0) um, 0..1, with a penumbra `penumbra` rad wide."""
        shape = np.shape(x)
        x = np.ascontiguousarray(x, np.float64).ravel()
        y = np.ascontiguousarray(y, np.float64).ravel()
        z0 = np.ascontiguousarray(np.broadcast_to(np.asarray(z0, np.float64), shape)).ravel()
        out = np.zeros(x.size)
        if self.cx.size and x.size:
            ce, se = math.cos(elev), math.sin(elev)
            reach = 2.0 * float(self.r.max()) / max(math.tan(elev), 1e-3) + float(self.r.max())
            _occlusion_nb(x, y, z0, self.cx, self.cy, self.r, self.start, self.members, self.x0, self.y0,
                          1.0 / self.bin, self.nx, self.ny, to_source[0] * ce, to_source[1] * ce, se,
                          reach, 0.5 * self.bin, penumbra, out)
        return out.reshape(shape)


@_njit()
def _sites_nb(seed, i0, i1, j0, j1, jitter, out_x, out_y, out_h):
    k = 0
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            z = seed ^ (np.uint64(i) * np.uint64(0x9E3779B97F4A7C15)) ^ (np.uint64(j) * np.uint64(0xC2B2AE3D27D4EB4F))
            z = splitmix64_kernel(z)
            fx = float((z >> np.uint64(11)) & np.uint64(0xFFFFF)) / 1048576.0
            fy = float((z >> np.uint64(31)) & np.uint64(0xFFFFF)) / 1048576.0
            out_x[k] = i + 0.5 + jitter * (fx - 0.5)
            out_y[k] = j + 0.5 + jitter * (fy - 0.5)
            out_h[k] = z
            k += 1


def lattice_sites(seed: int, cell_um: float, x0: float, y0: float, x1: float, y1: float, jitter: float = 0.4):
    """The jittered-lattice sites (x um, y um, hash) of every cell overlapping [x0, x1] x [y0, y1]."""
    i0, i1 = int(math.floor(x0 / cell_um)), int(math.floor(x1 / cell_um))
    j0, j1 = int(math.floor(y0 / cell_um)), int(math.floor(y1 / cell_um))
    n = (i1 - i0 + 1) * (j1 - j0 + 1)
    xs, ys, hs = np.empty(n), np.empty(n), np.empty(n, np.uint64)
    _sites_nb(np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF), i0, i1, j0, j1, float(jitter), xs, ys, hs)
    return xs * cell_um, ys * cell_um, hs
