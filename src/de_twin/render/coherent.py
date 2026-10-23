"""Coherent 4D-STEM: probe wavefunction x specimen transmission, far field on the detector.

The kinematic STEM renderer (:mod:`de_twin.render.stem`) draws each scan point's pattern
from a cache of disk patterns; its bright-field disk has no interference, so neither
ptychography nor tilt-corrected bright field (tcBF / parallax) can work on it. This module
simulates the wave-optical 4D-STEM experiment instead.

Model (per scan position ``r_p``)
---------------------------------
::

    probe     psi_p(k) = A(|k|) exp(-i chi(k))            A: condenser aperture |k| <= alpha/lambda
                                                          (edge softened over one simulation pixel)
    chi(k)    from OpticsState.probe_aberrations (CEOS set, de_twin.optics.aberrations):
              C1 = effective defocus (negative = underfocus, the twin's TEM/CTF convention),
              A1 = condenser stigmator (OpticsConfig.condenser_stig_nm_per_unit, 1000 nm/unit),
              C3 (1.2 mm uncorrected, or what a probe corrector leaves: C3 ~ 0, C5, ...),
              beam tilt k_t: chi(k + k_t) - chi(k_t) (off-axis coma/astigmatism, and a probe
              shift = grad chi(k_t) / 2pi, e.g. defocus x tilt).
    exit wave psi(r)   = psi_p(r - r_p) t(r)              projection approximation, single slice
    pattern   I(k)     = |F[psi](k)|^2                    (norm: sum = 1 for a vacuum pattern)
    detector  electrons per binned pixel = pattern_e_per_s x integral of I over the pixel

``t(r) = A(r) exp(i phi(r))`` is :func:`de_twin.render.tem.transmission_function`, the same
object the TEM imaging model uses (mass-thickness amplitude, mean-inner-potential phase,
world-locked amorphous phase texture, grains at their effective orientation incl. stage tilt),
called with ``coherent_k_max = K_t`` (the object band limit, default ``k_det + alpha/lambda``:
everything that can reach the detector). Scattering a band-limited coherent simulation can
represent stays coherent:

* Bragg beams with ``|g| <= K_t`` form a phase grating of kinematic strength
  (``a_g = sqrt(2 f_g)`` per Friedel pair, scaled with the local thickness), so crystal disks
  interfere with the direct beam;
* the mass-thickness diffuse part below ``theta_c = lambda K_t`` is a world-locked random phase
  field with the screened-Rutherford power spectrum (elastic scattering of the amorphous
  potential, one frozen configuration).

Only what lies beyond ``K_t`` is removed from ``t`` and added back incoherently (Bragg disks of the
high-order reflections, the Wentzel tail beyond ``theta_c``), weighted by the probe intensity
over the window. Totals, BF and ADF signals therefore match the kinematic model (tested to a
few %), while the part of the pattern ptychography and tcBF use is coherent.

Sampling
--------
The detector delivers ``(Hd, Wd) = output_shape / hw_binning`` pixels of
``dk_b = recip_pixel_inv_nm * binning`` 1/nm. The simulation uses a square grid of ``n``
pixels with reciprocal pitch ``dk = recip_pixel_inv_nm * g / m`` (``g = gcd(bx, by)``), so
each binned detector pixel integrates ``(m bx/g) x (m by/g)`` simulation pixels, and::

    real-space window  W  = 1/dk = m / (recip * g)   >= 2 (1.1 R_geo + tail lambda/alpha + 3 sigma_s)
    real-space pixel   dx = 1 / (n dk)
    k Nyquist          n dk / 2 >= max(k_det, (k_det + alpha/lambda + K_t) / 2)
                       (t is band-limited to K_t: nothing wraps around onto the detector)

``R_geo = max |grad chi| / 2pi`` over the aperture is the geometric probe radius (``|C1| alpha``
for a defocused probe; ``C3 alpha^3`` for an uncorrected one), so a tcBF-style probe at 1 um
defocus and 20 mrad needs a ~50 nm window. ``m`` (the oversampling factor) is the smallest
integer giving that window; the grid is capped by ``RenderConfig.coherent_max_grid``
(``Sampling.truncated`` reports a window that had to be cut). Diffraction shift and
descan place the optic axis anywhere on the detector: the integer part moves the crop of the
simulation grid, the fractional part is a phase ramp on the probe (an exact k-shift).

Partial coherence
-----------------
* temporal: the focal spread ``Delta`` (1 sigma, from Cc and the energy spread) as a Gauss-Hermite
  quadrature over defocus (1, 3 or 5 samples, by the phase ``pi lambda Delta (alpha/lambda)^2``);
* spatial: a Gaussian effective source (``OpticsState.source_size_nm`` FWHM, from
  ``OpticsConfig.stem_source_size_nm`` scaled by 1/spot) as a 3x3 Gauss-Hermite quadrature of
  probe positions.
The weighted sample probes are decomposed into orthogonal modes (eigenvectors of their Gram
matrix, i.e. the mixed-state / coherent-mode decomposition of the partially coherent probe);
modes are kept until ``coherent_mode_power`` of the intensity (at most ``coherent_max_modes``)
and every pattern is the incoherent sum ``sum_m w_m |F[phi_m t]|^2``. At the defaults (200 kV,
Cc 1.4 mm, 0.7 eV, 0.04 nm source at spot 3) a 20 mrad probe keeps ~66 % in its first mode
(the source is comparable to the diffraction limit): single-mode ePIE still reaches a phase
correlation ~0.85, mixed-state ePIE with the ground-truth modes ~0.9. A larger spot number
shrinks the source (and the current).

Limits (documented)
-------------------
Projection approximation (one slice): valid while the specimen is thinner than the probe's
depth of field ``~lambda/alpha^2`` (6 nm at 20 mrad, 25 nm at 10 mrad) - thick particles are
approximated (no channelling / dynamical thickness dependence beyond ``exp(i phi)``, no HOLZ);
Bragg phases per beam are hashed, not the crystal's structure-factor phases. Reflections beyond
K_t and the Wentzel tail beyond ``lambda K_t`` are incoherent. No inelastic (plasmon) scattering
model, no strain affine (the ``strain`` layer is ignored here), no per-point intensity jitter,
crystalline pixels without their own grain have no Bragg scattering. Scan positions
snap to the simulation grid (``dx``, ~0.02-0.05 nm; the true positions used are reported by
:meth:`CoherentStem.ground_truth`).
"""

from __future__ import annotations


import math
import threading
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
from scipy import fft as sfft

from ..optics.state import probe_aberrations_of
from ..specimen.materials import GRAINS_PER_MATERIAL, MATERIALS
from .diffraction import (THICKNESS_BIN_NM, Pattern, PatternOptions, PatternSet, annulus_fractions,
                          bucket_patterns, diffuse_annulus_fraction, render_pattern, screening_angle_mrad)
from .stem import _tilt_key, resolve_scan_point
from . import stem_nb as _snb
from .util import pool, shift_bilinear
from .coherent_setup import (Probe, Sampling, Tile, _placement, build_probe, build_tile, field_view, plan_sampling)

BLOCK_WORK = 4e8  # n^2 log2(n) x modes per block of scan points (~ 32 points of a 576^2, 4-mode grid)


# -------------------------------------------------------------- the engine
class CoherentStem:
    """Coherent 4D-STEM engine; owned by :class:`~de_twin.render.renderer.Renderer`.

    ``field_map(view) -> (FieldMap, token)`` rasterises the specimen on a fine view (the
    renderer's cached rasteriser)."""

    def __init__(self, cache, config, seed: int):
        self.cache = cache  # DiffractionCache (Bragg / diffuse add-on patterns)
        self.config = config
        self.seed = seed
        self._lock = threading.RLock()  # the caches below; the read-ahead thread shares them
        self._probes: "OrderedDict[tuple, Probe]" = OrderedDict()
        self._samplings: "OrderedDict[tuple, Sampling]" = OrderedDict()
        self._tiles: "OrderedDict[tuple, Tile]" = OrderedDict()
        self._blocks: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._inflight: dict = {}  # block key -> Future of the read-ahead computing it
        self._ahead = None  # single-thread executor of the read-ahead
        self._block_bytes = 0
        self.blocks_computed = 0
        self.points_computed = 0

    # ------------------------------------------------------------ keys
    def _cfg_key(self) -> tuple:
        c = self.config
        return (c.coherent_max_grid, c.coherent_probe_tail, c.coherent_mode_power, c.coherent_max_modes,
                c.coherent_focal_samples, c.coherent_focal_spread, c.coherent_source_size, c.coherent_beam_tilt,
                c.coherent_incoherent_scattering, c.diffuse_scattering, c.max_g_inv_nm, c.mip_phase,
                c.edge_taper_nm, c.refraction_loss, c.phase_texture_scale, c.texture_bandlimit_nm,
                c.amplitude_contrast, c.lattice_fringes, c.lattice_phase_rad, c.lattice_fringe_model, c.lattice_fringe_efficiency,
                c.diffraction_contrast_scale,
                c.add_descan, c.descan_ramp_scale, tuple(c.descan_ramp_px), c.film_halo,
                c.coherent_object_bandwidth_inv_nm)

    def _optics_key(self, optics) -> tuple:
        return (optics, probe_aberrations_of(optics).key(), tuple(optics.extras.get("hw_binning", (1, 1))))

    @staticmethod
    def _probe_key(optics) -> tuple:
        """What of the optics the sampling and the probe depend on (not the scan, the view or
        the stage: a stage move or a new scan step keeps the probe)."""
        return (optics.wavelength_nm, optics.convergence_mrad, tuple(optics.output_shape),
                optics.recip_pixel_inv_nm, tuple(optics.diffraction_center_px),
                probe_aberrations_of(optics).key(), tuple(optics.beam_tilt_mrad), optics.source_size_nm,
                optics.focal_spread_nm, tuple(optics.extras.get("hw_binning", (1, 1))))

    def sampling(self, optics) -> Sampling:
        """The simulation sampling (cheap: no probe is built)."""
        key = (self._probe_key(optics), self._cfg_key())
        with self._lock:
            p = self._probes.get(key)
            if p is not None:
                return p.sampling
            hit = self._samplings.get(key)
            if hit is None:
                hit = plan_sampling(optics, self.config)
                self._samplings[key] = hit
                while len(self._samplings) > 8:
                    self._samplings.popitem(last=False)
            return hit

    def probe(self, optics, build: bool = True) -> Optional[Probe]:
        key = (self._probe_key(optics), self._cfg_key())
        with self._lock:
            p = self._probes.get(key)
            if p is not None:
                self._probes.move_to_end(key)
                return p
        if not build:
            return None
        p = build_probe(optics, self.config, self.sampling(optics))  # outside the lock (prefetch thread)
        with self._lock:
            if key in self._probes:
                return self._probes[key]
            self._probes[key] = p
            while len(self._probes) > 2:
                self._probes.popitem(last=False)
            return p

    # ------------------------------------------------------------ tiles
    def _scan_world(self, optics, points) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, np.int64).reshape(-1, 2)
        x, y = optics.view.pixel_to_world(pts[:, 1], pts[:, 0])
        return np.asarray(x, float), np.asarray(y, float)

    def _full_scan_points(self, optics) -> np.ndarray:
        ny, nx = optics.view.shape
        iy, ix = np.mgrid[0:ny, 0:nx]
        return np.stack([ix.ravel(), iy.ravel()], 1)

    def _field_views(self, optics, s: Sampling, points) -> list:
        """[(indices into points, fine view)]: the whole scan field when it fits
        ``coherent_field_max_px``; else the scan is cut into a fixed grid of scan tiles (whole
        rows if they fit, else runs of one row) each of whose fields fits, and the points are
        grouped by scan tile - the same tiles whichever block asks (live frames and the
        datacube agree), and a sparse scan is never one tile spanning all its points."""
        limit = self.config.coherent_field_max_px
        ny, nx = optics.view.shape
        allx, ally = self._scan_world(optics, self._full_scan_points(optics))
        full = field_view(allx, ally, s, optics.view)
        pts = np.asarray(points, np.int64).reshape(-1, 2)
        if full.shape[0] * full.shape[1] <= limit:
            return [(np.arange(len(pts)), full)]
        R, C = self._scan_tiling(optics, s, limit)
        tid = (pts[:, 1] // R) * (-(-nx // C)) + pts[:, 0] // C
        out = []
        for t in dict.fromkeys(tid.tolist()):  # in order of first appearance
            r, c = divmod(t, -(-nx // C))
            iy, ix = np.mgrid[r * R:min(ny, (r + 1) * R), c * C:min(nx, (c + 1) * C)]
            xs, ys = self._scan_world(optics, np.stack([ix.ravel(), iy.ravel()], 1))
            out.append((np.flatnonzero(tid == t), field_view(xs, ys, s, optics.view)))
        return out

    def _scan_tiling(self, optics, s: Sampling, limit: int) -> tuple[int, int]:
        """(rows, columns) of scan points per scan tile: as many whole rows as fit
        ``limit`` fine pixels, else one row cut into as many columns as fit."""
        ny, nx = optics.view.shape

        def px(rows, cols):
            iy, ix = np.mgrid[0:rows, 0:cols]
            xs, ys = self._scan_world(optics, np.stack([ix.ravel(), iy.ravel()], 1))
            v = field_view(xs, ys, s, optics.view)
            return v.shape[0] * v.shape[1]

        def largest(fits, hi):  # the largest k in 1..hi with fits(k) (fits is monotone)
            lo = 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if fits(mid):
                    lo = mid
                else:
                    hi = mid - 1
            return lo
        if px(1, nx) <= limit:
            return largest(lambda r: px(r, nx) <= limit, ny), nx
        return 1, largest(lambda c: px(1, c) <= limit, nx)

    def _tile(self, ctx, optics, s: Sampling, view, build: bool = True) -> Optional[Tile]:
        fm, token = ctx.field_map(view)
        key = (token, view, optics.ht_kv, optics.thickness_tilt_factor, optics.alpha_rad, optics.beta_rad,
               optics.convergence_mrad, self.seed, self._cfg_key())
        with self._lock:
            tile = self._tiles.get(key)
            if tile is not None:
                self._tiles.move_to_end(key)
                return tile
        if not build:
            return None
        tile = build_tile(fm, optics, ctx.grains, ctx.crystallinity, self.config, self.seed,
                          s.object_bandwidth)  # outside the lock (prefetch thread)
        with self._lock:
            if key in self._tiles:
                return self._tiles[key]
            self._tiles[key] = tile
            while len(self._tiles) > 2:
                self._tiles.popitem(last=False)
            return tile

    def tile_for(self, ctx, optics, s: Sampling, points) -> Tile:
        """Transmission tile covering the probe windows of ``points`` (the whole scan field when
        it fits ``coherent_field_max_px``; ``points`` must then lie in one part)."""
        (_, view), *rest = self._field_views(optics, s, points)
        if rest:
            raise ValueError("coherent STEM: these scan points need several transmission tiles")
        return self._tile(ctx, optics, s, view)

    def _prepare(self, ctx, optics, points, build: bool = True):
        """(probe, [(indices, tile)]) for rendering ``points``; None when ``build`` is False and
        something is not cached yet."""
        probe = self.probe(optics, build)
        if probe is None:
            return None
        groups = []
        for idx, view in self._field_views(optics, probe.sampling, points):
            tile = self._tile(ctx, optics, probe.sampling, view, build)
            if tile is None:
                return None
            groups.append((idx, tile))
        return probe, groups

    # ------------------------------------------------------------ descan
    def _descan_px(self, optics, fm, ix: int, iy: int) -> tuple[float, float]:
        config = self.config
        dx = dy = 0.0
        ny, nx = optics.view.shape
        if config.add_descan and nx > 0 and ny > 0:
            xs, ys, dg = config.descan_ramp_px
            dx += config.descan_ramp_scale * (2.0 * (ix - nx / 2.0) / nx * xs + xs + iy * dg / ny)
            dy += config.descan_ramp_scale * (2.0 * (iy - ny / 2.0) / ny * ys + ys)
        if fm is not None and fm.descan is not None:
            dx += float(fm.descan[0, iy, ix])
            dy += float(fm.descan[1, iy, ix])
        return dx, dy

    # ------------------------------------------------------------ core
    def _weights(self, tile: Tile, probe: Probe) -> dict:
        """Probe-intensity-weighted Bragg / diffuse weights for every window position of the
        tile (valid correlation: element (r0, c0) is the window starting there)."""
        from scipy.signal import correlate

        key = id(probe)
        hit = tile.cache.get(key)
        if hit is not None and hit[0] is probe:
            return hit[1]
        out = {}
        if tile.bragg is not None:
            out["bragg"] = correlate(tile.bragg, probe.intensity, mode="valid", method="fft")
        for m, dw in tile.diffuse.items():
            out[m] = correlate(dw, probe.intensity, mode="valid", method="fft")
        tile.cache.clear()
        tile.cache[key] = (probe, out)
        return out

    def _compute(self, ctx, optics, probe: Probe, tile: Tile, points, fm_scan=None, *,
                 annuli_mrad=None, want_patterns=True) -> tuple:
        s = probe.sampling
        n, h2 = s.n, s.n // 2
        (hd, wd), (fx, fy) = s.det_shape, s.sub
        H, W = hd * fy, wd * fx
        xs, ys = self._scan_world(optics, points)
        rows, cols = tile.view.world_to_pixel(xs, ys)
        rows = np.rint(rows).astype(np.int64)
        cols = np.rint(cols).astype(np.int64)
        if (rows.min() - h2 < 0 or cols.min() - h2 < 0 or rows.max() + h2 > tile.t.shape[0]
                or cols.max() + h2 > tile.t.shape[1]):
            raise RuntimeError("coherent STEM: probe window outside the transmission tile")
        pts = np.asarray(points, np.int64).reshape(-1, 2)
        npts = len(pts)
        shifts = [self._descan_px(optics, fm_scan, int(ix), int(iy)) for ix, iy in pts]
        # crop start (array index of the first detector sub-pixel; k = 0 sits at n/2) per point
        place = []
        for sx, sy in shifts:
            ix, ex, iy, ey = _placement(s, (s.center_px[0] + sx, s.center_px[1] + sy))
            place.append((ix + h2, ex - probe.frac[0], iy + h2, ey - probe.frac[1]))
        place = np.array(place)
        any_ramp = bool(np.abs(place[:, [1, 3]]).max() > 1e-9)
        amask = None
        if annuli_mrad:
            f = (np.arange(n) - h2) * s.dk * (1000.0 * optics.wavelength_nm)
            kk = np.hypot(f[None, :], f[:, None])
            amask = [((kk >= a) & (kk < b)).astype(np.float32) for a, b in annuli_mrad]
        windows = np.lib.stride_tricks.sliding_window_view(tile.t, (n, n))
        work, mw = probe.work, probe.weights

        pats = np.zeros((npts, hd, wd), np.float32) if want_patterns else None
        anns = np.zeros((npts, len(amask)), np.float64) if amask is not None else None

        def crop_into(acc, I, k, x0, y0, w):
            # add w * I[y0:y0+H, x0:x0+W] into acc[k] (zero outside the simulated k range)
            ya, yb = max(y0, 0), min(y0 + H, n)
            xa, xb = max(x0, 0), min(x0 + W, n)
            if ya < yb and xa < xb:
                acc[k, ya - y0:yb - y0, xa - x0:xb - x0] += w * I[ya:yb, xa:xb]

        def sq(G):
            v = G.view(np.float32)
            v = v.reshape(v.shape[:-1] + (v.shape[-1] // 2, 2))
            return np.einsum("...i,...i->...", v, v)

        def run(lo, hi):
            B = hi - lo
            rr, cc = rows[lo:hi] - h2, cols[lo:hi] - h2
            ramp = None
            if any_ramp:
                ar = np.arange(n)
                ry = np.exp(-2j * np.pi * place[lo:hi, 3, None] * ar / n).astype(np.complex64)
                rx = np.exp(-2j * np.pi * place[lo:hi, 1, None] * ar / n).astype(np.complex64)
                ramp = ry[:, :, None] * rx[:, None, :]
            acc = np.zeros((B, H, W), np.float32) if want_patterns else None
            full = np.zeros((B, n, n), np.float32) if amask is not None else None
            x0s = place[lo:hi, 0].astype(int)
            y0s = place[lo:hi, 2].astype(int)
            simple = (want_patterns and (x0s == x0s[0]).all() and (y0s == y0s[0]).all()
                      and y0s[0] >= 0 and x0s[0] >= 0 and y0s[0] + H <= n and x0s[0] + W <= n)
            e = np.empty((B, n, n), np.complex64)
            for mode, w in zip(work, mw):
                for k in range(B):
                    np.multiply(windows[rr[k], cc[k]], mode, out=e[k])
                if ramp is not None:
                    e *= ramp
                F = sfft.fft2(e, norm="ortho", workers=1, overwrite_x=True)
                wf = np.float32(w)
                if simple:
                    I = sq(F[:, y0s[0]:y0s[0] + H, x0s[0]:x0s[0] + W])
                    if len(mw) == 1:
                        acc = I
                    else:
                        acc += wf * I
                    if full is not None:
                        full += wf * sq(F)
                    continue
                I = sq(F)
                if full is not None:
                    full += wf * I
                if want_patterns:
                    for k in range(B):
                        crop_into(acc, I[k], k, x0s[k], y0s[k], wf)
            if want_patterns:
                if fx == 1 and fy == 1:
                    pats[lo:hi] = acc
                else:
                    pats[lo:hi] = acc.reshape(B, hd, fy, wd, fx).sum(axis=(2, 4), dtype=np.float32)
            if full is not None:
                for j, m in enumerate(amask):
                    anns[lo:hi, j] = np.tensordot(full, m, axes=([1, 2], [0, 1]))

        if annuli_mrad:  # the annuli's k pixels, for the fused sums
            ann_idx = [tuple(np.ascontiguousarray(v, np.int64) for v in np.nonzero(m)) for m in amask]

        def run_nb(lo, hi):
            # fused: exit waves, FFT, w |F|^2 binned straight into the patterns / annulus sums
            B = hi - lo
            rr = np.ascontiguousarray(rows[lo:hi] - h2)
            cc = np.ascontiguousarray(cols[lo:hi] - h2)
            ry = rx = np.zeros((1, 1), np.complex64)
            if any_ramp:
                ar = np.arange(n)
                ry = np.exp(-2j * np.pi * place[lo:hi, 3, None] * ar / n).astype(np.complex64)
                rx = np.exp(-2j * np.pi * place[lo:hi, 1, None] * ar / n).astype(np.complex64)
            acc = np.zeros((B, hd, wd), np.float32) if want_patterns else None
            x0s = np.ascontiguousarray(place[lo:hi, 0].astype(np.int64))
            y0s = np.ascontiguousarray(place[lo:hi, 2].astype(np.int64))
            if amask is None and (int(x0s.min()) >= n or int(x0s.max()) + W <= 0 or int(y0s.min()) >= n
                                  or int(y0s.max()) + H <= 0):
                return  # the detector sees none of the simulated k range: zeros
            e = np.empty((B, n, n), np.complex64)
            for mode, w in zip(work, mw):
                _snb.fill_exit(tile.t, rr, cc, mode, ry, rx, any_ramp, e)
                F = sfft.fft2(e, norm="ortho", workers=1, overwrite_x=True)
                if want_patterns:
                    _snb.accumulate(F, y0s, x0s, H, W, fy, fx, np.float32(w), acc)
                if amask is not None:
                    for j, (iy_, ix_) in enumerate(ann_idx):
                        _snb.annulus_sums(F, iy_, ix_, float(w), anns[lo:hi], j)
            if want_patterns:
                pats[lo:hi] = acc

        nw = pool()._max_workers
        bs = int(max(1, min(64, (24 << 20) // (n * n * 8), -(-npts // (2 * nw)))))
        edges = list(range(0, npts, bs)) + [npts]
        jobs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        job = run_nb if _snb.AVAILABLE else run
        if len(jobs) == 1:
            job(*jobs[0])
        else:
            list(pool().map(lambda j: job(*j), jobs))

        # incoherent add-on weights (probe-intensity weighted over the window)
        r0, c0 = rows - h2, cols - h2
        if _snb.AVAILABLE:  # the points' own windows (no whole-tile correlation)
            wts = {}
            maps = ([("bragg", tile.bragg)] if tile.bragg is not None else []) + list(tile.diffuse.items())
            for m, a in maps:
                v = np.empty(npts)
                _snb.window_dots(a, np.ascontiguousarray(r0), np.ascontiguousarray(c0), probe.intensity, v)
                wts[m] = v
            per_point = True  # already one value per point
        else:
            wts = self._weights(tile, probe)
            per_point = False
        bragg_w = np.zeros(npts)
        keys: list = [None] * npts
        if "bragg" in wts:
            bragg_w = np.maximum(wts["bragg"] if per_point else wts["bragg"][r0, c0], 0.0).astype(np.float64)
            for k in np.flatnonzero(bragg_w > 1e-12):
                r, c = int(rows[k]), int(cols[k])
                g = int(tile.gid[r, c])
                if g < 0:  # probe centre off the grain: the grain dominating the window
                    prod = probe.intensity * tile.bragg[r - h2:r + h2, c - h2:c + h2]
                    jr, jc = divmod(int(np.argmax(prod)), n)
                    r, c = r - h2 + jr, c - h2 + jc
                    g = int(tile.gid[r, c])
                if g >= 0:
                    keys[k] = (g // GRAINS_PER_MATERIAL, g, int(tile.tbin[r, c]))
                else:
                    bragg_w[k] = 0.0
        diffuse = {m: np.maximum(v if per_point else v[r0, c0], 0.0).astype(np.float64)
                   for m, v in wts.items() if m != "bragg"}
        self.points_computed += npts
        return pats, anns, bragg_w, keys, diffuse, shifts

    # ------------------------------------------------------ add-on patterns
    def _bragg_pattern(self, ctx, optics, s: Sampling, key, center_b) -> Optional[np.ndarray]:
        hd, wd = s.det_shape
        rb = s.det_recip_px[0]
        ck = ("coh-bragg", key, (hd, wd), rb, round(center_b[0], 3), round(center_b[1], 3), s.object_bandwidth,
              optics.convergence_mrad, optics.alpha_rad, optics.beta_rad, optics.ht_kv, self.config.max_g_inv_nm,
              _tilt_key(optics))

        def render(_k):
            mid, gid, tb = key
            ps = bucket_patterns([mid], [gid], [tb * THICKNESS_BIN_NM], [1.0], ctx.grains, optics,
                                 PatternOptions(film_halo=False, max_g_inv_nm=self.config.max_g_inv_nm))
            sp = ps[0].spots
            sp = sp[np.hypot(sp[:, 0], sp[:, 1]) > max(1e-9, s.object_bandwidth)]
            tot = float(sp[:, 2].sum()) if len(sp) else 0.0
            if tot <= 0:
                return np.zeros((hd, wd), np.float32)
            sp = sp.copy()
            sp[:, 2] /= tot
            disk = optics.convergence_mrad / 1000.0 / optics.wavelength_nm / rb
            return render_pattern(Pattern(sp), (hd, wd), rb, disk, center=center_b)
        return self.cache.get(ck, render)

    def diffuse_cut_mrad(self, optics) -> float:
        """Inner angle of the incoherent diffuse background in coherent mode: the scattering
        angle ``lambda K_t`` beyond which the band-limited transmission function has none."""
        if not self.config.coherent_incoherent_scattering:
            return 0.0
        return 1000.0 * optics.wavelength_nm * self.sampling(optics).object_bandwidth

    def _diffuse_pattern(self, optics, s: Sampling, mid: int, center_b) -> np.ndarray:
        """Screened-Rutherford background beyond ``diffuse_cut_mrad`` (plane integral 1): the
        mass-thickness electrons are scattered to large angles; inside the bright-field cone the
        coherent wave already carries the specimen's (elastic) scattering."""
        hd, wd = s.det_shape
        rb = s.det_recip_px[0]
        cut = self.diffuse_cut_mrad(optics)
        ck = ("coh-diffuse", mid, (hd, wd), rb, round(center_b[0], 3), round(center_b[1], 3), optics.ht_kv,
              round(cut, 6))

        def render(_k):
            from scipy.special import erfc

            mrad_px = rb * 1000.0 * optics.wavelength_nm
            t0 = screening_angle_mrad(mid, optics.wavelength_nm)
            y = (np.arange(hd, dtype=np.float32) - np.float32(center_b[1]))[:, None]
            x = (np.arange(wd, dtype=np.float32) - np.float32(center_b[0]))[None, :]
            r = np.sqrt(x * x + y * y) * np.float32(mrad_px)
            p = np.float32(t0 * t0 / math.pi * mrad_px * mrad_px) / (r * r + np.float32(t0 * t0)) ** 2
            if cut > 0:
                p = p * (0.5 * erfc((cut - r) / (math.sqrt(2.0) * mrad_px))).astype(np.float32)
                p /= np.float32(t0 * t0 / (cut * cut + t0 * t0))
            return p.astype(np.float32)
        return self.cache.get(ck, render)

    # ------------------------------------------------------------- blocks
    def _compute_groups(self, ctx, optics, probe: Probe, groups, points, fm_scan=None, **kw) -> tuple:
        """:meth:`_compute` over the tiles of ``groups``, merged back into points order."""
        pts = np.asarray(points, np.int64).reshape(-1, 2)
        if len(groups) == 1:
            return self._compute(ctx, optics, probe, groups[0][1], pts, fm_scan, **kw)
        parts = [(idx, self._compute(ctx, optics, probe, tile, pts[idx], fm_scan, **kw)) for idx, tile in groups]
        npts = len(pts)

        def merged(i, shape_of):
            first = parts[0][1][i]
            if first is None:
                return None
            out = np.zeros((npts,) + shape_of(first), first.dtype)
            for idx, r in parts:
                out[idx] = r[i]
            return out
        pats = merged(0, lambda a: a.shape[1:])
        anns = merged(1, lambda a: a.shape[1:])
        bragg_w = merged(2, lambda a: ())
        keys: list = [None] * npts
        shifts: list = [None] * npts
        diffuse: dict = {}
        for idx, r in parts:
            for k, i in enumerate(idx):
                keys[i] = r[3][k]
                shifts[i] = r[5][k]
            for m, v in r[4].items():
                diffuse.setdefault(m, np.zeros(npts))[idx] = v
        return pats, anns, bragg_w, keys, diffuse, shifts

    def _block_size(self, optics, s: Sampling) -> int:
        """Points per block: about the same work per block whatever the grid (a first pattern
        waits for one block), a block at most 32 MB of patterns."""
        ny, nx = optics.view.shape
        total = max(1, ny * nx)
        per = s.det_shape[0] * s.det_shape[1] * 4
        work = s.n * s.n * max(1.0, math.log2(s.n)) * max(1, int(self.config.coherent_max_modes))
        b = int(min(256, max(16, BLOCK_WORK // work)))
        b = min(total, b, max(1, (32 << 20) // per))
        return int(b)

    def patterns(self, ctx, optics, points, fm_scan=None) -> np.ndarray:
        """(P, Hd, Wd) float32 binned detector patterns (electrons / binned pixel / s)."""
        probe, groups = self._prepare(ctx, optics, points)
        return self._patterns(ctx, optics, points, fm_scan, probe, groups)

    def _patterns(self, ctx, optics, points, fm_scan, probe: Probe, groups) -> np.ndarray:
        s = probe.sampling
        out, _, bragg_w, keys, diffuse, shifts = self._compute_groups(ctx, optics, probe, groups, points, fm_scan)
        cb = s.det_center_px
        bx, by = s.binning
        for k in range(len(out)):
            add = None
            if keys[k] is not None and bragg_w[k] > 0:
                add = np.float32(bragg_w[k]) * self._bragg_pattern(ctx, optics, s, keys[k], cb)
            for m, wm in diffuse.items():
                if wm[k] > 0:
                    d = np.float32(wm[k]) * self._diffuse_pattern(optics, s, m, cb)
                    add = d if add is None else add + d
            if add is not None:
                sx, sy = shifts[k]
                if sx or sy:
                    add = shift_bilinear(add, sx / bx, sy / by)
                out[k] += add
        out *= np.float32(optics.pattern_e_per_s)
        return out

    def _block_key(self, optics, fm_token, bsz: int, b: int) -> tuple:
        return (self._optics_key(optics), self._cfg_key(), fm_token, bsz, b)

    def _block_points(self, optics, bsz: int, b: int) -> np.ndarray:
        ny, nx = optics.view.shape
        idx = np.arange(b * bsz, min(ny * nx, (b + 1) * bsz))
        return np.stack([idx % nx, idx // nx], 1)

    def _store(self, key, blk: np.ndarray) -> None:
        with self._lock:
            if key in self._blocks:
                return
            blk.setflags(write=False)
            self.blocks_computed += 1
            self._blocks[key] = blk
            self._block_bytes += blk.nbytes
            budget = int(self.config.coherent_cache_mb) << 20
            while self._block_bytes > budget and len(self._blocks) > 1:
                _, v = self._blocks.popitem(last=False)
                self._block_bytes -= v.nbytes

    def _block(self, ctx, optics, fm_scan, fm_token, index: int,
               read_ahead: bool = False) -> tuple[np.ndarray, int]:
        s = self.sampling(optics)
        ny, nx = optics.view.shape
        bsz = self._block_size(optics, s)
        b = index // bsz
        key = self._block_key(optics, fm_token, bsz, b)
        with self._lock:
            blk = self._blocks.get(key)
            if blk is not None:
                self._blocks.move_to_end(key)
            fut = self._inflight.get(key) if blk is None else None
        if blk is None and fut is not None:
            try:
                blk = fut.result()  # being read ahead: wait for it (it takes no lock we hold)
            except Exception:  # noqa: BLE001 - compute it here instead
                blk = None
        if blk is None:
            with self._lock:  # a new view: what is read ahead for another one is not wanted
                for k, f in list(self._inflight.items()):
                    if k[:3] != key[:3] and f.cancel():
                        self._inflight.pop(k, None)
            blk = self.patterns(ctx, optics, self._block_points(optics, bsz, b), fm_scan)
            self._store(key, blk)
        if read_ahead:
            self._read_ahead(ctx, optics, fm_scan, fm_token, bsz, b, -(-(ny * nx) // bsz))
        return blk, index - b * bsz

    def _read_ahead(self, ctx, optics, fm_scan, fm_token, bsz: int, b: int, nblocks: int) -> None:
        """Compute the next ``coherent_read_ahead`` blocks of the scan in the background (live
        4D-STEM walks the scan in order), when their probe and tile are already built."""
        n_ahead = int(getattr(self.config, "coherent_read_ahead", 0))
        if n_ahead <= 0 or nblocks <= 1:
            return
        for d in range(1, min(n_ahead, nblocks - 1) + 1):
            nb_ = (b + d) % nblocks
            key = self._block_key(optics, fm_token, bsz, nb_)
            with self._lock:
                if key in self._blocks or key in self._inflight:
                    continue
            pts = self._block_points(optics, bsz, nb_)
            prep = self._prepare(ctx, optics, pts, build=False)
            if prep is None:
                return  # needs a new tile: left to the render thread
            if self._ahead is None:
                from concurrent.futures import ThreadPoolExecutor

                self._ahead = ThreadPoolExecutor(1, thread_name_prefix="de-twin-4dstem-ahead")

            def job(key=key, pts=pts, prep=prep):
                try:
                    blk = self._patterns(ctx, optics, pts, fm_scan, *prep)
                    self._store(key, blk)
                    return blk
                finally:
                    with self._lock:
                        self._inflight.pop(key, None)
            with self._lock:
                self._inflight[key] = self._ahead.submit(job)

    def prefetch(self, ctx, optics, fm_scan, fm_token) -> bool:
        """Build the probe and transmission tile of ``optics`` and compute its first block (a
        view the caller predicts), so the first pattern there is a cache hit. Runs on the
        caller's (prefetch) thread; the render thread waits for the block if it gets there
        first. Returns whether anything was computed."""
        s = self.sampling(optics)
        bsz = self._block_size(optics, s)
        key = self._block_key(optics, fm_token, bsz, 0)
        from concurrent.futures import Future

        fut = Future()
        fut.set_running_or_notify_cancel()  # running: the render thread waits, never cancels it
        with self._lock:
            if key in self._blocks or key in self._inflight:
                return False
            self._inflight[key] = fut
        try:
            blk = self.patterns(ctx, optics, self._block_points(optics, bsz, 0), fm_scan)
            self._store(key, blk)
            fut.set_result(blk)
            return True
        except BaseException as e:  # noqa: BLE001 - the waiting render thread computes it itself
            fut.set_exception(e)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def binned(self, ctx, fm_scan, fm_token, optics, scan_point, frame_index, specimen=None) -> np.ndarray:
        ix, iy = resolve_scan_point(optics, scan_point, frame_index, specimen, self.config)
        nx = optics.view.shape[1]
        blk, j = self._block(ctx, optics, fm_scan, fm_token, iy * nx + ix, read_ahead=scan_point is None)
        return blk[j]

    def render(self, ctx, fm_scan, fm_token, optics, scan_point, frame_index, specimen=None) -> np.ndarray:
        """float32 (h, w) = optics.output_shape, electrons / unbinned pixel / s."""
        pat = self.binned(ctx, fm_scan, fm_token, optics, scan_point, frame_index, specimen)
        return expand_binned(pat, optics)

    def datacube(self, ctx, optics, fm_scan=None, fm_token=None) -> np.ndarray:
        """(ny, nx, Hd, Wd) float32: every scan point's binned pattern (electrons / pixel / s)."""
        ny, nx = optics.view.shape
        s = self.sampling(optics)
        out = np.empty((ny * nx,) + s.det_shape, np.float32)
        i = 0
        while i < ny * nx:
            blk, j = self._block(ctx, optics, fm_scan, fm_token, i)
            k = min(len(blk) - j, ny * nx - i)
            out[i:i + k] = blk[j:j + k]
            i += k
        return out.reshape(ny, nx, *s.det_shape)

    # ------------------------------------------------------ virtual images
    def virtual_image(self, ctx, optics, inner_mrad: float, outer_mrad: float, fm_scan=None) -> np.ndarray:
        """Annular detector image over the scan: the coherent part integrated on the full
        simulation grid, the incoherent Bragg / diffuse parts analytically (not truncated by
        the camera, like the kinematic model)."""
        probe = self.probe(optics)
        s = probe.sampling
        ny, nx = optics.view.shape
        pts = self._full_scan_points(optics)
        out = np.zeros(ny * nx, np.float64)
        lam = optics.wavelength_nm
        bsz = max(1, self._block_size(optics, s))
        cut = self.diffuse_cut_mrad(optics)
        dfrac = {}
        for m in range(len(MATERIALS)):
            t0 = screening_angle_mrad(m, lam) if m else 1.0
            beyond = t0 * t0 / (cut * cut + t0 * t0)
            dfrac[m] = (diffuse_annulus_fraction(m, lam, max(inner_mrad, cut), max(outer_mrad, cut)) / beyond
                        if m and beyond > 0 else 0.0)
        bfrac: dict = {}
        for b0 in range(0, len(pts), bsz):
            chunk = pts[b0:b0 + bsz]
            _, groups = self._prepare(ctx, optics, chunk)
            _, anns, bragg_w, keys, diffuse, _ = self._compute_groups(
                ctx, optics, probe, groups, chunk, fm_scan, annuli_mrad=[(inner_mrad, outer_mrad)],
                want_patterns=False)
            v = anns[:, 0].copy()
            for k, key in enumerate(keys):
                if key is None or bragg_w[k] <= 0:
                    continue
                if key not in bfrac:
                    mid, gid, tb = key
                    ps = bucket_patterns([mid], [gid], [tb * THICKNESS_BIN_NM], [1.0], ctx.grains, optics,
                                         PatternOptions(film_halo=False, max_g_inv_nm=self.config.max_g_inv_nm))
                    sp = ps[0].spots
                    sp = sp[np.hypot(sp[:, 0], sp[:, 1]) > max(1e-9, s.object_bandwidth)]
                    tot = float(sp[:, 2].sum()) if len(sp) else 0.0
                    bfrac[key] = 0.0
                    if tot > 0:
                        sp = sp.copy()
                        sp[:, 2] /= tot
                        one = PatternSet(1, sp, np.zeros(len(sp), np.int64), np.zeros((0, 3)),
                                         np.zeros(0, np.int64))
                        bfrac[key] = float(annulus_fractions(one, optics.convergence_mrad, lam,
                                                             inner_mrad, outer_mrad)[0])
                v[k] += bragg_w[k] * bfrac[key]
            for m, wm in diffuse.items():
                v += wm * dfrac[m]
            out[b0:b0 + len(chunk)] = v
        return (out.reshape(ny, nx) * optics.pattern_e_per_s).astype(np.float32)

    # ---------------------------------------------------------- ground truth
    def ground_truth(self, ctx, optics) -> dict:
        probe = self.probe(optics)
        s = probe.sampling
        pts = self._full_scan_points(optics)
        allx, ally = self._scan_world(optics, pts)
        view = field_view(allx, ally, s, optics.view)
        fm, token = ctx.field_map(view)
        key = (token, view, optics.ht_kv, optics.thickness_tilt_factor, optics.alpha_rad, optics.beta_rad,
               optics.convergence_mrad, self.seed, self._cfg_key())
        tile = self._tiles.get(key)
        if tile is None:
            tile = build_tile(fm, optics, ctx.grains, ctx.crystallinity, self.config, self.seed,
                              s.object_bandwidth)
        rows, cols = tile.view.world_to_pixel(allx, ally)
        ny, nx = optics.view.shape
        pos = np.stack([np.rint(rows), np.rint(cols)], 1).astype(np.float64)  # snapped: exactly as simulated
        ab = probe.aberrations
        modes = probe.modes
        return {
            "object": tile.t, "dx_nm": s.dx_nm, "view": tile.view,
            "positions_px": pos, "positions_nm": pos * s.dx_nm, "scan_shape": (ny, nx),
            "probe": modes[0], "probe_modes": modes, "mode_weights": probe.weights,
            "coherent_fraction": probe.coherent_fraction,
            "wavelength_nm": optics.wavelength_nm, "convergence_mrad": optics.convergence_mrad,
            "alpha_mrad": optics.convergence_mrad,
            "aberrations": ab, "aberrations_polar": ab.polar(),
            "C1_nm": float(ab["C1"].real), "A1_nm": complex(ab["A1"]), "C3_nm": float(ab["C3"].real),
            "B2_nm": complex(ab["B2"]), "C5_nm": float(ab["C5"].real),
            "defocus_nm": float(optics.defocus_um * 1000.0),
            "beam_tilt_mrad": tuple(optics.beam_tilt_mrad), "focal_spread_nm": float(optics.focal_spread_nm),
            "source_size_nm": float(optics.source_size_nm),
            "scan_step_nm": optics.scan_step_um * 1000.0, "scan_rotation_rad": optics.scan_rotation_rad,
            "sampling": s,
            "detector_shape": s.det_shape, "detector_recip_px_inv_nm": s.det_recip_px,
            "detector_center_px": s.det_center_px,
        }


def expand_binned(pat: np.ndarray, optics) -> np.ndarray:
    """Binned (Hd, Wd) pattern -> unbinned output (h, w), each sensor pixel an equal share."""
    h, w = (int(v) for v in optics.output_shape)
    bx, by = (max(1, int(v)) for v in optics.extras.get("hw_binning", (1, 1)))
    if bx == 1 and by == 1 and pat.shape == (h, w):
        return np.array(pat, np.float32, copy=True)
    up = np.repeat(np.repeat(pat, by, axis=0), bx, axis=1) * np.float32(1.0 / (bx * by))
    out = np.zeros((h, w), np.float32)
    hh, ww = min(h, up.shape[0]), min(w, up.shape[1])
    out[:hh, :ww] = up[:hh, :ww]
    return out


def choose_model(optics, config, sampling_fn: Callable[[], Sampling]) -> str:
    """``RenderConfig.stem_model`` resolved for these optics ("coherent" | "kinematic")."""
    m = str(config.stem_model).lower()
    if m in ("coherent", "kinematic"):
        return m
    try:
        s = sampling_fn()
    except Exception:  # noqa: BLE001 - degenerate geometry: fall back to the robust model
        return "kinematic"
    if s.truncated or s.n > config.coherent_max_grid:
        return "kinematic"
    hd, wd = s.det_shape
    if hd * wd <= config.coherent_auto_max_pattern_px:
        return "coherent"
    # larger patterns: when the scan oversamples the probe (ptychography-style overlap)
    if 0 < optics.scan_step_um * 1000.0 < 2.0 * s.probe_size_nm:
        return "coherent"
    return "kinematic"
