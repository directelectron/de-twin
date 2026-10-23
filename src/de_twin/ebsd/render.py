"""EBSD rendering: per scan point, project the grain's master pattern onto the detector.

Per scan point (one per raster pixel of the EBSD optics' view, pitch = scan step on the
tilt-corrected sample surface):

1. the microstructure under the beam (:meth:`de_twin.specimen.bulk.BulkSurface.sample`): grain,
   phase, orientation, the neighbouring grain and the distance to the boundary between them,
   the surface quality and, for a strained specimen, the deformation gradient;
2. near a boundary the neighbour's pattern is mixed in with the fraction of the interaction
   volume's footprint (an ellipse ~0.02 x the Kanaya-Okayama range across the tilt axis,
   1 / cos(tilt) longer along it) that lies across the boundary;
3. the pattern centre for the beam position (:meth:`EbsdGeometry.pattern_centre`: linear in
   the beam offset);
4. one threaded numba pass (:func:`de_twin.ebsd.kernels.render`) over the binned detector grid:
   gnomonic direction of each (supersampled) pixel, through the inverse deformation gradient
   and the grain orientation into the crystal, master-pattern lookup, the backscatter
   background (cosine + forward lobe, x the backscatter yield and the beam current), the
   pixel's solid angle and the band modulation ``1 + band_contrast * quality * m``; for a
   camera frame it also writes the detector's Poisson means (``means``), so the detector
   starts its frame from them without another pass.

Steps 1-3 are done once per scan (:class:`EbsdTables`, :class:`EbsdPlan`: ~100 bytes per scan
point, +72 for a strained specimen) and kept; master patterns are process-wide
(:mod:`de_twin.ebsd.master`) unless the twin was given its own (``user_masters``).

A specimen without :meth:`sample` (a TEM specimen) is read from its field map: phase and
grain-table orientation per scan point, no boundary mixing.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..specimen.materials import GRAINS_PER_MATERIAL, material
from . import kernels
from .geometry import EbsdGeometry, background_parameters, backscatter_yield, boundary_mix, footprint_um
from .master import MasterPattern, master_for

ELECTRON_C = 1.602176634e-19
_NO_F = np.zeros((1, 3, 3), np.float32)
F32 = np.float32


@dataclass
class EbsdTables:
    """Per scan point (ny, nx) microstructure of one scan."""

    x_um: np.ndarray  # beam position on the sample (sample frame), float64
    y_um: np.ndarray
    material: np.ndarray  # uint8, 0 = vacuum
    material2: np.ndarray
    grain_id: np.ndarray  # int32, -1 = none
    grain_id2: np.ndarray
    orient: np.ndarray  # (ny, nx) int32 index into matrices (-1 = no crystal)
    orient2: np.ndarray
    matrices: np.ndarray  # (K, 3, 3) crystal -> sample
    mix: np.ndarray  # float32 weight of the neighbour (0 .. 0.5)
    quality: np.ndarray  # float32 surface quality (0, 1]
    F: Optional[np.ndarray] = None  # (ny, nx, 3, 3) float32 deformation gradient (sample frame)
    gnd: Optional[np.ndarray] = None  # (ny, nx) float32 GND density proxy, 1/m^2

    @property
    def nbytes(self) -> int:
        return int(sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray)))


@dataclass
class EbsdPlan:
    """Everything the kernel needs for every point of a scan (flattened, row-major)."""

    bank: np.ndarray  # (B, Nf, Nf) uint8 cube-map faces of every phase
    face0: np.ndarray  # (P,) int32 bank index of each phase's first face
    full: np.ndarray  # (P,) bool: six faces (non-centrosymmetric), else three
    offs: np.ndarray  # (P,) code -> standardised value
    scales: np.ndarray
    ph1: np.ndarray  # (n,) int32 phase of the grain (-1 none)
    ph2: np.ndarray
    o1: np.ndarray  # (n,) int32 orientation index
    o2: np.ndarray
    mix: np.ndarray  # (n,) float32
    contrast: np.ndarray  # (n,) float32 band modulation (band_contrast * quality)
    es: np.ndarray  # (n,) float32 backscattered electrons per second
    finv: np.ndarray  # (n, 3, 3) float32 inverse deformation gradient, or (1, 3, 3)
    has_f: bool
    dx: np.ndarray  # (n,) float32 beam offset from the scan centre, µm
    dy: np.ndarray
    matrices: np.ndarray
    pc_lin: np.ndarray  # (9,) pattern centre (sensor px) linear in the beam offset
    s2dT: np.ndarray
    normal: np.ndarray
    fwd: np.ndarray
    lobe: float
    kappa: float
    masters: list = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return int(sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray)))


def ebsd_geometry(optics) -> EbsdGeometry:
    g = optics.extras.get("ebsd")
    return g if isinstance(g, EbsdGeometry) else EbsdGeometry()


def _key(optics) -> tuple:
    g = ebsd_geometry(optics)
    return (optics.view, round(optics.ht_kv, 4), g.sample_tilt_deg, g.boundary_blend, g.interaction_scale)


def pc_linear(geometry: EbsdGeometry, sensor_shape, pixel_um: float, pc=None) -> np.ndarray:
    """The pattern centre in sensor pixels as ``a + b dx + c dy`` (beam offset µm): (x0, dx0/ddx,
    dx0/ddy, y0, ..., dd, ...). ``pc``: a fixed Bruker pattern centre instead (no slope)."""
    H, W = (float(v) for v in sensor_shape)
    if pc is not None:
        return np.array([pc[0] * W, 0, 0, pc[1] * H, 0, 0, pc[2] * H, 0, 0], np.float64)
    p0 = geometry.pattern_centre(0.0, 0.0, (H, W), pixel_um)
    px = geometry.pattern_centre(1.0, 0.0, (H, W), pixel_um) - p0
    py = geometry.pattern_centre(0.0, 1.0, (H, W), pixel_um) - p0
    s = np.array([W, H, H])
    return np.array([[p0[i] * s[i], px[i] * s[i], py[i] * s[i]] for i in range(3)], np.float64).ravel()


class EbsdRenderer:
    """Owns the EBSD scan tables and plans; renders patterns, pattern stacks and ground truth.

    ``user_masters``: material id -> list of user :class:`MasterPattern` (any energy), used in
    place of the kinematic master when one is within ``USER_MASTER_KV_TOLERANCE`` of the beam
    energy (see :meth:`de_twin.twin.DigitalTwin.set_master_pattern`)."""

    USER_MASTER_KV_TOLERANCE = 0.5

    def __init__(self, seed: int = 0, user_masters: Optional[dict] = None):
        self.seed = int(seed)
        self.user_masters = user_masters if user_masters is not None else {}
        self._tables: "OrderedDict[tuple, EbsdTables]" = OrderedDict()
        self._plans: "OrderedDict[tuple, EbsdPlan]" = OrderedDict()
        self._buffers: dict = {}
        self._lock = threading.RLock()
        self.patterns_rendered = 0

    # ------------------------------------------------------------------ masters
    def master(self, material_id: int, kv: float, geometry: EbsdGeometry) -> MasterPattern:
        users = self.user_masters.get(int(material_id)) or ()
        best = min(users, key=lambda m: abs(m.kv - kv), default=None)
        if best is not None and abs(best.kv - kv) <= self.USER_MASTER_KV_TOLERANCE:
            return best
        return master_for(material_id, kv, geometry)

    def _bank(self, materials, kv, g):
        """(bank, face0, full, offs, scales, masters): the cube maps of ``materials`` stacked."""
        masters = [self.master(int(m), kv, g) for m in materials]
        n = max((m.cube.shape[1] for m in masters), default=3)
        planes, face0, full = [], [], []
        for m in masters:
            if m.cube.shape[1] != n:
                raise ValueError(
                    "master patterns of one scan must share a resolution (set "
                    "EbsdGeometry.master_resolution_deg to the user master's)"
                )
            face0.append(len(planes))
            full.append(m.cube.shape[0] == 6)
            planes.extend(m.cube)
        bank = np.stack(planes) if planes else np.zeros((3, 3, 3), np.uint8)
        return (
            bank,
            np.array(face0 or [0], np.int32),
            np.array(full or [False], np.bool_),
            np.array([m.offset for m in masters] or [0.0]),
            np.array([m.scale for m in masters] or [0.0]),
            masters,
        )

    # ------------------------------------------------------------------ tables
    def tables(self, specimen, optics, field_map=None, grains=None) -> EbsdTables:
        key = _key(optics)
        with self._lock:
            t = self._tables.get(key)
            if t is not None:
                self._tables.move_to_end(key)
                return t
        t = self._build(specimen, optics, field_map, grains)
        with self._lock:
            self._tables[key] = t
            while len(self._tables) > 2:
                self._tables.popitem(last=False)
        return t

    def _build(self, specimen, optics, field_map, grains) -> EbsdTables:
        g = ebsd_geometry(optics)
        view = optics.view
        ny, nx = view.shape
        X, Y = view.world_grid()
        x, y = np.ascontiguousarray(X, np.float64), np.ascontiguousarray(Y, np.float64)
        if callable(getattr(specimen, "sample", None)):
            s = specimen.sample(x.ravel(), y.ravel(), blend=True)
            keys = np.concatenate([s["hash"], s["hash2"]])
            phases = np.concatenate([s["phase"], s["phase2"]])
            uk, first, inv = np.unique(keys, return_index=True, return_inverse=True)
            mats = specimen.orientations(uk, phases[first])
            n = x.size
            o1, o2 = inv[:n].astype(np.int32), inv[n:].astype(np.int32)
            mix = np.zeros(n, np.float32)
            if g.boundary_blend:
                m1 = s["material"]
                for mid in np.unique(m1):
                    sel = m1 == mid
                    mm = material(int(mid))
                    rx, ry = footprint_um(
                        mm.z_eff, mm.density_g_cm3, optics.ht_kv, g.sample_tilt_deg, g.interaction_scale
                    )
                    mix[sel] = boundary_mix(s["boundary_um"][sel], s["normal"][sel], rx, ry)
            mix[o1 == o2] = 0.0
            F = s.get("F")
            gnd = s.get("gnd")
            return EbsdTables(
                x,
                y,
                s["material"].reshape(ny, nx),
                s["material2"].reshape(ny, nx),
                s["grain_id"].reshape(ny, nx),
                s["grain_id2"].reshape(ny, nx),
                o1.reshape(ny, nx),
                o2.reshape(ny, nx),
                mats,
                mix.reshape(ny, nx),
                s["quality"].reshape(ny, nx),
                None if F is None else np.asarray(F, np.float32).reshape(ny, nx, 3, 3),
                None if gnd is None else np.asarray(gnd, np.float32).reshape(ny, nx),
            )
        # a TEM specimen: the field map's phase and grain-table orientation
        fm = field_map
        gid = fm.grain_id.astype(np.int64)
        mat = fm.material_id.copy()
        crystalline = np.array([material(int(m)).crystal is not None for m in range(int(mat.max()) + 1)])
        ok = (gid >= 0) & (gid // GRAINS_PER_MATERIAL == mat) & crystalline[mat]
        if grains is not None:
            ok &= gid < len(grains)
        uniq, inv = np.unique(gid[ok], return_inverse=True)
        mats = grains.matrices[uniq] if (grains is not None and len(uniq)) else np.zeros((0, 3, 3))
        orient = np.full(gid.shape, -1, np.int32)
        orient[ok] = inv
        gid32 = np.where(ok, gid, -1).astype(np.int32)
        return EbsdTables(
            x,
            y,
            mat.astype(np.uint8),
            mat.astype(np.uint8),
            gid32,
            gid32.copy(),
            orient,
            orient.copy(),
            mats,
            np.zeros((ny, nx), np.float32),
            np.ones((ny, nx), np.float32),
        )

    # ------------------------------------------------------------------ plans
    def plan(self, specimen, optics, field_map=None, grains=None) -> EbsdPlan:
        g = ebsd_geometry(optics)
        key = (
            _key(optics),
            g,
            round(optics.ht_kv, 4),
            optics.extras["sensor_shape"],
            optics.extras["sensor_pixel_um"],
            float(optics.pattern_e_per_s),
            tuple((k, tuple(id(m) for m in v)) for k, v in sorted(self.user_masters.items())),
        )
        with self._lock:
            p = self._plans.get(key)
            if p is not None:
                self._plans.move_to_end(key)
                return p
        tab = self.tables(specimen, optics, field_map, grains)
        p = self._make_plan(tab, optics, g)
        with self._lock:
            self._plans[key] = p
            while len(self._plans) > 2:
                self._plans.popitem(last=False)
        return p

    def _make_plan(self, tab: EbsdTables, optics, g: EbsdGeometry) -> EbsdPlan:
        from ..specimen.materials import MATERIALS

        o1, o2 = tab.orient.ravel(), tab.orient2.ravel()
        m1, m2 = tab.material.ravel(), tab.material2.ravel()
        crystal = np.unique(np.concatenate([m1[o1 >= 0], m2[o2 >= 0]]))
        bank, face0, full, offs, scales, masters = self._bank(crystal, optics.ht_kv, g)
        lut = np.full(256, -1, np.int32)
        lut[crystal] = np.arange(len(crystal), dtype=np.int32)
        ph1 = np.where(o1 >= 0, lut[m1], -1).astype(np.int32)
        ph2 = np.where(o2 >= 0, lut[m2], -1).astype(np.int32)
        eta = np.array([backscatter_yield(mm.z_eff, g.sample_tilt_deg) for mm in MATERIALS])
        eta[0] = 0.0  # vacuum
        es = (float(optics.pattern_e_per_s) * eta[m1]).astype(F32)
        contrast = (float(g.band_contrast) * tab.quality.ravel()).astype(F32)
        cx, cy = optics.view.center_um
        has_f = tab.F is not None
        finv = np.linalg.inv(tab.F.reshape(-1, 3, 3).astype(np.float64)).astype(F32) if has_f else _NO_F
        lobe, kappa = self._background(optics, g)
        return EbsdPlan(
            bank,
            face0,
            full,
            offs,
            scales,
            ph1,
            ph2,
            o1.astype(np.int32),
            o2.astype(np.int32),
            tab.mix.ravel().astype(F32),
            contrast,
            es,
            np.ascontiguousarray(finv),
            has_f,
            (tab.x_um - cx).ravel().astype(F32),
            (tab.y_um - cy).ravel().astype(F32),
            np.ascontiguousarray(tab.matrices if len(tab.matrices) else np.zeros((1, 3, 3))),
            pc_linear(g, optics.extras["sensor_shape"], optics.extras["sensor_pixel_um"]),
            np.ascontiguousarray(g.sample_to_detector().T),
            g.surface_normal_d(),
            g.forward_direction_d(),
            lobe,
            kappa,
            masters,
        )

    @staticmethod
    def _background(optics, g: EbsdGeometry) -> tuple[float, float]:
        if not g.background:  # isotropic: e_s / 4 pi per steradian
            return 1.0, 0.0
        return background_parameters(optics.ht_kv, g.sample_tilt_deg)

    # ------------------------------------------------------------------ geometry per point
    def pattern_centre(self, optics, tab: EbsdTables, ix: int, iy: int) -> np.ndarray:
        g = ebsd_geometry(optics)
        cx, cy = optics.view.center_um
        return g.pattern_centre(
            tab.x_um[iy, ix] - cx,
            tab.y_um[iy, ix] - cy,
            optics.extras["sensor_shape"],
            optics.extras["sensor_pixel_um"],
        )

    def pattern_centres(self, optics, tab: EbsdTables) -> np.ndarray:
        g = ebsd_geometry(optics)
        cx, cy = optics.view.center_um
        return g.pattern_centre(
            tab.x_um - cx, tab.y_um - cy, optics.extras["sensor_shape"], optics.extras["sensor_pixel_um"]
        )

    @staticmethod
    def grid(optics) -> tuple[int, int, int, int, int, int]:
        """(h, w, ox, oy, bx, by): the binned detector grid and its sensor offset."""
        bx, by = optics.extras.get("hw_binning", (1, 1))
        oh, ow = optics.output_shape
        ox, oy = optics.extras.get("roi_origin", (0, 0))
        return max(1, oh // by), max(1, ow // bx), int(ox), int(oy), int(bx), int(by)

    def _run(self, plan: EbsdPlan, optics, ks, out, raw=False, lam=None, p0=None, t_nb=0.0) -> float:
        g = ebsd_geometry(optics)
        h, w, ox, oy, bx, by = self.grid(optics)
        want = lam is not None
        dummy = self._buffer("dummy", (1, 1, 1))
        rows = kernels.render(
            plan.bank,
            plan.face0,
            plan.full,
            plan.offs,
            plan.scales,
            plan.matrices,
            plan.o1,
            plan.o2,
            plan.ph1,
            plan.ph2,
            plan.mix,
            plan.contrast,
            plan.es,
            plan.finv,
            plan.has_f,
            plan.dx,
            plan.dy,
            plan.pc_lin,
            plan.s2dT,
            np.asarray(ks, np.int64),
            float(ox),
            float(oy),
            float(bx),
            float(by),
            int(max(1, g.supersample)),
            bool(raw),
            plan.normal,
            plan.fwd,
            float(plan.lobe),
            float(plan.kappa),
            float(t_nb),
            out,
            lam if want else dummy,
            p0 if want else dummy,
            want,
        )
        self.patterns_rendered += len(ks)
        self.last_rows = rows  # per-row dose of the last call (row totals for a sparse detector path)
        return float(rows.sum())

    def _buffer(self, name, shape):
        b = self._buffers.get(name)
        if b is None or b.shape != tuple(shape):
            b = self._buffers[name] = np.empty(shape, np.float32)
        return b

    # ------------------------------------------------------------------ one pattern
    def pattern(
        self,
        optics,
        matrix,
        pc,
        *,
        material_id: int,
        matrix2=None,
        material2: Optional[int] = None,
        mix: float = 0.0,
        quality: float = 1.0,
        raw: bool = False,
        F=None,
        out=None,
    ) -> np.ndarray:
        """One pattern on the optics' binned detector grid for a grain with crystal->sample
        ``matrix`` at Bruker pattern centre ``pc`` (and deformation gradient ``F``, sample
        frame). ``raw``: the projected (standardised) master values only. Otherwise electrons
        per sensor pixel per second."""
        g = ebsd_geometry(optics)
        h, w = self.grid(optics)[:2]
        ids = [int(material_id)] if material_id and matrix is not None else []
        two = bool(matrix2 is not None and material2 and mix > 0 and ids)
        if two and int(material2) not in ids:
            ids.append(int(material2))
        bank, face0, full, offs, scales, masters = self._bank(ids, optics.ht_kv, g)
        mats = (
            np.stack([np.asarray(matrix, np.float64)] + ([np.asarray(matrix2, np.float64)] if two else []))
            if ids
            else np.zeros((1, 3, 3))
        )
        lobe, kappa = self._background(optics, g)
        mid = int(material_id) if material_id else 0
        es = float(optics.pattern_e_per_s) * (
            backscatter_yield(material(mid).z_eff, g.sample_tilt_deg) if mid else 0.0
        )
        plan = EbsdPlan(
            bank,
            face0,
            full,
            offs,
            scales,
            np.array([0 if ids else -1], np.int32),
            np.array([ids.index(int(material2)) if two else -1], np.int32),
            np.array([0 if ids else -1], np.int32),
            np.array([1 if two else -1], np.int32),
            np.array([float(mix) if two else 0.0], F32),
            np.array([float(g.band_contrast) * float(quality)], F32),
            np.array([es], F32),
            np.linalg.inv(np.asarray(F, np.float64))[None].astype(F32) if F is not None else _NO_F,
            F is not None,
            np.zeros(1, F32),
            np.zeros(1, F32),
            mats,
            pc_linear(g, optics.extras["sensor_shape"], optics.extras["sensor_pixel_um"], pc),
            np.ascontiguousarray(g.sample_to_detector().T),
            g.surface_normal_d(),
            g.forward_direction_d(),
            lobe,
            kappa,
            masters,
        )
        buf = np.empty((1, h, w), np.float32) if out is None else out.reshape(1, h, w)
        self._run(plan, optics, [0], buf, raw=raw)
        return buf[0]

    # ------------------------------------------------------------------ scan points
    def render(self, specimen, optics, ix: int, iy: int, field_map=None, grains=None) -> np.ndarray:
        """The pattern at scan point (ix, iy): a new float32 (h, w) array on the binned grid,
        electrons per sensor pixel per second."""
        plan = self.plan(specimen, optics, field_map, grains)
        h, w = self.grid(optics)[:2]
        out = np.empty((1, h, w), np.float32)
        self._run(plan, optics, [iy * optics.view.shape[1] + ix], out)
        return out[0]

    def render_frame(self, specimen, optics, ix: int, iy: int, t_nb: float, field_map=None, grains=None):
        """The pattern at (ix, iy) for a camera frame, into reused buffers: ``(flux, means)``
        with ``means = (lam, p0, dose, t_nb, row_totals)`` the detector's Poisson means for an exposure of
        ``t_nb`` (frame time x sensor pixels per grid pixel): what ``Detector.expose(...,
        means=)`` starts from. The buffers are overwritten by the next call."""
        plan = self.plan(specimen, optics, field_map, grains)
        h, w = self.grid(optics)[:2]
        out = self._buffer("flux", (1, h, w))
        lam = self._buffer("lam", (1, h, w))
        p0 = self._buffer("p0", (1, h, w))
        dose = self._run(plan, optics, [iy * optics.view.shape[1] + ix], out, lam=lam, p0=p0, t_nb=t_nb)
        # row_totals: each row's mean electrons, what a sparse (electron-by-electron) detector path
        # builds its per-row alias tables from, made in the same pass
        return out[0], (lam[0], p0[0], dose, float(t_nb), self.last_rows)

    def raw_pattern(self, specimen, optics, ix: int, iy: int) -> np.ndarray:
        """The projected master values at scan point (ix, iy) (no background, noiseless)."""
        plan = self.plan(specimen, optics)
        h, w = self.grid(optics)[:2]
        out = np.empty((1, h, w), np.float32)
        self._run(plan, optics, [iy * optics.view.shape[1] + ix], out, raw=True)
        return out[0]

    # ------------------------------------------------------------------ stacks
    def patterns(
        self,
        specimen,
        optics,
        field_map=None,
        grains=None,
        *,
        out=None,
        dtype=np.float32,
        per_binned_pixel: bool = True,
        block: int = 64,
    ) -> np.ndarray:
        """Every scan point's pattern: (ny, nx, h, w). ``per_binned_pixel``: electrons per binned
        pixel per second (the 4D-STEM datacube's unit); else per sensor pixel. ``out`` may be a
        preallocated (e.g. memory-mapped) array of that shape. Rendered ``block`` points per
        threaded kernel call."""
        plan = self.plan(specimen, optics, field_map, grains)
        ny, nx = optics.view.shape
        h, w, _, _, bx, by = self.grid(optics)
        if out is None:
            out = np.empty((ny, nx, h, w), dtype)
        flat = out.reshape(ny * nx, h, w)
        k = np.float32(bx * by if per_binned_pixel else 1)
        n = ny * nx
        buf = np.empty((min(block, n), h, w), np.float32)
        for b0 in range(0, n, block):
            ks = np.arange(b0, min(b0 + block, n))
            self._run(plan, optics, ks, buf[: len(ks)])
            if k != 1:
                buf[: len(ks)] *= k
            flat[b0 : b0 + len(ks)] = buf[: len(ks)]
        return out

    # ------------------------------------------------------------------ ground truth
    def ground_truth(self, specimen, optics, field_map=None, grains=None) -> dict:
        """Per scan point: ``material_id`` (= phase id; 0 none), ``grain_id`` (-1 none),
        ``quaternion`` (orix rotation of the local lattice, *sample* frame, a >= 0: the grain's
        orientation, times the lattice rotation of a strained specimen; identity where there is
        no crystal), ``pattern_centre`` (Bruker pcx, pcy, pcz), ``band_contrast`` (the dominant
        grain's band modulation, ``band_contrast * quality * (1 - mix)``), ``quality`` (surface
        quality x (1 - mix)), ``mix`` / ``grain_id2`` / ``material_id2`` (the neighbour mixed in
        near a boundary), ``x_um`` / ``y_um`` (beam on the sample).

        A strained specimen adds ``deformation_gradient`` F (3x3, sample frame, maps reference
        lattice vectors to deformed ones: ``u = F u0``), its polar parts ``strain`` (``U - I``,
        symmetric: the Biot strain, the small strain to first order) and ``rotation`` (rotation
        vector of R, radians, sample frame; ``F = R U``), and ``gnd`` (a GND density proxy:
        the norm of the lattice-curvature (Nye) tensor / b, 1/m^2)."""
        from ..crystal import rotation_from_crystal_to_lab

        g = ebsd_geometry(optics)
        tab = self.tables(specimen, optics, field_map, grains)
        ny, nx = tab.material.shape
        q = np.zeros((ny, nx, 4))
        q[..., 0] = 1.0
        ok = tab.orient >= 0
        extra = {}
        if tab.F is not None:
            from scipy.spatial.transform import Rotation as _R

            F = tab.F.reshape(-1, 3, 3).astype(np.float64)
            u, s, vt = np.linalg.svd(F)
            R = u @ vt
            U = np.einsum("nji,nj,njk->nik", vt, s, vt)
            extra = {
                "deformation_gradient": F.reshape(ny, nx, 3, 3),
                "strain": (U - np.eye(3)).reshape(ny, nx, 3, 3),
                "rotation": _R.from_matrix(R).as_rotvec().reshape(ny, nx, 3),
                "gnd": tab.gnd.copy() if tab.gnd is not None else np.zeros((ny, nx), np.float32),
            }
            if ok.any():
                okf = ok.ravel()
                local = np.einsum("nij,njk->nik", R[okf], tab.matrices[tab.orient.ravel()[okf]])
                q.reshape(-1, 4)[okf] = rotation_from_crystal_to_lab(local)
        elif ok.any():
            quats = rotation_from_crystal_to_lab(tab.matrices)
            q[ok] = quats[tab.orient[ok]]
        dominant = (1.0 - tab.mix).astype(np.float32)
        quality = np.where(ok, tab.quality * dominant, 0.0).astype(np.float32)
        return {
            **extra,
            "material_id": np.where(ok, tab.material, 0).astype(np.uint8),
            "grain_id": np.where(ok, tab.grain_id, -1).astype(np.int32),
            "quaternion": q,
            "pattern_centre": self.pattern_centres(optics, tab),
            "band_contrast": (quality * np.float32(g.band_contrast)).astype(np.float32),
            "quality": quality,
            "mix": tab.mix.copy(),
            "grain_id2": tab.grain_id2.copy(),
            "material_id2": tab.material2.copy(),
            "x_um": tab.x_um.copy(),
            "y_um": tab.y_um.copy(),
            "view": optics.view,
            "pc_convention": "bruker",
            "sample_tilt_deg": g.sample_tilt_deg,
            "detector_tilt_deg": g.detector_tilt_deg,
        }
