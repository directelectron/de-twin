"""DigitalTwin: column + specimen + holder + detector on one clock."""

from __future__ import annotations

import dataclasses

import math
import threading
from typing import Iterator, Optional, Union

import numpy as np

from .clock import Clock
from .state import (
    AcquisitionRequest,
    ExposureMode,
    FrameMeta,
    HolderState,
    MicroscopeState,
    Projection,
    RenderMode,
    TemStem,
)


class DigitalTwin:
    """A complete simulated instrument.

    Parameters
    ----------
    specimen
        A preset/pattern name (see ``de_twin.specimen.PRESETS``/``PATTERNS``, and the bulk EBSD
        specimens ``de_twin.specimen.BULK_PRESETS``), a ``SpecimenConfig``, a
        ``BulkSurfaceConfig`` or a ready ``Specimen`` / ``BulkSurface``.
    camera
        A camera model name (``de_twin.detector.CAMERAS``) or ``CameraModel``.
    column
        A ``Column`` (default: a new simulated column on this twin's clock) or a
        ``MirrorColumn`` following real software.
    corrector
        ``None`` (uncorrected), ``"probe"``, ``"image"`` or ``"both"``: give the twin's own
        column a CEOS-like aberration corrector (see ``de_twin.column.corrector``).
    holder
        A holder object (duck type of autopilot's heater) or a spec string
        (``"none"``, ``"sim-heating"``, ``"impulse"``).
    clock
        Shared clock; default real time. Use ``ManualClock`` for deterministic
        offline runs or ``Clock(time_scale=...)`` to fast-forward.
    flood_for_gain
        Gain/vacuum exposure modes render a uniform flood (as if the operator
        moved to an empty area), which is what gain references expect.
    """

    def __init__(
        self,
        specimen: Union[str, object] = "Dense Au on holey C",
        camera: Union[str, object] = "DE16",
        *,
        column=None,
        corrector: Optional[str] = None,
        holder: Union[str, object, None] = None,
        seed: int = 0,
        calibration=None,
        optics_config=None,
        render_config=None,
        clock: Optional[Clock] = None,
        flood_for_gain: bool = True,
        detector_overrides: Optional[dict] = None,
        start_on_specimen: bool = True,
    ):
        from .column import Column
        from .detector import Detector, camera as camera_by_name
        from .holder import connect_holder
        from .optics import Calibration, OpticsConfig
        from .render import Renderer

        self.lock = threading.RLock()
        self._frame_serial = 0  # frames exposed so far; seeds the detector noise
        self.clock = clock or Clock()
        self.seed = int(seed)
        self.flood_for_gain = flood_for_gain

        self.specimen = _make_specimen(specimen)

        model = camera_by_name(camera) if isinstance(camera, str) else camera
        self.detector = Detector(model, seed=self.seed, **(detector_overrides or {}))

        if column is not None and corrector not in (None, "none"):
            raise ValueError("pass corrector= only when the twin builds its own column")
        self.column = column if column is not None else Column(
            clock=self.clock, corrector=corrector, seed=self.seed)
        if holder is None or isinstance(holder, str):
            holder = connect_holder(holder or "none", clock=self.clock)
        self.holder = holder

        self.calibration = calibration or Calibration.default()
        self.optics_config = optics_config or OpticsConfig()
        if self.optics_config.realism is not None and column is None:
            self.column.backlash_um = float(self.optics_config.realism.backlash_um)
        self.renderer = Renderer(self.specimen, render_config)
        #: material id -> user EBSD master patterns (``set_master_pattern``)
        self._user_masters: dict = {}
        self.renderer.ebsd_user_masters = self._user_masters
        # Only a column the twin built is its to move: a supplied one may be a mirror of
        # real hardware, whose stage and beam are the operator's.
        self._own_column = column is None
        if start_on_specimen and self._own_column:
            self.start_on_specimen()

    # ------------------------------------------------------------------ state
    @property
    def camera_model(self):
        return self.detector.model

    def microscope_state(self) -> MicroscopeState:
        return self.column.state()

    def holder_state(self, t: Optional[float] = None) -> HolderState:
        t = self.clock.now() if t is None else t
        state = getattr(self.holder, "state", None)
        return state(t) if callable(state) else HolderState(t_s=t)

    def start_on_specimen(self) -> None:
        """Point the column at the specimen, as an operator would on loading it.

        The stage goes to `Specimen.home_um` (the origin is often a grid bar, a chip frame or
        bare film), the beam to the preset's `start_dose_e_per_px_s` (by solving the
        Intensity — the default spread beam gives ~2 e- per live frame, which reads as pure
        noise), and the defocus to its `start_defocus_um`. Everything stays the operator's
        to change afterwards.
        """
        from .optics.derive import stage_for_view_center

        opts = getattr(self.specimen, "options", None)
        if opts is None or not hasattr(self.specimen, "home_um"):
            return  # a synthetic test specimen has no holder to find
        # `home_um` is a point on the SPECIMEN; the stage carries the specimen, so the stage
        # position that puts it on axis is not the point itself.
        x, y = stage_for_view_center(self.specimen.home_um(), self.column.state(), self.optics_config)
        self.column.move_stage(x=x, y=y)
        self.column.set_defocus_um(self._start_defocus_um(opts))
        target = float(opts.start_dose_e_per_px_s) or self.safe_dose_e_per_px_s()
        if target > 0:
            self.column.set("Intensity", self._intensity_for_dose(target))

    def _start_defocus_um(self, opts) -> float:
        """The preset's start defocus; for unstained proteins in ice with none set,
        1.5 um underfocus — a phase object shows almost nothing in focus, and that is
        how cryo is imaged."""
        df = float(opts.start_defocus_um)
        if df == 0.0 and getattr(self.specimen.config, "preparation", "") == "proteins"                 and float(getattr(opts, "negative_stain_fraction", 0.0)) < 0.5:
            return -1.5
        return df

    def safe_dose_e_per_px_s(self, fps: float = 40.0) -> float:
        """A dose rate the detector takes without saturating a frame at *fps*.

        A DE integrating pixel holds only a few electrons per frame (~206 ADU per e- at
        300 kV against ~2100 ADU of range), so its frames are meant to be sparse and an
        image is their sum over the exposure: ~35 % of that per frame here. A counting
        camera wants ~0.1 e- per pixel per sensor cycle, or coincidence loss sets in.
        """
        m = self.detector.model
        if m.hardware_counting:
            return 0.1 * float(m.hw_quanta) * fps
        ht = float(self.column.state().ht_kv or m.reference_ht_kv)
        per_e = float(m.adu_per_electron_at(ht)) if callable(getattr(m, "adu_per_electron_at", None))             else float(m.adu_per_electron)
        per_frame = max(0.0, (float(m.saturation_adu) - float(m.dark_offset_adu)) / max(per_e, 1e-9))
        return 0.35 * per_frame * fps

    def _intensity_for_dose(self, target: float) -> float:
        """The Intensity (C2) at which the beam gives *target* e-/px/s at the current
        magnification: dose falls monotonically as Intensity spreads the beam, so bisect
        on the optics (no rendering)."""
        req = self.request()
        state = self.column.state()
        lo, hi = 0.02, 0.98
        r = self.optics_config.realism
        if r is not None:  # past the C2 crossover, where spreading is monotonic
            lo = max(lo, r.crossover(int(state.spot_size), int(state.probe_mode)) + 1e-3)

        def dose(v: float) -> float:
            return float(self.optics(req, dataclasses.replace(state, intensity=v)).dose_e_per_px_s)

        if dose(lo) <= target:
            return lo
        if dose(hi) >= target:
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if dose(mid) > target:
                lo = mid
            else:
                hi = mid
        return round(0.5 * (lo + hi), 4)

    def set_specimen(self, specimen, start: bool = True) -> None:
        """Swap the specimen (e.g. a new preset) keeping column/holder/detector. With
        *start*, the column is pointed at the new specimen (`start_on_specimen`)."""
        from .render import Renderer

        specimen = _make_specimen(specimen)
        with self.lock:
            self.specimen = specimen
            self.renderer = Renderer(specimen, getattr(self.renderer, "config", None))
            self.renderer.ebsd_user_masters = self._user_masters
        if start and getattr(self, "_own_column", False):
            self.start_on_specimen()

    # ---------------------------------------------------------------- optics
    def optics(self, request: AcquisitionRequest, state: Optional[MicroscopeState] = None):
        from .optics import derive_optics

        state = state or self.column.state()
        return derive_optics(state, request, self.detector.model, self.calibration, self.optics_config)

    def beam_on_detector(self, request: AcquisitionRequest, state: MicroscopeState) -> bool:
        if not request.beam_reaches_detector:
            return False
        if state.beam_blanked or not state.column_valves_open or not state.ht_on:
            return False
        if state.screen_position != 0:  # screen down intercepts the beam
            return False
        return True

    def scan_point(self, request: AcquisitionRequest, frame_index: int) -> Optional[tuple[int, int]]:
        scan = request.scan
        if not scan.enabled:
            return None
        if scan.park:
            return tuple(scan.park_position)
        point = frame_index // max(1, scan.frames_per_point)
        if scan.points:
            return tuple(scan.points[point % len(scan.points)])
        nx, ny = scan.size
        point %= max(1, nx * ny)
        return (point % nx, point // nx)

    def _flood(self, optics, shape) -> np.ndarray:
        return np.full(shape, np.float32(optics.dose_e_per_px_s), np.float32)

    def flux(
        self,
        request: AcquisitionRequest,
        frame_index: int = 0,
        scan_point: Optional[tuple[int, int]] = None,
        *,
        state: Optional[MicroscopeState] = None,
        optics=None,
        time_s: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Noiseless electrons/pixel/second on the ROI, or None if no beam."""
        img, _ = self._flux(request, frame_index, scan_point, state=state, optics=optics, time_s=time_s)
        return img

    def _flux(self, request, frame_index=0, scan_point=None, *, state=None, optics=None, time_s=None,
              frame_time_s: Optional[float] = None):
        """``(flux, means)``: ``means`` (the detector's Poisson means, see ``Detector.expose``) when
        the renderer made them on the way (an EBSD frame with ``frame_time_s``), else None. That
        flux is a reused buffer, valid until the next frame."""
        with self.lock:
            t = self.clock.now() if time_s is None else time_s
            state = state or self.column.state()
            if not self.beam_on_detector(request, state):
                return None, None
            optics = optics or self.optics(request, state)
            roi = self.detector.roi(request)
            shape = (roi.h, roi.w)
            if self.flood_for_gain and request.exposure_mode in (ExposureMode.GAIN, ExposureMode.VACUUM):
                return self._flood(optics, shape), None
            self.specimen.update(t, self.holder_state(t))
            if scan_point is None:
                scan_point = self.scan_point(request, frame_index)
            if (optics.render_mode == RenderMode.EBSD and frame_time_s is not None and not optics.beam_blanked
                    and not self.detector.super_resolution(request)):
                bx, by = (max(1, int(v)) for v in (request.hw_binning or (1, 1)))
                nb = bx * by if (bx, by) != (1, 1) else 1
                return self.renderer.render_ebsd_frame(optics, frame_index, scan_point, t,
                                                       float(frame_time_s) * nb)
            img = self.renderer.render(optics, frame_index=frame_index, scan_point=scan_point, time_s=t)
            if self.renderer.config.prefetch and optics.render_mode == RenderMode.TEM_IMAGING:
                self._prefetch_next(request, state, t)
            if optics.render_mode == RenderMode.EBSD and img.shape == self.detector._grid_shape(request):
                return img, None  # rendered on the binned grid already (per sensor pixel: expose takes it)
            if img.shape != shape:
                img = _fit(img, shape)
            return img, None

    # -------------------------------------------------------------- prefetch
    _VIEW_FIELDS = ("magnification", "defocus_um")
    _STAGE_FIELDS = ("x_um", "y_um", "alpha_deg", "beta_deg")

    def _view_of(self, state: MicroscopeState) -> tuple:
        return (tuple(getattr(state, f) for f in self._VIEW_FIELDS)
                + tuple(getattr(state.stage, f) for f in self._STAGE_FIELDS))

    def _prefetch_next(self, request: AcquisitionRequest, state: MicroscopeState, t: float) -> None:
        """A new view: render, in the background, the one repeating the last change gives."""
        cur = self._view_of(state)
        last = getattr(self, "_pf_last", None)
        if last is not None and last[0] == cur:
            return
        self._pf_last = (cur, state.copy())
        if last is None:
            return
        prev = last[1]
        nxt = state.copy()
        changed = False
        if state.magnification != prev.magnification:
            ladder = list(self.column.mag_ladder() or ())
            if state.magnification in ladder and prev.magnification in ladder:
                i = 2 * ladder.index(state.magnification) - ladder.index(prev.magnification)
                if 0 <= i < len(ladder):
                    nxt.magnification = ladder[i]
                    changed = True
        for f in ("defocus_um",):
            d = getattr(state, f) - getattr(prev, f)
            if d:
                setattr(nxt, f, getattr(state, f) + d)
                changed = True
        for f in self._STAGE_FIELDS:
            d = getattr(state.stage, f) - getattr(prev.stage, f)
            if d:
                setattr(nxt.stage, f, getattr(state.stage, f) + d)
                changed = True
        if not changed:
            return
        optics = self.optics(request, nxt)
        pending = getattr(self, "_pf_future", None)
        if pending is not None:
            pending.cancel()  # not started yet: the newer prediction replaces it
        if getattr(self, "_pf_pool", None) is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pf_pool = ThreadPoolExecutor(1, thread_name_prefix="de-twin-prefetch")
        self._pf_future = self._pf_pool.submit(self.renderer.prefetch, optics, t)

    # ---------------------------------------------------------------- frames
    def frames(
        self,
        request: AcquisitionRequest,
        *,
        pace: bool = False,
        stop: Optional[threading.Event] = None,
    ) -> Iterator[tuple[np.ndarray, FrameMeta]]:
        """Raw detector frames for ``request``.

        TEM imaging re-reads the column at every buffer boundary (live view
        follows stage moves); diffraction and STEM freeze the optics for the
        whole acquisition, like DE-Server's VirtualSpecimen.
        ``total_frames == 0`` means "until ``stop`` is set".
        """
        frame_time = max(float(request.frame_time_s), 1e-6)
        fpb = max(1, int(request.frames_per_buffer))
        state = self.column.state()
        optics = self.optics(request, state)
        frozen = optics.render_mode != RenderMode.TEM_IMAGING
        source = self.ebsd_frame_source(request, state=state, optics=optics)
        if source is not None:
            yield from self._fused_frames(request, source, state, optics, pace=pace, stop=stop)
            return
        t_start = self.clock.now()
        i = 0
        while request.total_frames <= 0 or i < request.total_frames:
            if stop is not None and stop.is_set():
                return
            t = self.clock.now()
            if not frozen and i % fpb == 0 and i > 0:
                new_state = self.column.state()
                if new_state != state:
                    state = new_state
                    optics = self.optics(request, state)
            with self.lock:
                holder_state = self.holder_state(t)
                flux, means = self._flux(request, i, state=state, optics=optics, time_s=t,
                                         frame_time_s=frame_time)
                blanked = flux is None
                # Noise is seeded by a twin-wide frame counter, not the index within this
                # request: repeated acquisitions (live view) must not repeat their noise.
                # A request with its own seed is pinned instead: frame i by (seed, i).
                if request.seed is None:
                    serial, seed = self._frame_serial, None
                    self._frame_serial += 1
                else:
                    serial, seed = i, int(request.seed)
                raw, info = self.detector.expose(flux, frame_time, request, serial,
                                                 ht_kv=state.ht_kv, seed=seed, means=means)
            meta = FrameMeta(
                frame_index=i,
                time_s=t,
                exposure_s=frame_time,
                electrons_per_pixel=float(info.get("dose_e_per_px", 0.0)),
                render_mode=optics.render_mode,
                blanked=blanked,
                scan_point=self.scan_point(request, i),
                microscope=state,
                holder=holder_state,
            )
            if pace:
                self.clock.sleep_until(t_start + (i + 1) * frame_time)
            yield raw, meta
            i += 1

    def noise_serial(self, request: AcquisitionRequest, index: int) -> tuple[int, int]:
        """``(serial, seed)`` of frame *index* of *request*: the twin-wide frame count (taking the
        next one) and the detector's seed, or (index, request.seed) for a pinned acquisition."""
        if request.seed is not None:
            return int(index), int(request.seed)
        with self.lock:
            serial = self._frame_serial
            self._frame_serial += 1
        return serial, self.detector.seed

    def ebsd_frame_source(self, request: AcquisitionRequest, *, state: Optional[MicroscopeState] = None,
                          optics=None):
        """An :class:`~de_twin.ebsd.framesource.EbsdFrameSource` making *request*'s EBSD camera
        frames in one fused CPU pass each, at the column's current state (or *state*,
        *optics*); None when its frames are not that path (not an EBSD camera frame, the beam
        off the detector, a counting camera, supersampled binning...): :meth:`frames` then
        takes the general path. ``DE_TWIN_FUSED_EBSD=0`` turns it off."""
        import os

        from .ebsd import framesource

        if os.environ.get("DE_TWIN_FUSED_EBSD") == "0":
            return None
        with self.lock:
            state = state or self.column.state()
            optics = optics or self.optics(request, state)
            if optics.render_mode != RenderMode.EBSD or optics.beam_blanked or not framesource.usable(optics):
                return None
            if not self.beam_on_detector(request, state) or self.detector.super_resolution(request):
                return None
            if self.flood_for_gain and request.exposure_mode in (ExposureMode.GAIN, ExposureMode.VACUUM):
                return None
            exposure = self.detector.grid_exposure(max(float(request.frame_time_s), 1e-6), request,
                                                   ht_kv=state.ht_kv)
            if exposure is None:
                return None
            grid = self.renderer.ebsd.grid(optics)
            if tuple(grid[:2]) != tuple(exposure.shape):
                return None
            t = self.clock.now()
            self.specimen.update(t, self.holder_state(t))
            plan = self.renderer.ebsd_plan(optics, t)
        return framesource.EbsdFrameSource(plan, grid, exposure)

    #: Frames a worker makes per job: the interpreter is taken once a job (to start it and to
    #: hand it back), and a busy consumer holds it, so a job of several frames waits less.
    FRAMES_PER_JOB = 4

    def _fused_frames(self, request, source, state, optics, *, pace, stop):
        """:meth:`frames` through an `EbsdFrameSource`: frames are jobs for the worker threads
        (:mod:`de_twin.workers`), a few frames each, submitted ahead of the consumer into a
        ring of frame buffers. A buffer is used again only once the consumer has let go of
        every frame it held in it (a consumer that keeps frames gets new buffers)."""
        import sys
        from collections import deque

        from .workers import pool

        workers = pool()
        frame_time = max(float(request.frame_time_s), 1e-6)
        h, w = source.shape
        total = request.total_frames
        per_job = self.FRAMES_PER_JOB
        jobs_ahead = workers.threads + workers.threads // 2
        # points a running acquisition may still change (a steered list) are made one at a time
        if request.scan.points is not None and not isinstance(request.scan.points, (list, tuple, np.ndarray)):
            per_job, jobs_ahead = 1, 1
        # more buffers than jobs in flight: the one a job takes was filled a few jobs ago, not
        # the one whose frame the consumer still holds while it asks for the next
        ring = [None] * (jobs_ahead + 2)
        pending: deque = deque()  # (first frame, frames, slot, batch)
        t_start = self.clock.now()
        i = nxt = 0
        n_jobs = 0
        try:
            while total <= 0 or i < total:
                if stop is not None and stop.is_set():
                    return
                while len(pending) < jobs_ahead and (total <= 0 or nxt < total):
                    count = per_job if total <= 0 else min(per_job, total - nxt)
                    slot = n_jobs % len(ring)
                    buf = ring[slot]
                    if buf is None or sys.getrefcount(buf) > 3:  # held elsewhere than ring, buf and the call
                        buf = ring[slot] = np.empty((per_job, h, w), np.uint16)
                    ks, serials, seed = [], [], 0
                    for j in range(nxt, nxt + count):
                        ks.append(self.renderer.ebsd_flat_point(optics, self.scan_point(request, j), j))
                        serial, seed = self.noise_serial(request, j)
                        serials.append(serial)
                    pending.append((nxt, count, slot, source.submit(workers, buf, ks, seed, serials)))
                    nxt += count
                    n_jobs += 1
                first, count, slot, batch = pending.popleft()
                (electrons, _), = batch.wait()
                chunk = ring[slot]
                for b in range(count):
                    if stop is not None and stop.is_set():
                        return
                    t = self.clock.now()
                    meta = FrameMeta(
                        frame_index=i,
                        time_s=t,
                        exposure_s=frame_time,
                        electrons_per_pixel=float(electrons[b]) / source.sensor_pixels,
                        render_mode=optics.render_mode,
                        blanked=False,
                        scan_point=self.scan_point(request, i),
                        microscope=state,
                        holder=self.holder_state(t),
                    )
                    if pace:
                        self.clock.sleep_until(t_start + (i + 1) * frame_time)
                    yield chunk[b], meta
                    i += 1
                del chunk
        finally:
            # let the jobs finish with their buffers; not at interpreter exit, when the worker
            # threads have stopped and a generator left open is closed by the collector
            if not sys.is_finalizing():
                for _, _, _, batch in pending:
                    batch.wait_done(10.0)
            self.renderer.ebsd.patterns_rendered += nxt

    def reset_serial(self) -> None:
        """Restart the running frame count that seeds the detector noise, so the next
        acquisition's noise is the same as a fresh twin's (see ``AcquisitionRequest.seed``
        to pin one acquisition without touching the count)."""
        with self.lock:
            self._frame_serial = 0

    def acquire(self, request: AcquisitionRequest, *, pace: bool = False) -> np.ndarray:
        """Sum of all frames of ``request`` (uint32/float64 like a DE-Server integrated image)."""
        total = None
        for raw, _ in self.frames(request, pace=pace):
            total = raw.astype(np.float64) if total is None else total + raw
        return total

    def raw_frame(self, frame_time_s: float = 0.025, **request_kw) -> np.ndarray:
        """One raw detector frame (offset, noise, defects and all)."""
        req = self.request(frame_time_s=frame_time_s, total_frames=1, **request_kw)
        raw, _ = next(self.frames(req))
        return raw

    def snap(self, exposure_s: float = 1.0, *, fps: float = 40.0, units: str = "electrons",
             **request_kw) -> np.ndarray:
        """An integrated, dark/gain-corrected image, the way DE-Server delivers one.

        ``exposure_s`` is split into frames at ``fps``, each corrected with
        references the twin measures on first use (see :attr:`processor`).
        ``units``: "electrons" | "adu" | "raw". ``seed=`` (a request field) pins the
        image's noise: the same seed gives the same image whatever came before.
        """
        from .processing import frames_for_exposure

        frame_time, n = frames_for_exposure(exposure_s, fps)
        req = self.request(frame_time_s=frame_time, total_frames=n, **request_kw)
        return self.processor.acquire(req, units=units)

    @property
    def processor(self):
        """Reference store + correction (DE-Server-like processing)."""
        if getattr(self, "_processor", None) is None:
            from .processing import Processor

            self._processor = Processor(self)
        return self._processor

    def virtual_image(self, request: AcquisitionRequest, inner_mrad: float, outer_mrad: float) -> np.ndarray:
        """STEM virtual detector image for the request's scan (noiseless)."""
        with self.lock:
            state = self.column.state()
            optics = self.optics(request, state)
            self.specimen.update(self.clock.now(), self.holder_state())
            return self.renderer.virtual_image(optics, inner_mrad, outer_mrad)

    def datacube(self, request: AcquisitionRequest) -> np.ndarray:
        """Noiseless 4D-STEM data for the request's scan: float32 (scan ny, nx, Hd, Wd) at the
        detector's *binned* output shape, electrons per binned pixel per second (multiply by the
        dwell time for electrons per pattern). The same patterns :meth:`frames` exposes. In EBSD
        mode, the EBSD pattern stack (:meth:`ebsd_patterns`)."""
        with self.lock:
            state = self.column.state()
            optics = self.optics(request, state)
            t = self.clock.now()
            self.specimen.update(t, self.holder_state(t))
            return self.renderer.datacube(optics, time_s=t)

    # ------------------------------------------------------------------ EBSD
    @property
    def ebsd(self):
        """The EBSD geometry (:class:`~de_twin.ebsd.EbsdGeometry`) when the twin is an SEM with
        an EBSD detector, else None."""
        return getattr(self.optics_config, "ebsd", None)

    def configure_ebsd(self, geometry=None, *, beam_kv: Optional[float] = None, **fields):
        """Make the twin an SEM with this camera as its EBSD detector, and return the geometry.

        ``geometry``: an :class:`~de_twin.ebsd.EbsdGeometry` (default: the current one, or the
        defaults: 70 deg sample tilt, Bruker pattern centre (0.5, 0.3, 0.6), 0.1 nA); ``fields``
        change it (``sample_tilt_deg=70, pcz=0.55, ...``). ``beam_kv`` sets the column's high
        tension (EBSD is done at 5-30 kV); left out, a twin column outside 5-30 kV is set to
        20 kV when EBSD is enabled. ``configure_ebsd(enabled=False)`` returns to a TEM (the HT
        stays where it is).
        """
        from .ebsd import EbsdGeometry

        g = geometry if geometry is not None else (self.ebsd or EbsdGeometry())
        if fields:
            g = g.replace(**fields)
        with self.lock:
            self.optics_config.ebsd = g
            if (beam_kv is None and g.enabled and getattr(self, "_own_column", False)
                    and not 5.0 <= float(self.column.state().ht_kv) <= 30.0):
                beam_kv = 20.0
            if beam_kv is not None:
                if not getattr(self, "_own_column", False):
                    raise ValueError("beam_kv sets the twin's own column's HT; a supplied column's is "
                                     "the operator's")
                # Choosing the SEM's energy builds the instrument; the column's operator guard
                # on HT (allow_ht) is for remote control, so go round it here.
                self.column._set_local("HighTension", float(beam_kv) * 1000.0)
        return g

    def set_master_pattern(self, phase, source, *, energy_kv: Optional[float] = None, kv: Optional[float] = None,
                           projection: Optional[str] = None, hemisphere: Optional[str] = None, energies=None,
                           weights=None, centrosymmetric: Optional[bool] = None, validate: bool = True):
        """Use your own EBSD master pattern for ``phase`` instead of the kinematic one, and return
        it (a :class:`~de_twin.ebsd.MasterPattern`; ``nbytes``, ``symmetry_error``).

        ``phase``: a material id or name (``"Nickel"``). ``source``: an EMsoft ``EMEBSDmaster``
        HDF5 file, a kikuchipy ``EBSDMasterPattern`` saved to HDF5, or a NumPy array — (n, n),
        (2, n, n) north/south, (nE, n, n) with ``energies``, (2, nE, n, n) — with ``projection``
        ("lambert" | "stereographic") and ``hemisphere`` ("north" | "south" | "both").
        ``energy_kv`` picks (interpolates) an energy bin; by default the Monte-Carlo
        energy-weighted sum when the file has one, else the highest energy. ``kv``: the beam energy
        it is used at (default the selected / highest energy, else the column's HT; it is used
        within 0.5 kV of it). It is resampled and quantised to ``EbsdGeometry.master_resolution_deg``
        (a few MB) and validated against the phase's point group (see :mod:`de_twin.ebsd.io`).
        """
        from .ebsd import EbsdGeometry
        from .ebsd.io import load_master_pattern

        mid = _material_id(phase)
        g = self.ebsd or EbsdGeometry()
        if kv is None and energy_kv is None and energies is None and not isinstance(source, (str, bytes)) \
                and not hasattr(source, "__fspath__"):
            kv = float(self.column.state().ht_kv)
        mp = load_master_pattern(source, mid, resolution_deg=g.master_resolution_deg, energy_kv=energy_kv, kv=kv,
                                 projection=projection, hemisphere=hemisphere, energies=energies,
                                 weights=weights, centrosymmetric=centrosymmetric, validate=validate)
        with self.lock:
            lst = [m for m in self._user_masters.get(mid, []) if abs(m.kv - mp.kv) > 1e-6]
            self._user_masters[mid] = lst + [mp]
        return mp

    def clear_master_patterns(self, phase=None) -> None:
        """Back to kinematic master patterns (for ``phase``, or all)."""
        with self.lock:
            if phase is None:
                self._user_masters.clear()
            else:
                self._user_masters.pop(_material_id(phase), None)

    def master_patterns(self) -> list:
        """The user master patterns in use: dicts of phase, kv, source, bytes, hemispheres,
        symmetry error."""
        from .specimen.materials import material

        return [{"phase": material(mid).name, "material_id": mid, "kv": m.kv, "source": m.source,
                 "nbytes": m.nbytes, "hemispheres": 1 if m.south is None else 2,
                 "symmetry_error": getattr(m, "symmetry_error", float("nan"))}
                for mid, lst in sorted(self._user_masters.items()) for m in lst]

    def ebsd_patterns(self, request: AcquisitionRequest, *, out=None, dtype=np.float32) -> np.ndarray:
        """Noiseless EBSD patterns of the request's scan: (scan ny, nx, h, w) at the detector's
        *binned* output shape, electrons per binned pixel per second (x the dwell time for
        electrons per pattern): the same patterns :meth:`frames` exposes. ``out``: a
        preallocated (e.g. ``np.memmap``) array to fill instead of allocating
        ``ny * nx * h * w * itemsize`` bytes."""
        with self.lock:
            state = self.column.state()
            optics = self.optics(request, state)
            t = self.clock.now()
            self.specimen.update(t, self.holder_state(t))
            return self.renderer.ebsd_patterns(optics, time_s=t, out=out, dtype=dtype)

    def stem_model(self, request: AcquisitionRequest) -> str:
        """"coherent" or "kinematic": the STEM model the renderer uses for this request."""
        with self.lock:
            return self.renderer.stem_model(self.optics(request))

    def ground_truth_ptychography(self, request: AcquisitionRequest) -> dict:
        """Ground truth of the coherent 4D-STEM simulation of ``request`` (to score
        ptychography / tcBF):

        * ``object``: complex transmission t(r) over the scanned field at the simulation pixel
          ``dx_nm`` (lab frame: x = detector columns); ``view``: its world placement;
        * ``positions_px`` / ``positions_nm``: (N, 2) probe centres (row, col) in that array, in
          scan order (row-major), exactly as simulated;
        * ``probe`` (principal mode), ``probe_modes`` / ``mode_weights`` (partial coherence):
          complex (n, n) at ``dx_nm``, centred at (n/2, n/2), ``sum |P|^2 = 1``;
        * ``wavelength_nm``, ``convergence_mrad`` (alpha), ``aberrations`` (the full probe set,
          :class:`~de_twin.optics.aberrations.Aberrations`, incl. C1 = defocus and A1 = condenser
          stig) and ``C1_nm``, ``A1_nm``, ``B2_nm``, ``C3_nm``, ``C5_nm``; ``focal_spread_nm``,
          ``source_size_nm``, ``scan_step_nm``, ``scan_rotation_rad``;
        * ``detector_shape``, ``detector_recip_px_inv_nm``, ``detector_center_px`` (binned) and
          ``sampling`` (:class:`~de_twin.render.coherent.Sampling`).

        the test helper ``tests/reconstruction.object_on_grid`` resamples the object to a reconstruction's pixel.
        """
        with self.lock:
            state = self.column.state()
            optics = self.optics(request, state)
            t = self.clock.now()
            self.specimen.update(t, self.holder_state(t))
            return self.renderer.ground_truth_ptychography(optics, time_s=t)

    def calibration_truth(self, request: Optional[AcquisitionRequest] = None) -> dict:
        """What SerialEM's calibrations should find at the column's current state: the true
        pixel size against the nominal one, the image rotation on the camera, the image-shift
        matrix, the magnification's image offset, the stage's backlash error and how far the
        specimen is off eucentric height. For closed-loop tests: calibrate the twin, then
        compare. An ideal column (no ``OpticsConfig.realism``) reports the identities.

        ``image_rotation_deg`` is the view's world -> camera rotation: image features turn by
        minus it in (x right, y down) pixel coordinates."""
        request = request or self.request()
        state = self.column.state()
        o = self.optics(request, state)
        cfg = self.optics_config
        r = cfg.realism
        imaging = o.render_mode == RenderMode.TEM_IMAGING
        out = {
            "magnification": float(state.magnification),
            "mag_mode": str(state.mag_mode),
            "nominal_pixel_nm": float(o.extras.get("nominal_pixel_nm", o.specimen_pixel_nm)),
            "true_pixel_nm": float(o.specimen_pixel_nm),
            "image_rotation_deg": math.degrees(o.view.rotation_rad) if imaging else 0.0,
            "is_matrix_um_per_unit": [[1.0, 0.0], [0.0, 1.0]],
            "mag_offset_um": (0.0, 0.0),
            "stage_error_um": (float(state.stage_error_um.x), float(state.stage_error_um.y)),
            "backlash_um": float(self.column.backlash_um) if hasattr(self.column, "backlash_um") else 0.0,
            "eucentric_offset_um": float(state.stage.z_um + cfg.stage_offset_um[2] - cfg.eucentric_height_um),
            "flip": (bool(cfg.flip_x), bool(cfg.flip_y)),
        }
        out["bs_matrix_um_per_unit"] = [[1.0, 0.0], [0.0, 1.0]]
        out["crossover_intensity"] = None
        if r is not None:
            out["bs_matrix_um_per_unit"] = r.bs_matrix().tolist()
            out["crossover_intensity"] = r.crossover(int(state.spot_size), int(state.probe_mode))
        out["is_coma_mrad_per_is_um"] = [[0.0, 0.0], [0.0, 0.0]]
        out["is_astig_nm_per_is_um"] = (0.0, 0.0)
        if r is not None and imaging:
            t = r.truth(state.mag_mode, state.magnification)
            out["is_matrix_um_per_unit"] = t["is_matrix_um_per_unit"]
            out["mag_offset_um"] = tuple(t["mag_offset_um"])
            # per micrometre of image shift at the SPECIMEN (IS matrix applied)
            out["is_coma_mrad_per_is_um"] = [list(r.is_tilt_mrad(1.0, 0.0)), list(r.is_tilt_mrad(0.0, 1.0))]
            a = r.is_astig_nm(1.0, 0.0)
            out["is_astig_nm_per_is_um"] = (a.real, a.imag)
        return out

    def ground_truth(self, request: Optional[AcquisitionRequest] = None, *, as_arrays: bool = False):
        """Phase + effective orientation (grain x stage tilt) per STEM scan point, or per raster
        pixel of a TEM view, for scoring orientation maps (e.g. pyxem against
        ``CrystalLibrary.template_library``). An orix ``CrystalMap`` (phase id = material id,
        ``-1`` = not indexed: vacuum/amorphous; props ``grain_id``, ``thickness_nm``; x/y in nm),
        or the raw arrays with ``as_arrays=True``.

        In EBSD mode, per scan point: phase, grain id, the orientation in the *sample* frame
        (x_s scan x, y_s scan y, z_s into the sample; orix convention, see
        :mod:`de_twin.ebsd.geometry`), the Bruker pattern centre and a band-contrast proxy
        (CrystalMap props ``grain_id``, ``band_contrast``, ``quality``, ``mix``, ``pcx``,
        ``pcy``, ``pcz``; x/y in µm)."""
        with self.lock:
            state = self.column.state()
            optics = self.optics(request or self.request(), state)
            t = self.clock.now()
            self.specimen.update(t, self.holder_state(t))
            gt = self.renderer.ground_truth(optics, time_s=t)
        if as_arrays:
            return gt
        from orix.crystal_map import CrystalMap, PhaseList
        from orix.quaternion import Rotation

        from .crystal import phase_for

        if "pattern_centre" in gt:  # EBSD: sample-frame orientations per scan point
            gid, mat = gt["grain_id"].ravel(), gt["material_id"].ravel()
            ids = [int(i) for i in np.unique(mat[gid >= 0])]
            ny, nx = gt["grain_id"].shape
            step = gt["view"].pixel_um
            y, x = np.mgrid[0:ny, 0:nx] * step
            pc = gt["pattern_centre"].reshape(-1, 3)
            return CrystalMap(Rotation(gt["quaternion"].reshape(-1, 4)),
                              np.where(gid >= 0, mat.astype(int), -1),
                              x=x.ravel(), y=y.ravel(),
                              phase_list=PhaseList([phase_for(i) for i in ids], ids=ids) if ids else None,
                              prop={"grain_id": gid, "band_contrast": gt["band_contrast"].ravel(),
                                    "quality": gt["quality"].ravel(), "mix": gt["mix"].ravel(),
                                    "pcx": pc[:, 0], "pcy": pc[:, 1], "pcz": pc[:, 2]}, scan_unit="um")
        gid, mat = gt["grain_id"].ravel(), gt["material_id"].ravel()
        ids = [int(i) for i in np.unique(mat[gid >= 0])]
        ny, nx = gt["grain_id"].shape
        step = gt["view"].pixel_um * 1000.0
        y, x = np.mgrid[0:ny, 0:nx] * step
        return CrystalMap(Rotation(gt["quaternion"].reshape(-1, 4)), np.where(gid >= 0, mat.astype(int), -1),
                          x=x.ravel(), y=y.ravel(),
                          phase_list=PhaseList([phase_for(i) for i in ids], ids=ids) if ids else None,
                          prop={"grain_id": gid, "thickness_nm": gt["thickness_nm"].ravel()}, scan_unit="nm")

    def request(self, **kw) -> AcquisitionRequest:
        """An AcquisitionRequest for this twin's camera."""
        kw.setdefault("camera_model", self.detector.model.name)
        return AcquisitionRequest(**kw)

    def close(self) -> None:
        for obj in (self.holder, self.column):
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best effort
                    pass


def _material_id(phase) -> int:
    """A material id from an id or a (case-insensitive) material name."""
    from .specimen.materials import MATERIALS

    if isinstance(phase, str):
        key = phase.strip().lower()
        for m in MATERIALS:
            if m.name.lower() == key:
                return int(m.id)
        raise KeyError(f"unknown material {phase!r}")
    return int(phase)


def _make_specimen(specimen):
    """A specimen from a preset name, a config or a ready specimen."""
    from .specimen import Specimen, SpecimenConfig, from_name
    from .specimen.bulk import BulkSurface, BulkSurfaceConfig, bulk_from_name

    if isinstance(specimen, str):
        bulk = bulk_from_name(specimen)
        return BulkSurface(bulk) if bulk is not None else Specimen(from_name(specimen))
    if isinstance(specimen, BulkSurfaceConfig):
        return BulkSurface(specimen)
    if isinstance(specimen, SpecimenConfig):
        return Specimen(specimen)
    return specimen


def _fit(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize (nearest/linear) a render to the detector ROI if they disagree."""
    from scipy.ndimage import zoom

    zy = shape[0] / img.shape[0]
    zx = shape[1] / img.shape[1]
    out = zoom(img, (zy, zx), order=1)
    if out.shape != shape:  # rounding
        fixed = np.zeros(shape, np.float32)
        h = min(shape[0], out.shape[0])
        w = min(shape[1], out.shape[1])
        fixed[:h, :w] = out[:h, :w]
        out = fixed
    # zoom scales per-pixel values unchanged; flux per pixel scales with pixel area
    return (out / (zy * zx)).astype(np.float32)


__all__ = ["DigitalTwin", "AcquisitionRequest", "ExposureMode", "Projection", "TemStem"]
