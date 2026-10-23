"""DE-Server-like frame processing for in-process use.

A real DE camera never hands back a single long raw exposure. DE-Server
integrates many short frames and corrects each one with dark and gain
references that the operator acquired earlier. This module does the same
through the twin, so the references are *measured* (noisy, finite frame
count, fps-dependent dark current) rather than read from the detector's
ground truth. That makes them realistic inputs for reference-quality and
automation work.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace

import numpy as np

from .state import AcquisitionRequest, ExposureMode


def _key(request: AcquisitionRequest, roi) -> tuple:
    return (roi.x, roi.y, roi.w, roi.h, tuple(request.hw_binning), round(request.frame_time_s, 9),
            request.counting)


@dataclass
class References:
    dark: np.ndarray  # mean dark frame (ADU), output shape
    gain: np.ndarray | None  # flat-field, normalised to mean 1 (None = not acquired)
    n_dark: int
    n_gain: int
    frame_time_s: float


class Processor:
    """Reference acquisition + correction, keyed like DE-Server's RefFinder."""

    def __init__(self, twin, dark_frames: int = 20, gain_frames: int = 20,
                 gain_dose_e_per_px: float = 2000.0):
        self.twin = twin
        self.dark_frames = dark_frames
        self.gain_frames = gain_frames
        self.gain_dose_e_per_px = gain_dose_e_per_px
        self._refs: dict[tuple, References] = {}

    def _mean(self, request: AcquisitionRequest, mode: ExposureMode, n: int) -> np.ndarray:
        req = replace(request, exposure_mode=mode, total_frames=n, scan=replace(request.scan, enabled=False))
        acc = None
        for raw, _ in self.twin.frames(req):
            acc = raw.astype(np.float64) if acc is None else acc + raw
        return (acc / n).astype(np.float32)

    def lookup(self, request: AcquisitionRequest) -> References | None:
        """Stored references matching ``request`` (no acquisition)."""
        return self._refs.get(_key(request, self.twin.detector.roi(request)))

    def store_dark(self, request: AcquisitionRequest, mean_dark: np.ndarray, n_frames: int) -> References:
        key = _key(request, self.twin.detector.roi(request))
        old = self._refs.get(key)
        refs = References(np.asarray(mean_dark, np.float32), old.gain if old else None, n_frames,
                          old.n_gain if old else 0, request.frame_time_s)
        self._refs[key] = refs
        return refs

    def store_gain(self, request: AcquisitionRequest, mean_flood: np.ndarray, n_frames: int) -> References:
        """Store a gain reference from a mean flood frame (dark-subtracted here)."""
        refs = self.lookup(request)
        if refs is None:
            raise RuntimeError("take a dark reference before the gain reference")
        flood = np.asarray(mean_flood, np.float32) - refs.dark
        mean = float(flood.mean())
        if mean <= 0:
            raise RuntimeError("gain reference has no signal (beam blanked or column closed?)")
        refs.gain, refs.n_gain = np.clip(flood / mean, 0.05, None).astype(np.float32), n_frames
        return refs

    def take_dark(self, request: AcquisitionRequest, n_frames: int | None = None) -> References:
        n = n_frames or self.dark_frames
        return self.store_dark(request, self._mean(request, ExposureMode.DARK, n), n)

    def take_gain(self, request: AcquisitionRequest, n_frames: int | None = None) -> References:
        n = n_frames or self.gain_frames
        self.references(request, gain=False)
        flood = self._mean(request, ExposureMode.GAIN, n)
        sat = getattr(self.twin.detector.model, "saturation_adu", None)
        if sat and float(np.median(flood)) > 0.8 * sat:
            warnings.warn(f"gain reference flood is near saturation (median {np.median(flood):.0f} ADU, "
                          f"saturation {sat}); lower the dose or frame time", RuntimeWarning, stacklevel=2)
        return self.store_gain(request, flood, n)

    def synthesize_gain(self, request: AcquisitionRequest, dose_e_per_px: float | None = None) -> References:
        """A well-dosed gain reference without simulating hundreds of flood frames.

        A good gain reference integrates thousands of electrons per pixel, which
        at a DE sensor's ~10 e/px/frame full well means hundreds of frames. This
        returns the detector's true flat field with the residual shot noise such a
        reference would have: relative noise sqrt((1 + cv^2) / dose).
        """
        from .hashing import SeedKind, rng_for

        dose = float(dose_e_per_px or self.gain_dose_e_per_px)
        refs = self.references(request, gain=False)
        truth = self.twin.detector.expected_gain(request).astype(np.float32)
        truth /= float(truth.mean())
        cv = getattr(self.twin.detector.model, "adu_cv", 0.7)
        rng = rng_for(self.twin.seed, SeedKind.GAIN, int(dose))
        noise = rng.standard_normal(truth.shape, dtype=np.float32) * np.float32(np.sqrt((1 + cv * cv) / dose))
        refs.gain = np.clip(truth * (1 + noise), 0.05, None).astype(np.float32)
        refs.n_gain = 0
        return refs

    def references(self, request: AcquisitionRequest, *, gain: bool = True) -> References:
        """References for ``request``; missing ones are provided like a prepared operator would.

        The dark reference is measured through the twin. A missing gain reference is
        :meth:`synthesize_gain` (well dosed); use :meth:`take_gain` or a Gain-mode
        acquisition to measure one frame by frame instead.
        """
        roi = self.twin.detector.roi(request)
        refs = self._refs.get(_key(request, roi))
        if refs is None:
            refs = self.take_dark(request)
        if gain and refs.gain is None:
            refs = self.synthesize_gain(request)
        return refs

    def clear(self) -> None:
        self._refs.clear()

    def correct(self, raw: np.ndarray, refs: References) -> np.ndarray:
        out = raw.astype(np.float32) - refs.dark
        if refs.gain is not None:
            out /= refs.gain
        return out

    def acquire(
        self,
        request: AcquisitionRequest,
        *,
        units: str = "electrons",
        gain: bool = True,
        pace: bool = False,
    ) -> np.ndarray:
        """Integrated, dark/gain-corrected image of ``request``.

        ``units``: "electrons" (divide by the camera's ADU/e at the column's HT),
        "adu" (corrected ADU) or "raw" (plain sum, no correction).
        """
        if units == "raw":
            return self.twin.acquire(request, pace=pace)
        refs = self.references(request, gain=gain)
        total = None
        for raw, _ in self.twin.frames(request, pace=pace):
            c = self.correct(raw, refs)
            total = c if total is None else total + c
        if units == "electrons" and not request.counting and not self.twin.detector.model.hardware_counting:
            ht = self.twin.column.state().ht_kv
            total = total / self.twin.detector.model.adu_per_electron_at(ht)
        return total


def frames_for_exposure(exposure_s: float, fps: float) -> tuple[float, int]:
    """(frame_time_s, n_frames) that integrate ``exposure_s`` at ``fps``."""
    frame_time = 1.0 / fps
    n = max(1, math.ceil(exposure_s * fps - 1e-9))
    return min(frame_time, exposure_s), n
