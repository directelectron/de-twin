"""A CEOS-like spherical-aberration corrector (probe and/or image side).

The twin's column can carry a probe (STEM, CESCOR-like, CEOS port 7072) and/or
an image (TEM, CETCOR-like, port 7071) corrector. Each side is a
:class:`CorrectorUnit` that behaves like one CEOS-RPC server as DE-TEM-Channel's
``CorrectorService`` sees it (``instruments/corrector``, ``core/CorrectorService``):

* **Aberrations.** ``residual = native + offset + drift`` (CEOS names, complex nm,
  :class:`~de_twin.optics.aberrations.Aberrations`). ``native`` is the uncorrected
  lens (C3 ~ 1.2 mm, C5 a few mm, small B2/A2/S3/A3; seeded), ``offset`` is what the
  corrector's elements add (it cancels C3 and trims the low orders), ``drift`` a
  seeded random walk on the column's clock. The residual EXCLUDES the operator's
  focus and stigmators (``defocus_um``, stage z, objective / condenser stig), which
  the optics adds as C1 / A1 -- but a *measurement* sees the total, like the real one.
* **Measurement** (``measureC1A1``, ``acquireTableau``) is a Zemlin tableau done on the
  aberration function: at each beam tilt of the tableau the induced defocus and
  two-fold astigmatism (the second derivatives of chi about the tilt) are "measured"
  with per-image noise, a tilt-calibration error and a measurement range (images out
  of range are lost), then a weighted least-squares fit up to ``maxFit`` returns the
  coefficients and their 1-sigma confidence. Truncation, noise and angle all matter
  as on the instrument: C5 leaks into a Standard (A4) fit, a Fast (9 mrad) tableau
  cannot see third order, an out-of-tune column loses its outer images.
* **Correction** (``correctAberration``) follows the CEOS reference semantics
  (``value`` omitted = the corrector's own state-of-correction estimate; ``target``
  omitted = zero; "No value for aberration X available") and is imperfect: a fixed
  per-element gain / rotation calibration error plus jitter, and parasitic coupling
  into related aberrations (B2 -> C1/A1, A2 -> A1/B2, C3 -> S3/A3/B2, ...).
* **Timing.** Commands are posted and run on the injected clock (C1A1 ~2 s, tableau
  1 s + 0.4 s per image, correction 0.5 s, alignment file 1 s); ``status`` is
  0 idle / 1 running / 2 done / 3 failed / 6 absent, as ``getCorrectorStatus``.
* **Tune perturbations.** An HT change detunes badly (like a cold start); a probe-mode /
  function-mode / TEM-STEM change kicks C1/A1/B2/A2 a little.

Units on the CEOS side are metres (radians for WD); everything in-process is nm.
The x, y of a CEOS vector are the real / imaginary parts of the complex coefficient.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..optics.aberrations import NAMES, TERMS, Aberrations
from .ceos import (
    CEOS_ORDER,
    CONFIG_DEFAULTS,
    CORRECT_ERRORS,
    CORRECT_ERRORS_DEFAULT,
    COUPLING,
    DETUNE_HT,
    DETUNE_MODE,
    DONE,
    DRIFT_RATES,
    FAILED,
    IDLE,
    IMAGE_PORT,
    IMAGE_SIGMA_NM,
    IMAGE_SIGMA_REL,
    MEASUREMENT_RANGE_M,
    PROBE_PORT,
    PROTOCOL_VERSION,
    RUNNING,
    SELECT_ERROR_SCALE,
    T_ALIGNMENT,
    T_CONFIG,
    T_CORRECT,
    T_MEASURE_C1A1,
    T_READBACK,
    T_RECONNECT,
    T_TABLEAU_OVERHEAD,
    T_TABLEAU_PER_IMAGE,
    TABLEAU_RINGS,
    TABLEAU_TYPES,
    TILT_CAL_ERROR,
    TILT_JITTER_MRAD,
    CorrectorError,
    MeasurementResult,
    _design_rows,
    _Drift,
    _params,
    _random_coeff,
    _round,
    _server_error,
    _symmetry,
    format_aberration_set,
    induced_c1a1,
    native_aberrations,
    pi4_angle_mrad,
    tableau_tilts,
    tuned_residual,
)

# ---------------------------------------------------------------------------
# One CEOS server
# ---------------------------------------------------------------------------


@dataclass
class _Pending:
    op: str
    t_done: float
    run: Callable[[float], tuple[bool, str, str]]  # t -> (ok, result text, error)


class CorrectorUnit:
    """One corrector (one CEOS-RPC server): probe (STEM, port 7072) or image (TEM, 7071).

    Normally built by :class:`Corrector`; everything is in nm / seconds of the
    injected clock. Thread-safe through the owner's lock.
    """

    def __init__(
        self,
        side: str,
        *,
        seed: int = 0,
        clock: Callable[[], float],
        lock: threading.RLock,
        tuned: bool = True,
        ht_kv: float = 200.0,
        drift_rates: dict[str, float] | None = None,
        drift_step_s: float = 1.0,
        operator: Callable[[str], tuple[float, complex]] | None = None,
        beam_check: Callable[[str], str | None] | None = None,
        label: Callable[[], str] | None = None,
        host: str = "127.0.0.1",
    ):
        if side not in ("probe", "image"):
            raise ValueError("side must be 'probe' or 'image'")
        self.side = side
        self.port = PROBE_PORT if side == "probe" else IMAGE_PORT
        self.corrector_type = "CESCOR" if side == "probe" else "CETCOR"
        self.mode = "STEM" if side == "probe" else "TEM"
        self.host = host
        self.seed = int(seed)
        self.lock = lock
        self._now = clock
        self._operator = operator or (lambda side: (0.0, 0j))
        self._beam_check = beam_check or (lambda side: None)
        self._label = label or (lambda: "default")
        self.ht_kv = float(ht_kv)
        tag = 2 if side == "probe" else 1
        self._rng_events = np.random.default_rng([self.seed, tag, 11])
        self._rng_measure = np.random.default_rng([self.seed, tag, 12])
        self._rng_correct = np.random.default_rng([self.seed, tag, 13])
        calibration = np.random.default_rng([self.seed, tag, 14])

        self.native = native_aberrations(self.seed, side)
        target = tuned_residual(self._rng_events)
        target = target + {"C5": 0j}  # C5 stays native
        self.offset = target - self.native.only(n for n in self.native.coeffs if n != "C5")
        self.tuned_at_start = bool(tuned)
        if not tuned:
            self._kick(DETUNE_HT)
        self.drift_rates = dict(DRIFT_RATES if drift_rates is None else drift_rates)
        self._drift = _Drift(self.drift_rates, [self.seed, tag, 15], self._now(), drift_step_s)

        # fixed calibration errors of the elements and the tableau tilt
        self._cal: dict[str, tuple[float, float]] = {}
        for n in NAMES:
            g, rot, _ = CORRECT_ERRORS.get(n, CORRECT_ERRORS_DEFAULT)
            self._cal[n] = (
                float(calibration.normal(0.0, g)),
                float(calibration.normal(0.0, math.radians(rot))),
            )
        self._tilt_cal = complex(
            1.0 + calibration.normal(0.0, TILT_CAL_ERROR), calibration.normal(0.0, TILT_CAL_ERROR)
        )

        # CorrectorService-shaped state
        self.state = IDLE
        self.last_operation = "none"
        self.last_error = ""
        self.last_result = ""
        self.alignment_file = ""
        self.aberrations: list[tuple[str, tuple[float, float]]] = []  # last measurement, CEOS
        self.last_measurement: MeasurementResult | None = None
        self.soc: dict[str, complex] = {}  # state of correction, nm (rad for WD)
        self.config = dict(CONFIG_DEFAULTS)
        self.wd = 0j  # true corrector beam tilt (rad)
        self.beam_tilt_commanded = 0j
        self.beam_tilt_known = False
        self._pending: _Pending | None = None
        self.history: list[tuple[float, str, bool]] = []

    # ------------------------------------------------------------ aberrations
    def residual(self, t: float | None = None) -> Aberrations:
        """Ground truth: native + corrector offset + drift at time ``t`` (default now)."""
        with self.lock:
            self._advance()
            return self._residual(self._now() if t is None else t)

    def _residual(self, t: float) -> Aberrations:
        return self.native + self.offset + self._drift.at(t)

    def total(self, t: float | None = None) -> Aberrations:
        """What a measurement sees: residual + the operator's focus (C1) and stig (A1)."""
        with self.lock:
            t = self._now() if t is None else t
            c1, a1 = self._operator(self.side)
            return self._residual(t) + {"C1": c1, "A1": a1}

    def set_residual(self, name: str, value_nm: complex) -> None:
        """Test / expert knob: make the residual ``name`` exactly ``value_nm`` now."""
        with self.lock:
            cur = self._residual(self._now())[name]
            self.offset = self.offset + {name: complex(value_nm) - cur}

    def _kick(self, table: dict[str, float], scale: float = 1.0) -> None:
        self.offset = self.offset + {
            n: _random_coeff(self._rng_events, n, v * scale * self._rng_events.uniform(0.3, 1.0))
            for n, v in table.items()
        }

    def perturb(self, kind: str = "mode", scale: float = 1.0) -> None:
        """Detune: ``"ht"`` (HT change, cold start) or ``"mode"`` (probe/function mode)."""
        with self.lock:
            self._advance()
            self._kick(DETUNE_HT if kind == "ht" else DETUNE_MODE, scale)
            if kind == "ht":
                self.soc.clear()

    # ------------------------------------------------------------ status
    def _advance(self) -> None:
        p = self._pending
        if p is None or self._now() < p.t_done:
            return
        self._pending = None
        try:
            ok, text, err = p.run(p.t_done)
        except CorrectorError as exc:
            ok, text, err = False, "", str(exc)
        self.state = DONE if ok else FAILED
        self.last_result = text if ok else ""
        self.last_error = "" if ok else err
        self.history.append((p.t_done, p.op, ok))

    def status(self) -> int:
        with self.lock:
            self._advance()
            return self.state

    @property
    def busy(self) -> bool:
        return self.status() == RUNNING

    def time_to_idle(self) -> float:
        with self.lock:
            self._advance()
            return max(0.0, self._pending.t_done - self._now()) if self._pending else 0.0

    def info(self) -> str:
        """``CorrectorInfo::toString`` (getCorrectorInfo)."""
        rng = ",".join(f"{k}:{v:.9g}" for k, v in MEASUREMENT_RANGE_M.items())
        return (
            f"correctorType={self.corrector_type};mode={self.mode};host={self.host};"
            f"version=v1.9-0-de_twin;kernelVersion=0-00-00;currentLabel={self._label()};"
            f"currentSetting=de_twin;protocolVersion={PROTOCOL_VERSION};measurementRange={rng}"
        )

    def endpoint(self) -> str:
        """``CorrectorFactory::describe``."""
        return (
            f"{self.corrector_type} {self.mode} corrector at {self.host}:{self.port} "
            f"(protocol {PROTOCOL_VERSION})"
        )

    def aberrations_string(self) -> str:
        with self.lock:
            self._advance()
            return format_aberration_set(self.aberrations)

    def aberration(self, name: str) -> tuple[float, float, int]:
        """getAberration: (x, y, status) from the last measurement (metres / rad)."""
        with self.lock:
            self._advance()
            for n, (x, y) in self.aberrations:
                if n == name:
                    return x, y, 1
            return 0.0, 0.0, 0

    # ------------------------------------------------------------ posting
    def _post(self, op: str, duration: float, run) -> tuple[bool, str]:
        with self.lock:
            self._advance()
            if self.state == RUNNING:
                return False, "the corrector is busy"
            self.state = RUNNING
            self.last_operation = op
            self.last_result = ""
            self.last_error = ""
            self._pending = _Pending(op, self._now() + float(duration), run)
            return True, ""

    def wait(self, timeout: float | None = None, sleeper: Callable[[float], None] | None = None) -> int:
        """Wait (on the clock) until the running command finishes. Returns the state."""
        start = self._now()
        while True:
            rem = self.time_to_idle()
            if rem <= 0:
                return self.status()
            if timeout is not None:
                left = timeout - (self._now() - start)
                if left <= 0:
                    return self.status()
                rem = min(rem, left)
            if sleeper is not None:
                sleeper(rem + 1e-9)
            else:
                time.sleep(min(rem, 0.05) + 1e-4)

    # measureC1A1 ------------------------------------------------------------
    def post_measure_c1a1(self) -> tuple[bool, str]:
        return self._post("measureC1A1", T_MEASURE_C1A1, lambda t: self._run_measure(t, "C1A1"))

    def post_tableau(
        self, tab_type: str = "", angle_mrad: float = 0.0, max_fit: str = ""
    ) -> tuple[bool, str]:
        tab = tab_type or "Standard"
        if tab not in TABLEAU_TYPES:
            # the reference server raises inside the command: accepted, then failed
            err = _server_error(f'Invalid Tableau type "{tab}", valid types are: {list(TABLEAU_TYPES)}')
            return self._post("acquireTableau", 0.05, lambda t: (False, "", err))
        if max_fit and max_fit not in CEOS_ORDER:
            err = _server_error(f'MaxFit aberration "{max_fit}" is unknown')
            return self._post("acquireTableau", 0.05, lambda t: (False, "", err))
        n_images = 1 + sum(n for _, n in TABLEAU_RINGS[tab])
        dur = T_TABLEAU_OVERHEAD + T_TABLEAU_PER_IMAGE * n_images
        return self._post(
            "acquireTableau", dur, lambda t: self._run_measure(t, "tableau", tab, angle_mrad, max_fit)
        )

    def _run_measure(
        self, t: float, kind: str, tab: str = "Standard", angle_mrad: float = 0.0, max_fit: str = ""
    ) -> tuple[bool, str, str]:
        why = self._beam_check(self.side)
        if why:
            return False, "", _server_error(why)
        result = self.simulate_measurement(kind, tab, angle_mrad, max_fit, t=t)
        if isinstance(result, str):
            return False, "", _server_error(result)
        self.last_measurement = result
        self.aberrations = result.ceos()
        for n, (x, y) in self.aberrations:
            self.soc[n] = complex(x * 1e9, y * 1e9)
        return True, result.to_string(), ""

    def simulate_measurement(
        self,
        kind: str = "tableau",
        tab: str = "Standard",
        angle_mrad: float = 0.0,
        max_fit: str = "",
        t: float | None = None,
    ) -> MeasurementResult | str:
        """Run the tableau / C1A1 analysis on the current aberrations (no side effects
        other than the measurement noise stream). Returns a result or an error string."""
        t = self._now() if t is None else t
        rng = self._rng_measure
        c1_op, a1_op = self._operator(self.side)
        truth = (self._residual(t) + {"C1": c1_op, "A1": a1_op}).coeffs
        if kind == "C1A1":
            names = ("C1", "A1")
            tilts = [0j]
            angle = 0.0
            max_fit = "A1"
        else:
            angle = float(angle_mrad) if angle_mrad and angle_mrad > 0 else TABLEAU_TYPES[tab][0]
            max_fit = max_fit or TABLEAU_TYPES[tab][1]
            upto = CEOS_ORDER[: CEOS_ORDER.index(max_fit) + 1]
            names = tuple(n for n in upto if n in TERMS)
            tilts = tableau_tilts(tab, angle)
        params = _params(names)
        rows, obs, sigma = [], [], []
        lost = 0
        rng_c1 = MEASUREMENT_RANGE_M["C1"] * 1e9
        rng_a1 = MEASUREMENT_RANGE_M["A1"] * 1e9
        for i, tilt in enumerate(tilts):
            actual = tilt * self._tilt_cal + (
                complex(*rng.normal(0.0, TILT_JITTER_MRAD * 1e-3, 2)) if tilt != 0 else 0j
            )
            c1, a1 = induced_c1a1(truth, actual)
            if abs(c1) > rng_c1 or abs(a1) > rng_a1:
                lost += 1
                if i == 0:
                    return "aberrations out of the measurement range (C1/A1 > {:.3g} m)".format(
                        MEASUREMENT_RANGE_M["C1"]
                    )
                continue
            s_c1 = math.hypot(IMAGE_SIGMA_NM, IMAGE_SIGMA_REL * abs(c1))
            s_a1 = math.hypot(IMAGE_SIGMA_NM, IMAGE_SIGMA_REL * abs(a1)) / math.sqrt(2.0)
            m = (c1 + rng.normal(0.0, s_c1), a1.real + rng.normal(0.0, s_a1), a1.imag + rng.normal(0.0, s_a1))
            rows.append(_design_rows(params, tilt))
            obs.extend(m)
            sigma.extend((s_c1, s_a1, s_a1))
        n_ok = len(tilts) - lost
        if kind == "tableau" and (lost > len(tilts) // 3 or 3 * n_ok < 2 * len(params)):
            return (
                f"tableau failed: {lost} of {len(tilts)} images out of the measurement "
                f"range; reduce the tilt angle or use a Fast tableau"
            )
        a = np.vstack(rows)
        w = 1.0 / np.asarray(sigma)
        aw = a * w[:, None]
        bw = np.asarray(obs) * w
        x, *_ = np.linalg.lstsq(aw, bw, rcond=None)
        try:
            cov = np.linalg.inv(aw.T @ aw)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(aw.T @ aw)
        coeffs: dict[str, complex] = {}
        var: dict[str, float] = {}
        for j, (n, unit) in enumerate(params):
            coeffs[n] = coeffs.get(n, 0j) + x[j] * unit
            var[n] = var.get(n, 0.0) + float(cov[j, j])
        n_img = len(tilts)
        return MeasurementResult(
            kind=kind,
            aberrations=Aberrations(coeffs),
            sigma_nm={n: math.sqrt(max(v, 0.0)) for n, v in var.items()},
            tab_type="" if kind == "C1A1" else tab,
            angle_mrad=angle,
            max_fit=max_fit,
            images=n_img,
            images_lost=lost,
            t_start=t,
            t_done=t,
            side=self.side,
            names=names,
        )

    # correctAberration ------------------------------------------------------
    def post_correct(
        self,
        name: str,
        value: complex | None = None,
        target: complex | None = None,
        select: str = "",
        *,
        beam_tilt_managed: bool = False,
    ) -> tuple[bool, str]:
        """``value`` / ``target`` in nm (rad for WD); None = omitted."""
        if not name:
            return False, "no aberration name was given"
        return self._post(
            "correctAberration",
            T_CORRECT,
            lambda t: self._run_correct(t, name, value, target, select, beam_tilt_managed),
        )

    def _run_correct(
        self, t: float, name: str, value, target, select: str, managed: bool
    ) -> tuple[bool, str, str]:
        if name not in CEOS_ORDER:
            return False, "", _server_error(f'Unknown aberration "{name}"')
        if value is not None:
            self.soc[name] = complex(value)
        if name not in self.soc:
            return False, "", _server_error(f"No value for aberration {name} available")
        tgt = complex(target) if target is not None else 0j
        applied = self.soc[name] - tgt
        if name == "WD":
            self.wd -= applied
        elif name in TERMS:
            self._apply_correction(name, applied, select)
        # We (image shift) and WG: nothing the aberration function sees
        if target is not None:
            self.soc[name] = tgt
        else:
            self.soc.pop(name, None)
        if name == "WD" and not managed:
            self.beam_tilt_known = False
        return True, "", ""

    def _apply_correction(self, name: str, applied: complex, select: str = "") -> None:
        rng = self._rng_correct
        g0, r0 = self._cal[name]
        jit = CORRECT_ERRORS.get(name, CORRECT_ERRORS_DEFAULT)[2]
        scale = SELECT_ERROR_SCALE.get((select or "").lower(), 1.0)
        gain = 1.0 + scale * (g0 + rng.normal(0.0, jit))
        if _round(name):
            change = -applied.real * gain
            self.offset = self.offset + {name: complex(change, 0.0)}
        else:
            n = max(_symmetry(name), 1)
            rot = scale * (r0 + rng.normal(0.0, math.radians(jit * 30.0))) * n
            change = -applied * gain * complex(math.cos(rot), math.sin(rot))
            self.offset = self.offset + {name: change}
        magnification = abs(applied)
        for dst, k in COUPLING.get(name, {}).items():
            self.offset = self.offset + {
                dst: _random_coeff(rng, dst, k * magnification * rng.uniform(0.5, 1.5))
            }

    # alignment / config / housekeeping ---------------------------------------
    def alignment_json(self) -> str:
        return json.dumps(
            {
                "format": "de_twin corrector alignment",
                "version": 1,
                "correctorType": self.corrector_type,
                "mode": self.mode,
                "label": self._label(),
                "offset": {n: [v.real * 1e-9, v.imag * 1e-9] for n, v in self.offset.coeffs.items()},
                "WD": [self.wd.real, self.wd.imag],
            },
            sort_keys=True,
        )

    def _load_alignment(self, data: str) -> tuple[bool, str, str]:
        try:
            d = json.loads(data)
            off = {n: complex(float(x) * 1e9, float(y) * 1e9) for n, (x, y) in d["offset"].items()}
            ab = Aberrations(off)
            wd = d.get("WD", [0.0, 0.0])
        except (ValueError, KeyError, TypeError) as exc:
            return False, "", _server_error(f"invalid alignment file ({exc})")
        self.offset = ab
        self.wd = complex(float(wd[0]), float(wd[1]))
        self.beam_tilt_known = False
        return True, "", ""

    def post_fetch_alignment(self) -> tuple[bool, str]:
        def run(t):
            self.alignment_file = self.alignment_json()
            return True, "", ""

        return self._post("getAlignmentFile", T_ALIGNMENT, run)

    def post_put_alignment(self, data: str) -> tuple[bool, str]:
        if not data:
            return False, "the alignment file was empty"
        return self._post("putAlignmentFile", T_ALIGNMENT, lambda t: self._load_alignment(data))

    def post_get_config(self, name: str) -> tuple[bool, str]:
        if not name:
            return False, "no configuration option name was given"

        def run(t):
            key = name.lower()
            if key not in self.config:
                return False, "", _server_error(f'Unknown configuration option "{name}"')
            return True, "true" if self.config[key] else "false", ""

        return self._post("getConfigOption", T_CONFIG, run)

    def post_set_config(self, name: str, value: str) -> tuple[bool, str]:
        if not name:
            return False, "no configuration option name was given"

        def run(t):
            key = name.lower()
            if key not in self.config:
                return False, "", _server_error(f'Unknown configuration option "{name}"')
            v = str(value).strip().lower()
            if v in ("true", "1", "yes", "on"):
                b = True
            elif v in ("false", "0", "no", "off"):
                b = False
            else:
                try:
                    b = bool(float(v))
                except ValueError:
                    b = bool(v)
            self.config[key] = b
            return True, "", ""

        return self._post("setConfigOption", T_CONFIG, run)

    def post_readback(self) -> tuple[bool, str]:
        return self._post("readBack", T_READBACK, lambda t: (True, "", ""))

    def post_reconnect(self) -> tuple[bool, str]:
        def run(t):
            self.beam_tilt_known = False
            return True, "", ""

        return self._post("reconnect", T_RECONNECT, run)

    # beam tilt (WD) -----------------------------------------------------------
    def set_beam_tilt(
        self, x: float, y: float, relative: bool = True, wait_s: float = 0.0, sleeper=None
    ) -> tuple[bool, bool, str]:
        """CorrectorService::setBeamTilt (radians). Returns (completed, still_running, why_not).
        ``wait_s == 0``: posts; ``still_running`` True means accepted."""
        tgt = complex(float(x), float(y))
        ok, why = self.post_correct("WD", value=0j if relative else None, target=tgt, beam_tilt_managed=True)
        if not ok:
            return False, False, why
        if wait_s <= 0:
            # the tracked value is only updated on a confirmed completion
            return False, True, ""
        state = self.wait(wait_s, sleeper)
        if state == RUNNING:
            return False, True, "still running"
        if state != DONE:
            return False, False, self.last_error or "the corrector command failed"
        with self.lock:
            if relative and self.beam_tilt_known:
                self.beam_tilt_commanded += tgt
            else:
                self.beam_tilt_commanded = tgt
            self.beam_tilt_known = True
        return True, False, ""


# ---------------------------------------------------------------------------
# The column's corrector
# ---------------------------------------------------------------------------


class Corrector:
    """Probe and/or image corrector of a column.

    >>> col = Column(corrector="probe", seed=1, clock=ManualClock())
    >>> col.corrector.pi4_angle_mrad()          # ~25-30 mrad when tuned
    >>> r = col.corrector.measure("Standard")   # tableau; advances a ManualClock
    >>> col.corrector.correct("B2"); col.corrector.correct("A2", fraction=0.7)

    ``kind`` is ``"probe"``, ``"image"`` or ``"both"``. ``served`` is the side the
    DE-TEM-Channel face serves (the channel keeps one endpoint; the probe corrector
    by default). Every method takes ``side=`` to address the other one in-process.
    """

    def __init__(
        self,
        kind: str = "probe",
        *,
        clock=None,
        seed: int = 0,
        tuned: bool = True,
        ht_kv: float = 200.0,
        lock: threading.RLock | None = None,
        drift_rates: dict[str, float] | None = None,
        drift_step_s: float = 1.0,
        operator: Callable[[str], tuple[float, complex]] | None = None,
        beam_check: Callable[[str], str | None] | None = None,
        label: Callable[[], str] | None = None,
        served: str | None = None,
        sleeper: Callable[[float], None] | None = None,
    ):
        kind = normalize_kind(kind)
        if kind == "none":
            raise ValueError("Corrector needs kind 'probe', 'image' or 'both'")
        self.kind = kind
        self.lock = lock or threading.RLock()
        if clock is None:
            t0 = time.perf_counter()
            clock = lambda: time.perf_counter() - t0
        elif not callable(clock) or hasattr(clock, "now"):
            sleeper = sleeper or getattr(clock, "sleep", None)
            clock = clock.now
        self._now = clock
        self._sleeper = sleeper
        self.units: dict[str, CorrectorUnit] = {}
        sides = ("probe", "image") if kind == "both" else (kind,)
        for side in sides:
            self.units[side] = CorrectorUnit(
                side,
                seed=seed,
                clock=clock,
                lock=self.lock,
                tuned=tuned,
                ht_kv=ht_kv,
                drift_rates=drift_rates,
                drift_step_s=drift_step_s,
                operator=operator,
                beam_check=beam_check,
                label=label,
            )
        self.served_side = served if served in self.units else sides[0]
        self.ht_kv = float(ht_kv)

    # ------------------------------------------------------------ access
    @property
    def served(self) -> CorrectorUnit:
        return self.units[self.served_side]

    def unit(self, side: str | None = None) -> CorrectorUnit:
        side = side or self.served_side
        if side not in self.units:
            raise CorrectorError(f"no {side} corrector on this column (fitted: {self.kind})")
        return self.units[side]

    def has(self, side: str) -> bool:
        return side in self.units

    def residual_dict(self, side: str) -> dict:
        """For MicroscopeState.{probe,image}_aberrations ({} when that side has none)."""
        if side not in self.units:
            return {}
        return dict(self.units[side].residual().coeffs)

    def get(self, side: str | None = None) -> Aberrations:
        """Ground-truth residual of a side (nm)."""
        return self.unit(side).residual()

    def set(self, name: str, value_nm: complex, side: str | None = None) -> None:
        """Set a residual coefficient exactly (an ideal correction; tests / scenarios)."""
        self.unit(side).set_residual(name, value_nm)

    def pi4_angle_mrad(self, side: str | None = None) -> float:
        return pi4_angle_mrad(self.get(side), self.ht_kv)

    def status(self, side: str | None = None) -> int:
        return self.unit(side).status()

    @property
    def busy(self) -> bool:
        return any(u.busy for u in self.units.values())

    @property
    def beam_tilt(self) -> tuple[float, float, bool]:
        """Commanded corrector beam tilt (WD) of the served side, rad: (x, y, known)."""
        u = self.served
        return u.beam_tilt_commanded.real, u.beam_tilt_commanded.imag, u.beam_tilt_known

    def set_beam_tilt(
        self, x_rad: float, y_rad: float, relative: bool = True, side: str | None = None
    ) -> None:
        completed, _, why = self.unit(side).set_beam_tilt(
            x_rad, y_rad, relative, wait_s=5.0, sleeper=self._sleeper
        )
        if not completed:
            raise CorrectorError(why or self.unit(side).last_error)

    def endpoints(self) -> str:
        return "\n".join(
            u.endpoint() for u in (self.served, *[v for k, v in self.units.items() if k != self.served_side])
        )

    # ------------------------------------------------------------ blocking ops
    def _finish(self, u: CorrectorUnit, accepted: tuple[bool, str]) -> None:
        ok, why = accepted
        if not ok:
            raise CorrectorError(why)
        state = u.wait(None, self._sleeper)
        if state != DONE:
            raise CorrectorError(u.last_error or "the corrector command failed")

    def measure(
        self, tab_type: str = "Standard", angle_mrad: float = 0.0, max_fit: str = "", side: str | None = None
    ) -> MeasurementResult:
        """Acquire a tableau (``"Fast"``/``"Standard"``/``"Enhanced"``) or, with
        ``tab_type="C1A1"``, a single C1/A1 measurement. Blocks on the clock (a
        ManualClock is advanced). Raises :class:`CorrectorError` on failure."""
        u = self.unit(side)
        if tab_type.upper() == "C1A1":
            acc = u.post_measure_c1a1()
        else:
            acc = u.post_tableau(tab_type, angle_mrad, max_fit)
        self._finish(u, acc)
        assert u.last_measurement is not None
        return u.last_measurement

    def correct(
        self,
        name: str,
        fraction: float = 1.0,
        *,
        side: str | None = None,
        select: str = "",
        value_nm: complex | None = None,
    ) -> Aberrations:
        """Correct ``fraction`` of the corrector's current estimate of ``name`` (from the
        last measurement / state of correction), i.e. CEOS ``correctAberration`` with
        ``target = (1 - fraction) * estimate``. Returns the new ground-truth residual."""
        u = self.unit(side)
        with self.lock:
            est = complex(value_nm) if value_nm is not None else u.soc.get(name)
        target = None if fraction == 1.0 or est is None else est * (1.0 - float(fraction))
        self._finish(u, u.post_correct(name, value=value_nm, target=target, select=select))
        return u.residual()

    def save_alignment(self, side: str | None = None) -> str:
        u = self.unit(side)
        self._finish(u, u.post_fetch_alignment())
        return u.alignment_file

    def load_alignment(self, data: str, side: str | None = None) -> None:
        u = self.unit(side)
        self._finish(u, u.post_put_alignment(data))

    def wait(self, timeout: float | None = None, side: str | None = None) -> int:
        return self.unit(side).wait(timeout, self._sleeper)

    # ------------------------------------------------------------ column events
    def perturb(self, kind: str = "mode", scale: float = 1.0) -> None:
        for u in self.units.values():
            u.perturb(kind, scale)

    def on_ht_change(self, ht_kv: float) -> None:
        self.ht_kv = float(ht_kv)
        for u in self.units.values():
            u.ht_kv = float(ht_kv)
        self.perturb("ht")


def normalize_kind(kind: Any) -> str:
    k = str(kind or "none").strip().lower()
    if k in ("", "none", "false", "0", "no"):
        return "none"
    if k in ("probe", "stem", "cescor"):
        return "probe"
    if k in ("image", "tem", "cetcor"):
        return "image"
    if k in ("both", "probe+image", "image+probe"):
        return "both"
    raise ValueError(f"corrector must be none | probe | image | both, not {kind!r}")
