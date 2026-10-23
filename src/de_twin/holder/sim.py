"""SimHeatingHolder: a MEMS heating chip (DENS Wildfire-like), optionally biasing.

Physics (assumptions documented here, all parameters overridable):

* **Controller.** Impulse runs a closed loop on the chip's 4-point Pt sensor.
  ``set()`` steps the setpoint; ``ramp()`` moves it linearly at the requested
  rate (or over the requested time).
* **Chip response.** The membrane has a tiny heat capacity, so the measured
  temperature follows the setpoint with a first-order lag of time constant
  ``tau_s`` (default 5 ms; DENS quote heating rates up to ~1e6 degC/s and
  millisecond settling for Wildfire chips, i.e. tau of order 1-10 ms). With
  ``overshoot > 0`` (fraction, e.g. 0.05) the response is second order
  (natural frequency 1/tau, damping from the overshoot) to mimic an
  aggressively tuned PID.
* **Heater.** Pt resistance ``R(T) = R0 (1 + alpha T)`` with T in degC
  (R0 = 500 ohm, alpha = 0.0030 /degC: thin-film Pt, lower than bulk 0.00385).
  Electrical power ``P = G (T - T_amb) + eps*sigma*A (T^4 - T_amb^4) + C dT/dt``
  with G = 3e-5 W/K (about 30 mW at 1000 degC, typical of MEMS heaters) and
  ``C = G * tau``. Heater current ``sqrt(P / R)``.
* **Bias channel (optional).** A semiconducting sample between the bias
  contacts: ``R_s = R_s0 exp(-(T - T_amb) / T_decay)`` (1 Mohm, 200 degC, the
  same law as autopilot's SimHeater), current ``V / R_s``.
* **Noise.** Deterministic (hash of seed and clock millisecond), 0.01 degC rms.

Time comes from an injected clock (callable or ``.now()``), so a virtual
clock fast-forwards anneals; the response is integrated exactly (first order)
or with RK4 over the transient and analytically afterwards (second order).
"""

from __future__ import annotations

import logging
import math
import threading
import time

from ..hashing import hash_seed, normal_from_hash
from ..state import HolderState
from .base import (
    AMBIENT_C,
    DEFAULT_RAMP_RATE,
    DEFAULT_TOLERANCE_C,
    ChannelUnsupported,
    Reading,
    clock_fn,
)

log = logging.getLogger(__name__)

SIGMA = 5.670374419e-8


class SimHeatingHolder:
    """Simulated MEMS heating (and optionally biasing) holder."""

    real = False

    def __init__(
        self,
        reason: str = "Simulated MEMS heating holder (de_twin).",
        *,
        clock=None,
        bias: bool = False,
        tau_s: float = 0.005,
        overshoot: float = 0.0,
        ambient_c: float = AMBIENT_C,
        max_c: float = 1300.0,
        r0_ohm: float = 500.0,
        alpha_per_c: float = 0.0030,
        conductance_w_per_k: float = 3e-5,
        emissivity_area_m2: float = 0.3 * 1e-6,
        sample_r0_ohm: float = 1.0e6,
        sample_decay_c: float = 200.0,
        noise_c: float = 0.01,
        seed: int = 0,
    ):
        self.reason = reason
        self._now = clock_fn(clock)
        self.channels = ("temperature", "bias") if bias else ("temperature",)
        self.tau_s = float(tau_s)
        self.overshoot = float(overshoot)
        self.ambient_c = float(ambient_c)
        self.max_c = float(max_c)
        self.r0_ohm = float(r0_ohm)
        self.alpha_per_c = float(alpha_per_c)
        self.conductance = float(conductance_w_per_k)
        self.eps_area = float(emissivity_area_m2)
        self.sample_r0_ohm = float(sample_r0_ohm)
        self.sample_decay_c = float(sample_decay_c)
        self.noise_c = float(noise_c)
        self.seed = int(seed)
        self._lock = threading.RLock()
        t = self._clock()
        self._t = t
        self._y = self.ambient_c  # chip temperature
        self._v = 0.0  # dT/dt
        # setpoint: moves from _sp0 at _sp_t0 towards _target at _rate (inf = step)
        self._sp0 = self.ambient_c
        self._sp_t0 = t
        self._target = self.ambient_c
        self._rate = math.inf
        self._bias_v = 0.0
        self._flags: list[tuple[float, str]] = []

    # -------------------------------------------------------------- clock
    def _clock(self) -> float:
        return self._now() if self._now else time.monotonic()

    def _stamp(self) -> float:
        return self._now() if self._now else time.time()

    # ----------------------------------------------------------- setpoint
    def _setpoint(self, t: float) -> float:
        if not math.isfinite(self._rate):
            return self._target
        span = self._target - self._sp0
        moved = self._rate * max(0.0, t - self._sp_t0)
        return self._target if moved >= abs(span) else self._sp0 + math.copysign(moved, span)

    def _ramp_end(self) -> float:
        if not math.isfinite(self._rate) or self._rate <= 0:
            return self._sp_t0
        return self._sp_t0 + abs(self._target - self._sp0) / self._rate

    def _slope(self, t: float) -> float:
        if not math.isfinite(self._rate) or t >= self._ramp_end():
            return 0.0
        return math.copysign(self._rate, self._target - self._sp0)

    # ----------------------------------------------------------- dynamics
    def _advance(self, t: float) -> None:
        if t <= self._t:
            return
        while self._t < t:
            end = self._ramp_end()
            t_next = min(t, end) if end > self._t else t
            self._integrate(self._t, t_next)
            self._t = t_next

    def _integrate(self, t0: float, t1: float) -> None:
        dt = t1 - t0
        if dt <= 0:
            return
        u0 = self._setpoint(t0)
        b = self._slope(t0)
        tau = max(self.tau_s, 1e-9)
        if self.overshoot <= 0:
            # y' = (u - y)/tau with u = u0 + b s: exact solution
            e = math.exp(-dt / tau)
            u1 = u0 + b * dt
            self._y = u1 - b * tau + (self._y - u0 + b * tau) * e
            self._v = b + (self._y - (u1 - b * tau)) * (-1.0 / tau)
            return
        os_ = min(max(self.overshoot, 1e-6), 0.99)
        ln = math.log(os_)
        zeta = -ln / math.sqrt(math.pi**2 + ln**2)
        w = 1.0 / tau
        transient = 40.0 * tau / max(zeta, 0.05)
        h = tau / 20.0
        s = 0.0
        y, v = self._y, self._v

        def acc(ss, yy, vv):
            u = u0 + b * ss
            return w * w * (u - yy) - 2 * zeta * w * vv

        limit = min(dt, transient)
        while s < limit:
            hh = min(h, limit - s)
            k1y, k1v = v, acc(s, y, v)
            k2y, k2v = v + 0.5 * hh * k1v, acc(s + 0.5 * hh, y + 0.5 * hh * k1y, v + 0.5 * hh * k1v)
            k3y, k3v = v + 0.5 * hh * k2v, acc(s + 0.5 * hh, y + 0.5 * hh * k2y, v + 0.5 * hh * k2v)
            k4y, k4v = v + hh * k3v, acc(s + hh, y + hh * k3y, v + hh * k3v)
            y += hh / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
            v += hh / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            s += hh
        if dt > limit:  # transient has died out: steady ramp tracking
            y = u0 + b * dt - 2 * zeta * b / w
            v = b
        self._y, self._v = y, v

    # ------------------------------------------------------------- physics
    def heater_resistance(self, temp_c: float) -> float:
        return self.r0_ohm * (1.0 + self.alpha_per_c * temp_c)

    def heater_power(self, temp_c: float, dtdt: float = 0.0) -> float:
        tk, ta = temp_c + 273.15, self.ambient_c + 273.15
        p = self.conductance * (temp_c - self.ambient_c) + 0.9 * SIGMA * self.eps_area * (tk**4 - ta**4)
        p += self.conductance * self.tau_s * dtdt
        return max(0.0, p)

    def sample_resistance(self, temp_c: float) -> float:
        return self.sample_r0_ohm * math.exp(-max(0.0, temp_c - self.ambient_c) / self.sample_decay_c)

    def _noise(self, t: float) -> float:
        if self.noise_c <= 0:
            return 0.0
        return self.noise_c * float(normal_from_hash(hash_seed(self.seed, 77, round(t * 1000))))

    # ---------------------------------------------------------- duck type
    @property
    def busy(self) -> bool:
        with self._lock:
            t = self._clock()
            self._advance(t)
            return t < self._ramp_end() or abs(self._target - self._y) > DEFAULT_TOLERANCE_C

    def _sample(self, t: float):
        with self._lock:
            self._advance(t)
            temp = self._y + self._noise(t)
            return temp, self._target, self._v, self._bias_v

    def read(self) -> Reading:
        t = self._clock()
        temp, target, dtdt, bias = self._sample(t)
        stamp = self._stamp()
        power = self.heater_power(temp, dtdt)
        r_heater = self.heater_resistance(temp)
        if "bias" in self.channels and bias:
            r_s = self.sample_resistance(temp)
            return Reading(
                measured=temp,
                target=target,
                t=stamp,
                power=power,
                resistance=r_s,
                voltage=bias,
                current=bias / r_s,
                source="bias",
                bias_target=bias,
            )
        current = math.sqrt(power / r_heater) if power > 0 else 0.0
        return Reading(
            measured=temp,
            target=target,
            t=stamp,
            power=power,
            resistance=r_heater,
            voltage=current * r_heater,
            current=current,
            source="heater",
            bias_target=bias if "bias" in self.channels else None,
        )

    def _check(self, channel: str) -> None:
        if channel not in self.channels:
            raise ChannelUnsupported(f"This holder has no {channel!r} channel ({self.channels}).")

    def _retarget(self, value: float, rate: float) -> None:
        t = self._clock()
        self._advance(t)
        self._sp0 = self._setpoint(t)
        self._sp_t0 = t
        self._target = min(max(float(value), -273.0), self.max_c)
        self._rate = rate

    def set(self, value: float, *, channel: str = "temperature") -> None:
        self._check(channel)
        with self._lock:
            if channel == "bias":
                self._bias_v = float(value)
                return
            self._retarget(value, math.inf)

    def ramp(
        self,
        value: float,
        *,
        rate_c_per_s: float | None = None,
        seconds: float | None = None,
        channel: str = "temperature",
    ) -> None:
        self._check(channel)
        with self._lock:
            if channel == "bias":
                self._bias_v = float(value)  # a supply steps in microseconds
                return
            t = self._clock()
            self._advance(t)
            start = self._setpoint(t)
            if seconds and seconds > 0 and not rate_c_per_s:
                rate = abs(float(value) - start) / float(seconds) or math.inf
            else:
                rate = float(rate_c_per_s or DEFAULT_RAMP_RATE)
            self._retarget(value, rate)

    def stop(self) -> None:
        """Hold where the setpoint has got to (not "go cold")."""
        with self._lock:
            t = self._clock()
            self._advance(t)
            self._retarget(self._setpoint(t), math.inf)

    def flag(self, text: str) -> None:
        self._flags.append((self._stamp(), str(text)))
        log.info("holder flag: %s", text)

    @property
    def flags(self) -> list[tuple[float, str]]:
        return list(self._flags)

    def describe(self) -> str:
        what = "heating + biasing" if "bias" in self.channels else "heating"
        return f"Simulated MEMS {what} holder"

    def wait_for_control(self, timeout_s: float = 30.0) -> bool:
        return True

    def state(self, t: float | None = None) -> HolderState:
        """HolderState at clock time ``t`` (default now). ``t`` in the past reads the
        current state (the model does not rewind)."""
        t = self._clock() if t is None else float(t)
        temp, target, dtdt, bias = self._sample(t)
        power = self.heater_power(temp, dtdt)
        biasing = "bias" in self.channels
        if biasing and bias:
            current = bias / self.sample_resistance(temp)
        else:
            r = self.heater_resistance(temp)
            current = math.sqrt(power / r) if power > 0 else 0.0
        return HolderState(
            kind="heating_biasing" if biasing else "heating",
            temperature_c=temp,
            target_c=target,
            power_w=power,
            resistance_ohm=self.heater_resistance(temp),
            bias_v=bias,
            current_a=current,
            t_s=t,
        )

    def close(self) -> None:
        pass
