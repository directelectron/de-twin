"""In-situ holders: the de_autopilot heater duck type plus ``state(t) -> HolderState``.

* :class:`NoHolder` - a plain holder at room temperature.
* :class:`SimHeatingHolder` - MEMS heating (optionally biasing) chip model.
* :class:`ImpulseFollower` - a DENSsolutions holder (or DENS's simulator) via impulsePy.
* :func:`connect_holder` - ``"none" | "sim-heating" | "sim-biasing" | "impulse"``.
"""

from __future__ import annotations

import logging

from .base import (
    DEFAULT_RAMP_RATE,
    DEFAULT_TOLERANCE_C,
    ChannelUnsupported,
    HeaterUnavailable,
    NoHolder,
    Reading,
)
from .impulse import ImpulseFollower
from .sim import SimHeatingHolder

log = logging.getLogger(__name__)


def connect_holder(spec=None, *, clock=None, strict: bool = False, **kwargs):
    """Build a holder from a spec string (or return a holder object unchanged).

    ``"none"`` (or None/"") -> :class:`NoHolder`; ``"sim-heating"`` (aliases
    "sim", "heating") -> :class:`SimHeatingHolder`; ``"sim-biasing"`` (alias
    "sim-heating-biasing") -> the same with a bias channel; ``"impulse"`` ->
    :class:`ImpulseFollower`. If impulsePy is missing or Impulse does not answer,
    a :class:`SimHeatingHolder` stands in with the reason (autopilot's
    behaviour) unless ``strict=True``. Extra kwargs go to the holder.
    """
    if spec is not None and not isinstance(spec, str):
        return spec
    key = (spec or "none").strip().lower()
    if key in ("none", "no", "off", ""):
        return NoHolder(clock=clock)
    if key in ("sim-heating", "sim", "heating", "sim_heating"):
        return SimHeatingHolder(clock=clock, **kwargs)
    if key in ("sim-biasing", "sim-heating-biasing", "biasing", "heating-biasing"):
        kwargs.setdefault("bias", True)
        return SimHeatingHolder(clock=clock, **kwargs)
    if key in ("impulse", "dens", "impulsepy"):
        try:
            follower = ImpulseFollower(kwargs.pop("impulse", None), clock=clock)
            try:
                log.info("Impulse status: %s", follower._impulse.getStatus())
            except AttributeError:
                pass
            return follower
        except Exception as e:
            if strict:
                raise
            reason = (
                f"impulsePy is not installed ({e})"
                if isinstance(e, ImportError)
                else f"Impulse did not answer ({type(e).__name__}: {e})"
            )
            log.warning("%s; using the simulated heating holder", reason)
            return SimHeatingHolder(reason=reason, clock=clock, **kwargs)
    raise ValueError(f"unknown holder spec {spec!r} (none | sim-heating | sim-biasing | impulse)")


__all__ = [
    "DEFAULT_RAMP_RATE",
    "DEFAULT_TOLERANCE_C",
    "ChannelUnsupported",
    "HeaterUnavailable",
    "ImpulseFollower",
    "NoHolder",
    "Reading",
    "SimHeatingHolder",
    "connect_holder",
]
