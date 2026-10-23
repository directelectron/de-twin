"""In-situ holders: autopilot heater duck type + HolderState."""

import math
import sys
import types

import pytest

from de_twin.clock import ManualClock
from de_twin.holder import (
    ChannelUnsupported,
    HeaterUnavailable,
    ImpulseFollower,
    NoHolder,
    Reading,
    SimHeatingHolder,
    connect_holder,
)
from de_twin.state import HolderState

DUCK = ("real", "reason", "channels", "busy", "read", "set", "ramp", "stop", "flag", "describe",
        "wait_for_control", "close", "state")


@pytest.fixture
def clock():
    return ManualClock()


def test_duck_type_everywhere(clock):
    for h in (NoHolder(clock=clock), SimHeatingHolder(clock=clock), connect_holder("impulse", clock=clock)):
        for attr in DUCK:
            assert hasattr(h, attr), (type(h).__name__, attr)
        assert isinstance(h.state(), HolderState)
        assert isinstance(h.read(), Reading)


def test_no_holder(clock):
    h = connect_holder("none", clock=clock)
    assert isinstance(h, NoHolder) and h.channels == ()
    assert h.state(3.0) == HolderState(kind="none", t_s=3.0)
    with pytest.raises(HeaterUnavailable):
        h.set(100)


def test_step_response_first_order(clock):
    h = SimHeatingHolder(clock=clock, tau_s=0.005, noise_c=0.0)
    h.set(525.0)
    clock.advance(0.005)  # one time constant: 63.2 % of the step
    assert h.state().temperature_c == pytest.approx(25 + 500 * (1 - math.exp(-1)), rel=1e-6)
    clock.advance(0.1)
    assert h.state().temperature_c == pytest.approx(525.0, abs=1e-6)
    assert not h.busy


def test_ramp_rate_and_lag(clock):
    h = SimHeatingHolder(clock=clock, noise_c=0.0)
    h.ramp(425.0, rate_c_per_s=10.0)
    clock.advance(20.0)
    st = h.state()
    # setpoint at 225 degC, chip lags by rate*tau
    assert st.temperature_c == pytest.approx(225.0 - 10.0 * h.tau_s, abs=1e-6)
    assert h.busy
    clock.advance(30.0)
    assert h.state().temperature_c == pytest.approx(425.0, abs=1e-6)
    assert not h.busy
    h.ramp(25.0, seconds=40.0)
    clock.advance(20.0)
    assert h.state().temperature_c == pytest.approx(225.0, abs=0.1)


def test_fast_forward_is_exact_and_cheap(clock):
    h = SimHeatingHolder(clock=clock, noise_c=0.0)
    h.ramp(1000.0, rate_c_per_s=1.0)
    clock.advance(3600.0)
    assert h.state().temperature_c == pytest.approx(1000.0)


def test_overshoot(clock):
    h = SimHeatingHolder(clock=clock, noise_c=0.0, overshoot=0.1, tau_s=0.005)
    h.set(125.0)
    peak = 0.0
    for _ in range(200):
        clock.advance(0.0005)
        peak = max(peak, h.state().temperature_c)
    assert peak == pytest.approx(135.0, rel=0.02)  # 10 % of the 100 degC step
    clock.advance(1.0)
    assert h.state().temperature_c == pytest.approx(125.0, abs=1e-3)


def test_electrical_readout(clock):
    h = SimHeatingHolder(clock=clock, noise_c=0.0)
    cold = h.state()
    assert cold.resistance_ohm == pytest.approx(h.r0_ohm * (1 + h.alpha_per_c * 25))
    assert cold.power_w == pytest.approx(0.0, abs=1e-9)
    h.set(800)
    clock.advance(1.0)
    hot = h.state()
    assert hot.resistance_ohm == pytest.approx(500 * (1 + 0.003 * 800))
    assert 0.02 < hot.power_w < 0.05  # tens of mW, MEMS-like
    r = h.read()
    assert r.source == "heater" and r.current == pytest.approx(math.sqrt(r.power / r.resistance))
    assert r.settled
    with pytest.raises(ChannelUnsupported):
        h.set(1.0, channel="bias")


def test_bias_channel(clock):
    h = connect_holder("sim-biasing", clock=clock, noise_c=0.0)
    assert h.channels == ("temperature", "bias")
    h.set(2.0, channel="bias")
    cold = h.read()
    h.set(400)
    clock.advance(1)
    hot = h.read()
    assert cold.source == hot.source == "bias"
    assert hot.current > 5 * cold.current  # semiconductor sample conducts more hot
    st = h.state()
    assert st.kind == "heating_biasing" and st.bias_v == 2.0


def test_noise_is_deterministic(clock):
    a = SimHeatingHolder(clock=clock, seed=3)
    b = SimHeatingHolder(clock=clock, seed=3)
    clock.advance(1.234)
    ta, tb = a.state().temperature_c, b.state().temperature_c
    assert ta == tb and ta != 25.0 and abs(ta - 25.0) < 0.1


def test_stop_holds(clock):
    h = SimHeatingHolder(clock=clock, noise_c=0.0)
    h.ramp(300, rate_c_per_s=10)
    clock.advance(5)
    h.stop()
    clock.advance(100)
    assert h.state().temperature_c == pytest.approx(75.0, abs=0.1)
    h.flag("mark")
    assert h.flags[-1][1] == "mark"


def test_connect_holder_specs(clock):
    assert isinstance(connect_holder(None), NoHolder)
    assert isinstance(connect_holder("sim-heating", clock=clock), SimHeatingHolder)
    obj = object()
    assert connect_holder(obj) is obj
    with pytest.raises(ValueError):
        connect_holder("plasma")


class _FakeData:
    def __init__(self, d):
        self.d = d
        self.flags = []

    def getLastData(self):
        return self.d

    def setFlag(self, text):
        self.flags.append(text)


class _FakeChannel:
    def __init__(self, data):
        self.data = _FakeData(data)
        self.busy = False
        self.calls = []

    def set(self, v):
        self.calls.append(("set", v))

    def startRamp(self, v, kind, rate):
        self.calls.append(("ramp", v, kind, rate))

    def stopRamp(self):
        self.calls.append(("stop",))


def _fake_impulse(bias=False):
    mod = types.SimpleNamespace()
    mod.heat = _FakeChannel({"temperature": 301.5, "targetTemperature": 300.0, "power": 0.012,
                             "resistance": 950.0, "voltage": 3.4, "current": 0.0036})
    mod.bias = _FakeChannel({"voltage": 1.0, "current": 2e-6}) if bias else None
    mod.waitForControl = lambda: None
    mod.getStatus = lambda: "OK"
    mod.disconnect = lambda: None
    return mod


def test_impulse_follower_with_fake_module(clock):
    mod = _fake_impulse(bias=True)
    h = ImpulseFollower(mod, clock=clock)
    assert h.real and h.channels == ("temperature", "bias")
    r = h.read()
    assert r.measured == 301.5 and r.source == "bias" and r.current == 2e-6
    h.ramp(500, rate_c_per_s=5)
    h.set(1.5, channel="bias")
    h.stop()
    assert mod.heat.calls == [("ramp", 500.0, "rampRate", 5.0), ("stop",)]
    assert mod.bias.calls == [("set", 1.5)]
    st = h.state(7.0)
    assert st.temperature_c == 301.5 and st.t_s == 7.0 and st.kind == "heating_biasing"
    assert h.wait_for_control(1.0)
    h.flag("acq")
    assert mod.heat.data.flags == ["acq"]


def test_connect_impulse_uses_module(monkeypatch, clock):
    monkeypatch.setitem(sys.modules, "impulsePy", _fake_impulse())
    h = connect_holder("impulse", clock=clock)
    assert isinstance(h, ImpulseFollower) and h.channels == ("temperature",)
    with pytest.raises(ChannelUnsupported):
        h.set(1, channel="bias")


def test_impulse_missing_falls_back(monkeypatch, clock):
    monkeypatch.setitem(sys.modules, "impulsePy", None)  # import raises ImportError
    h = connect_holder("impulse", clock=clock)
    assert isinstance(h, SimHeatingHolder) and "impulsePy" in h.reason
    with pytest.raises(ImportError):
        connect_holder("impulse", clock=clock, strict=True)


def test_autopilot_planrun_drives_sim_holder(clock):
    impulse = pytest.importorskip("de_automate.impulse") if _autopilot_on_path() else pytest.skip(
        "de_autopilot not found")
    h = SimHeatingHolder(clock=clock, noise_c=0.0)
    plan = impulse.Plan.staircase(25, 125, steps=2, rate=10, hold_s=5)
    run = impulse.PlanRun(plan, h, start=25, now=clock.now())
    t = clock.now()
    while not run.finished and t < 100:
        run.tick(h.read(), now=t)
        clock.advance(0.5)
        t = clock.now()
    assert run.finished
    assert h.state().temperature_c == pytest.approx(125.0, abs=0.01)


def _autopilot_on_path() -> bool:
    import os

    path = r"C:\Users\CarterFrancis\PycharmProjects\de_autopilot"
    if os.path.isdir(path) and path not in sys.path:
        sys.path.append(path)
    try:
        import de_automate.impulse  # noqa: F401
        return True
    except Exception:
        return False


def test_a_hanging_impulsepy_import_times_out(monkeypatch):
    """impulsePy connects to Impulse at import and blocks for good when it is not running."""
    import threading

    from de_twin.holder import impulse as imp

    release = threading.Event()

    class Hang:
        def find_spec(self, name, path=None, target=None):
            if name == "impulsePy":
                release.wait(10)

    finder = Hang()
    monkeypatch.setattr(sys, "meta_path", [finder] + sys.meta_path)
    monkeypatch.delitem(sys.modules, "impulsePy", raising=False)
    monkeypatch.setattr(imp, "_IMPORT", {})
    try:
        with pytest.raises(TimeoutError):
            imp.import_impulse(timeout_s=0.2)
        with pytest.raises(TimeoutError):  # still stuck: fails at once
            imp.import_impulse(timeout_s=5)
        holder = connect_holder("impulse")
        assert not getattr(holder, "real", False) and "did not answer" in holder.reason
    finally:
        release.set()
