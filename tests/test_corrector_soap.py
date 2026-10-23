"""The corrector on the DE-TEM-Channel SOAP face, the SOAP client, MirrorColumn and deapi."""

import glob
import os
import re

import pytest

from de_twin.clock import Clock, ManualClock
from de_twin.column import Column, MirrorColumn, SoapTemChannelClient
from de_twin.column.ceos import ABSENT, DONE, FAILED, RUNNING, parse_aberration_set
from de_twin.column.soap import envelope, parse_body
from de_twin.faces.temchannel_soap import TemChannelServer
from de_twin.optics.aberrations import Aberrations

GSOAP = r"C:\direct_electron\DE-TEM-Channel\gsoapFiles"
CORRECTOR_OPS = (
    "getCorrectorPresent",
    "getCorrectorStatus",
    "getCorrectorInfo",
    "getCorrectorEndpoints",
    "getCorrectorError",
    "getCorrectorResult",
    "getAberrations",
    "getAberration",
    "getAlignmentFile",
    "correctorMeasureC1A1",
    "correctorAcquireTableau",
    "correctorCorrectAberration",
    "correctorFetchAlignmentFile",
    "correctorPutAlignmentFile",
    "correctorGetConfigOption",
    "correctorSetConfigOption",
    "correctorReadBack",
    "correctorReconnect",
    "setCorrectorBeamTilt",
    "getCorrectorBeamTilt",
)


def fast_column(corrector="probe", seed=1):
    """A real-time clock running 20x, so posted commands finish in wall tenths of a second."""
    col = Column(clock=Clock(time_scale=20.0), corrector=corrector, seed=seed)
    col.set_tem_stem(1)
    return col


@pytest.fixture
def corrected():
    col = fast_column()
    srv = TemChannelServer(col, host="127.0.0.1", port=0).start()
    yield col, srv, SoapTemChannelClient("127.0.0.1", srv.port, timeout=10)
    srv.stop()


def _handle(srv, method, **fields):
    code, xml = srv.handle(envelope(method, list(fields.items())))
    assert code == 200, xml
    return parse_body(xml)


# ------------------------------------------------------------------ absent
def test_absent_corrector_answers_not_present():
    srv = TemChannelServer(Column(clock=ManualClock()), port=0)
    assert _handle(srv, "getCorrectorPresent", val=0) == ("getintResponse", {"intResult": "0"})
    assert _handle(srv, "getCorrectorStatus", val=0)[1]["intResult"] == str(ABSENT)
    assert _handle(srv, "getCorrectorInfo", val=0) == ("getCorrectorInfoResponse", {"r": ""})
    assert _handle(srv, "getAberrations", val=0) == ("getAberrationsResponse", {"r": ""})
    assert _handle(srv, "getAberration", name="C1")[1] == {"x": "0", "y": "0", "status": "0"}
    for op in (
        "correctorMeasureC1A1",
        "correctorAcquireTableau",
        "correctorCorrectAberration",
        "setCorrectorBeamTilt",
        "correctorFetchAlignmentFile",
    ):
        assert _handle(srv, op) == ("setResult", {"success": "0"}), op
    # reconnect is the one command allowed without a corrector (it re-probes)
    assert _handle(srv, "correctorReconnect", val=0)[1]["success"] == "1"
    assert _handle(srv, "getCorrectorStatus", val=0)[1]["intResult"] == str(ABSENT)
    assert _handle(srv, "getCorrectorError", val=0)[1]["r"] == "no corrector responded"


# ------------------------------------------------------------------ recorded envelopes
def _samples():
    return sorted(
        p
        for p in glob.glob(os.path.join(GSOAP, "*.req.xml"))
        if any(f".{op}.req.xml" in p for op in CORRECTOR_OPS)
    )


@pytest.mark.skipif(not os.path.isdir(GSOAP), reason="DE-TEM-Channel gsoapFiles not available")
@pytest.mark.parametrize("corrector", [None, "probe", "image", "both"])
def test_replay_recorded_corrector_envelopes(corrector):
    col = Column(clock=ManualClock(), corrector=corrector, seed=2)
    srv = TemChannelServer(col, port=0)
    replayed = set()
    for req_path in _samples():
        res_path = req_path.replace(".req.xml", ".res.xml")
        code, xml = srv.handle(open(req_path, "rb").read())
        assert code == 200, (req_path, xml[:300])
        want_el, want = parse_body(open(res_path, "rb").read())
        got_el, got = parse_body(xml)
        assert got_el == want_el, (req_path, got_el, want_el)
        assert list(got) == list(want), (req_path, list(got), list(want))
        for tag, recorded in want.items():
            assert re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL), (req_path, tag)
            if recorded not in ("", None) and tag != "r":
                float(got[tag])
        replayed.add(os.path.basename(req_path).split(".")[1])
    assert replayed == set(CORRECTOR_OPS)


# ------------------------------------------------------------------ round trip
def test_soap_round_trip_through_the_client(corrected):
    col, _srv, c = corrected
    assert c.corrector_present()
    assert c.corrector_status() == 0
    info = c.corrector_info()
    assert info["correctorType"] == "CESCOR" and info["mode"] == "STEM"
    assert info["protocolVersion"] == "4" and "STEM" in info["currentLabel"]
    assert c.corrector_endpoints() == ["CESCOR STEM corrector at 127.0.0.1:7072 (protocol 4)"]
    assert c.aberrations() == []  # nothing measured yet

    # post, poll, fetch
    assert c.corrector_acquire_tableau("Fast")
    assert c.corrector_status() == RUNNING
    assert not c.corrector_measure_c1a1()  # one command at a time: refused while busy
    assert c.corrector_wait(10) == DONE
    items = c.aberrations()
    assert [n for n, _ in items] == ["C1", "A1", "A2", "B2"]  # CEOS order up to maxFit B2
    assert parse_aberration_set(c.corrector_result()) == items
    assert re.fullmatch(r"(\w\w=[-+.\deE]+,[-+.\deE]+;?)+", c.corrector_result())
    truth = col.corrector.get()
    x, y, known = c.aberration("B2")
    assert known and abs(complex(x, y) * 1e9 - truth["B2"]) < 30  # metres on the wire
    assert c.aberration("C3") == (0.0, 0.0, False)

    # synchronous correction (waitMs): the outcome in one round trip
    b2_before = abs(truth["B2"])
    col.corrector.set("B2", 300 + 0j)
    assert c.corrector_correct("B2", value=(300e-9, 0.0), wait_ms=3000)
    assert abs(col.corrector.get()["B2"]) < 0.15 * 300
    assert b2_before < 300
    # posted correction of something the corrector has no value for: accepted, then failed
    assert c.corrector_correct("A3")
    assert c.corrector_wait(10) == FAILED
    assert c.corrector_error() == "Server error: No value for aberration A3 available (-32000)"
    assert c.corrector_result() == ""


def test_soap_beam_tilt_config_and_alignment(corrected):
    col, _srv, c = corrected
    assert c.get_corrector_beam_tilt() == (0.0, 0.0, False)
    assert c.set_corrector_beam_tilt(1e-4, 0.0, relative=True, wait_ms=2000)
    assert c.set_corrector_beam_tilt(0.0, 2e-4, relative=True, wait_ms=2000)
    x, y, known = c.get_corrector_beam_tilt()
    assert known and (x, y) == pytest.approx((1e-4, 2e-4))
    assert col.corrector.served.wd == pytest.approx(1e-4 + 2e-4j)
    # waitMs=0: accepted and running; the commanded value only moves on completion
    assert c.set_corrector_beam_tilt(1e-4, 0.0, wait_ms=0)
    assert c.corrector_wait(10) == DONE

    assert c.corrector_get_config_option("detilt")
    assert c.corrector_wait(10) == DONE and c.corrector_result() == "true"
    assert c.corrector_set_config_option("detilt", False)
    assert c.corrector_wait(10) == DONE
    assert c.corrector_get_config_option("detilt") and c.corrector_wait(10) == DONE
    assert c.corrector_result() == "false"

    assert c.alignment_file() == ""
    assert c.corrector_fetch_alignment_file() and c.corrector_wait(10) == DONE
    data = c.alignment_file()
    assert '"correctorType": "CESCOR"' in data
    good = col.corrector.get()
    col.corrector.perturb("ht")
    assert c.corrector_put_alignment_file(data) and c.corrector_wait(10) == DONE
    assert abs((col.corrector.get() - good)["C3"]) < 100
    assert not c.corrector_put_alignment_file("")  # refused before posting
    assert c.corrector_read_back() and c.corrector_wait(10) == DONE
    assert c.corrector_reconnect() and c.corrector_wait(10) == DONE
    assert c.get_corrector_beam_tilt()[2] is False  # a reconnect forgets the commanded tilt


def test_image_side_served_when_asked():
    col = Column(clock=ManualClock(), corrector="both", seed=2)
    srv = TemChannelServer(col, port=0, corrector_side="image")
    info = _handle(srv, "getCorrectorInfo", val=0)[1]["r"]
    assert "correctorType=CETCOR;mode=TEM" in info
    eps = _handle(srv, "getCorrectorEndpoints", val=0)[1]["r"].splitlines()
    assert eps[0].endswith(":7071 (protocol 4)") and eps[1].endswith(":7072 (protocol 4)")
    # default: the probe corrector, like the channel's preferredPort 0
    srv = TemChannelServer(col, port=0)
    assert "mode=STEM" in _handle(srv, "getCorrectorInfo", val=0)[1]["r"]


def test_soap_tableau_arguments_and_errors():
    clk = ManualClock()
    col = Column(clock=clk, corrector="probe", seed=4)
    col.set_tem_stem(1)
    srv = TemChannelServer(col, port=0)
    ok = _handle(srv, "correctorAcquireTableau", tabType="Enhanced", angle=30.0, maxFit="C5")
    assert ok[1]["success"] == "1"
    assert _handle(srv, "getCorrectorStatus", val=0)[1]["intResult"] == str(RUNNING)
    clk.advance(60)
    assert _handle(srv, "getCorrectorStatus", val=0)[1]["intResult"] == str(DONE)
    names = [n for n, _ in parse_aberration_set(_handle(srv, "getAberrations", val=0)[1]["r"])]
    assert names[-1] == "C5" and "A5" not in names
    assert col.corrector.served.last_measurement.angle_mrad == 30.0
    _handle(srv, "correctorAcquireTableau", tabType="Huge", angle=0.0, maxFit="")
    clk.advance(1)
    assert _handle(srv, "getCorrectorStatus", val=0)[1]["intResult"] == str(FAILED)
    assert 'Invalid Tableau type "Huge"' in _handle(srv, "getCorrectorError", val=0)[1]["r"]
    # the probe corrector cannot measure in TEM mode
    col.set_tem_stem(0)
    _handle(srv, "correctorMeasureC1A1", val=0)
    clk.advance(3)
    assert _handle(srv, "getCorrectorStatus", val=0)[1]["intResult"] == str(FAILED)
    assert "STEM" in _handle(srv, "getCorrectorError", val=0)[1]["r"]


# ------------------------------------------------------------------ mirror
def test_mirror_reads_the_real_correctors_aberrations(corrected):
    col, srv, c = corrected
    col.set_defocus_um(0.02)  # the operator's focus is in the measured C1 ...
    m = MirrorColumn("127.0.0.1", srv.port, start=False, corrector_every=1)
    try:
        s = m.state()
        assert s.corrector == "probe" and s.probe_aberrations == {}  # nothing measured yet
        assert m.corrector_info["mode"] == "STEM"
        assert c.corrector_acquire_tableau("Standard") and c.corrector_wait(10) == DONE
        m.poll()
        s = m.state()
        truth = col.corrector.get()
        got = Aberrations(s.probe_aberrations)
        assert s.defocus_um == pytest.approx(0.02)
        assert got["C1"] == pytest.approx(truth["C1"], abs=5.0)  # ... and taken out again
        assert abs(got["B2"] - truth["B2"]) < 20 and abs(got["A2"] - truth["A2"]) < 20
        assert got["C3"].real == pytest.approx(truth["C3"].real, abs=2000)
        assert s.image_aberrations == {}
    finally:
        m.close()


def test_mirror_of_a_column_without_corrector():
    col = Column(clock=ManualClock())
    srv = TemChannelServer(col, host="127.0.0.1", port=0).start()
    try:
        m = MirrorColumn("127.0.0.1", srv.port, start=False, corrector_every=1)
        s = m.state()
        assert s.corrector == "none" and s.probe_aberrations == {}
        m.close()
    finally:
        srv.stop()


# ------------------------------------------------------------------ deapi
def test_deapi_exposes_corrector_state():
    deapi = pytest.importorskip("deapi")
    from test_faces_deapi import StubTwin

    from de_twin.faces.deapi_server import TwinDeapiServer

    twin = StubTwin()
    twin.column = Column(clock=twin.clock, corrector="probe", seed=1)
    srv = TwinDeapiServer(twin, port=0, pace=False).start()
    client = deapi.Client()
    client.usingMmf = False
    try:
        client.connect(port=srv.port)
        assert client["Instrument Corrector"] == "probe"
        assert 20 < client["Instrument Corrector Pi/4 Angle (mrad)"] < 35
        assert 0 < client["Instrument Corrector Residual B2 (nm)"] < 60
        assert 1 < client["Instrument Corrector Residual C5 (mm)"] < 4
        client.disconnect()
    finally:
        srv.stop()


# ------------------------------------------------------------------ CLI
def test_cli_builds_a_corrected_twin(monkeypatch):
    from de_twin import cli

    captured = {}

    class FakeTwin:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr("de_twin.twin.DigitalTwin", FakeTwin)
    args = cli.argparse.Namespace(
        specimen="x",
        camera="DE16",
        seed=3,
        holder="none",
        time_scale=1.0,
        mirror_temchannel=None,
        corrector="both",
        corrector_cold=False,
    )
    cli._build_twin(args)
    col = captured["column"]
    assert col.corrector.kind == "both" and col.seed == 3
