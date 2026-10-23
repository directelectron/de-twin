"""TEM-Channel SOAP face: client round trips, de_microscope, recorded samples, mirror."""

import glob
import os
import re
import sys
import time
import urllib.request

import pytest

from de_twin.clock import ManualClock
from de_twin.column import (
    BATCH_KEYS,
    DESERVER_BATCH,
    Column,
    MirrorColumn,
    SoapTemChannelClient,
    state_from_batch,
)
from de_twin.column.soap import PROPERTIES, parse_body, split_batch
from de_twin.faces.temchannel_soap import TemChannelServer
from de_twin.state import ProbeMode, Projection, TemStem

SAMPLE_DIRS = [
    r"C:\direct_electron\DE-TEM-Channel\gsoapFiles",
    r"C:\direct_electron\DE-TEM-Channel\docs\soap-samples",
    r"C:\direct_electron\virtual-specimen\code\source\CameraServer\Instrument\gsoap",
]
DE_MICROSCOPE = r"C:\Users\CarterFrancis\PycharmProjects\de_microscope"


@pytest.fixture
def column():
    return Column(stage_speed_um_s=1e5, tilt_speed_deg_s=1e4)


@pytest.fixture
def server(column):
    srv = TemChannelServer(column, host="127.0.0.1", port=0).start()
    yield srv
    srv.stop()


@pytest.fixture
def client(server):
    return SoapTemChannelClient("127.0.0.1", server.port, timeout=5)


# --------------------------------------------------------------------- batch


def test_batch_formats(server, column, client):
    column.set_stage(x=12.5, y=-3.25, z=1.0, alpha=10.0)
    column.wait_idle()
    column.set("ImageShift", (0.5, -0.125))
    d = client.batch_dict(BATCH_KEYS)
    assert d["GUN_HT"] == "200000"  # volts
    assert d["PROJ_Magnification"] == "20000"
    assert d["PROJ_CameraLength"] == "25"  # cm
    assert d["STAGE_PositionFull"] == "12.5;-3.25;1;10;0"
    assert d["ILLUM_ImageShift"] == "0.5;-0.125"
    assert d["ILLUM_ConvergenceAngle"] == "0.080"  # parallel TEM illumination, alpha 3
    assert d["INSTRUMENT_Type"] == "JEOL"
    assert d["STAGE_Position"] == ""  # never implemented by the channel
    assert d["PROJ_ImageShift"] == "0" and d["STAGE_Status"] == "0"
    column.set("TemStemMode", 1)
    column.set("Magnification", 1.2e6)
    assert client.execute_batch(["PROJ_Magnification", "PROJ_SubMode"]) == ["1.2e+06", "2"]
    assert client.batch_dict(["ILLUM_ConvergenceAngle"])["ILLUM_ConvergenceAngle"] == "22.000"  # STEM table


def test_batch_answers_what_deserver_accepts(server, client):
    """DE-Server only accepts an answer that splits into exactly totalCommands fields."""
    out = client.call(
        "executeBatchCommands", commands=",".join(DESERVER_BATCH), totalCommands=len(DESERVER_BATCH)
    )["r"]
    fields = split_batch(out, len(DESERVER_BATCH))
    assert fields is not None and all(f != "" for f in fields)
    # totalCommands beyond what was sent is clamped, unknown keys answer empty
    out = client.call("executeBatchCommands", commands="GUN_HT,NOPE", totalCommands=9)["r"]
    assert out == "200000,"


def test_parse_batch_like_deserver(server, column, client):
    column.set("ProjectionMode", 2)
    column.set("CameraLength", 40)
    column.set("ProbeMode", 2)
    column.set("Defocus", -2.5)
    state, info = state_from_batch(client.batch_dict())
    assert state.projection == Projection.DIFFRACTION and state.mag_mode == "DIFF"
    assert state.camera_length_mm == 400.0
    assert state.probe_mode == ProbeMode.NBD
    assert state.defocus_um == -2.5
    assert state.ht_kv == 200.0
    assert info["raw_sub_mode"] == 4


# ------------------------------------------------------------ client round trip

ROUNDTRIP = {
    "Magnification": (50000.0, 50000.0),
    "CameraLength": None,  # diffraction only, tested separately
    "Defocus": (-1.25, -1.25),
    "SpotSize": (2, 2),
    "ProbeMode": (1, 1),
    "AlphaSelector": (2, 2),
    "CondenserApertureIndex": (3, 3),
    "Intensity": (0.625, 0.625),
    "ImageShift": ((0.25, -0.5), (0.25, -0.5)),
    "BeamShift": ((1.5, 2.0), (1.5, 2.0)),
    "BeamTilt": ((0.1, -0.2), (0.1, -0.2)),
    "ObjectiveStig": ((0.05, 0.01), (0.05, 0.01)),
    "CondenserStig": ((-0.3, 0.3), (-0.3, 0.3)),
    "DiffractionShift": ((0.25, 0.125), (0.25, 0.125)),
    "StagePosition": (
        {"x": 5.0, "y": -7.0, "z": 1.0, "a": 3.0, "b": -2.0},
        {"x": 5.0, "y": -7.0, "z": 1.0, "a": 3.0, "b": -2.0},
    ),
    "StageX": (11.0, 11.0),
    "StageY": (-4.0, -4.0),
    "StageZ": (2.0, 2.0),
    "StageA": (1.0, 1.0),
    "StageB": (-1.0, -1.0),
    "TemStemMode": (1, 1),
    "ProjectionMode": (1, 1),
    "SubMode": (2, 2),
    "BeamBlank": (1, 1),
    "ScreenPosition": (1, 1),
    "ColumnValvesOpen": (0, 0),
    "StageMode": (1, 1),
    "ScanRotation": (12.5, 12.5),
    "DiffractionFocus": (0.75, 0.75),
    "GunShift": ((0.1, 0.2), (0.1, 0.2)),
    "GunTilt": ((0.3, 0.4), (0.3, 0.4)),
    "MagnificationIndex": None,
}


def test_every_property_roundtrips_through_the_server(server, column, client):
    for name, (read_spec, set_spec, _) in PROPERTIES.items():
        if read_spec is not None:
            client.get(name)  # every getter answers and parses
        if set_spec is None or ROUNDTRIP.get(name) is None:
            continue
        value, expect = ROUNDTRIP[name]
        if name == "TemStemMode":
            continue  # changes the mode for everything else; tested below
        assert client.set(name, value), name
        column.wait_idle()
        got = client.get(name)
        if isinstance(expect, dict):
            assert got == pytest.approx(expect), name
        elif isinstance(expect, tuple):
            assert tuple(got) == pytest.approx(expect), name
        else:
            assert got == pytest.approx(expect), name
    assert client.status("StageX") == 2  # Done


def test_modes_and_camera_length(server, column, client):
    assert client.set("ProjectionMode", 2)
    assert client.set("CameraLength", 80)
    assert client.get("CameraLength") == 80
    assert not client.set("Magnification", 50000)  # refused in TEM diffraction
    assert client.set("TemStemMode", 1)
    assert client.get("TemStemMode") == 1 and client.get("SubMode") == 4
    assert column.state().tem_stem == TemStem.STEM
    assert client.set("MagnificationIndex", 3)
    assert client.get("Magnification") == 5000


def test_service_range_checks_and_refusals(server, client):
    assert not client.set("BeamTilt", (1.5, 0))
    assert not client.set("ObjectiveStig", (0, -2))
    assert client.status("BeamTilt") in (0, 2, 3)
    assert not client.set("ScreenPosition", 5)
    assert client.get_property("BeamBlank") == ("0", 0)  # never set -> Idle


def test_stage_status_running_then_done():
    clock = ManualClock()
    col = Column(clock=clock)
    with TemChannelServer(col, host="127.0.0.1", port=0) as srv:
        c = SoapTemChannelClient("127.0.0.1", srv.port)
        assert c.set_stage_position(x=100.0)
        assert c.status("StageX") == 1 and c.operation_status(0) == 1
        assert c.batch_dict(["OP_Status"])["OP_Status"] == "1"
        assert c.status("StageY") == 0  # not commanded
        clock.advance(0.5)
        assert c.get("StageX") == pytest.approx(50.0)
        clock.advance(1.0)
        assert c.status("StageX") == 2 and c.get("StageX") == 100.0
        assert c.set_stage_position(y=500.0)
        assert c.stop()
        assert c.operation_status(0) == 4


def test_fei_vendor_on_the_wire():
    col = Column(instrument_type="FEI")
    with TemChannelServer(col, host="127.0.0.1", port=0) as srv:
        c = SoapTemChannelClient("127.0.0.1", srv.port)
        d = c.batch_dict(["INSTRUMENT_Type", "PROJ_SubMode", "ILLUM_ProbeMode", "PROJ_ScreenPosition"])
        assert d == {
            "INSTRUMENT_Type": "FEI",
            "PROJ_SubMode": "2",
            "ILLUM_ProbeMode": "1",
            "PROJ_ScreenPosition": "2",
        }
        state, _ = state_from_batch(c.batch_dict())
        assert state.probe_mode == ProbeMode.MICROPROBE and state.screen_position == 0


def test_http_details(server):
    url = f"http://127.0.0.1:{server.port}/"
    with urllib.request.urlopen(url, timeout=5) as r:
        assert r.status == 200
    bad = urllib.request.Request(url, data=b"<not-soap/>", headers={"Content-Type": "text/xml"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(bad, timeout=5)
    assert ei.value.code == 500


# -------------------------------------------------------------- de_microscope


def _de_microscope():
    if not os.path.isdir(DE_MICROSCOPE):
        pytest.skip("de_microscope not found")
    if DE_MICROSCOPE not in sys.path:
        sys.path.insert(0, DE_MICROSCOPE)
    try:
        import de_microscope
    except Exception as exc:
        pytest.skip(f"de_microscope not importable: {exc}")
    return de_microscope


def test_de_microscope_client(server, column):
    dm = _de_microscope()
    scope = dm.Microscope("127.0.0.1", port=server.port, timeout=5)
    errors = {}
    for name, prop in dm.PROPERTIES.items():
        if prop.get is None:
            continue
        try:
            scope[name]
        except Exception as exc:
            errors[name] = exc
    assert not errors, errors
    assert scope["InstrumentType"] == "JEOL"
    assert scope.vendor == "JEOL"
    scope["Magnification"] = 60000  # waits on getProperty status (ASYNC_SETTABLE)
    assert scope["Magnification"] == 60000
    scope["ImageShift"] = (0.5, -0.5)
    assert scope["ImageShift"] == (0.5, -0.5)
    scope["StagePosition"] = {"x": 25.0, "y": 5.0}
    assert scope.wait("StageX") == 2
    assert scope["StageX"] == 25.0 and scope["StageY"] == 5.0
    scope["Defocus"] = -3.0
    assert column.state().defocus_um == -3.0
    scope["SpotSize"] = 4
    assert column.state().spot_size == 4
    assert scope.screen_positions() == {"up": 0, "down": 1}
    with pytest.raises(dm.MicroscopeError):
        scope["BeamTilt"] = (5, 5)
    assert scope.stop()


def test_de_microscope_real_column(server, column):
    dm = _de_microscope()
    from de_microscope.column import RealColumn

    rc = RealColumn(dm.Microscope("127.0.0.1", port=server.port, timeout=5), "twin")
    rc.move_stage(40.0, -10.0)
    assert rc.stage_xy() == (40.0, -10.0)
    rc.tilt_to(5.0)
    assert rc.tilt() == 5.0
    vals = rc.values()
    assert vals["InstrumentType"] == "JEOL" and vals["Magnification"] == 20000


# ------------------------------------------------------------ recorded samples


def _samples():
    out = []
    for d in SAMPLE_DIRS:
        out += sorted(glob.glob(os.path.join(d, "*.req.xml")))
    return out


@pytest.mark.skipif(not _samples(), reason="no recorded SOAP samples on this machine")
def test_replay_recorded_samples(column):
    srv = TemChannelServer(column, port=0)  # handle() needs no socket
    replayed = 0
    for req_path in _samples():
        res_path = req_path.replace(".req.xml", ".res.xml")
        if not os.path.exists(res_path):
            continue
        request = open(req_path, "rb").read()
        code, xml = srv.handle(request)
        assert code == 200, (req_path, xml[:300])
        want_el, want = parse_body(open(res_path, "rb").read())
        got_el, got = parse_body(xml)
        assert got_el == want_el, (req_path, got_el, want_el)
        assert list(got) == list(want), (req_path, list(got), list(want))
        for tag, recorded in want.items():
            # the regex extraction de_microscope uses finds every recorded tag
            assert re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL), (req_path, tag)
            if recorded not in ("", None) and tag != "r":
                float(got[tag])  # numeric where the sample is numeric
        replayed += 1
    assert replayed >= 70


# ---------------------------------------------------------------------- mirror


def test_mirror_follows_and_forwards():
    a = Column(stage_speed_um_s=1e5)
    with TemChannelServer(a, host="127.0.0.1", port=0) as srv:
        mirror = MirrorColumn("127.0.0.1", srv.port, period_s=0.05)
        try:
            # changes on A appear in the mirror
            a.set("Magnification", 80000)
            a.set_stage(x=33.0, y=-12.0)
            a.wait_idle()
            a.set("ProbeMode", 3)
            a.set("BeamShift", (0.2, 0.3))
            deadline = time.time() + 3
            while time.time() < deadline:
                s = mirror.state()
                if s.magnification == 80000 and s.stage.x_um == 33.0:
                    break
                time.sleep(0.05)
            s = mirror.state()
            assert s.magnification == 80000.0
            assert (s.stage.x_um, s.stage.y_um) == (33.0, -12.0)
            assert s.probe_mode == ProbeMode.CBD
            assert (s.beam_shift_um.x, s.beam_shift_um.y) == (0.2, 0.3)
            assert s.convergence_semi_angle_mrad == pytest.approx(a.convergence_mrad())
            # sets on the mirror reach A (and read back immediately)
            mirror.set("Defocus", -4.0)
            assert a.state().defocus_um == -4.0 and mirror.state().defocus_um == -4.0
            mirror.set_stage(x=1.0, y=2.0)
            assert mirror.wait_idle(5)
            assert a.stage_position()[:2] == (1.0, 2.0)
            assert mirror.get("StageX") == 1.0
            mirror.set_function_mode("DIFF")
            assert a.state().projection == Projection.DIFFRACTION
            mirror.set_camera_length_mm(800)
            assert a.get("CameraLength") == 80.0
            with pytest.raises(Exception):
                mirror.set("BeamTilt", (4, 4))  # the channel refuses
            changes = []
            mirror.on_change(lambda n, v: changes.append(n))
            a.set("SpotSize", 1)
            time.sleep(0.3)
            assert "SpotSize" in changes
            assert mirror.connected and mirror.polls > 2
        finally:
            mirror.close()


def test_mirror_of_unreachable_channel_does_not_raise():
    m = MirrorColumn("127.0.0.1", 1, period_s=10, timeout=0.2, start=False)
    assert not m.connected
    assert m.state().instrument_type == "JEOL"
