"""The column as an SEM in EBSD mode: what it reports, what it refuses, and how the twin
keeps it and the EBSD geometry in step, in process and over the DE-TEM-Channel face."""

from __future__ import annotations

import pytest

from de_twin import DigitalTwin
from de_twin.clock import ManualClock
from de_twin.column import Column, ColumnRefused, SoapTemChannelClient, is_sem
from de_twin.column.properties import SAMPLE_TILT_RANGE_DEG, WORKING_DISTANCE_RANGE_MM
from de_twin.ebsd import EbsdGeometry
from de_twin.faces.temchannel_soap import TemChannelServer


@pytest.fixture
def column():
    return Column(clock=ManualClock())


def test_a_tem_has_no_working_distance_or_sample_tilt(column):
    assert not column.sem and not is_sem(column.get("InstrumentType"))
    assert column.get("WorkingDistance") == -1.0
    assert column.get("SampleTilt") == -1.0
    with pytest.raises(ColumnRefused):
        column.set("WorkingDistance", 15.0)


@pytest.mark.parametrize(
    ("vendor", "reported"), [("JEOL", "JEOL SEM (EIK)"), ("FEI", "Thermo Fisher / FEI xT")]
)
def test_an_sem_reports_what_de_tem_channels_sem_backends_report(vendor, reported):
    column = Column(clock=ManualClock(), instrument_type=vendor)
    column.enter_sem(working_distance_mm=12.0, sample_tilt_deg=70.0)
    values = column.values()
    assert values["InstrumentType"] == reported and is_sem(reported)
    assert (values["TemStemMode"], values["ProjectionMode"], values["SubMode"]) == (1, 0, 0)
    assert (values["WorkingDistance"], values["SampleTilt"]) == (12.0, 70.0)


def test_an_sem_refuses_projector_writes_and_settings_out_of_range(column):
    column.enter_sem(working_distance_mm=15.0, sample_tilt_deg=70.0)
    for name, value in (("ProjectionMode", 2), ("SubMode", 4), ("TemStemMode", 0)):
        with pytest.raises(ColumnRefused):
            column.set(name, value)
    column.set("TemStemMode", 1)
    with pytest.raises(ColumnRefused):
        column.set("WorkingDistance", 0.0)
    with pytest.raises(ColumnRefused):
        column.enter_sem(working_distance_mm=15.0, sample_tilt_deg=95.0)
    assert column.get("SampleTilt") == 70.0


def test_leaving_the_sem_returns_the_tem_it_was(column):
    column.set("Magnification", 50000)
    column.enter_sem(working_distance_mm=15.0, sample_tilt_deg=70.0)
    column.leave_sem()
    assert column.get("InstrumentType") == "JEOL"
    assert (column.get("TemStemMode"), column.get("ProjectionMode")) == (0, 1)
    assert column.get("Magnification") == 50000


def test_the_column_takes_the_ranges_the_ebsd_geometry_documents():
    described = EbsdGeometry.describe()
    for field, limits in (
        ("working_distance_mm", WORKING_DISTANCE_RANGE_MM),
        ("sample_tilt_deg", SAMPLE_TILT_RANGE_DEG),
    ):
        assert (described[field]["lo"], described[field]["hi"]) == limits


def test_the_twins_column_is_an_sem_in_ebsd_mode_and_follows_the_geometry():
    twin = DigitalTwin(camera="Celeritas")
    assert not twin.column.sem
    twin.configure_ebsd(working_distance_mm=18.0, sample_tilt_deg=65.0)
    assert twin.column.sem
    assert (twin.column.get("WorkingDistance"), twin.column.get("SampleTilt")) == (18.0, 65.0)
    twin.column.set("WorkingDistance", 22.0)
    twin.column.set("SampleTilt", 70.0)
    assert (twin.ebsd.working_distance_mm, twin.ebsd.sample_tilt_deg) == (22.0, 70.0)
    twin.configure_ebsd(enabled=False)
    assert not twin.column.sem and twin.column.get("InstrumentType") == "JEOL"


def test_the_de_tem_channel_face_serves_the_sem():
    twin = DigitalTwin(camera="Celeritas")
    twin.configure_ebsd(working_distance_mm=15.0, sample_tilt_deg=70.0)
    server = TemChannelServer(twin.column, host="127.0.0.1", port=0).start()
    try:
        client = SoapTemChannelClient("127.0.0.1", server.port, timeout=5)
        batch = client.batch_dict(["INSTRUMENT_Type", "PROJ_TemStemMode", "PROJ_Mode", "PROJ_Defocus"])
        assert batch == {
            "INSTRUMENT_Type": "JEOL SEM (EIK)",
            "PROJ_TemStemMode": "1",
            "PROJ_Mode": "0",
            "PROJ_Defocus": "0.015",  # the working distance, in metres
        }
        assert float(client.call("getDeFocusValue", val=1)["floatResult"]) == pytest.approx(0.015)
        assert client.get_property("WorkingDistance")[0] == "15"
        assert client.get_property("SampleTilt")[0] == "70"
        assert client.call("setDefocus", **{"defocus-m": 0.02})["success"] == "1"
        assert twin.ebsd.working_distance_mm == pytest.approx(20.0)
        assert client.call("setProjectionMode", mode=2)["success"] == "0"
    finally:
        server.stop()
