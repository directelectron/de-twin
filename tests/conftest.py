"""Test configuration.

``RenderConfig.stem_model`` defaults to "auto", which picks the coherent 4D-STEM model for
small (<= 256^2) patterns or ptychography-like scans. The STEM tests written for the
kinematic model (disk caches, intensity jitter, probe-footprint blend, descan ramp, virtual
image speed) exercise that model on such geometries, so outside the coherent test modules
"auto" resolves to "kinematic" here. Explicit ``stem_model="coherent"`` is never overridden.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _auto_stem_model_is_kinematic_for_legacy_tests(request, monkeypatch):
    name = request.module.__name__.rsplit(".", 1)[-1]
    if name.startswith("test_coherent"):
        yield
        return
    from de_twin.render import coherent

    original = coherent.choose_model

    def choose(optics, cfg, sampling_fn):
        if str(cfg.stem_model).lower() == "auto":
            return "kinematic"
        return original(optics, cfg, sampling_fn)

    monkeypatch.setattr(coherent, "choose_model", choose)
    import de_twin.render.renderer as renderer

    monkeypatch.setattr(renderer, "choose_model", choose)
    yield


# ---------------------------------------------------------------- factories
@pytest.fixture
def make_twin():
    """Builds a ``DigitalTwin`` on a manual clock; ``magnification`` is set on its column."""
    from de_twin.clock import ManualClock
    from de_twin.twin import DigitalTwin

    def build(name="Dense Au on holey C", *, seed=1, magnification=None, camera="DESim", **twin_options):
        twin = DigitalTwin(name, camera=camera, clock=ManualClock(), seed=seed, **twin_options)
        if magnification is not None:
            twin.column.set("Magnification", magnification)
        return twin

    return build


@pytest.fixture
def make_realistic_twin(make_twin):
    """A twin whose column is as imperfect as a real one (or ideal with ``realistic=False``)."""
    from de_twin.optics import OpticsConfig

    def build(name="Dense Au on holey C", realistic=True, seed=7, magnification=20000.0):
        config = OpticsConfig.realistic(seed=seed) if realistic else OpticsConfig()
        return make_twin(name, magnification=magnification, optics_config=config)

    return build


@pytest.fixture
def make_optics():
    """Derives ``Optics`` from state attributes given as keywords (``stage_x`` / ``stage_y`` in um)."""
    from de_twin.optics import Calibration, OpticsConfig, derive_optics
    from de_twin.render.testing import StubCamera
    from de_twin.state import AcquisitionRequest, MicroscopeState

    def build(*, camera=None, config=None, request=None, **state):
        microscope = MicroscopeState()
        for key, value in state.items():
            if key in ("stage_x", "stage_y"):
                setattr(microscope.stage, key[-1] + "_um", value)
            else:
                setattr(microscope, key, value)
        return derive_optics(
            microscope,
            request or AcquisitionRequest(),
            camera or StubCamera(sensor_shape=(512, 512)),
            Calibration.default(),
            config or OpticsConfig(),
        )

    return build


@pytest.fixture
def make_stem_optics():
    """Optics for a STEM scan of ``size`` positions on a 256^2 detector region."""
    from de_twin.optics import Calibration, OpticsConfig, derive_optics
    from de_twin.render.testing import StubCamera
    from de_twin.state import AcquisitionRequest, MicroscopeState, Roi, TemStem

    def build(scan=True, size=(32, 32), step_um=0.002, camera_length_mm=13.0, config=None, **state):
        microscope = MicroscopeState()
        microscope.tem_stem = TemStem.STEM
        microscope.camera_length_mm = camera_length_mm
        microscope.convergence_semi_angle_mrad = 10.0
        for key, value in state.items():
            setattr(microscope, key, value)
        request = AcquisitionRequest(hw_roi=Roi(1920, 1920, 256, 256))
        request.scan.enabled = scan
        request.scan.size = size
        request.scan.step_um = step_um
        return derive_optics(
            microscope, request, StubCamera(), Calibration.default(), config or OpticsConfig()
        )

    return build


@pytest.fixture
def make_state():
    """A ``MicroscopeState`` with the given attributes set."""
    from de_twin.state import MicroscopeState

    def build(**attributes):
        state = MicroscopeState()
        for name, value in attributes.items():
            setattr(state, name, value)
        return state

    return build


@pytest.fixture
def make_derived_optics():
    """``derive_optics`` with a default state, request, camera, calibration and config."""
    from de_twin.optics import Calibration, OpticsConfig, derive_optics
    from de_twin.render.testing import StubCamera
    from de_twin.state import AcquisitionRequest, MicroscopeState

    def build(state=None, request=None, camera=None, config=None, calibration=None):
        return derive_optics(
            state or MicroscopeState(),
            request or AcquisitionRequest(),
            camera or StubCamera(),
            calibration or Calibration.default(),
            config or OpticsConfig(),
        )

    return build
