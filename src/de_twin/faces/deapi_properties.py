"""The properties a deapi client reads and writes: deapi's table plus the twin's and the instrument's."""

from __future__ import annotations

import logging
import time

import numpy as np

from .deapi_compat import DynamicProperty
from .deapi_protocol import (
    _PROJECT_NAMES,
    DEFAULT_EXPOSURE_S,
    EBSD_PROPERTIES,
    TEMPERATURE_PROP,
    _property_key,
    _to_bool,
)
from .deapi_request import (
    OUTPUT_BUFFER_PROPERTY,
    SCAN_COIL_PROPERTIES,
    SCAN_FLIP_PROPERTIES,
    SIMULATOR_SOURCE_PROPERTY,
    model_counting,
)

log = logging.getLogger(__name__)


class PropertiesMixin:
    """The properties a deapi client reads and writes: deapi's table plus the twin's and the instrument's."""

    def _install_properties(self, default_fps: float) -> None:
        model = self.twin.detector.model
        h, w = (int(v) for v in model.sensor_shape)
        max_fps = float(getattr(model, "max_fps", 0) or 0) or (
            1.0 / float(getattr(model, "full_frame_time_s", 0.01) or 0.01)
        )
        v = self._values
        v["camera_name"]._value = str(model.name)
        for axis, n in (("x", w), ("y", h)):
            v[f"sensor_size_{axis}_pixels"]._value = str(n)
            v[f"sensor_size_{axis}_pixels"].default_value = str(n)
            v[f"hardware_roi_size_{axis}"]._value = str(n)
            v[f"hardware_roi_size_{axis}"].options = f"1.0, {float(n)}, '1 - {n}'"
            v[f"hardware_roi_offset_{axis}"].options = f"0.0, {float(n - 1)}, '0 - {n - 1}'"
        v["exposure_mode"].options = "'Dark', 'Gain', 'Normal*', 'Trial', 'Vacuum'"
        # hardware binning: X also sets Y (deapi's set_also passes a sympy float that its
        # own Set check then rejects, so the coupling is done here)
        self._hw_bin = [1, 1]

        def set_bin(axis):
            def setter(value):
                b = int(float(value))
                if b not in (1, 2, 4):
                    log.warning("hardware binning %s not supported", value)
                    return
                if axis == 0:
                    self._hw_bin[:] = [b, b]
                else:
                    self._hw_bin[1] = b

            return setter

        for i, axis in enumerate("XY"):
            self._put(
                DynamicProperty(
                    f"Hardware Binning {axis}",
                    (lambda i=i: self._hw_bin[i]),
                    set_bin(i),
                    data_type="Integer",
                    value_type="Set",
                    options="'1*', '2'",
                    default_value="1",
                    category="Basic",
                )
            )
        # exposure: frames per second / frame count / exposure time stay consistent
        self._fps = min(float(default_fps), max_fps * 10)
        # A 0.5 s exposure to begin with, not one frame: a DE frame is sparse by
        # design (a pixel holds a few electrons), and an image is the sum of frames
        # over the exposure. One frame per image reads as pure noise.
        self._frame_count = max(1, round(DEFAULT_EXPOSURE_S * self._fps))

        def set_fps(value):
            exposure = self._frame_count / self._fps
            self._fps = min(max(float(value), 0.06), 1e6)
            self._frame_count = max(1, round(exposure * self._fps))

        def set_count(value):
            self._frame_count = max(1, int(float(value)))

        def set_exposure(value):
            self._frame_count = max(1, round(float(value) * self._fps))

        self._put(
            DynamicProperty(
                "Frames Per Second",
                lambda: self._fps,
                set_fps,
                data_type="Float",
                value_type="Range",
                options="0.06, 1000000.0, '0.06 - 1000000.0'",
                category="Basic",
            )
        )
        self._put(
            DynamicProperty(
                "Frame Count",
                lambda: self._frame_count,
                set_count,
                data_type="Integer",
                value_type="Range",
                options="1.0, 100000000.0, '1 - 100000000'",
                category="Basic",
            )
        )
        self._put(
            DynamicProperty(
                "Exposure Time (seconds)",
                lambda: self._frame_count / self._fps,
                set_exposure,
                data_type="Float",
                value_type="Range",
                options="0.0, 1000000.0, '0.0 - 1000000.0'",
                category="Basic",
            )
        )
        # twin extras
        self._put(
            DynamicProperty(
                TEMPERATURE_PROP,
                self._get_temperature,
                self._set_temperature,
                data_type="Float",
                category="Simulator",
            )
        )
        self._put(
            DynamicProperty(
                "Simulator Virtual Specimen",
                lambda: self._specimen_name,
                self._set_specimen,
                category="Simulator",
            )
        )
        for name, value in SCAN_COIL_PROPERTIES.items():
            self._plain(name, value, "Float", category="Scan")
        self._plain(OUTPUT_BUFFER_PROPERTY, 0.0, "Float", category="Scan")
        for name in SCAN_FLIP_PROPERTIES:
            self._plain(name, "Off", category="Scan")
        clock = getattr(self.twin, "clock", None)
        self._put(
            DynamicProperty(
                "Twin Clock (seconds)",
                lambda: clock.now() if hasattr(clock, "now") else time.time(),
                data_type="Float",
            )
        )
        self._put(DynamicProperty("Twin Holder", lambda: self._holder_state().kind))
        self.auto_references = True
        self._put(
            DynamicProperty(
                "Simulator Auto References",
                lambda: "On" if self.auto_references else "Off",
                lambda v: setattr(self, "auto_references", _to_bool(v)),
                category="Simulator",
                options="On, Off",
            )
        )
        self._install_ebsd_properties()
        self._install_instrument_properties()
        self.virtual_masks = [np.ones((self._image_shape()), np.int8) for _ in range(5)]

    def _install_ebsd_properties(self) -> None:
        """``Simulator EBSD ...``: the twin's EBSD geometry (``DigitalTwin.configure_ebsd``), so a
        deapi client can switch the twin to an SEM-EBSD system and set its pattern centre. A
        real DE-Server has none of these: its metadata carries no EBSD geometry."""
        twin = self.twin
        if not callable(getattr(twin, "configure_ebsd", None)):
            return
        from ..ebsd import EbsdGeometry

        def current():
            return twin.ebsd or EbsdGeometry(enabled=False)

        def setter(field):
            def set_field(value):
                if field == "enabled" or isinstance(getattr(EbsdGeometry, field, None), bool):
                    value = _to_bool(value)
                twin.configure_ebsd(current(), **{field: value})

            return set_field

        descriptions = EbsdGeometry.describe()
        for field, label in EBSD_PROPERTIES.items():
            if isinstance(descriptions[field]["default"], bool):
                self._put(
                    DynamicProperty(
                        f"Simulator EBSD {label}",
                        (lambda field=field: "On" if getattr(current(), field) else "Off"),
                        setter(field),
                        category="Simulator",
                        options="On, Off",
                    )
                )
            else:
                self._put(
                    DynamicProperty(
                        f"Simulator EBSD {label}",
                        (lambda field=field: getattr(current(), field)),
                        setter(field),
                        data_type="Float",
                        category="Simulator",
                    )
                )

        def set_beam_kv(value):
            twin.configure_ebsd(current(), beam_kv=float(value))

        # user master pattern: a file on the twin's machine (deapi properties are strings: no upload)
        self._ebsd_master_phase = "Nickel"
        self._ebsd_master_status = "kinematic"

        def set_master_file(value):
            path = str(value).strip()
            try:
                if path:
                    master = twin.set_master_pattern(self._ebsd_master_phase, path)
                    self._ebsd_master_status = (
                        f"{self._ebsd_master_phase}: {path} at {master.kv:g} kV, {master.nbytes / 1e6:.1f} MB"
                    )
                else:
                    twin.clear_master_patterns(self._ebsd_master_phase)
                    self._ebsd_master_status = "kinematic"
            except Exception as exc:  # reported through the status property
                self._ebsd_master_status = f"error: {exc}"
                log.warning("EBSD master pattern %s: %s", path, exc)

        self._put(
            DynamicProperty(
                "Simulator EBSD Master Pattern Phase",
                lambda: self._ebsd_master_phase,
                lambda v: setattr(self, "_ebsd_master_phase", str(v)),
                category="Simulator",
            )
        )
        self._put(
            DynamicProperty(
                "Simulator EBSD Master Pattern File", lambda: "", set_master_file, category="Simulator"
            )
        )
        self._put(
            DynamicProperty(
                "Simulator EBSD Master Pattern Status", lambda: self._ebsd_master_status, category="Simulator"
            )
        )
        self._put(
            DynamicProperty(
                "Simulator EBSD Beam Energy (kV)",
                lambda: twin.column.state().ht_kv,
                set_beam_kv,
                data_type="Float",
                category="Simulator",
            )
        )

    def _install_instrument_properties(self) -> None:
        col = self.twin.column

        def st():
            return col.state()

        def g(name):
            return lambda: col.get(name)

        from ..column.soap import DESERVER_BATCH
        from .temchannel_soap import TemChannelServer

        batcher = TemChannelServer(col, port=0)  # formatter only, never started

        def metadata():
            fields = []
            for key in DESERVER_BATCH:
                val = batcher.batch_field(key)
                if not val:
                    continue
                fields.append(f"{key}={val.replace(';', ',')}")
            return ";".join(fields)

        def project_name():
            s = st()
            return _PROJECT_NAMES.get((col.function_mode, int(s.tem_stem)), "Unknown")

        def binning(axis):
            # DE-Server reports pixel sizes per *binned* pixel: hardware x software binning
            b = 1.0
            for name in (f"Hardware Binning {axis}", f"Binning {axis}"):
                try:
                    b *= float(self[name] or 1)
                except Exception:
                    pass
            return b

        def pixel_nm(axis="X"):
            try:
                px = float(self.twin.calibration.specimen_pixel_nm(st(), self.twin.detector.model))
            except Exception:
                return -1.0
            return px * binning(axis)

        def recip_px():
            try:
                return float(self.twin.calibration.recip_pixel_inv_nm(st(), self.twin.detector.model))
            except Exception:
                return -1.0

        ro = [
            ("Instrument Metadata", metadata, "String"),
            ("Instrument Project Name", project_name, "String"),
            ("Instrument Project Mode", g("ProjectionMode"), "Integer"),
            ("Instrument Project Sub Mode", g("SubMode"), "Integer"),
            ("Instrument Project Magnification", g("Magnification"), "Float"),
            ("Instrument Project Camera Length (centimeters)", g("CameraLength"), "Float"),
            ("Instrument Project TEMorSTEM Mode", g("TemStemMode"), "Integer"),
            ("Instrument Accelerating Voltage (V)", g("HighTension"), "Integer"),
            ("Instrument Screen Position", g("ScreenPosition"), "Integer"),
            ("Instrument Beam Blanking", g("BeamBlank"), "Integer"),
            ("Instrument Stage Position X (micrometers)", g("StageX"), "Float"),
            ("Instrument Stage Position Y (micrometers)", g("StageY"), "Float"),
            ("Instrument Stage Position Z (micrometers)", g("StageZ"), "Float"),
            ("Instrument Stage Tilt Alpha (degrees)", g("StageA"), "Float"),
            ("Instrument Stage Tilt Beta (degrees)", g("StageB"), "Float"),
            ("Instrument Beam Shift X (micrometers)", lambda: col.get("BeamShift")[0], "Float"),
            ("Instrument Beam Shift Y (micrometers)", lambda: col.get("BeamShift")[1], "Float"),
            ("Instrument Image Shift X (micrometers)", lambda: col.get("ImageShift")[0], "Float"),
            ("Instrument Image Shift Y (micrometers)", lambda: col.get("ImageShift")[1], "Float"),
            ("Instrument Defocus (micrometers)", g("Defocus"), "Float"),
            ("Instrument Spot Size", g("SpotSize"), "Integer"),
            ("Instrument Probe Mode", lambda: st().probe_mode.name, "String"),
            ("Instrument Convergence Semi-Angle (mrad)", g("ConvergenceAngle"), "Float"),
            ("Instrument Alpha Selector", g("AlphaSelector"), "Integer"),
            ("Instrument Condenser Aperture Index", g("CondenserApertureIndex"), "Integer"),
            ("Instrument Type", g("InstrumentType"), "String"),
            ("Specimen Pixel Size X (nanometers)", lambda: pixel_nm("X"), "Float"),
            ("Specimen Pixel Size Y (nanometers)", lambda: pixel_nm("Y"), "Float"),
            ("Diffraction Pixel Size X", recip_px, "Float"),
            ("Diffraction Pixel Size Y", recip_px, "Float"),
        ]

        # [twin] aberration corrector (read only): which side(s), and the residual of the
        # probe side (image side when only an image corrector is fitted). Ground truth,
        # from MicroscopeState, so a MirrorColumn shows the real corrector's last reading.
        def corr_ab():
            s = st()
            side = "image" if s.corrector == "image" else "probe"
            return s, side, (s.image_aberrations if side == "image" else s.probe_aberrations)

        def corr_mag(name, scale=1.0):
            def get():
                _, _, ab = corr_ab()
                return abs(complex(ab.get(name, 0j))) * scale

            return get

        def corr_pi4():
            s, _, ab = corr_ab()
            if not ab:
                return 0.0
            from ..column.ceos import pi4_angle_mrad
            from ..optics.aberrations import Aberrations

            return pi4_angle_mrad(Aberrations(ab), s.ht_kv)

        ro += [
            ("Instrument Corrector", lambda: st().corrector, "String"),
            ("Instrument Corrector Residual C1 (nm)", corr_mag("C1"), "Float"),
            ("Instrument Corrector Residual A1 (nm)", corr_mag("A1"), "Float"),
            ("Instrument Corrector Residual B2 (nm)", corr_mag("B2"), "Float"),
            ("Instrument Corrector Residual A2 (nm)", corr_mag("A2"), "Float"),
            ("Instrument Corrector Residual C3 (um)", corr_mag("C3", 1e-3), "Float"),
            ("Instrument Corrector Residual S3 (um)", corr_mag("S3", 1e-3), "Float"),
            ("Instrument Corrector Residual A3 (um)", corr_mag("A3", 1e-3), "Float"),
            ("Instrument Corrector Residual C5 (mm)", corr_mag("C5", 1e-6), "Float"),
            ("Instrument Corrector Pi/4 Angle (mrad)", corr_pi4, "Float"),
        ]
        for name, getter, dt in ro:
            self._put(DynamicProperty(name, getter, data_type=dt, category="Instrument"))
        # Counting cameras: DE-Server's "Centroiding Mode" (Standard | Super-resolution).
        # Super-resolution reads out 2x the ROI each way, and the image size says so.
        if model_counting(self.twin):
            self._centroiding = "Standard"

            def set_centroiding(value):
                v = str(value).strip()
                if v.lower() not in ("standard", "super-resolution"):
                    log.warning("centroiding mode %s not supported", value)
                    return
                self._centroiding = "Super-resolution" if "super" in v.lower() else "Standard"

            self._put(
                DynamicProperty(
                    "Centroiding Mode",
                    lambda: self._centroiding,
                    set_centroiding,
                    data_type="String",
                    value_type="Set",
                    options="'Standard*', 'Super-resolution'",
                    default_value="Standard",
                    category="Basic",
                )
            )
            for axis in ("x", "y"):
                orig = self._values.get(f"image_size_{axis}_pixels")
                if orig is None:
                    continue

                def size(orig=orig):
                    n = int(float(orig.value))
                    return (
                        2 * n
                        if (self._centroiding == "Super-resolution" and tuple(self._hw_bin) == (1, 1))
                        else n
                    )

                self._put(DynamicProperty(orig.name, size, data_type="Integer", category=orig.category))

        # Provenance: frames from here are simulated, and by what
        from .. import __version__

        self._put(
            DynamicProperty(
                SIMULATOR_SOURCE_PROPERTY,
                lambda: f"de-twin {__version__}",
                data_type="String",
                category="Twin",
            )
        )

    def _holder_state(self):
        holder = self.twin.holder
        fn = getattr(holder, "state", None)
        if callable(fn):
            return fn()
        from ..state import HolderState

        return HolderState()

    def _get_temperature(self) -> float:
        return float(self._holder_state().temperature_c)

    def _set_temperature(self, value) -> None:
        """Written by autopilot's ServerFurnace: the twin's holder follows it.
        A holder without a temperature channel is swapped for a simulated
        heating holder on the twin's clock (the twin now has a furnace)."""
        holder = self.twin.holder
        if "temperature" not in tuple(getattr(holder, "channels", ())):
            from ..holder import SimHeatingHolder

            holder = SimHeatingHolder(
                reason="created by a deapi temperature write", clock=getattr(self.twin, "clock", None)
            )
            with self.twin.lock:
                self.twin.holder = holder
        holder.set(float(value))

    def _set_specimen(self, value) -> None:
        name = str(value)
        setter = getattr(self.twin, "set_specimen", None)
        if callable(setter):
            setter(name)
        self._specimen_name = name

    def _property_specifications(self, command):
        name = command.command[0].parameter[0].p_string
        prop = self._values.get(_property_key(name))
        if prop is None:
            return (self._ack(command, f"Property not in Server: '{name}'", error=True),)
        vt = str(prop.value_type)
        opts: list = []
        raw = prop.options if isinstance(prop.options, str) else ",".join(map(str, prop.options or []))
        if vt == "Range" and raw:
            parts = raw.split(",")[:2]
            try:
                opts = [float(parts[0]), float(parts[1])]
            except (ValueError, IndexError):
                opts = []
        elif vt == "Set" and raw:
            opts = [o.replace("*", "").replace("'", "").replace("`", "").strip() for o in raw.split(",")]
        read_only = vt in ("ReadOnly", "READ_ONLY", "Read Only", "Server")
        values = [
            str(prop.data_type),
            vt,
            *opts,
            str(prop.category),
            str(prop.default_value or ""),
            str(prop.value),
            bool(read_only),
        ]
        return (self._ack(command, *values),)
