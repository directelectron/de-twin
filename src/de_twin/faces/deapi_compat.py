"""What the deapi face needs from the optional ``deapi`` package, with stand-ins when it is absent."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

try:  # deapi is an optional dependency of the twin
    from deapi.buffer_protocols import pb
    from deapi.simulated_server.fake_server import FakeServer, Property, add_parameter
    from deapi.version import commandVersion

    try:
        from deapi.version import fake_server_software_version as _SERVER_VERSION
    except ImportError:  # older deapi
        _SERVER_VERSION = "2.8.0.11901"
except ImportError:  # pragma: no cover - exercised only without deapi
    pb = None
    FakeServer = object  # type: ignore[assignment,misc]
    Property = object  # type: ignore[assignment,misc]
    add_parameter = None
    commandVersion = 0
    _SERVER_VERSION = "2.8.0.11901"

log = logging.getLogger(__name__)


def _sends_scan_pattern_index(server_version: str) -> bool:
    """DE-Server 2.8.0 build 12073+ appends ``current_scan_pattern_idx`` to GET_RESULT's
    attributes (command version 16). The face reports the installed deapi's fake-server
    version, so it answers the way that DE-Server build would."""
    try:
        major, minor, patch, build = (int(p) for p in str(server_version).split(".")[:4])
    except ValueError:
        return False
    return (major, minor, patch) >= (2, 8, 0) and build >= 12073 and commandVersion >= 16


class DynamicProperty(Property):  # type: ignore[misc]
    """A property whose value is computed (and optionally applied) by callables."""

    def __init__(
        self,
        name: str,
        getter: Callable[[], Any],
        setter: Callable[[Any], None] | None = None,
        *,
        data_type: str = "String",
        category: str = "Twin",
        options: str = "",
        value_type: str | None = None,
        default_value: str = "",
        server=None,
    ):
        super().__init__(
            name=name,
            value="",
            data_type=data_type,
            category=category,
            value_type=value_type or ("ReadOnly" if setter is None else "AllowAll"),
            options=options,
            default_value=default_value,
            server=server,
        )
        self._getter = getter
        self._setter = setter

    @property  # type: ignore[override]
    def value(self):
        v = self._getter()
        if self.data_type == "Integer":
            return str(round(float(v)))
        if self.data_type == "Float":
            return str(float(v))
        return str(v)

    @value.setter
    def value(self, value):
        if self._setter is None:
            log.warning("Property %s is read only", self.name)
            return
        self._setter(value)
