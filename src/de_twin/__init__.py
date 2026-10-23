"""de-twin: digital twin of a Direct Electron camera on a TEM.

Column, specimen, in-situ holder and detector on one clock, for developing automated
microscope control without a microscope, a camera or DE-Server::

    from de_twin import DigitalTwin

    twin = DigitalTwin("Dense Au on holey C", camera="DE16")
    image = twin.snap(1.0)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("de-twin")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

__all__ = ["AcquisitionRequest", "Clock", "DigitalTwin", "ExposureMode", "ManualClock", "__version__"]


def __getattr__(name):
    # Lazy so `import de_twin` stays cheap (the specimen/crystal stack is imported on first use).
    if name == "DigitalTwin":
        from .twin import DigitalTwin

        return DigitalTwin
    if name in ("AcquisitionRequest", "ExposureMode"):
        from . import state

        return getattr(state, name)
    if name in ("Clock", "ManualClock"):
        from . import clock

        return getattr(clock, name)
    raise AttributeError(f"module 'de_twin' has no attribute {name!r}")
