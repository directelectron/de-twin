"""de-twin: digital twin of a Direct Electron camera on a TEM.

Column, specimen, in-situ holder and detector on one clock, for developing automated
microscope control without a microscope, a camera or DE-Server::

    from de_twin import DigitalTwin

    twin = DigitalTwin("Dense Au on holey C", camera="DE16")
    image = twin.snap(1.0)
"""

import os as _os
import sys as _sys
from importlib.metadata import PackageNotFoundError, version as _version


def _keep_off_the_gpu() -> None:
    """Keep the twin's process from initialising CUDA.

    The twin renders on the CPU, and the GPU belongs to whoever reads its frames (DE-Server,
    then Autopilot). Nothing of the twin's needs CUDA, but its crystallography stack does
    (diffsims asks ``numba.cuda.is_available()`` on import, which calls ``cuInit`` and so
    starts the driver in this process). ``NUMBA_DISABLE_CUDA`` makes numba answer "no" without
    touching the driver. ``DE_TWIN_ALLOW_CUDA=1`` leaves the process alone (a notebook that
    wants numba.cuda next to the twin)."""
    if _os.environ.get("DE_TWIN_ALLOW_CUDA") == "1" or "numba.cuda" in _sys.modules:
        return
    _os.environ["NUMBA_DISABLE_CUDA"] = "1"
    config = _sys.modules.get("numba.core.config")
    if config is not None:  # numba already read its environment
        config.DISABLE_CUDA = 1


_keep_off_the_gpu()

try:
    __version__ = _version("de-twin")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

__all__ = ["DigitalTwin", "AcquisitionRequest", "ExposureMode", "Clock", "ManualClock", "__version__"]


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
