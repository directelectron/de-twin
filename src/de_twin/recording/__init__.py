"""Recorded datasets replayed as the twin's camera.

Load one with :meth:`DigitalTwin.load_recording`: the twin's frames are then the
recording's, through every face (the deapi face, and the shared memory that feeds a real
DE-Server), and its sensor is the recording's frame. :mod:`.formats` says what is read and
from where, :mod:`.replay` where each frame comes from, :mod:`.clock` when.
"""

from .clock import RecordedClock
from .files import FrameFiles, FramePart
from .formats import Recording, SideNeeded, Unreadable, open_recording
from .replay import Replay

__all__ = [
    "FrameFiles",
    "FramePart",
    "RecordedClock",
    "Recording",
    "Replay",
    "SideNeeded",
    "Unreadable",
    "open_recording",
]
