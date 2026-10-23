"""Errors the simulated column raises."""

from __future__ import annotations



class ColumnRefused(ValueError):
    """The column understood the request and refused it (the Dummy's ``false``)."""
