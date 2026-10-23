Magnification and camera-length ladders are one type, ``de_twin.column.ladders.Ladder``,
which carries both rules for picking the nearest rung (``snap_linear`` for the column's
detents, ``snap_logarithmic`` for calibration tables). ``Column.mag_ladder()``,
``Column.cl_ladder()``, ``Calibration.mag_ladder()`` and ``Calibration.cl_ladder()`` return it,
and the calibration's built-in tables moved to ``de_twin.column.ladders``.
