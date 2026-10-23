``Column.mag_ladder()`` and ``Column.cl_ladder()`` now return lists, empty when the mode has
no such ladder (they returned ``None``), and ``cl_ladder()`` is in millimeters like
``Calibration.cl_ladder()`` (it was centimeters). In ``de_twin.column.ladders``,
``DIFF_CAMLENGTHS_CM`` and ``STEM_CAMLENGTHS_CM`` became ``DIFF_CAMERA_LENGTHS_MM`` and
``STEM_CAMERA_LENGTHS_MM``. The ``CameraLength`` property is still centimeters, as on the
DE-TEM-Channel protocol.
