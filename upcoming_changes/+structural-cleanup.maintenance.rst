Structural cleanup with no change to rendered output: one shared numba kernel decorator
(``de_twin.parallel.numba_jit``) replaces the per-module copies, the column-realism classes
share their derived values, and unused helpers (``log_snap``, ``resample_object``,
``upsample2x``, ``point_in_polygon``, ``AmorphousStructure``) are removed.
