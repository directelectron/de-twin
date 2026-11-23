Replay a recorded dataset as the twin's camera: ``DigitalTwin.load_recording`` serves a
DE-Server MRC movie (with its ``_info.txt``), an EDAX ``.up1``/``.up2`` file or an Autopilot
run folder through every face, the deapi face and the shared memory that feeds a real
DE-Server alike. The sensor is the recording's frame; each scan point reads the recorded
point its coil voltages put the beam on (subsampling, ROIs, XY arrays and the pixel offset
included); and which recorded scan a frame comes from follows recorded time, moved by the
frames acquired and by the wall clock between acquisitions (``searching``,
``series_started``). ``de-twin serve --recording``.
