# Feeding a real DE-Server

DE-Server can take its camera frames from the twin and process them exactly as camera
frames: dark and gain references, bad pixels, counting, saving. DE-Server's only
twin-related code is a small external frame source (`ExternalFrameSource.h/.cpp`, about
220 lines, in the repository's `cpp/` folder) behind the test pattern
**External Frame Source (Shared Memory)**.

1. In DE-Server's `configurations/server.xml`, set `GrabberType="Software"`. Keep the
   `Camera Type` of the model you want to emulate; its geometry, topology and bit depth are
   used.
2. Point *Instrument Client Address* at the machine running the twin, so DE-Server's
   microscope metadata comes from the twin's column.
3. Run `de-twin serve --shm --soap 5002 --camera DE16`.
4. In DE-MC, select the test pattern **External Frame Source (Shared Memory)** and acquire.

## How it works

DE-Server creates the shared memory `DE_ExternalFrames`. Before each acquisition it writes
the request (frame size after hardware ROI and binning, sensor, frame time, frame count,
exposure mode, scan size) and increments a request id. The twin renders frames for exactly
that request and writes them into a ring of slots; DE-Server copies them into its grab
buffer in order. If no frame arrives within 2 × frame time + 1 s, DE-Server logs a warning
and delivers a blank frame, so an acquisition never hangs.

The layout is documented at the top of `cpp/ExternalFrameSource.h`, and its Python mirror
is `de_twin.transport.shm_layout`. The tests compile the C++ and check both agree.
