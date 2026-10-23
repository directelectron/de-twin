"""Command line: ``de-twin serve | snap | list``.

``de-twin serve`` hosts one twin and any combination of faces::

    # Everything a laptop needs: SOAP microscope + deapi fake server
    de-twin serve --soap 5002 --deapi 13240

    # Feed a real DE-Server (test pattern "External Frame Source (Shared Memory)")
    # and let DE-Server poll the twin's column as its TEM-Channel
    de-twin serve --shm --soap 5002 --camera DE16

    # ... as a live view (Ground Crew): fast previews while the stage or mag changes
    de-twin serve --shm --soap 5002 --camera DE16 --interactive

    # Follow a real / Dummy DE-TEM-Channel instead of simulating the column,
    # and a real DENS Impulse holder
    de-twin serve --shm --mirror-temchannel 192.168.0.10 --holder impulse

    # A probe-corrected column (CEOS-like corrector on the SOAP face)
    de-twin serve --soap 5002 --deapi 13240 --corrector probe
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading


def _build_twin(args):
    from .clock import Clock
    from .twin import DigitalTwin

    clock = Clock(time_scale=args.time_scale)
    column = None
    if args.mirror_temchannel:
        from .column import MirrorColumn

        host, _, port = args.mirror_temchannel.partition(":")
        column = MirrorColumn(host=host, port=int(port or 5002), clock=clock)
        if args.corrector != "none":
            print(
                "--corrector ignored: the column mirrors a real DE-TEM-Channel (its corrector "
                "is read over SOAP)",
                file=sys.stderr,
            )
    elif args.corrector != "none":
        from .column import Column

        column = Column(
            clock=clock,
            corrector=args.corrector,
            seed=args.seed,
            corrector_options={"tuned": not args.corrector_cold},
        )
    render_config = None
    if getattr(args, "interactive", False):
        from .render import RenderConfig

        render_config = RenderConfig(interactive=True, prefetch=True)
    return DigitalTwin(
        specimen=args.specimen,
        camera=args.camera,
        column=column,
        holder=args.holder,
        seed=args.seed,
        clock=clock,
        render_config=render_config,
    )


def _add_twin_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--specimen", default="Dense Au on holey C", help="preset or pattern name")
    p.add_argument("--camera", default="DE16", help="camera model (see `de-twin list`)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holder", default="none", help="none | sim-heating | impulse")
    p.add_argument("--time-scale", type=float, default=1.0, help="simulated seconds per wall second")
    p.add_argument(
        "--mirror-temchannel",
        metavar="HOST[:PORT]",
        help="follow a real/Dummy DE-TEM-Channel instead of simulating the column",
    )
    p.add_argument(
        "--corrector",
        default="none",
        choices=("none", "probe", "image", "both"),
        help="fit a CEOS-like aberration corrector to the simulated column",
    )
    p.add_argument(
        "--corrector-cold",
        action="store_true",
        help="start the corrector out of tune (cold start) instead of tuned",
    )


def cmd_serve(args) -> int:
    twin = _build_twin(args)
    stoppers = []
    if args.soap:
        if args.mirror_temchannel:
            print("--soap ignored: the column mirrors a real DE-TEM-Channel", file=sys.stderr)
        else:
            from .faces.temchannel_soap import TemChannelServer

            server = TemChannelServer(twin.column, host=args.host, port=args.soap).start()
            stoppers.append(server.stop)
            corr = getattr(twin.column, "corrector", None)
            extra = f" (corrector: {corr.kind}, serving {corr.served_side})" if corr else ""
            print(f"TEM-Channel SOAP face on {args.host}:{args.soap}{extra}")
    if args.shm is not None:
        from .faces.shm_face import ShmFace

        face = ShmFace(twin, name=args.shm, reuse=args.reuse, threads=args.threads).start()
        stoppers.append(face.stop)
        print(f"shared-memory frame source '{args.shm}' ({twin.detector.model.name})")
    if args.deapi:
        from .faces import deapi_server

        t = threading.Thread(target=deapi_server.serve, args=(twin, args.deapi), daemon=True)
        t.start()
        print(f"deapi fake server on 127.0.0.1:{args.deapi}")
    if not stoppers and not args.deapi:
        print("nothing to serve: pass --soap, --shm and/or --deapi", file=sys.stderr)
        return 2
    print("started .... ", flush=True)

    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    try:
        signal.signal(signal.SIGTERM, lambda *_: done.set())
    except (ValueError, AttributeError):  # pragma: no cover
        pass
    while not done.wait(0.5):
        pass
    for stop in reversed(stoppers):
        stop()
    twin.close()
    return 0


def cmd_snap(args) -> int:
    import numpy as np

    twin = _build_twin(args)
    if args.magnification:
        twin.column.set("Magnification", args.magnification)
    if args.defocus is not None:
        twin.column.set("Defocus", args.defocus)
    if args.stage:
        x, y = (float(v) for v in args.stage.split(","))
        twin.column.set_stage(x=x, y=y)
        twin.column.wait_idle() if hasattr(twin.column, "wait_idle") else None
    frame = twin.snap(args.exposure)
    if args.out.endswith(".npy"):
        np.save(args.out, frame)
    else:
        try:
            import tifffile

            tifffile.imwrite(args.out, frame)
        except ImportError:
            np.save(args.out + ".npy", frame)
            args.out += ".npy"
    print(f"{args.out}: {frame.shape} {frame.dtype} mean={frame.mean():.1f}")
    return 0


def cmd_list(args) -> int:
    from .detector import CAMERAS
    from .specimen import PATTERNS, PRESETS

    print("Cameras:")
    for name, m in CAMERAS.items():
        print(f"  {name:14s} {m.sensor_shape[1]}x{m.sensor_shape[0]}  {m.bit_depth}-bit")
    print("Specimen presets:")
    for name in PRESETS:
        print(f"  {name}")
    print("Specimen patterns:")
    for name in PATTERNS:
        print(f"  {name}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="de-twin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="host the twin and its faces")
    _add_twin_args(p)
    p.add_argument("--host", default="0.0.0.0", help="bind address for the SOAP face")
    p.add_argument(
        "--soap",
        type=int,
        nargs="?",
        const=5002,
        default=None,
        help="serve the DE-TEM-Channel SOAP API (default port 5002)",
    )
    p.add_argument(
        "--deapi",
        type=int,
        nargs="?",
        const=13240,
        default=None,
        help="serve a deapi-compatible fake DE-Server (default port 13240)",
    )
    p.add_argument(
        "--shm",
        nargs="?",
        const="DE_ExternalFrames",
        default=None,
        metavar="NAME",
        help="publish frames into the shared-memory ring for DE-Server",
    )
    p.add_argument(
        "--reuse",
        type=int,
        default=1,
        metavar="N",
        help="shared memory: publish each rendered frame up to N times, so the stream "
        "keeps up with fast frame rates (a real camera's GB/s); 1 = every frame unique",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="shared memory: cores the detector physics may use (leave the rest for DE-Server's processing)",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="live view: while the stage or magnification keeps changing, show a fast "
        "low-resolution render and the full one once the view settles "
        "(RenderConfig.interactive)",
    )
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("snap", help="render one frame to a file")
    _add_twin_args(p)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--magnification", type=float)
    p.add_argument("--defocus", type=float, help="um")
    p.add_argument("--stage", help="x,y in um")
    p.add_argument("--out", default="snap.npy")
    p.set_defaults(func=cmd_snap)

    p = sub.add_parser("list", help="list cameras and specimens")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
