"""
Opt-in diagnostic output to stdout, selected by channel.

    python -m src.GUI.pipeline_editor.main --log bus,metrics
    python run_pipeline_headless.py p.json --log all

Channels:
    timing    per-step execution time after each run / batch
    bus       every message posted, with the node that sent it
    metrics   every metric node's value, per frame
    warnings  executor warnings (skipped nodes, clipped output)
    progress  one line per frame during a batch
    params    each node's parameter values at the start of a run

`timing` is ON by default, matching the behaviour before this flag
existed. --log REPLACES the set, so pass `--log timing,bus` to keep it
alongside another channel, `--log all` for everything, or `--log none`
for silence.

Everything routes through log() so the GUI, playback, batch runs and the
headless runner all produce identical output — the executor emits it, no
caller has to remember to.
"""
import sys

CHANNELS = ("timing", "bus", "metrics", "warnings", "progress", "params")

_enabled = {"timing"}


def enable(spec: str) -> None:
    """Apply a --log spec: comma-separated channel names, 'all', or
    'none'. Unknown names raise ValueError so a typo is caught at
    startup instead of silently producing no output."""
    global _enabled
    spec = (spec or "").strip().lower()
    if not spec or spec == "none":
        _enabled = set()
        return
    if spec == "all":
        _enabled = set(CHANNELS)
        return
    names = {n.strip() for n in spec.split(",") if n.strip()}
    unknown = names - set(CHANNELS)
    if unknown:
        raise ValueError(
            f"unknown --log channel(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(CHANNELS)}, all, none")
    _enabled = names


def is_on(channel: str) -> bool:
    return channel in _enabled


def log(channel: str, message: str) -> None:
    if channel in _enabled:
        print(message, flush=True)


def add_argument(parser) -> None:
    """Register --log on an argparse parser."""
    parser.add_argument(
        "--log", default="timing", metavar="CHANNELS",
        help=("comma-separated diagnostic channels to print to stdout: "
              + ", ".join(CHANNELS) + "; or 'all' / 'none' "
              "(default: timing)"))


def take_from_argv(argv: list = None) -> list:
    """Pull --log out of an argv list, apply it, and return the argv with
    it removed — so it can be stripped before handing the rest to
    QApplication, which would otherwise complain about an unknown flag.
    Accepts '--log X' and '--log=X'."""
    argv = list(sys.argv if argv is None else argv)
    out, i = [], 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--log" and i + 1 < len(argv):
            enable(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--log="):
            enable(arg.split("=", 1)[1])
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


PER_FRAME_CHANNELS = ("bus", "metrics", "warnings", "progress", "params")


def per_frame_active() -> bool:
    """True if any channel prints a line per frame. Callers that render
    an in-place progress counter (\r ...) must stay quiet when this is
    true, or the counter and the log lines overwrite each other."""
    return any(c in _enabled for c in PER_FRAME_CHANNELS)


def frame_tag(frame_index: int, total_frames: int) -> str:
    return f"[{frame_index}]" if total_frames <= 1 else \
        f"[{frame_index}/{total_frames - 1}]"
