"""
Opt-in diagnostic output to stdout, selected by channel.

    python -m src.GUI.pipeline_editor.main --log bus,metrics
    python run_pipeline_headless.py p.json --log all

Channels:
    timing    per-step execution time after each run / batch
    control   every control value sent along a control edge
    meta      each node's outgoing frame metadata
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

CHANNELS = ("timing", "control", "meta", "metrics", "warnings",
            "progress", "params")

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


def add_argument(parser, default: str = "timing") -> None:
    """Register --log on an argparse parser.

    `default` lets a tool pick the channel set that suits it: the GUI
    wants timing, a regression harness wants warnings."""
    parser.add_argument(
        "--log", default=default, metavar="CHANNELS",
        help=("comma-separated diagnostic channels to print to stdout: "
              + ", ".join(CHANNELS) + "; or 'all' / 'none' "
              f"(default: {default})"))


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


#: Channels that print a line for EVERY frame. `warnings` is not one of
#: them: it is deduped per batch, so a clean run prints nothing at all —
#: suppressing a progress bar for it would mean never showing one, since
#: `warnings` is the default for the headless runner.
PER_FRAME_CHANNELS = ("control", "meta", "metrics", "progress", "params")


def per_frame_active() -> bool:
    """True if any channel prints a line per frame. A progress bar must
    stay quiet then, or the bar and the log lines overwrite each other."""
    return any(c in _enabled for c in PER_FRAME_CHANNELS)


def print_timing(rows: list, context: str, per_frame: bool = True) -> None:
    """The per-step timing table, on the `timing` channel.

    `rows` is Pipeline.timing_summary(). Lived in three copies — the GUI
    and both CLI tools — which is two too many for something whose only
    job is to format a list.
    """
    if not is_on("timing") or not rows:
        return
    label = "mean ms/frame" if per_frame else "ms"
    width = max(len(name) for name, _m, _t, _s in rows)
    grand = sum(mean for _n, mean, _t, _s in rows) or 1.0
    print(f"\n--- step timing: {context} ---")
    print(f"{'step'.ljust(width)}  {label:>13}  {'share':>6}")
    for name, mean, _total, share in rows:
        print(f"{name.ljust(width)}  {mean:13.2f}  {share:5.1%}")
    print(f"{'TOTAL'.ljust(width)}  {grand:13.2f}\n", flush=True)


def frame_ticker(total: int, label: str = "", every: float = 2.0,
                 stream=None):
    """A callable to invoke once per frame; prints an occasional line.

    Plain lines rather than a redrawn bar: they survive being piped to a
    file, they appear correctly in IDE consoles that do not emulate a
    terminal, and they need no dependency in a deployed build.

    Time-based rather than every-Nth-frame, so a slow pipeline still
    reports and a fast one does not flood — and it stays silent when a
    per-frame channel is already printing.
    """
    import sys as _sys
    import time as _time
    stream = stream or _sys.stderr
    quiet = per_frame_active()
    state = {"n": 0, "t0": _time.perf_counter(), "last": 0.0}

    def tick(count: int = 1):
        state["n"] += count
        if quiet:
            return
        now = _time.perf_counter()
        done = state["n"]
        if now - state["last"] < every and done < total:
            return
        state["last"] = now
        elapsed = now - state["t0"]
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        tail = "" if done < total else "  done"
        stream.write(f"  {label} {done}/{total} frames  "
                     f"{rate:.1f} f/s  eta {eta:.0f}s{tail}\n")
        stream.flush()

    return tick


def frame_tag(frame_index: int, total_frames: int) -> str:
    return f"[{frame_index}]" if total_frames <= 1 else \
        f"[{frame_index}/{total_frames - 1}]"
