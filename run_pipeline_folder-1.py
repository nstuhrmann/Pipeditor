"""
Run every pipeline in a folder, headless, and report on all of them.

    python run_pipeline_folder.py pipelines/
    python run_pipeline_folder.py pipelines/ --frames 20 --csv results.csv
    python run_pipeline_folder.py pipelines/ --log warnings

Built as a regression harness: a failure in one pipeline is recorded and
the run continues, so one broken file doesn't hide the state of the other
twenty. The exit code is non-zero if anything failed, which is what makes
it usable from a build script.

For each pipeline it reports frames processed, wall time, warnings, the
slowest step, and every metric's final value — so a change in an
algorithm shows up as a changed number rather than as "it still runs".
"""
import argparse
import csv
import glob
import os
import sys
import time
import traceback

# Populate STEP_REGISTRY. Pipelines are stored as class names, so every
# step module must be imported before a load can resolve them.
import src.GUI.pipeline_editor.steps  # noqa: F401

from src.GUI.pipeline_editor import run_log
from src.GUI.pipeline_editor.pipeline import Pipeline


class Outcome:
    __slots__ = ("path", "ok", "frames", "seconds", "warnings", "metrics",
                 "slowest", "slowest_ms", "error")

    def __init__(self, path):
        self.path = path
        self.ok = False
        self.frames = 0
        self.seconds = 0.0
        self.warnings: list = []
        self.metrics: dict = {}
        self.slowest = ""
        self.slowest_ms = 0.0
        self.error = ""


def run_one(path: str, max_frames: int = 0, single: bool = False) -> Outcome:
    outcome = Outcome(path)
    started = time.perf_counter()
    try:
        pipeline = Pipeline.load(path)
        total = pipeline.total_frames()

        if max_frames and total > max_frames:
            # Cap long sequences so a folder of 500-frame videos is still
            # a usable smoke test. Cancelling after N frames goes through
            # the normal path, so sinks are still closed properly.
            counter = {"n": 0}

            def should_cancel():
                counter["n"] += 1
                return counter["n"] > max_frames
            result = pipeline.run_sequence(should_cancel=should_cancel)
        elif single or total <= 1:
            result = pipeline.run()
        else:
            result = pipeline.run_sequence()

        outcome.frames = max(1, result.frames_processed)
        outcome.warnings = list(result.warnings)
        outcome.metrics = {
            pipeline.nodes[nid].display_name: value
            for nid, value in result.metrics.items() if nid in pipeline.nodes
        }
        rows = pipeline.timing_summary()
        if rows:
            outcome.slowest, outcome.slowest_ms = rows[0][0], rows[0][1]
        outcome.ok = True
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        if run_log.is_on("warnings"):
            traceback.print_exc()
    outcome.seconds = time.perf_counter() - started
    return outcome


def report(outcomes: list):
    width = max((len(os.path.basename(o.path)) for o in outcomes), default=8)
    print()
    print(f"{'pipeline'.ljust(width)}  {'status':>7}  {'frames':>6}  "
          f"{'time s':>7}  {'warn':>4}  slowest step")
    print("-" * (width + 46))
    for o in outcomes:
        name = os.path.basename(o.path).ljust(width)
        status = "OK" if o.ok else "FAIL"
        slow = f"{o.slowest} ({o.slowest_ms:.1f} ms)" if o.slowest else ""
        print(f"{name}  {status:>7}  {o.frames:>6}  {o.seconds:>7.2f}  "
              f"{len(o.warnings):>4}  {slow}")

    for o in outcomes:
        if o.metrics:
            print(f"\n  {os.path.basename(o.path)} metrics:")
            for name, value in o.metrics.items():
                shown = f"{value:.6g}" if isinstance(value, float) else value
                print(f"    {name}: {shown}")
    for o in outcomes:
        if o.warnings:
            print(f"\n  {os.path.basename(o.path)} warnings:")
            for w in o.warnings:
                print(f"    {w}")
    for o in outcomes:
        if o.error:
            print(f"\n  {os.path.basename(o.path)} FAILED: {o.error}")

    failed = [o for o in outcomes if not o.ok]
    warned = [o for o in outcomes if o.ok and o.warnings]
    print()
    print(f"{len(outcomes)} pipeline(s): {len(outcomes) - len(failed)} ok, "
          f"{len(failed)} failed, {len(warned)} with warnings, "
          f"{sum(o.seconds for o in outcomes):.1f} s total")
    return len(failed)


def write_csv(outcomes: list, path: str):
    """One row per pipeline, with metrics flattened into columns — so a
    run can be diffed against a previous one."""
    metric_names = sorted({n for o in outcomes for n in o.metrics})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pipeline", "ok", "frames", "seconds", "warnings",
                    "slowest_step", "slowest_ms", "error"] + metric_names)
        for o in outcomes:
            w.writerow([os.path.basename(o.path), int(o.ok), o.frames,
                        f"{o.seconds:.3f}", len(o.warnings), o.slowest,
                        f"{o.slowest_ms:.3f}", o.error]
                       + [o.metrics.get(n, "") for n in metric_names])


def main():
    parser = argparse.ArgumentParser(
        description="Run every pipeline in a folder headless.")
    parser.add_argument("folder", help="folder containing pipeline .json files")
    parser.add_argument("--pattern", default="*.json",
                        help="filename pattern (default: *.json)")
    parser.add_argument("--frames", type=int, default=0, metavar="N",
                        help="cap sequences at N frames (0 = run them fully)")
    parser.add_argument("--single", action="store_true",
                        help="run one frame per pipeline, never a sequence")
    parser.add_argument("--csv", default="", metavar="FILE",
                        help="also write the results as CSV")
    run_log.add_argument(parser)
    args = parser.parse_args()

    try:
        run_log.enable(args.log)
    except ValueError as exc:
        parser.error(str(exc))

    paths = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if not paths:
        print(f"no files matching '{args.pattern}' in {args.folder}",
              file=sys.stderr)
        return 2

    outcomes = []
    for path in paths:
        print(f"=== {os.path.basename(path)} ===", flush=True)
        outcomes.append(run_one(path, max_frames=args.frames,
                                single=args.single))

    failed = report(outcomes)
    if args.csv:
        write_csv(outcomes, args.csv)
        print(f"results written to {args.csv}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
