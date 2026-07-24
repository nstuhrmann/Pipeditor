"""
Run a pipeline saved from the GUI (File > Save) without Qt at all.

    python run_pipeline_headless.py my_pipeline.json
    python run_pipeline_headless.py my_pipeline.json --log all

--log takes the same channels as the GUI (see run_log.py): timing, bus,
metrics, warnings, progress, params — or 'all' / 'none'.
"""
import argparse
import sys

# Populates STEP_REGISTRY by importing every module under steps/ — must
# happen before Pipeline.load(), which looks classes up by name.
import src.GUI.pipeline_editor.steps  # noqa: F401

from src.GUI.pipeline_editor import run_log
from src.GUI.pipeline_editor.pipeline import Pipeline


def _print_timing(pipeline, context: str, use_mean: bool):
    if not run_log.is_on("timing"):
        return
    rows = [(n.display_name,
             n.timing.mean_ms if use_mean else n.timing.last_ms,
             n.timing.max_ms)
            for n in pipeline.nodes.values() if n.timing.count]
    if not rows:
        return
    rows.sort(key=lambda r: r[1], reverse=True)
    grand = sum(r[1] for r in rows) or 1.0
    label = "mean ms/frame" if use_mean else "ms"
    width = max(len(r[0]) for r in rows)
    print(f"\n--- step timing: {context} ---")
    print(f"{'step'.ljust(width)}  {label:>13}  {'share':>6}  {'max ms':>8}")
    for name, val, mx in rows:
        print(f"{name.ljust(width)}  {val:13.2f}  {val / grand:5.1%}  {mx:8.2f}")
    print(f"{'TOTAL'.ljust(width)}  {grand:13.2f}\n", flush=True)


def main(path: str):
    pipeline = Pipeline.load(path)

    total = pipeline.total_frames()
    print(f"Loaded '{path}': {len(pipeline.nodes)} nodes, "
         f"{total} frame(s) to process")

    warnings: list[str] = []

    if total == 1:
        # Single-frame pipeline (plain image sources, no video/stack).
        results = pipeline.run(warnings_out=warnings)
        _print_timing(pipeline, "single run", use_mean=False)
    else:
        # Video / image-stack source: process every frame. This is the
        # exact call the GUI's "Process Full Sequence" makes.
        def on_progress(done, tot):
            # Any per-frame channel prints its own lines; this in-place
            # counter would overwrite / interleave with them.
            if not run_log.per_frame_active():
                print(f"\r  frame {done}/{tot}", end="", flush=True)

        processed = pipeline.run_sequence(
            on_progress=on_progress, warnings_out=warnings)
        print(f"\ndone: {processed}/{total} frames processed")
        _print_timing(pipeline, f"batch, {processed} frame(s)",
                      use_mean=True)

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a saved ImgPipe pipeline without the GUI.")
    parser.add_argument("pipeline", help="pipeline .json saved from the editor")
    run_log.add_argument(parser)
    args = parser.parse_args()
    try:
        run_log.enable(args.log)
    except ValueError as exc:
        parser.error(str(exc))
    main(args.pipeline)
