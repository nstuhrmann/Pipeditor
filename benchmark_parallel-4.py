"""
Measure whether the parallel executor is worth it, on YOUR machine.

    python benchmark_parallel.py my_pipeline.json
    python benchmark_parallel.py pipelines/ --workers 1,2,4,8,14
    python benchmark_parallel.py my_pipeline.json --frames 40 --repeats 3

Reports, per pipeline:
  * serial wall time (the baseline you have today)
  * parallel wall time at each worker count, with speedup
  * the per-step timing table, so a disappointing result can be read
    rather than guessed at
  * whether the parallel result was IDENTICAL to the serial one

That last check runs every time. A speedup that changes the numbers is
not a speedup, and the failure would otherwise be silent.

Thread oversubscription
-----------------------
OpenCV already spreads a single call across every core. Running N pool
workers on top of that gives N x cv2_threads threads competing for the
same cores, which is usually SLOWER than serial. This script sets
cv2.setNumThreads(1) inside the parallel runs by default so the pool
owns the cores, and reports both settings with --both-thread-modes so
you can see the difference rather than take my word for it.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import src.GUI.pipeline_editor.steps  # noqa: F401  (registers every step)

from src.GUI.pipeline_editor.pipeline import Pipeline
from src.GUI.pipeline_editor.parallel import iter_sequence_parallel
from src.GUI.pipeline_editor import run_log


def _cap_frames(pipeline, limit: int):
    """Shorten every sequence source, so a benchmark on a 500-frame video
    doesn't take a coffee break per configuration."""
    if not limit:
        return
    for node in pipeline.nodes.values():
        for name in ("num_frames", "frames"):
            if name in node.step.values and node.step.KIND == "source":
                node.step.values[name] = min(node.step.values[name], limit)


def _drain(stream, total: int, label: str):
    """Time a run and fingerprint every frame, with a progress bar.

    The bar is constructed OUTSIDE the timed region and only updated
    inside it: setup cost would otherwise land in the measurement, and a
    benchmark must not measure its own instrument."""
    frames = []
    tick = run_log.frame_ticker(total, label, every=3.0)
    t0 = time.perf_counter()
    for index, _total, frame in stream:
        frames.append(_fingerprint(index, frame))
        tick()
    elapsed = time.perf_counter() - t0
    return elapsed, frames


def run_serial(pipeline, label: str = "serial"):
    return _drain(pipeline.iter_sequence(), pipeline.total_frames(), label)


def run_parallel(pipeline, workers: int, label: str = ""):
    return _drain(iter_sequence_parallel(pipeline, workers=workers),
                  pipeline.total_frames(),
                  label or f"workers={workers}")


def _fingerprint(index, frame):
    """Enough of a frame to prove two runs agree, without keeping every
    image alive for the whole benchmark."""
    return (
        index,
        tuple(sorted((nid, float(np.asarray(img, np.float64).sum()))
                     for nid, img in frame.images.items()
                     if isinstance(img, np.ndarray))),
        tuple(sorted(frame.metrics.items())),
    )


def timing_table(pipeline, top: int = 6) -> list:
    rows = pipeline.timing_summary()
    return rows[:top]


def parallel_fraction(pipeline) -> float:
    """Share of frame time in nodes the scheduler may run several
    workers on — classified exactly as parallel.py does, so the bound
    below describes THIS executor rather than some other design."""
    from src.GUI.pipeline_editor.base_step import StatefulStep
    total = 0.0
    par = 0.0
    for node in pipeline.nodes.values():
        if not node.timing.count:
            continue
        step = node.step
        serial = (isinstance(step, StatefulStep)
                  or step.KIND in ("sink", "source")
                  or step.EMITS_CONTROL or step.ACCEPTS_CONTROL)
        total += node.timing.mean_ms
        if not serial:
            par += node.timing.mean_ms
    return (par / total) if total else 0.0


def amdahl(par_fraction: float, workers: int) -> float:
    serial = 1.0 - par_fraction
    return 1.0 / (serial + par_fraction / max(1, workers))


def bench_one(path: str, worker_counts: list, frames_cap: int,
              repeats: int, both_modes: bool) -> bool:
    print(f"\n=== {os.path.basename(path)}")
    pipeline = Pipeline.load(path)
    _cap_frames(pipeline, frames_cap)
    total = pipeline.total_frames()

    # A live source returns a new frame on every read, so two runs of the
    # same graph legitimately differ. Comparing them would report a fault
    # that is not one — and hide a real difference in the noise.
    live = [n.display_name for n in pipeline.nodes.values()
            if not n.step.DETERMINISTIC]
    if live:
        print(f"  NOTE: {', '.join(live)} is a live source — results "
              f"cannot be compared between runs, only timings.")

    # One throwaway pass before any timing. The serial baseline would
    # otherwise run first and pay every one-off cost — lazy imports,
    # first-touch allocations, cold caches, a source opening its file —
    # making the parallel runs that follow look faster than they are.
    # It showed as speedup ABOVE the theoretical bound at one worker,
    # where there is no concurrency at all.
    run_serial(pipeline, "warm-up")

    # Serial baseline, best of `repeats` — the fastest run is the least
    # polluted by whatever else the machine was doing.
    serial_t, serial_fp = None, None
    for r in range(repeats):
        t, fp = run_serial(pipeline, f"serial {r + 1}/{repeats}")
        if serial_t is None or t < serial_t:
            serial_t, serial_fp = t, fp
    rows = timing_table(pipeline)
    per_frame = serial_t / max(1, total)
    print(f"  {total} frames | serial {serial_t:.2f}s "
          f"({per_frame * 1000:.1f} ms/frame)")
    if rows:
        print("  slowest steps:")
        for name, mean, _tot, share in rows:
            print(f"      {name:28s} {mean:7.2f} ms  {share:5.1%}")
        # Two different bounds, and confusing them is easy: sum/max is
        # the cap if each node had ONE worker, which is not what this
        # executor does. Multi-worker on a stateless node removes that
        # cap, so the bound that actually applies is Amdahl over the
        # share of time in nodes that may run concurrently.
        cap = sum(m for _n, m, _t, _s in pipeline.timing_summary())
        top = rows[0][1] or 1e-9
        pf = parallel_fraction(pipeline)
        print(f"  {pf:.0%} of frame time is parallelizable "
              f"({1 - pf:.0%} serialized: sources, sinks, stateful, "
              f"control)")
        print(f"  bounds: stage-pipelining alone {cap / top:.2f}x  |  "
              + "  ".join(f"{w}w {amdahl(pf, w):.2f}x"
                          for w in worker_counts))

    modes = [("cv2=1", 1)] if not both_modes else [("cv2=1", 1),
                                                   ("cv2=auto", None)]
    ok_all = True
    for label, cv2_threads in modes:
        for workers in worker_counts:
            prev = cv2.getNumThreads() if cv2 is not None else None
            if cv2 is not None and cv2_threads is not None:
                cv2.setNumThreads(cv2_threads)
            try:
                best_t, fp = None, None
                for r in range(repeats):
                    t, f = run_parallel(
                        pipeline, workers,
                        f"{label} w={workers} {r + 1}/{repeats}")
                    if best_t is None or t < best_t:
                        best_t, fp = t, f
            finally:
                if cv2 is not None and prev is not None:
                    cv2.setNumThreads(prev)
            identical = fp == serial_fp
            if live:
                flag = "   (not compared: live source)"
            else:
                ok_all &= identical
                flag = "" if identical else "   *** RESULT DIFFERS ***"
            print(f"  {label:9s} workers={workers:<3} {best_t:6.2f}s  "
                  f"speedup {serial_t / best_t:5.2f}x{flag}")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark the parallel executor against the serial one.")
    ap.add_argument("path", help="a pipeline .json, or a folder of them")
    ap.add_argument("--workers", default="",
                    help="comma-separated worker counts "
                         "(default: 1,2,4,<cpus>)")
    ap.add_argument("--frames", type=int, default=0, metavar="N",
                    help="cap sequence sources at N frames")
    ap.add_argument("--repeats", type=int, default=2,
                    help="runs per configuration; the fastest is reported")
    ap.add_argument("--both-thread-modes", action="store_true",
                    help="also measure with OpenCV threading left on, to "
                         "show the oversubscription cost")
    args = ap.parse_args()

    cpus = os.cpu_count() or 4
    if args.workers:
        counts = [int(w) for w in args.workers.split(",") if w.strip()]
    else:
        counts = sorted({1, 2, 4, cpus})

    if os.path.isdir(args.path):
        paths = sorted(glob.glob(os.path.join(args.path, "*.json")))
    elif os.path.isfile(args.path):
        paths = [args.path]
    else:
        print(f"no such file or folder: {args.path}", file=sys.stderr)
        return 2
    if not paths:
        print("no pipelines found", file=sys.stderr)
        return 2

    print(f"cpus={cpus}", end="")
    if cv2 is not None:
        print(f"  cv2 threads={cv2.getNumThreads()}", end="")
    print(f"  repeats={args.repeats}")

    all_ok = True
    for path in paths:
        try:
            all_ok &= bench_one(path, counts, args.frames, args.repeats,
                                args.both_thread_modes)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            all_ok = False

    print()
    if not all_ok:
        print("A parallel run did not match the serial one — the timings "
              "above are meaningless until that is fixed.")
        return 1
    print("All parallel runs matched the serial results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
