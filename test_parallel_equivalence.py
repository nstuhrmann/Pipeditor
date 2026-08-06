"""
The acceptance test for the parallel executor.

The serial path is the reference implementation — it is the one that has
been run in anger for months — so "correct" here means BIT-IDENTICAL to
it, not merely plausible. Anything else and a parallel run silently
becomes a different experiment from the batch it is meant to reproduce.

    python test_parallel_equivalence.py

Exits non-zero on the first mismatch. Run it before touching anything in
parallel.py; it is meant to fail until the scheduler works.
"""
import sys

import numpy as np

import src.GUI.pipeline_editor.steps  # noqa: F401  (registers every step)
from src.GUI.pipeline_editor.base_step import (
    ParamSpec, ProcessingStep, StatefulStep, STEP_REGISTRY, register_step,
)
from src.GUI.pipeline_editor.pipeline import Pipeline, EDGE_CONTROL


# ---------------------------------------------------------------------------
# Steps that exist only to make the graphs below interesting
# ---------------------------------------------------------------------------

@register_step
class RampSource(ProcessingStep):
    """Deterministic per-index content, so a frame that arrives out of
    order is detectable rather than merely suspicious."""
    NAME = "Ramp Source"
    CATEGORY = "Test"
    KIND = "source"
    PARAMS = [ParamSpec("frames", "Frames", "int", default=12,
                        min_value=1, max_value=500, step=1),
              ParamSpec("size", "Size", "int", default=48,
                        min_value=8, max_value=512, step=8)]

    def frame_count(self):
        return int(self.p.frames)

    def process(self):
        n = int(self.p.size)
        base = 0.05 + 0.9 * (self.ctx.index / max(1, int(self.p.frames) - 1))
        img = np.full((n, n), base, np.float32)
        # A per-index watermark: any frame mix-up changes these pixels.
        img[0, : min(n, self.ctx.index + 1)] = 1.0
        self.emit(src_index=self.ctx.index)
        return img


@register_step
class Slow(ProcessingStep):
    """Stateless and deliberately uneven: cost depends on the frame, so
    workers finish out of order and any ordering bug is exposed."""
    NAME = "Slow"
    CATEGORY = "Test"
    PARAMS = [ParamSpec("work", "Work", "int", default=3,
                        min_value=0, max_value=200, step=1)]

    def process(self, image):
        a = np.asarray(image, np.float32)
        # Odd frames cost more, so completion order != submission order.
        rounds = int(self.p.work) * (2 if self.ctx.index % 2 else 1)
        for _ in range(rounds):
            a = a * 1.0000001
        return np.clip(a, 0.0, 1.0)


@register_step
class RunningSum(StatefulStep):
    """Order-sensitive by construction: its output encodes the exact
    sequence of indices it was fed."""
    NAME = "Running Sum"
    CATEGORY = "Test"

    def reset(self):
        self._acc = 0.0
        self._seen = []

    def advance(self, image):
        self._seen.append(self.ctx.index)
        self._acc += float(np.asarray(image).mean())
        self.emit(running_sum=self._acc,
                  order="-".join(str(i) for i in self._seen[-4:]))
        return image


@register_step
class SumMetric(ProcessingStep):
    NAME = "Sum Metric"
    CATEGORY = "Test"
    KIND = "metric"

    def process(self, image):
        return float(np.asarray(image, np.float64).sum())


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def graph_stateless_wide(frames=12):
    """The case the pool exists for: one source fanning into parallel
    chains that rejoin. Fully parallelizable."""
    p = Pipeline()
    src = p.add_node(STEP_REGISTRY["RampSource"]())
    src.step.set_param_values({"frames": frames})
    a = p.add_node(STEP_REGISTRY["Slow"]())
    b = p.add_node(STEP_REGISTRY["Slow"]())
    b.step.set_param_values({"work": 9})
    diff = p.add_node(STEP_REGISTRY["AbsDiff"]())
    m = p.add_node(STEP_REGISTRY["SumMetric"]())
    p.add_edge(src.id, a.id, 0)
    p.add_edge(src.id, b.id, 0)
    p.add_edge(a.id, diff.id, 0)
    p.add_edge(b.id, diff.id, 1)
    p.add_edge(diff.id, m.id, 0)
    return p


def graph_stateful_chain(frames=12):
    """A stateful node between stateless ones: must stay strictly in
    order however the neighbours are scheduled."""
    p = Pipeline()
    src = p.add_node(STEP_REGISTRY["RampSource"]())
    src.step.set_param_values({"frames": frames})
    a = p.add_node(STEP_REGISTRY["Slow"]())
    acc = p.add_node(STEP_REGISTRY["RunningSum"]())
    b = p.add_node(STEP_REGISTRY["Slow"]())
    m = p.add_node(STEP_REGISTRY["SumMetric"]())
    p.add_edge(src.id, a.id, 0)
    p.add_edge(a.id, acc.id, 0)
    p.add_edge(acc.id, b.id, 0)
    p.add_edge(b.id, m.id, 0)
    return p


def graph_control_loop(frames=12):
    """A feedback loop: no cross-frame parallelism is available at all,
    and the result must still match the serial run exactly."""
    p = Pipeline()
    cam = p.add_node(STEP_REGISTRY["CameraSimulator"]())
    cam.step.set_param_values(dict(resolution="320x240", num_frames=frames,
                                   illum_level=0.1, seed=1))
    ae = p.add_node(STEP_REGISTRY["AutoExposure"]())
    mon = p.add_node(STEP_REGISTRY["AEMonitor"]())
    noise = p.add_node(STEP_REGISTRY["SpatialNoise"]())
    p.add_edge(cam.id, ae.id, 0)
    p.add_edge(ae.id, mon.id, 0)
    p.add_edge(mon.id, noise.id, 0)
    p.add_edge(ae.id, cam.id, 0, kind=EDGE_CONTROL)
    return p


def graph_mixed(frames=10):
    """Temporal filtering plus a gated averager plus metrics — the shape
    of a real evaluation pipeline."""
    p = Pipeline()
    src = p.add_node(STEP_REGISTRY["RampSource"]())
    src.step.set_param_values({"frames": frames})
    noise = p.add_node(STEP_REGISTRY["AddNoise"]())
    noise.step.set_param_values(dict(sigma_const=0.02, seed=3))
    iir = p.add_node(STEP_REGISTRY["TemporalIIR"]())
    gate = p.add_node(STEP_REGISTRY["GatedTemporalAverage"]())
    sn = p.add_node(STEP_REGISTRY["SpatialNoise"]())
    sh = p.add_node(STEP_REGISTRY["Sharpness"]())
    p.add_edge(src.id, noise.id, 0)
    p.add_edge(noise.id, iir.id, 0)
    p.add_edge(iir.id, gate.id, 0)
    p.add_edge(gate.id, sn.id, 0)
    p.add_edge(sn.id, sh.id, 0)
    return p


GRAPHS = {
    "stateless_wide": graph_stateless_wide,
    "stateful_chain": graph_stateful_chain,
    "control_loop": graph_control_loop,
    "mixed": graph_mixed,
}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def collect(pipeline, parallel: bool, workers: int = 4) -> dict:
    """Everything a run produces, in a comparable form. Per-FRAME, not
    just the final state: a scheduler can land on the right answer at
    the end while getting the middle wrong."""
    frames = []
    if parallel:
        from src.GUI.pipeline_editor.parallel import iter_sequence_parallel
        stream = iter_sequence_parallel(pipeline, workers=workers)
    else:
        stream = pipeline.iter_sequence()
    for index, _total, frame in stream:
        frames.append({
            "index": index,
            "images": {nid: np.asarray(img).copy()
                       for nid, img in frame.images.items()
                       if isinstance(img, np.ndarray)},
            "metrics": dict(frame.metrics),
            "meta": {nid: dict(m) for nid, m in frame.meta.items()},
            "warnings": list(frame.warnings),
        })
    return {"frames": frames}


def compare(name: str, serial: dict, par: dict) -> list:
    problems = []
    sf, pf = serial["frames"], par["frames"]
    if len(sf) != len(pf):
        return [f"{name}: {len(pf)} frames from parallel, {len(sf)} serial"]

    for s, p in zip(sf, pf):
        if s["index"] != p["index"]:
            problems.append(f"{name}: frame order {p['index']} != {s['index']}")
            continue
        i = s["index"]
        if set(s["images"]) != set(p["images"]):
            problems.append(f"{name} frame {i}: different node set")
            continue
        for nid, img in s["images"].items():
            other = p["images"][nid]
            if img.shape != other.shape:
                problems.append(f"{name} frame {i} {nid[:8]}: shape "
                                f"{other.shape} != {img.shape}")
            elif not np.array_equal(img, other):
                d = float(np.abs(img.astype(np.float64)
                                 - other.astype(np.float64)).max())
                problems.append(f"{name} frame {i} {nid[:8]}: image differs "
                                f"(max {d:.3e})")
        for nid, v in s["metrics"].items():
            if nid not in p["metrics"]:
                problems.append(f"{name} frame {i}: metric {nid[:8]} missing")
            elif p["metrics"][nid] != v:
                problems.append(f"{name} frame {i} {nid[:8]}: metric "
                                f"{p['metrics'][nid]!r} != {v!r}")
        for nid, m in s["meta"].items():
            if p["meta"].get(nid) != m:
                problems.append(f"{name} frame {i} {nid[:8]}: metadata "
                                f"{p['meta'].get(nid)!r} != {m!r}")
        if s["warnings"] != p["warnings"]:
            problems.append(f"{name} frame {i}: warnings differ")
    return problems


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    total = 0
    for name, build in GRAPHS.items():
        # ONE pipeline, run twice. Node ids are fresh uuids per build, so
        # two separately-built graphs cannot be compared by id at all —
        # and reusing the object also proves the run is repeatable, since
        # iter_sequence() resets state before each pass.
        pipeline = build()
        serial = collect(pipeline, parallel=False)
        try:
            par = collect(pipeline, parallel=True, workers=workers)
        except ImportError:
            print("parallel.py not present yet — nothing to compare against")
            return 2
        except Exception as exc:
            print(f"  {name:16s} PARALLEL RAISED {type(exc).__name__}: {exc}")
            total += 1
            continue
        problems = compare(name, serial, par)
        status = "OK" if not problems else f"{len(problems)} MISMATCH"
        print(f"  {name:16s} {len(serial['frames']):>3} frames  {status}")
        for p in problems[:6]:
            print(f"      {p}")
        total += len(problems)

    print()
    print("identical to the serial executor" if not total
          else f"{total} mismatch(es)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
