"""
Pipeline model and executor.

Three things flow through a graph, and each has its own path:

  images    along DATA edges, forward, one array per port.
  metadata  with the frame it describes, forward. The executor merges
            and re-attaches it, so a step that knows nothing about
            metadata still passes it along untouched.
  control   along CONTROL edges, delivered on the target's NEXT
            execution. Drawn backwards in the graph (AE -> camera), so a
            feedback loop is visible instead of being an invisible global
            channel. The one-frame delay is inherent: within a frame the
            camera runs before the AE that measures it.

There is deliberately no global message bus any more: forward
information is metadata, backward information is a control edge, and
both are tied to specific nodes rather than to a shared namespace, so
two independent camera/AE pairs in one graph cannot collide.
"""
import json
import time
import uuid

import numpy as np

from src.GUI.pipeline_editor import run_log
from src.GUI.pipeline_editor.array_utils import (
    Frame, METRIC_META_PREFIX,
)
from src.GUI.pipeline_editor.base_step import (
    FrameContext, STEP_REGISTRY, to_float01,
)

EDGE_DATA = "data"
EDGE_CONTROL = "control"


class StepTiming:
    """Per-node execution timing. Per-NODE, not per-class: two instances
    of the same step with different parameters time differently."""
    __slots__ = ("last_ms", "total_ms", "max_ms", "count")

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_ms = 0.0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.count = 0

    def add(self, ms: float):
        self.last_ms = ms
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)
        self.count += 1

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


class RunResult:
    """Everything one run (or one batch) produced.

    Replaces the previous pile of `*_out` list/dict arguments, which grew
    one entry per feature and forced every caller to pre-allocate
    containers it might not want.
    """
    __slots__ = ("images", "metrics", "meta", "warnings", "metric_series",
                 "frames_processed", "cancelled")

    def __init__(self):
        self.images: dict = {}          # node_id -> Frame
        self.metrics: dict = {}         # node_id -> value (last frame)
        self.meta: dict = {}            # node_id -> metadata dict
        self.warnings: list = []
        self.metric_series: dict = {}   # node_id -> [(frame_index, value)]
        self.frames_processed: int = 0
        self.cancelled: bool = False

    def __getitem__(self, node_id):     # result[nid] -> image
        return self.images[node_id]

    def __contains__(self, node_id):
        return node_id in self.images


class PipelineNode:
    def __init__(self, step, node_id: str | None = None, pos=(0, 0),
                 number: int = 1, bypassed: bool = False):
        self.step = step
        self.id = node_id or str(uuid.uuid4())
        self.pos = pos
        self.number = number
        self.bypassed = bypassed
        self.timing = StepTiming()
        self.last_meta: dict = {}
        self.last_emitted: dict = {}
        self.last_control: dict = {}
        # Control values waiting for this node's next execution, and the
        # index of the frame this node last saw (for FrameContext).
        self.inbox: dict = {}
        self._last_index = None

    @property
    def class_name(self) -> str:
        return type(self.step).__name__

    @property
    def display_name(self) -> str:
        return f"{self.step.NAME} {self.number}"


class Pipeline:
    def __init__(self):
        self.nodes: dict[str, PipelineNode] = {}
        # (from_id, to_id, to_port, kind) — kind is EDGE_DATA or EDGE_CONTROL
        self.edges: list[tuple] = []
        self._name_counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Graph editing
    # ------------------------------------------------------------------
    def add_node(self, step, pos=(0, 0)) -> PipelineNode:
        node = PipelineNode(step, pos=pos,
                            number=self._next_number(step.NAME))
        self.nodes[node.id] = node
        return node

    def _next_number(self, name: str) -> int:
        n = self._name_counters.get(name, 0) + 1
        self._name_counters[name] = n
        return n

    def _note_number(self, name: str, number: int):
        self._name_counters[name] = max(self._name_counters.get(name, 0),
                                        number)

    def remove_node(self, node_id: str):
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges
                      if e[0] != node_id and e[1] != node_id]

    def add_edge(self, from_id: str, to_id: str, to_port: int = 0,
                 kind: str = EDGE_DATA):
        edge = (from_id, to_id, to_port, kind)
        if kind == EDGE_DATA:
            # One source per input port; a control edge doesn't occupy one.
            self.edges = [e for e in self.edges
                          if not (e[1] == to_id and e[2] == to_port
                                  and e[3] == EDGE_DATA)]
        if edge not in self.edges:
            self.edges.append(edge)

    def remove_edge(self, from_id: str, to_id: str, to_port: int = 0,
                    kind: str = None):
        self.edges = [e for e in self.edges
                      if not (e[0] == from_id and e[1] == to_id
                              and e[2] == to_port
                              and (kind is None or e[3] == kind))]

    def set_edge_kind(self, from_id: str, to_id: str, to_port: int,
                      kind: str):
        self.edges = [(f, t, p, kind) if (f, t, p) == (from_id, to_id, to_port)
                      else (f, t, p, k) for f, t, p, k in self.edges]

    def data_edges(self) -> list:
        return [e for e in self.edges if e[3] == EDGE_DATA]

    def control_edges(self) -> list:
        return [e for e in self.edges if e[3] == EDGE_CONTROL]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _topological_order(self) -> list[str]:
        """Data edges only. Control edges are excluded by design: they
        normally point backwards, and ordering by them would either
        create a cycle or force the controller to run before the thing
        it controls."""
        indeg = {nid: 0 for nid in self.nodes}
        adj: dict[str, list] = {nid: [] for nid in self.nodes}
        for f, t, _p, _k in self.data_edges():
            if f in self.nodes and t in self.nodes:
                adj[f].append(t)
                indeg[t] += 1
        queue = [nid for nid, d in indeg.items() if d == 0]
        order = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for nxt in adj[nid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(self.nodes):
            raise RuntimeError("Pipeline contains a cycle in its data edges.")
        return order

    def _context_for(self, node: PipelineNode, frame_index: int,
                     total_frames: int, in_sequence: bool) -> FrameContext:
        last = node._last_index
        return FrameContext(
            index=frame_index,
            total=total_frames,
            in_sequence=in_sequence,
            is_first=last is None,
            is_rerun=(last == frame_index),
            jumped=(last is not None and frame_index != last + 1
                    and frame_index != last),
        )

    def run(self, frame_index: int = 0, total_frames: int = 1,
            on_step_done=None, in_sequence: bool = False) -> RunResult:
        """Execute the graph once, for one frame."""
        result = RunResult()
        result.frames_processed = 1
        tag = run_log.frame_tag(frame_index, total_frames)

        if run_log.is_on("params"):
            for node in self.nodes.values():
                vals = ", ".join(f"{k}={v!r}"
                                 for k, v in node.step.get_param_values().items())
                run_log.log("params", f"{tag} param {node.display_name}: {vals}")

        order = self._topological_order()

        predecessors: dict[str, dict] = {nid: {} for nid in self.nodes}
        for f, t, port, _k in self.data_edges():
            if f in self.nodes and t in self.nodes:
                predecessors[t][port] = f

        skipped: set = set()
        outbox: dict = {}      # node_id -> control values it sent this frame

        def warn(msg: str):
            if msg not in result.warnings:
                result.warnings.append(msg)

        for nid in order:
            node = self.nodes[nid]
            step = node.step
            is_source = step.IS_SOURCE
            is_metric = step.IS_METRIC
            n_inputs = step.N_INPUTS
            pred_map = predecessors[nid]

            step.ctx = self._context_for(node, frame_index, total_frames,
                                         in_sequence)
            step.inbox = node.inbox
            step._out_meta = None
            step._outbox = None

            if is_source:
                inputs, in_metas = [], []
            else:
                inputs, in_metas = [], []
                problem = None
                for i in range(n_inputs):
                    val, problem = self._resolve_input(
                        i, n_inputs, pred_map, result.images, skipped, step)
                    if problem is not None:
                        break
                    inputs.append(val)
                    in_metas.append(result.meta.get(pred_map.get(i), {}))
                if problem is not None:
                    warn(f"Skipped '{node.display_name}': {problem}")
                    skipped.add(nid)
                    continue

            if node.bypassed and not is_source and not is_metric:
                out = inputs[0]
                node.timing.add(0.0)
            else:
                t0 = time.perf_counter()
                try:
                    out = step.process(*inputs)
                except Exception as exc:
                    raise RuntimeError(
                        f"Error in '{node.display_name}': {exc}") from exc
                node.timing.add((time.perf_counter() - t0) * 1000.0)

            if is_metric and not isinstance(out, np.ndarray):
                # A metric measures; it does not draw. The value travels
                # as metadata (so any number of metrics accumulate side
                # by side without overlapping) and the image passes
                # through untouched. Insert an "Annotate Metrics" step to
                # render them.
                #
                # It goes through the step's own _out_meta rather than
                # straight into `merged`, so it is emitted by exactly the
                # same path as any other metadata — otherwise the value
                # reaches downstream nodes but never shows up as
                # something THIS node emitted.
                result.metrics[nid] = out
                if step._out_meta is None:
                    step._out_meta = {}
                step._out_meta[METRIC_META_PREFIX + node.display_name] = out
                out = inputs[0] if inputs else np.zeros((1, 1), np.float32)

            # Metadata: everything the inputs carried, plus what this step
            # emitted. Port 0 wins a key collision (it is the primary
            # input); a step that needs both reads image_a.meta directly.
            merged = {}
            for m in reversed(in_metas):
                merged.update(m)
            if step._out_meta:
                merged.update(step._out_meta)

            if isinstance(out, np.ndarray):
                if (np.issubdtype(out.dtype, np.floating) and out.size
                        and (float(out.max()) > 1.0 or float(out.min()) < 0.0)):
                    warn(f"'{node.display_name}': float output outside "
                         f"[0, 1] was clipped — scale the step's output "
                         f"(e.g. divide by 255) or return uint8/uint16.")
                out = Frame(to_float01(out), merged)

            result.images[nid] = out
            result.meta[nid] = merged
            node.last_meta = merged
            # What THIS node contributed, for the canvas side-channel
            # display — distinct from `merged`, which includes everything
            # inherited from upstream.
            node.last_emitted = dict(step._out_meta or {})
            node.last_control = dict(step._outbox or {})
            node._last_index = frame_index
            if step._outbox:
                outbox[nid] = dict(step._outbox)
            if on_step_done:
                on_step_done(nid, out)

        # Deliver control values along control edges, for the NEXT frame.
        for node in self.nodes.values():
            node.inbox = {}
        for f, t, _port, kind in self.edges:
            if kind != EDGE_CONTROL or f not in outbox:
                continue
            target = self.nodes.get(t)
            if target is not None:
                target.inbox.update(outbox[f])

        self._log_frame(tag, result, outbox, in_sequence)
        return result

    def _log_frame(self, tag: str, result: RunResult, outbox: dict,
                   in_sequence: bool = False):
        if run_log.is_on("metrics"):
            for mid, value in result.metrics.items():
                name = self.nodes[mid].display_name if mid in self.nodes else mid
                run_log.log("metrics", f"{tag} metric {name}: {value}")
        if run_log.is_on("control"):
            for sender, values in outbox.items():
                name = (self.nodes[sender].display_name
                        if sender in self.nodes else sender)
                run_log.log("control", f"{tag} control {name} -> {values}")
        if run_log.is_on("meta"):
            for nid, m in result.meta.items():
                if m:
                    name = (self.nodes[nid].display_name
                            if nid in self.nodes else nid)
                    run_log.log("meta", f"{tag} meta {name}: {m}")
        # Inside a batch, run_sequence() logs the DISTINCT warnings
        # instead — a node skipped on every frame would otherwise print
        # the same line once per frame.
        if not in_sequence:
            for w in result.warnings:
                run_log.log("warnings", f"{tag} warn {w}")

    @staticmethod
    def _port_label(port_index: int, n_inputs: int, step=None) -> str:
        """Human-readable port name matching the on-canvas labels."""
        declared = tuple(getattr(step, "INPUT_LABELS", ()) or ())
        if port_index < len(declared) and declared[port_index]:
            return f"input {declared[port_index]}"
        if n_inputs <= 1:
            return "input"
        letters = "ABCDEFGH"
        return (f"input {letters[port_index]}"
                if port_index < len(letters) else f"input {port_index}")

    def _resolve_input(self, port_index: int, n_inputs: int,
                       pred_map: dict, images: dict, skipped: set,
                       step=None):
        label = self._port_label(port_index, n_inputs, step)
        from_id = pred_map.get(port_index)
        if from_id is None:
            return None, f"{label} is not connected."
        if from_id in skipped:
            src = self.nodes.get(from_id)
            return None, (f"{label} comes from "
                          f"'{src.display_name if src else from_id}', "
                          f"which was skipped.")
        val = images.get(from_id)
        if not isinstance(val, np.ndarray):
            src = self.nodes.get(from_id)
            src_name = src.display_name if src else from_id
            return None, (
                f"{label} is connected to '{src_name}', which returned "
                f"{type(val).__name__} instead of an image — that step "
                f"produced no output (check its settings, e.g. an unset "
                f"or unreadable file path).")
        return val, None

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------
    def total_frames(self) -> int:
        counts = [n.step.frame_count() for n in self.nodes.values()
                  if n.step.IS_SOURCE]
        counts = [c for c in counts if c > 1]
        return min(counts) if counts else 1

    def begin_sequence(self, total_frames: int):
        for node in self.nodes.values():
            node.inbox = {}
            node._last_index = None
            node.step.begin_sequence(total_frames)

    def end_sequence(self):
        for node in self.nodes.values():
            node.step.end_sequence()

    def reset_timings(self):
        for node in self.nodes.values():
            node.timing.reset()

    def timing_summary(self) -> list:
        rows = [(n.display_name, n.timing.mean_ms, n.timing.total_ms)
                for n in self.nodes.values() if n.timing.count]
        grand = sum(t for _, _, t in rows) or 1.0
        rows.sort(key=lambda r: r[2], reverse=True)
        return [(name, mean, total, total / grand)
                for name, mean, total in rows]

    def run_sequence(self, on_frame_done=None, on_progress=None,
                     should_cancel=None) -> RunResult:
        """Run every frame of the shortest sequence source. Just a loop
        around run(), so there is one code path for graph execution."""
        total = self.total_frames()
        self.reset_timings()
        self.begin_sequence(total)

        batch = RunResult()
        seen_warnings: set = set()
        try:
            for i in range(total):
                if should_cancel is not None and should_cancel():
                    batch.cancelled = True
                    break
                frame = self.run(frame_index=i, total_frames=total,
                                 in_sequence=True)
                batch.images = frame.images
                batch.meta = frame.meta
                batch.metrics = frame.metrics
                for nid, value in frame.metrics.items():
                    batch.metric_series.setdefault(nid, []).append((i, value))
                for w in frame.warnings:
                    if w not in seen_warnings:
                        seen_warnings.add(w)
                        batch.warnings.append(w)
                        run_log.log("warnings",
                                    f"[{i}/{total - 1}] warn {w}")
                batch.frames_processed += 1
                run_log.log("progress",
                            f"[{i}/{total - 1}] frame done "
                            f"({batch.frames_processed}/{total})")
                if on_frame_done:
                    on_frame_done(i, total, frame)
                if on_progress:
                    on_progress(batch.frames_processed, total)
        finally:
            self.end_sequence()
            self._dump_metric_csvs(batch)
        return batch

    def _dump_metric_csvs(self, batch: RunResult):
        for nid, series in batch.metric_series.items():
            node = self.nodes.get(nid)
            if node is None or not series:
                continue
            vals = node.step.get_param_values()
            if not vals.get("dump_csv"):
                continue
            path = (vals.get("csv_path") or "").strip()
            if not path:
                path = node.display_name.replace(" ", "_") + ".csv"
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("frame,value\n")
                    for frame_index, v in series:
                        f.write(f"{frame_index},{v}\n")
            except Exception as exc:
                batch.warnings.append(
                    f"'{node.display_name}': could not write CSV "
                    f"to '{path}': {exc}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "class_name": n.class_name,
                    "params": n.step.get_param_values(),
                    "pos": list(n.pos),
                    "number": n.number,
                    "bypassed": n.bypassed,
                }
                for n in self.nodes.values()
            ],
            "edges": [list(e) for e in self.edges],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline":
        pipeline = cls()
        for n in data["nodes"]:
            step_cls = STEP_REGISTRY.get(n["class_name"])
            if step_cls is None:
                raise ValueError(
                    f"Unknown step type '{n['class_name']}' — "
                    "is the module imported?")
            step = step_cls()
            step.set_param_values(n.get("params", {}))
            number = n.get("number")
            if number is None:
                number = pipeline._next_number(step.NAME)
            else:
                pipeline._note_number(step.NAME, number)
            node = PipelineNode(step, node_id=n["id"],
                                pos=tuple(n.get("pos", (0, 0))),
                                number=number,
                                bypassed=n.get("bypassed", False))
            pipeline.nodes[node.id] = node
        for e in data.get("edges", []):
            e = list(e)
            if len(e) == 2:            # very old format
                e = [e[0], e[1], 0, EDGE_DATA]
            elif len(e) == 3:          # pre-control-edge format
                e = [e[0], e[1], e[2], EDGE_DATA]
            pipeline.edges.append(tuple(e))
        return pipeline

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
