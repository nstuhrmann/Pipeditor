"""
Pipeline data model.

Edges are stored as (from_id, to_id, to_port_index) triples, which
allows fan-out (one output → many inputs) and multi-input nodes
(metric steps with ports A and B).
"""
from __future__ import annotations
import json
import uuid
import numpy as np
from src.GUI.pipeline_editor.base_step import STEP_REGISTRY, to_float01


def _annotate_metric(image: np.ndarray, value) -> np.ndarray:
    """Render the metric value as a text overlay onto (a copy of) the
    metric's first input. Text scale follows image height; drawn with a
    dark outline for readability on any content."""
    import cv2   # lazy: keep pipeline.py importable without OpenCV
    img = np.ascontiguousarray(image.copy())
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    h = img.shape[0]
    scale = max(0.4, h / 400.0)
    thickness = max(1, int(round(h / 300.0)))
    org = (8, int(28 * scale) + 6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if img.ndim == 2:
        outline, fill = 0.0, 1.0
    else:
        outline, fill = (0.0, 0.0, 0.0), (1.0, 0.85, 0.2)
    cv2.putText(img, text, org, font, scale, outline, thickness + 2,
                cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, fill, thickness, cv2.LINE_AA)
    return img


class MessageBus:
    """
    Side-band signaling between steps, for control flow that isn't image
    data — e.g. an auto-exposure step posting a correction that a camera
    control step applies on the NEXT frame:

        # in AutoExposure.process():
        self.bus.post("exposure_correction", factor)

        # in CameraControl.process():
        corr = self.bus.pop("exposure_correction", None)
        if corr is not None:
            self._apply_to_camera(corr)

    Semantics: post() overwrites the topic's value (latest wins), get()
    reads without consuming, pop() consumes. The bus persists across the
    frames of a batch (that's what makes feedback loops work: measure
    frame N, act on frame N+1) and is cleared at the start of every
    batch so stale preview-time messages don't leak in. Execution is
    single-threaded per frame, so no locking is needed.
    """
    def __init__(self):
        self._topics: dict = {}

    def post(self, topic: str, value):
        self._topics[topic] = value

    def get(self, topic: str, default=None):
        return self._topics.get(topic, default)

    def pop(self, topic: str, default=None):
        return self._topics.pop(topic, default)

    def clear(self):
        self._topics.clear()


class PipelineNode:
    def __init__(self, step, node_id: str | None = None, pos=(0, 0),
                 number: int = 1, bypassed: bool = False):
        self.id = node_id or str(uuid.uuid4())
        self.step = step
        self.pos = pos
        # 1-based sequential index among nodes of the same step NAME,
        # e.g. "Gaussian Blur 1", "Gaussian Blur 2" — assigned once at
        # creation time by Pipeline.add_node / restored on load.
        self.number = number
        # When True, Pipeline.run() skips process() entirely for this node
        # and passes its first input straight through — lets you A/B a step
        # in/out of the graph without rewiring edges.
        self.bypassed = bypassed

    @property
    def class_name(self):
        return self.step.__class__.__name__

    @property
    def display_name(self) -> str:
        return f"{self.step.NAME} {self.number}"


class Pipeline:
    def __init__(self):
        self.nodes: dict[str, PipelineNode] = {}
        # edges: list of (from_node_id, to_node_id, to_port_index)
        self.edges: list[tuple[str, str, int]] = []
        # step.NAME -> highest instance number assigned so far, so titles
        # like "Gaussian Blur 1", "Gaussian Blur 2" never repeat, even
        # after nodes are deleted or a pipeline is reloaded.
        self._name_counters: dict[str, int] = {}
        # side-band control messages between steps (see MessageBus)
        self.bus = MessageBus()

    # ------------------------------------------------------------------
    # Graph editing
    # ------------------------------------------------------------------

    def add_node(self, step, pos=(0, 0)) -> PipelineNode:
        number = self._next_number(step.NAME)
        node = PipelineNode(step, pos=pos, number=number)
        self.nodes[node.id] = node
        return node

    def _next_number(self, name: str) -> int:
        n = self._name_counters.get(name, 0) + 1
        self._name_counters[name] = n
        return n

    def _note_number(self, name: str, number: int):
        """Ensure future _next_number() calls continue past a restored number."""
        if number > self._name_counters.get(name, 0):
            self._name_counters[name] = number

    def remove_node(self, node_id: str):
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges
                      if e[0] != node_id and e[1] != node_id]

    def add_edge(self, from_id: str, to_id: str, to_port: int = 0):
        if from_id == to_id:
            return
        # Only one source per input port
        self.edges = [e for e in self.edges
                      if not (e[1] == to_id and e[2] == to_port)]
        self.edges.append((from_id, to_id, to_port))

    def remove_edge(self, from_id: str, to_id: str, to_port: int = 0):
        self.edges = [e for e in self.edges
                      if not (e[0] == from_id and e[1] == to_id
                              and e[2] == to_port)]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _topological_order(self) -> list[str]:
        incoming = {nid: 0 for nid in self.nodes}
        for f, t, _ in self.edges:
            incoming[t] += 1
        queue = [nid for nid, c in incoming.items() if c == 0]
        order = []
        remaining = list(self.edges)
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            outgoing = [e for e in remaining if e[0] == nid]
            for edge in outgoing:
                remaining.remove(edge)
                incoming[edge[1]] -= 1
                if incoming[edge[1]] == 0:
                    queue.append(edge[1])
        if len(order) != len(self.nodes):
            raise ValueError("Pipeline contains a cycle — not allowed.")
        return order

    def run(self, image: np.ndarray = None, on_step_done=None,
            frame_index: int = 0, total_frames: int = 1,
            warnings_out: "list[str] | None" = None,
            metric_values_out: "dict | None" = None
            ) -> dict[str, "np.ndarray | float | str"]:
        """
        Execute the pipeline once. Returns a dict mapping node_id to its
        output. For metric nodes the output is a float/str; for all
        others it is a numpy array.
        on_step_done(node_id, result) is called after each step if provided.

        `image` is accepted for backward compatibility but no longer used:
        sources generate/load their own data.

        frame_index/total_frames are only meaningful to IS_SEQUENCE_AWARE
        steps (video/image-stack sources and sinks) — every other step
        ignores them entirely. A plain single-frame preview call (the
        normal "Run" / Live Update path) leaves these at their defaults,
        which is indistinguishable from "frame 0 of 1" to such steps.
        run_sequence() below is what actually varies them across a batch.

        Wiring problems don't abort the run and don't silently substitute
        a dummy image (the old behavior, which produced plausible-looking
        garbage): a node with an unconnected input, or an input fed by a
        skipped node or by a node without an image output (a metric), is
        SKIPPED — it gets no entry in the results dict, and a
        human-readable reason is appended to warnings_out (if given).
        This keeps a half-wired scratch node from blocking live updates
        for the rest of the graph.

        A step that raises inside its own process() DOES abort the run,
        as RuntimeError with the failing node's display name in the
        message.
        """
        order = self._topological_order()

        # predecessors[to_id][port_index] = from_id
        predecessors: dict[str, dict[int, str]] = {nid: {} for nid in self.nodes}
        for f, t, p in self.edges:
            predecessors[t][p] = f

        results: dict[str, object] = {}
        skipped: set[str] = set()

        def warn(msg: str):
            if warnings_out is not None:
                warnings_out.append(msg)

        for nid in order:
            node = self.nodes[nid]
            is_source = getattr(node.step, "IS_SOURCE", False)
            is_metric = getattr(node.step, "IS_METRIC", False)
            n_inputs = getattr(node.step, "N_INPUTS", 1)
            pred_map = predecessors[nid]   # {port_index: from_id}

            node.step.bus = self.bus   # side-band messaging (see MessageBus)
            if getattr(node.step, "IS_SEQUENCE_AWARE", False):
                node.step.set_frame_index(frame_index, total_frames)

            if is_source:
                inputs = [np.zeros((1, 1, 3), dtype=np.uint8)]
            else:
                inputs = []
                problem = None
                for i in range(n_inputs):
                    val, problem = self._resolve_input(
                        i, n_inputs, pred_map, results, skipped)
                    if problem is not None:
                        break
                    inputs.append(val)
                if problem is not None:
                    warn(f"Skipped '{node.display_name}': {problem}")
                    skipped.add(nid)
                    continue

            # Bypass isn't meaningful for sources (nothing to pass through)
            # or metrics (no single output to pass through as); the UI
            # already prevents toggling it for those, this is just a
            # matching safety net.
            if node.bypassed and not is_source and not is_metric:
                out = inputs[0]
            else:
                try:
                    out = node.step.run(inputs)
                except Exception as exc:
                    # Attribute the failure to the node so the user knows
                    # *where* in the graph it happened, not just what.
                    raise RuntimeError(
                        f"Error in '{node.display_name}': {exc}") from exc

            if is_metric and not isinstance(out, np.ndarray):
                # A metric's graph output is its FIRST INPUT with the
                # value burned in as a text overlay — displayable and
                # chainable (e.g. into a writer) instead of a dead end.
                # The raw value is reported separately for the on-node
                # label and CSV dumping.
                if metric_values_out is not None:
                    metric_values_out[nid] = out
                out = _annotate_metric(inputs[0], out)

            # Inter-module contract: every image is float32 in [0, 1].
            # Int outputs are scaled by their dtype max; float outputs
            # outside [0, 1] violate the contract and are clipped with a
            # warning — never silently rescaled.
            if isinstance(out, np.ndarray):
                if (np.issubdtype(out.dtype, np.floating) and out.size
                        and (float(out.max()) > 1.0 or float(out.min()) < 0.0)):
                    # Stable text (no numbers) so run_sequence's dedup
                    # collapses this to one entry per node per batch.
                    warn(f"'{node.display_name}': float output outside "
                         f"[0, 1] was clipped — scale the step's output "
                         f"(e.g. divide by 255) or return uint8/uint16.")
                out = to_float01(out)
            results[nid] = out
            if on_step_done:
                on_step_done(nid, out)

        return results

    @staticmethod
    def _port_label(port_index: int, n_inputs: int) -> str:
        """Human-readable port name matching the on-canvas labels."""
        if n_inputs <= 1:
            return "input"
        letters = "ABCDEFGH"
        return (f"input {letters[port_index]}"
                if port_index < len(letters) else f"input {port_index}")

    def _resolve_input(self, port_index: int, n_inputs: int,
                       pred_map: dict, results: dict, skipped: set):
        """Resolve one input port to an upstream image. Returns
        (array, None) on success, or (None, reason) when the port can't
        be satisfied — an unconnected port, an upstream node that was
        itself skipped, or an upstream node without an image output."""
        label = self._port_label(port_index, n_inputs)
        from_id = pred_map.get(port_index)
        if from_id is None:
            return None, f"{label} is not connected."
        if from_id in skipped:
            src = self.nodes.get(from_id)
            src_name = src.display_name if src else from_id
            return None, (f"{label} comes from '{src_name}', "
                          f"which was skipped.")
        val = results.get(from_id)
        if not isinstance(val, np.ndarray):
            src = self.nodes.get(from_id)
            src_name = src.display_name if src else from_id
            return None, (f"{label} is connected to '{src_name}', which "
                          f"outputs a metric value "
                          f"({type(val).__name__}), not an image.")
        return val, None

    # ------------------------------------------------------------------
    # Sequence (video / image-stack) batch execution
    # ------------------------------------------------------------------

    def total_frames(self) -> int:
        """Number of frames available across every IS_SEQUENCE_AWARE source
        node. Ordinary single-image pipelines (no such source) → 1. If more
        than one sequence source exists, the shortest one wins — like
        processing stops when the shortest input runs out."""
        counts = [node.step.frame_count() for node in self.nodes.values()
                 if getattr(node.step, "IS_SOURCE", False)
                 and getattr(node.step, "IS_SEQUENCE_AWARE", False)]
        return min(counts) if counts else 1

    def begin_sequence(self, total_frames: int):
        for node in self.nodes.values():
            if getattr(node.step, "IS_SEQUENCE_AWARE", False):
                node.step.begin_sequence(total_frames)

    def end_sequence(self):
        for node in self.nodes.values():
            if getattr(node.step, "IS_SEQUENCE_AWARE", False):
                node.step.end_sequence()

    def run_sequence(self, on_frame_done=None, on_progress=None,
                     should_cancel=None,
                     warnings_out: "list[str] | None" = None,
                     metric_values_out: "dict | None" = None,
                     metric_series_out: "dict | None" = None) -> int:
        """
        Runs every frame of the shortest sequence source through the
        pipeline — deliberately just a loop around run(), so there is one
        code path for graph execution whether it's a single-frame preview
        or a full batch; nothing about how a node processes a frame
        differs between the two.

        on_frame_done(frame_index, total, results) — full results dict,
            e.g. for progress-preview thumbnails.
        on_progress(frames_done, total) — lighter-weight, for a progress bar.
        should_cancel() -> bool — checked before every frame; return True
            to stop early (e.g. user clicked Cancel).
        warnings_out — like run()'s, but deduplicated: a node skipped on
            every frame produces one entry, not one per frame.
        metric_values_out — filled with the LAST processed frame's metric
            values (node_id -> value), for the on-node labels.
        metric_series_out — filled with EVERY frame's metric values
            (node_id -> [(frame_index, value), ...]), e.g. for plotting
            or for optimizing a parameter against a whole sequence.

        Metric nodes whose 'dump_csv' parameter is enabled get their
        per-frame values written to their 'csv_path' (or
        '<node name>.csv' if unset) when the batch ends — including after
        cancellation, so partial series aren't lost.

        Returns the number of frames actually processed (may be less than
        total_frames() if cancelled). begin_sequence()/end_sequence() are
        called exactly once each, and end_sequence() always runs — even on
        an exception or cancellation — so writers/captures never leak.
        """
        total = self.total_frames()
        self.bus.clear()   # no stale preview-time messages in the batch
        self.begin_sequence(total)
        processed = 0
        seen_warnings: set[str] = set()
        metric_series: dict[str, list] = {}   # nid -> [(frame, value), ...]
        try:
            for i in range(total):
                if should_cancel is not None and should_cancel():
                    break
                frame_warnings: "list[str] | None" = (
                    [] if warnings_out is not None else None)
                frame_metrics: dict = {}
                results = self.run(frame_index=i, total_frames=total,
                                   warnings_out=frame_warnings,
                                   metric_values_out=frame_metrics)
                for nid, v in frame_metrics.items():
                    metric_series.setdefault(nid, []).append((i, v))
                if metric_values_out is not None:
                    metric_values_out.clear()
                    metric_values_out.update(frame_metrics)
                if frame_warnings:
                    for w in frame_warnings:
                        if w not in seen_warnings:
                            seen_warnings.add(w)
                            warnings_out.append(w)
                processed += 1
                if on_frame_done:
                    on_frame_done(i, total, results)
                if on_progress:
                    on_progress(processed, total)
        finally:
            self.end_sequence()
            if metric_series_out is not None:
                metric_series_out.clear()
                metric_series_out.update(metric_series)
            self._dump_metric_csvs(metric_series, warnings_out)
        return processed

    def _dump_metric_csvs(self, metric_series: dict,
                          warnings_out: "list[str] | None"):
        """Write per-frame metric values to CSV for metric nodes whose
        auto-injected 'dump_csv' parameter is enabled."""
        for nid, series in metric_series.items():
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
                    for frame, v in series:
                        f.write(f"{frame},{v}\n")
            except Exception as exc:
                if warnings_out is not None:
                    warnings_out.append(
                        f"'{node.display_name}': could not write CSV "
                        f"to '{path}': {exc}")

    # ------------------------------------------------------------------
    # Serialisation
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
                    "is the module imported?"
                )
            step = step_cls()
            step.set_param_values(n.get("params", {}))
            # Older saved pipelines won't have a "number" field — fall back
            # to auto-assigning one so titles still come out sequential.
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
            # support old 2-element edges from previous versions
            if len(e) == 2:
                e = [e[0], e[1], 0]
            pipeline.edges.append(tuple(e))
        return pipeline

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
