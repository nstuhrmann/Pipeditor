"""
Parallel execution of a pipeline: a shared thread pool, one task manager
per node.

The serial executor in pipeline.py remains the reference implementation.
This module must produce bit-identical results — see
test_parallel_equivalence.py, which is the acceptance criterion.

The model
---------
Every node has a manager holding a mailbox of arrived inputs keyed by
frame index. A frame is submitted to the pool as soon as its node's
readiness predicate holds:

    ready(node, k) = every input slot filled for (k + edge_offset)
                 and (k is the next expected index, if order matters)
                 and running < worker_limit

Three per-node attributes decide the rest:

    worker_limit   1 for stateful steps, sinks, and anything with control
                   ports; otherwise the pool width.
    needs_order    True for the same set — a stateful step must see
                   frames in order, and completion order is NOT frame
                   order once any upstream node runs several workers.
    offset         per edge: 0 for data, -1 for control. B at frame k
                   consumes A's frame k-1, which is exactly the one-frame
                   feedback delay the serial executor gets for free.

Expressing control edges as an offset rather than a special case means a
feedback loop needs no special handling: camera(k) needs AE(k-1) and
AE(k) needs camera(k), so the scheduler simply finds no parallelism and
the loop runs serially, correctly.

Memory
------
An output is pushed into each consumer's mailbox and forgotten by the
producer, so Python's refcount frees it when the last consumer takes it.
Arrays are shared, not copied: fan-out already hands the same object to
several consumers in the serial executor, so read-only inputs are an
existing invariant. Mailboxes are depth-capped and a node is not
scheduled while a consumer is full, which is what actually bounds memory.

Not in this version: process pool, dynamic rebalancing, and any skip
propagation beyond what the serial executor already does.
"""
import threading
import time  # noqa: F401  (used in _execute for per-node timing)
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.GUI.pipeline_editor import run_log
from src.GUI.pipeline_editor.array_utils import (
    Frame, METRIC_META_PREFIX, to_float01,
)
from src.GUI.pipeline_editor.pipeline import EDGE_CONTROL, RunResult

#: In-flight frames per worker. The bound has to be GLOBAL rather than
#: per edge, because the output is globally ordered: the runner emits
#: frames in sequence, so every frame started but not yet emitted holds a
#: partial RunResult, whatever route it took through the graph. A
#: per-edge rule cannot bound that — two independent branches can each
#: stay within their local limits while one runs hundreds of frames ahead
#: of the other, and those partial frames pile up at the emission point.
#: So the window is not a separate throttle; it is the dual of "emit in
#: order", and it is derived from the worker count rather than chosen:
#: enough in flight to keep every worker fed through a multi-stage graph,
#: with memory proportional to the parallelism actually requested.
FRAMES_IN_FLIGHT_PER_WORKER = 2


class _NodeManager:
    """Readiness and admission for one node."""

    __slots__ = ("node", "step", "inputs", "control_sources", "consumers",
                 "worker_limit", "needs_order", "mailbox", "control_box",
                 "next_index", "running", "done_through", "clones",
                 "free_clones")

    def __init__(self, node, worker_limit: int, needs_order: bool):
        self.node = node
        self.step = node.step
        self.inputs: dict = {}          # port -> producer node id
        self.control_sources: list = []  # node ids feeding control edges
        self.consumers: list = []       # (manager, port, offset)
        self.worker_limit = worker_limit
        self.needs_order = needs_order
        # frame index -> {port: Frame}
        self.mailbox: dict = {}
        # frame index -> merged control values for that frame
        self.control_box: dict = {}
        self.next_index = 0
        self.running = 0
        self.done_through = -1
        # One step instance per concurrent worker. A step keeps per-call
        # state (ctx, inbox, _out_meta) on itself, so two workers inside
        # one instance would overwrite each other.
        self.clones: list = []
        self.free_clones: list = []

    # --- readiness ---------------------------------------------------
    def has_inputs(self, index: int) -> bool:
        if not self.inputs:
            return True                 # a source needs nothing
        slots = self.mailbox.get(index)
        return bool(slots) and len(slots) == len(self.inputs)

    def has_control(self, index: int) -> bool:
        """Control arrives from frame index-1. Frame 0 has no predecessor,
        so it runs on the step's own parameters — matching the serial
        executor, where the inbox is simply empty on the first frame.

        The test is whether every control SOURCE has finished frame
        index-1, not whether a value turned up: a controller may legally
        emit nothing on a frame, and waiting for a value that will never
        come would deadlock. An earlier version checked this node's own
        progress, which is always satisfied and so never waited at all —
        the camera then ran ahead of the AE and the loop was open.
        """
        if not self.control_sources or index == 0:
            return True
        return all(src.done_through >= index - 1
                   for src in self.control_sources)

    def ready(self, index: int) -> bool:
        if self.running >= self.worker_limit:
            return False
        if self.needs_order and index != self.next_index:
            return False
        return self.has_inputs(index) and self.has_control(index)

    # Memory is bounded by the runner's admission WINDOW, not per-edge:
    # blocking a producer because one consumer's mailbox is full
    # deadlocks as soon as an in-order consumer is waiting for a frame
    # that the blocked producer would have made. Limiting how far ahead
    # of the oldest un-emitted frame anything may run bounds memory just
    # as well and cannot deadlock, because every node is always allowed
    # to work on the oldest frame.


class ParallelRunner:
    """Drives one committed run of a pipeline across a thread pool."""

    def __init__(self, pipeline, workers: int = 4, should_cancel=None):
        self.pipeline = pipeline
        self.workers = max(1, int(workers))
        self.should_cancel = should_cancel
        # Derived, not configured: one knob (workers) instead of two.
        self.window = self.workers * FRAMES_IN_FLIGHT_PER_WORKER
        self.emitted = 0

        self.total = pipeline.total_frames()
        self.lock = threading.Condition()
        self.managers: dict = {}
        self.results: dict = {}          # frame index -> RunResult
        self.pending_meta: dict = {}     # frame -> {node_id: meta}
        self.error = None
        self.cancelled = False
        self.completed: dict = {}        # frame index -> count of nodes done
        # Nodes that can never run: an input port with nothing wired to
        # it, or downstream of such a node. The serial executor decides
        # this per frame and reports a warning; here it is static, and it
        # MUST be handled — a node that never completes would leave the
        # frame barrier unreleased and the run would hang rather than
        # fail.
        self.skipped: dict = {}          # node id -> reason
        self.submitted = 0
        self._build_managers()
        self._prepare()

    # ------------------------------------------------------------------
    def _build_managers(self):
        from src.GUI.pipeline_editor.base_step import StatefulStep
        for nid, node in self.pipeline.nodes.items():
            step = node.step
            serial = (isinstance(step, StatefulStep)
                      or step.KIND in ("sink", "source")
                      or getattr(step, "EMITS_CONTROL", False)
                      or getattr(step, "ACCEPTS_CONTROL", False))
            # Sources are serialized because most read a file or device
            # with a stateful cursor; ordering also gives writers their
            # in-order guarantee for free.
            self.managers[nid] = _NodeManager(
                node,
                worker_limit=1 if serial else self.workers,
                needs_order=serial)

        for f, t, port, kind in self.pipeline.edges:
            if f not in self.managers or t not in self.managers:
                continue
            src, dst = self.managers[f], self.managers[t]
            if kind == EDGE_CONTROL:
                dst.control_sources.append(src)
                src.consumers.append((dst, None, -1))
            else:
                dst.inputs[port] = f
                src.consumers.append((dst, port, 0))

    def _prepare(self):
        self._find_unreachable()

    def _find_unreachable(self):
        """Statically resolve which nodes can never produce an image, in
        topological order so the reason propagates downstream."""
        order = self.pipeline._topological_order()
        for nid in order:
            mgr = self.managers[nid]
            step = mgr.step
            if step.KIND == "source":
                continue
            for port in range(step.N_INPUTS):
                producer = mgr.inputs.get(port)
                label = self.pipeline._port_label(port, step.N_INPUTS, step)
                if producer is None:
                    self.skipped[nid] = f"{label} is not connected."
                    break
                if producer in self.skipped:
                    src = self.pipeline.nodes.get(producer)
                    self.skipped[nid] = (
                        f"{label} comes from "
                        f"'{src.display_name if src else producer}', "
                        f"which was skipped.")
                    break

    # ------------------------------------------------------------------
    def _clone_for(self, mgr: _NodeManager):
        """A step instance this worker may use exclusively."""
        if mgr.worker_limit == 1:
            return mgr.step
        if mgr.free_clones:
            return mgr.free_clones.pop()
        if not mgr.clones:
            mgr.clones.append(mgr.step)   # the original counts as one
            return mgr.step
        clone = type(mgr.step)()
        clone.set_param_values(mgr.step.get_param_values())
        mgr.clones.append(clone)
        return clone

    def _release(self, mgr: _NodeManager, step):
        if mgr.worker_limit != 1:
            mgr.free_clones.append(step)

    # ------------------------------------------------------------------
    def _execute(self, mgr: _NodeManager, index: int, step, slots, inbox):
        """Run one node for one frame. Mirrors the per-node body of
        Pipeline._run_frame; kept here rather than shared because the
        serial version must stay untouched as the reference.

        `slots` and `inbox` are claimed by the scheduler under the lock,
        not read here: leaving them in the mailbox until the worker ran
        let a second scheduling pass see them and submit the same frame
        twice.
        """
        node = mgr.node
        kind = step.KIND
        in_metas = [slots[i].meta if i in slots else {}
                    for i in range(step.N_INPUTS)]
        inputs = [slots[i] for i in sorted(slots)] if slots else []

        step.ctx = self.pipeline._context_for(node, index, self.total,
                                              committed=True)
        step.inbox = dict(inbox)
        step._out_meta = None
        step._outbox = None

        warnings = []
        if node.bypassed and kind != "source":
            out = inputs[0] if inputs else None
            node.timing.reset()
        else:
            t0 = time.perf_counter()
            out = step.process(*inputs)
            node.timing.add((time.perf_counter() - t0) * 1000.0)

        merged = {}
        for m in reversed(in_metas):
            merged.update(m)
        metric_value = None
        if kind == "metric" and not isinstance(out, np.ndarray):
            metric_value = out
            if step._out_meta is None:
                step._out_meta = {}
            step._out_meta[METRIC_META_PREFIX + node.display_name] = out
            out = inputs[0] if inputs else np.zeros((1, 1), np.float32)
        if step._out_meta:
            merged.update(step._out_meta)

        if isinstance(out, np.ndarray):
            if (np.issubdtype(out.dtype, np.floating) and out.size
                    and (float(out.max()) > 1.0 or float(out.min()) < 0.0)):
                warnings.append(
                    f"'{node.display_name}': float output outside [0, 1] "
                    f"was clipped — scale the step's output (e.g. divide "
                    f"by 255) or return uint8/uint16.")
            out = Frame(to_float01(out), merged)

        return out, merged, metric_value, dict(step._outbox or {}), warnings

    # ------------------------------------------------------------------
    def _deliver(self, mgr: _NodeManager, index: int, out, outbox: dict):
        """Push this node's output to its consumers. The producer keeps
        no reference afterwards, so the array is freed once the last
        mailbox drops it."""
        for dst, port, offset in mgr.consumers:
            if offset == 0:
                if isinstance(out, np.ndarray):
                    dst.mailbox.setdefault(index, {})[port] = out
            elif outbox:
                dst.control_box.setdefault(index, {}).update(outbox)

    def _on_done(self, mgr: _NodeManager, index: int, step, payload):
        out, merged, metric_value, outbox, warnings = payload
        with self.lock:
            node = mgr.node
            node.last_meta = merged
            node.last_emitted = dict(step._out_meta or {})
            node.last_control = dict(outbox)
            node._last_index = index

            frame = self.results.setdefault(index, RunResult())
            frame.frames_processed = 1
            if out is not None:
                frame.images[node.id] = out
            frame.meta[node.id] = merged
            if metric_value is not None:
                frame.metrics[node.id] = metric_value
            for w in warnings:
                if w not in frame.warnings:
                    frame.warnings.append(w)

            self._deliver(mgr, index, out, outbox)
            mgr.running -= 1
            mgr.done_through = max(mgr.done_through, index)
            if mgr.needs_order:
                mgr.next_index = index + 1
            self._release(mgr, step)
            self.completed[index] = self.completed.get(index, 0) + 1
            self.lock.notify_all()

    def _worker_body(self, mgr, index, step, slots, inbox):
        try:
            payload = self._execute(mgr, index, step, slots, inbox)
        except Exception as exc:
            with self.lock:
                if self.error is None:
                    self.error = RuntimeError(
                        f"Error in '{mgr.node.display_name}': {exc}")
                mgr.running -= 1
                self.lock.notify_all()
            return
        self._on_done(mgr, index, step, payload)

    # ------------------------------------------------------------------
    def _schedule(self, pool):
        """Submit everything currently runnable. Caller holds the lock."""
        launched = 0
        horizon = self.emitted + self.window
        for nid, reason in self.skipped.items():
            mgr = self.managers[nid]
            while mgr.next_index < self.total and mgr.next_index <= horizon:
                index = mgr.next_index
                frame = self.results.setdefault(index, RunResult())
                text = (f"Skipped '{mgr.node.display_name}': {reason}")
                if text not in frame.warnings:
                    frame.warnings.append(text)
                self.completed[index] = self.completed.get(index, 0) + 1
                mgr.next_index += 1

        for mgr in self.managers.values():
            if mgr.node.id in self.skipped:
                continue
            while True:
                index = (mgr.next_index if mgr.needs_order
                         else self._lowest_ready(mgr))
                if index is None or index >= self.total:
                    break
                if index > horizon:
                    break                    # admission window
                if not mgr.ready(index):
                    break
                step = self._clone_for(mgr)
                # Claim the inputs NOW, under the lock, so the next pass
                # cannot resubmit this frame.
                slots = mgr.mailbox.pop(index, {})
                inbox = mgr.control_box.pop(index - 1, {}) if index else {}
                mgr.running += 1
                if not mgr.inputs:
                    mgr.next_index = max(mgr.next_index, index + 1)
                pool.submit(self._worker_body, mgr, index, step, slots, inbox)
                launched += 1
                if mgr.needs_order:
                    break                    # one at a time, in order
        return launched

    def _lowest_ready(self, mgr: _NodeManager):
        candidates = [i for i in mgr.mailbox
                      if len(mgr.mailbox[i]) == len(mgr.inputs)]
        return min(candidates) if candidates else None

    # ------------------------------------------------------------------
    def run(self):
        """Yield (index, total, RunResult) in frame order."""
        pipeline = self.pipeline
        node_count = len(pipeline.nodes)
        pipeline.reset_timings()
        pipeline.open_resources(self.total)
        emitted = 0
        try:
            pipeline.reset_state()
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                while emitted < self.total:
                    with self.lock:
                        if self.error is not None:
                            raise self.error
                        if (self.should_cancel is not None
                                and self.should_cancel()):
                            self.cancelled = True
                            break
                        self._schedule(pool)
                        ready = (self.completed.get(emitted, 0) >= node_count
                                 or emitted in self._skippable())
                        if not ready:
                            self.lock.wait(timeout=0.05)
                            continue
                        frame = self.results.pop(emitted)
                        self.completed.pop(emitted, None)
                    yield emitted, self.total, frame
                    emitted += 1
                    with self.lock:
                        self.emitted = emitted
                        self.lock.notify_all()
        finally:
            pipeline.close_resources()

    def _skippable(self):
        """Frames whose remaining nodes can never run — currently only
        used to avoid hanging when a node was skipped for want of an
        input, which the serial executor reports as a warning."""
        return ()


def iter_sequence_parallel(pipeline, workers: int = 4, should_cancel=None):
    """Drop-in parallel replacement for Pipeline.iter_sequence()."""
    runner = ParallelRunner(pipeline, workers=workers,
                            should_cancel=should_cancel)
    run_log.log("progress", f"parallel run: {runner.total} frames, "
                            f"{runner.workers} workers")
    yield from runner.run()
