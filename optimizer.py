"""
Parameter optimization: search the allowed range of selected step
parameters to minimize or maximize a metric node's value.

Deliberately dependency-free (no scipy) -- your deployment has already
had enough trouble with extra packages, and the pipeline objective is
non-differentiable and mildly noisy anyway, which rules out the
gradient-based methods scipy would add.

Two built-in search strategies, both gradient-free:

  random    uniform sampling of the whole box. Global, unbiased, good at
            finding the right basin, poor at pinning down the optimum.
  pattern   Hooke-Jeeves coordinate pattern search: probe +/- a step in
            each dimension, move if better, otherwise halve the step.
            Local, deterministic, good at refining -- but it only ever
            finds the basin it started in.

The default "random+pattern" spends the first third of the budget
sampling globally, then refines the best point found. That combination
handles the usual shape of these problems (a broad basin plus a shallow
optimum) far better than either half alone.

Everything here is Qt-free so it can be tested headlessly and reused
from a CLI; optimize_dialog.py is the UI on top.
"""
from dataclasses import dataclass, field
import math
import random


@dataclass
class OptimTarget:
    """One parameter to optimize."""
    node_id: str
    param_name: str
    lo: float
    hi: float
    is_int: bool = False
    label: str = ""

    def clamp(self, x: float):
        x = max(self.lo, min(self.hi, x))
        return int(round(x)) if self.is_int else float(x)


@dataclass
class OptimResult:
    best_values: dict = field(default_factory=dict)   # (node_id, param) -> value
    best_score: float = float("inf")                  # in MINIMIZED form
    best_metric: float = float("nan")                 # as the metric reported it
    start_metric: float = float("nan")
    evaluations: int = 0
    history: list = field(default_factory=list)       # [(eval_idx, metric)]
    cancelled: bool = False
    message: str = ""


def optimizable_params(pipeline) -> list:
    """Every numeric parameter in the graph that has a finite range --
    the only ones a search can be run over. Returns OptimTargets with
    labels ready for display."""
    out = []
    for nid, node in pipeline.nodes.items():
        for spec in node.step.PARAMS:
            if spec.kind not in ("int", "float"):
                continue
            if spec.min_value is None or spec.max_value is None:
                continue
            if spec.max_value <= spec.min_value:
                continue
            out.append(OptimTarget(
                node_id=nid,
                param_name=spec.name,
                lo=float(spec.min_value),
                hi=float(spec.max_value),
                is_int=(spec.kind == "int"),
                label=f"{node.display_name} — {spec.label}"))
    return out


def metric_nodes(pipeline) -> list:
    """(node_id, display_name) for every metric node — the possible
    optimization objectives."""
    return [(nid, node.display_name)
            for nid, node in pipeline.nodes.items()
            if getattr(node.step, "IS_METRIC", False)]


class ParameterOptimizer:
    """
    Evaluates the pipeline repeatedly with different parameter values.

    mode="frame"     one pipeline.run() per evaluation (fast)
    mode="sequence"  a full run_sequence() per evaluation, with the
                     metric aggregated across frames. This is
                     N_evals x N_frames pipeline runs -- accurate, but
                     budget it carefully.
    """

    def __init__(self, pipeline, targets: list, metric_node_id: str,
                 direction: str = "minimize", mode: str = "frame",
                 frame_index: int = 0, total_frames: int = 1,
                 aggregate: str = "mean", method: str = "random+pattern",
                 max_evals: int = 60, seed: int = 0):
        self.pipeline = pipeline
        self.targets = list(targets)
        self.metric_node_id = metric_node_id
        self.direction = direction
        self.mode = mode
        self.frame_index = frame_index
        self.total_frames = total_frames
        self.aggregate = aggregate
        self.method = method
        self.max_evals = max_evals
        self.rng = random.Random(seed)

        self._evals = 0
        self._history = []
        self._best_x = None
        self._best_score = float("inf")
        self._on_progress = None
        self._should_cancel = None
        self._cancelled = False

    # --- parameter application ---------------------------------------
    def _original_values(self) -> dict:
        out = {}
        for t in self.targets:
            step = self.pipeline.nodes[t.node_id].step
            out[(t.node_id, t.param_name)] = step.get_param_values().get(
                t.param_name)
        return out

    def _apply(self, x: list):
        for t, v in zip(self.targets, x):
            step = self.pipeline.nodes[t.node_id].step
            step.set_param_values({t.param_name: t.clamp(v)})

    def restore(self, values: dict):
        for (nid, pname), v in values.items():
            node = self.pipeline.nodes.get(nid)
            if node is not None:
                node.step.set_param_values({pname: v})

    # --- objective ----------------------------------------------------
    def _aggregate(self, series: list) -> float:
        vals = [v for _, v in series if isinstance(v, (int, float))]
        if not vals:
            return float("nan")
        if self.aggregate == "min":
            return float(min(vals))
        if self.aggregate == "max":
            return float(max(vals))
        if self.aggregate == "last":
            return float(vals[-1])
        return float(sum(vals) / len(vals))

    def measure(self) -> float:
        """Run the pipeline once (or once per frame) and return the
        metric as reported. NaN if it couldn't be obtained."""
        try:
            if self.mode == "sequence":
                series_out: dict = {}
                self.pipeline.run_sequence(
                    metric_series_out=series_out,
                    should_cancel=self._should_cancel)
                series = series_out.get(self.metric_node_id, [])
                return self._aggregate(series)
            values: dict = {}
            self.pipeline.run(frame_index=self.frame_index,
                              total_frames=self.total_frames,
                              metric_values_out=values)
            v = values.get(self.metric_node_id)
            return float(v) if isinstance(v, (int, float)) else float("nan")
        except Exception:
            # A parameter combination that makes a step raise is simply a
            # bad point, not a reason to abort the whole search.
            return float("nan")

    def _score(self, x: list) -> float:
        """Objective in MINIMIZED form (maximize is handled by negation).
        Unusable points score +inf so the search walks away from them."""
        if self._should_cancel is not None and self._should_cancel():
            self._cancelled = True
            return float("inf")
        self._apply(x)
        metric = self.measure()
        self._evals += 1
        if math.isnan(metric):
            score = float("inf")
        else:
            score = -metric if self.direction == "maximize" else metric
        self._history.append((self._evals, metric))
        if score < self._best_score:
            self._best_score = score
            self._best_x = [t.clamp(v) for t, v in zip(self.targets, x)]
        if self._on_progress is not None:
            self._on_progress(self._evals, self.max_evals, metric,
                              self._best_metric())
        return score

    def _best_metric(self) -> float:
        if self._best_score == float("inf"):
            return float("nan")
        return (-self._best_score if self.direction == "maximize"
                else self._best_score)

    # --- search strategies --------------------------------------------
    def _budget_left(self) -> int:
        return max(0, self.max_evals - self._evals)

    def _stop(self) -> bool:
        return self._cancelled or self._budget_left() <= 0

    def _random_search(self, n: int):
        for _ in range(n):
            if self._stop():
                return
            x = [t.clamp(self.rng.uniform(t.lo, t.hi)) for t in self.targets]
            self._score(x)

    def _pattern_search(self, x0: list):
        """Hooke-Jeeves: probe each axis, keep improvements, shrink the
        step when a full sweep finds nothing better."""
        x = list(x0)
        base = self._best_score
        steps = [(t.hi - t.lo) * 0.25 for t in self.targets]
        min_steps = [max((t.hi - t.lo) * 1e-3, 1.0 if t.is_int else 0.0)
                     for t in self.targets]
        while not self._stop():
            improved = False
            for i, t in enumerate(self.targets):
                if self._stop():
                    break
                for sign in (1.0, -1.0):
                    if self._stop():
                        break
                    trial = list(x)
                    trial[i] = t.clamp(x[i] + sign * steps[i])
                    if trial[i] == x[i]:
                        continue
                    s = self._score(trial)
                    if s < base:
                        base, x, improved = s, trial, True
                        break
            if not improved:
                steps = [s * 0.5 for s in steps]
                if all(s <= m for s, m in zip(steps, min_steps)):
                    return

    # --- entry point ---------------------------------------------------
    def run(self, on_progress=None, should_cancel=None) -> OptimResult:
        self._on_progress = on_progress
        self._should_cancel = should_cancel
        self._cancelled = False

        original = self._original_values()
        if not self.targets:
            return OptimResult(message="No parameters selected.")

        start_metric = self.measure()
        self._evals = 0          # the baseline probe isn't part of the budget

        try:
            if self.method == "random":
                self._random_search(self._budget_left())
            elif self.method == "pattern":
                x0 = [t.clamp(original[(t.node_id, t.param_name)] or t.lo)
                      for t in self.targets]
                self._score(x0)
                self._pattern_search(x0)
            else:   # random+pattern
                n_random = max(2, self.max_evals // 3)
                self._random_search(n_random)
                if self._best_x is not None and not self._stop():
                    self._pattern_search(self._best_x)
        finally:
            # Always leave the pipeline as we found it; the caller
            # decides whether to apply the result.
            self.restore(original)

        best_values = {}
        if self._best_x is not None:
            best_values = {(t.node_id, t.param_name): v
                           for t, v in zip(self.targets, self._best_x)}

        return OptimResult(
            best_values=best_values,
            best_score=self._best_score,
            best_metric=self._best_metric(),
            start_metric=start_metric,
            evaluations=self._evals,
            history=list(self._history),
            cancelled=self._cancelled,
            message=("Cancelled." if self._cancelled else
                     "No usable result — every evaluation failed."
                     if not best_values else "OK"))
