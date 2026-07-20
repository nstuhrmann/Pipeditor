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
class Objective:
    """One metric node contributing to the combined score.

    weight is in [-1, 1] and the optimizer ALWAYS minimizes the weighted
    sum, so the sign expresses the goal:
        +1  minimize this metric      (e.g. Delta E)
        -1  maximize this metric      (e.g. SSIM, TMQI)
         0  ignore it
    Intermediate values trade the objectives off against each other.
    """
    node_id: str
    weight: float = 1.0
    label: str = ""


@dataclass
class OptimResult:
    best_values: dict = field(default_factory=dict)   # (node_id, param) -> value
    best_score: float = float("inf")                  # weighted sum, minimized
    start_score: float = float("nan")
    best_metrics: dict = field(default_factory=dict)  # node_id -> raw value
    start_metrics: dict = field(default_factory=dict)
    evaluations: int = 0
    history: list = field(default_factory=list)       # [(eval_idx, score)]
    cancelled: bool = False
    message: str = ""


def bayes_available() -> bool:
    """True if scikit-optimize is installed, i.e. the "bayes" method can
    be offered. It is an OPTIONAL dependency: it pulls in scikit-learn,
    which is heavy and awkward to bundle into a frozen build, so the
    built-in methods stay the default."""
    try:
        import skopt  # noqa: F401
        return True
    except Exception:
        return False


def available_methods() -> list:
    methods = ["random+pattern", "random", "pattern"]
    if bayes_available():
        methods.append("bayes")
    return methods


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

    def __init__(self, pipeline, targets: list, objectives=None,
                 metric_node_id: str = None, direction: str = "minimize",
                 mode: str = "frame",
                 frame_index: int = 0, total_frames: int = 1,
                 aggregate: str = "mean", method: str = "random+pattern",
                 max_evals: int = 60, seed: int = 0,
                 normalize: bool = False):
        """objectives: list[Objective] -- the weighted sum that gets
        minimized. metric_node_id/direction remain as a single-objective
        shorthand.

        normalize=True divides each metric by |its starting value| before
        weighting, so metrics on very different scales (Delta E ~0-100 vs
        SSIM ~0-1) contribute comparably and the weights express relative
        importance rather than being swamped by whichever metric happens
        to have the larger numbers."""
        self.pipeline = pipeline
        self.targets = list(targets)
        if objectives:
            self.objectives = list(objectives)
        elif metric_node_id is not None:
            self.objectives = [Objective(
                metric_node_id,
                -1.0 if direction == "maximize" else 1.0)]
        else:
            self.objectives = []
        self.normalize = normalize
        self._norm = {}          # node_id -> divisor from the baseline
        self.mode = mode
        self.frame_index = frame_index
        self.total_frames = total_frames
        self.aggregate = aggregate
        self.method = method
        self.max_evals = max_evals
        self.seed = seed
        self.rng = random.Random(seed)

        self._evals = 0
        self._history = []
        self._best_x = None
        self._best_score = float("inf")
        self._best_metrics = {}
        self._x_start = []       # current values, as a search candidate
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

    def measure(self) -> dict:
        """Run the pipeline once (or once per frame) and return every
        objective metric's value: {node_id: float}. Missing/failed
        metrics come back as NaN."""
        out = {o.node_id: float("nan") for o in self.objectives}
        try:
            if self.mode == "sequence":
                series_out: dict = {}
                self.pipeline.run_sequence(
                    metric_series_out=series_out,
                    should_cancel=self._should_cancel)
                for o in self.objectives:
                    out[o.node_id] = self._aggregate(
                        series_out.get(o.node_id, []))
                return out
            values: dict = {}
            self.pipeline.run(frame_index=self.frame_index,
                              total_frames=self.total_frames,
                              metric_values_out=values)
            for o in self.objectives:
                v = values.get(o.node_id)
                out[o.node_id] = (float(v) if isinstance(v, (int, float))
                                  else float("nan"))
            return out
        except Exception:
            # A parameter combination that makes a step raise is simply a
            # bad point, not a reason to abort the whole search.
            return out

    def combine(self, metrics: dict) -> float:
        """Weighted sum, always minimized. Any NaN among the WEIGHTED
        objectives makes the whole point unusable (+inf) -- a partial
        score would let the search prefer combinations that simply broke
        one of the metrics."""
        total = 0.0
        for o in self.objectives:
            if not o.weight:
                continue
            v = metrics.get(o.node_id, float("nan"))
            if v is None or math.isnan(v):
                return float("inf")
            total += o.weight * (v / self._norm.get(o.node_id, 1.0))
        return total

    def _score(self, x: list) -> float:
        """Objective in MINIMIZED form (maximize is handled by negation).
        Unusable points score +inf so the search walks away from them."""
        if self._should_cancel is not None and self._should_cancel():
            self._cancelled = True
            return float("inf")
        self._apply(x)
        metrics = self.measure()
        score = self.combine(metrics)
        self._evals += 1
        self._history.append((self._evals, score))
        if score < self._best_score:
            self._best_score = score
            self._best_metrics = dict(metrics)
            self._best_x = [t.clamp(v) for t, v in zip(self.targets, x)]
        if self._on_progress is not None:
            self._on_progress(self._evals, self.max_evals, score,
                              self._best_score)
        return score

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

    def _bayes_search(self):
        """Bayesian optimization via scikit-optimize's gp_minimize.

        Worth the optional dependency because every evaluation here is a
        full pipeline run (in sequence mode, one run PER FRAME), so
        sample efficiency matters far more than optimizer overhead: this
        typically finds a good point in tens of evaluations where random
        search needs hundreds.

        Two adaptations are needed:
          * a GP can't be fitted through +inf, which is what failed
            evaluations score -- they're mapped to a finite penalty
            derived from the worst score actually observed;
          * gp_minimize owns the loop, so cancellation goes through its
            callback (returning True stops it) rather than our own
            _stop() checks.
        """
        from skopt import gp_minimize                      # optional dep
        from skopt.space import Real, Integer

        space = [Integer(int(round(t.lo)), int(round(t.hi)))
                 if t.is_int else Real(float(t.lo), float(t.hi))
                 for t in self.targets]

        worst_finite = [0.0]

        def objective(x):
            s = self._score([float(v) for v in x])
            if math.isinf(s) or math.isnan(s):
                # Finite stand-in so the surrogate model still fits, but
                # clearly worse than anything real that's been seen.
                return abs(worst_finite[0]) * 10.0 + 1.0
            worst_finite[0] = max(worst_finite[0], abs(s))
            return s

        def cancel_cb(_res):
            return bool(self._should_cancel is not None
                        and self._should_cancel())

        # Start from the CURRENT parameter values: they're usually a
        # sensible point, and it anchors the surrogate model somewhere
        # meaningful instead of purely at random.
        x0 = None
        if self._x_start:
            x0 = [int(round(v)) if t.is_int else float(v)
                  for t, v in zip(self.targets, self._x_start)]

        n_calls = max(6, self.max_evals)
        gp_minimize(objective, space,
                    n_calls=n_calls,
                    n_initial_points=min(10, max(4, n_calls // 4)),
                    x0=x0,
                    random_state=self.seed,
                    callback=[cancel_cb])

    # --- entry point ---------------------------------------------------
    def run(self, on_progress=None, should_cancel=None) -> OptimResult:
        self._on_progress = on_progress
        self._should_cancel = should_cancel
        self._cancelled = False

        original = self._original_values()
        if not self.targets:
            return OptimResult(message="No parameters selected.")
        if not any(o.weight for o in self.objectives):
            return OptimResult(
                message="No objective — give at least one metric a "
                        "non-zero weight.")

        start_metrics = self.measure()
        if self.normalize:
            # Scale each metric by its own starting magnitude so the
            # weights express importance instead of being dominated by
            # whichever metric has the larger numeric range.
            for o in self.objectives:
                v = start_metrics.get(o.node_id)
                self._norm[o.node_id] = (abs(v) if v and not math.isnan(v)
                                         and abs(v) > 1e-12 else 1.0)
        start_score = self.combine(start_metrics)
        self._evals = 0          # the baseline probe isn't part of the budget

        # Register the starting point as a candidate. Without this the
        # search could finish with a result WORSE than the user's current
        # settings, and "Apply Result" would silently degrade them.
        self._x_start = [t.clamp(original.get((t.node_id, t.param_name))
                                 if isinstance(original.get(
                                     (t.node_id, t.param_name)), (int, float))
                                 else (t.lo + t.hi) / 2)
                         for t in self.targets]
        if start_score < self._best_score:
            self._best_score = start_score
            self._best_metrics = dict(start_metrics)
            self._best_x = list(self._x_start)

        try:
            if self.method == "bayes":
                self._bayes_search()
            elif self.method == "random":
                self._random_search(self._budget_left())
            elif self.method == "pattern":
                self._pattern_search(list(self._x_start))
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
            start_score=start_score,
            best_metrics=dict(self._best_metrics),
            start_metrics=dict(start_metrics),
            evaluations=self._evals,
            history=list(self._history),
            cancelled=self._cancelled,
            message=("Cancelled." if self._cancelled else
                     "No usable result — every evaluation failed."
                     if not best_values else "OK"))
