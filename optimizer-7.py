"""
Parameter optimization: search the allowed ranges of selected step
parameters to minimize a weighted sum of metric-node values.

Deliberately slim: no hand-rolled search loops, exactly three
established optimizers behind one interface, named after the functions
they wrap --

  minimize                 scipy.optimize.minimize (Powell, bounded),
                           starting from the CURRENT parameter values.
                           Sample-efficient local refinement; the right
                           tool for many correlated continuous
                           parameters (e.g. a colour correction matrix).
  differential_evolution   scipy.optimize.differential_evolution.
                           Global, population-based; use when the
                           landscape has several basins or the current
                           values are far from any optimum.
  gp_minimize              skopt.gp_minimize (optional dependency).
                           Global, model-based; wins when a single
                           evaluation is expensive (sequence mode, large
                           frames). Leave skopt out of a frozen build
                           and the method simply isn't offered.

Everything here is Qt-free so it can be tested headlessly and reused
from a CLI; optimize_dialog.py is the UI on top.
"""
from dataclasses import dataclass, field
import math


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


def local_available() -> bool:
    """True if scipy is installed, enabling the "local" method."""
    try:
        from scipy.optimize import minimize  # noqa: F401
        return True
    except Exception:
        return False


def available_methods() -> list:
    methods = []
    if local_available():
        methods += ["minimize", "differential_evolution"]
    if bayes_available():
        methods.append("gp_minimize")
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


class _AbortSearch(Exception):
    """Raised inside an objective wrapper to hard-stop an external
    optimizer (scipy) on cancellation/budget; caught by the caller."""


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
                 mode: str = "frame",
                 frame_index: int = 0, total_frames: int = 1,
                 aggregate: str = "mean", method: str = "minimize",
                 max_evals: int = 60, seed: int = 0,
                 normalize: bool = False):
        """objectives: list[Objective] -- the weighted sum that gets
        minimized.

        normalize=True divides each metric by |its starting value| before
        weighting, so metrics on very different scales (Delta E ~0-100 vs
        SSIM ~0-1) contribute comparably and the weights express relative
        importance rather than being swamped by whichever metric happens
        to have the larger numbers."""
        self.pipeline = pipeline
        self.targets = list(targets)
        self.objectives = list(objectives or [])
        self.normalize = normalize
        self._norm = {}          # node_id -> divisor from the baseline
        self.mode = mode
        self.frame_index = frame_index
        self.total_frames = total_frames
        self.aggregate = aggregate
        self.method = method
        self.max_evals = max_evals
        self.seed = seed

        self._evals = 0
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
        if score < self._best_score:
            self._best_score = score
            self._best_metrics = dict(metrics)
            self._best_x = [t.clamp(v) for t, v in zip(self.targets, x)]
        if self._on_progress is not None:
            self._on_progress(self._evals, self.max_evals, score,
                              self._best_score, self.best_snapshot())
        return score

    def best_snapshot(self) -> dict:
        """The current best point, for live progress reporting: each
        metric's own value plus the parameter values that produced it.
        The combined score alone hides which objective is actually
        improving and what the search is converging on."""
        values = {}
        if self._best_x is not None:
            values = {(t.node_id, t.param_name): v
                      for t, v in zip(self.targets, self._best_x)}
        return {"best_metrics": dict(self._best_metrics),
                "best_values": values}

    # --- search strategies --------------------------------------------
    def _stop(self) -> bool:
        return self._cancelled or self._evals >= self.max_evals

    def _local_search(self):
        """Bounded local search via scipy (Powell) from the CURRENT
        values.

        This is the right tool for problems like fitting a colour
        correction matrix: many continuous, strongly correlated
        parameters whose optimum lies near a sensible starting point.
        Coordinate pattern search struggles there because it only moves
        one axis at a time; Powell builds conjugate directions and moves
        diagonally, which fits correlated coefficients far better.
        """
        from scipy.optimize import minimize          # optional dep

        x0 = [float(v) for v in self._x_start]
        bounds = [(t.lo, t.hi) for t in self.targets]

        def f(x):
            if self._stop():
                # Powell has no cancellation callback; aborting via an
                # exception is the supported way out mid-run.
                raise _AbortSearch()
            s = self._score([float(v) for v in x])
            # scipy can't work with +inf; give failures a large finite cost
            return 1e12 if (math.isinf(s) or math.isnan(s)) else s

        try:
            minimize(f, x0, method="Powell", bounds=bounds,
                     options={"maxfev": self.max_evals, "xtol": 1e-4,
                              "ftol": 1e-6})
        except _AbortSearch:
            pass          # cancelled / budget spent; _best_x is recorded
        except Exception:
            # scipy bailing out for its own reasons — whatever was
            # evaluated is already recorded in _best_x.
            pass

    def _de_search(self):
        """Global search via scipy.optimize.differential_evolution.

        workers=1 is mandatory: the objective runs the pipeline, whose
        steps are stateful and not thread-safe. polish=False keeps the
        budget honest (polishing appends an unbudgeted local solve) --
        run "minimize" afterwards if you want local refinement. The
        budget itself is enforced by the objective wrapper aborting via
        _AbortSearch once max_evals is spent, since DE's maxiter/popsize
        only bound evaluations indirectly."""
        from scipy.optimize import differential_evolution   # optional dep

        bounds = [(t.lo, t.hi) for t in self.targets]
        x0 = [float(v) for v in self._x_start] if self._x_start else None

        def f(x):
            if self._stop():
                raise _AbortSearch()
            s = self._score([float(v) for v in x])
            return 1e12 if (math.isinf(s) or math.isnan(s)) else s

        try:
            differential_evolution(
                f, bounds, x0=x0,
                maxiter=10_000,          # budget is enforced via f()
                popsize=15, tol=1e-6,
                seed=self.seed, polish=False, workers=1,
                updating="immediate")
        except _AbortSearch:
            pass          # cancelled / budget spent; _best_x is recorded
        except Exception:
            pass          # scipy bailing out; evaluated points are kept

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
            if self.method == "gp_minimize":
                if not bayes_available():
                    raise ValueError(
                        "method 'gp_minimize' needs scikit-optimize "
                        "(pip install scikit-optimize)")
                self._bayes_search()
            elif self.method == "minimize":
                if not local_available():
                    raise ValueError("method 'minimize' needs scipy")
                self._local_search()
            elif self.method == "differential_evolution":
                if not local_available():
                    raise ValueError(
                        "method 'differential_evolution' needs scipy")
                self._de_search()
            else:
                raise ValueError(
                    f"unknown method '{self.method}' -- "
                    f"use one of {available_methods()}")
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
            cancelled=self._cancelled,
            message=("Cancelled." if self._cancelled else
                     "No usable result — every evaluation failed."
                     if not best_values else "OK"))
