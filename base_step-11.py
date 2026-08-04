"""
Base class for all image processing steps.

Writing a step is meant to be almost free:

    @register_step
    class GaussianBlur(ProcessingStep):
        NAME = "Gaussian Blur"
        CATEGORY = "Filter/Blur"
        PARAMS = [ParamSpec("sigma", "Sigma", "float", default=1.0,
                            min_value=0.0, max_value=20.0)]

        def process(self, image):
            return cv2.GaussianBlur(image, (0, 0), self.p.sigma)

process() receives ONLY frames — one argument per input port. Parameters
are read off `self.p`, which is generated from PARAMS, so a step never
restates its own defaults and can never be handed a keyword it doesn't
expect. (The previous design passed params as **kwargs, which forced
every step to choose between an exact signature that broke when the
framework injected a param, and a **kwargs catch-all that swallowed
typos silently. `self.p` has neither failure mode.)

Everything else a step might want is an attribute set by the executor
before the call:

    self.p            parameters            self.p.sigma
    self.ctx          frame context         self.ctx.index, .total, .is_rerun
    self.inbox        control values received from upstream control edges
    image.meta        metadata riding with that frame

and two methods to send things onward:

    self.emit(exposure_ms=8.0)     attach metadata to this frame
    self.control(gain=2.0)         send a control value BACKWARD along a
                                   control edge, delivered next frame
"""
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.GUI.pipeline_editor.array_utils import to_float01  # noqa: F401
# (re-exported: steps and the executor have always imported it from here)


@dataclass
class ParamSpec:
    name: str
    label: str
    kind: str          # "int"|"float"|"bool"|"choice"|"str"|"file"|"directory"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list] = None
    # For kind="file" only: a ready-made Qt file-dialog filter string,
    # e.g. "NUC files (*.nuc);;All files (*)". Empty -> all files.
    # kind="directory" ignores it.
    types: str = ""
    # For kind="float" only: digits shown in the spin box. Note this
    # also ROUNDS the stored value, so a param that needs fine values
    # (small sigma/gamma increments) must raise it. None -> derived from
    # `step` (0.05 -> 2, 0.001 -> 3), falling back to 3 when step is unset.
    decimals: Optional[int] = None
    #: One-line explanation, shown as the tooltip on this parameter's
    #: control in the dialog. The label says WHAT it is; the hint says
    #: what it does or what a sensible value looks like.
    hint: str = ""

    def display_decimals(self) -> int:
        """Digits to show for a float parameter, wherever it is shown.

        An explicit `decimals` wins. Otherwise derive from the step size,
        so step=0.05 gives 2 and step=0.001 gives 3 — a control showing
        fewer digits than its own step can't represent the values it
        produces. Falls back to 3 when there is no step.
        """
        if self.decimals is not None:
            return max(0, int(self.decimals))
        if not self.step or self.step <= 0:
            return 3
        # Never show FEWER than 3: dropping digits silently rounds
        # parameter values that were saved with more.
        return max(3, min(8, -math.floor(math.log10(self.step))))


@dataclass(frozen=True)
class FrameContext:
    """Why the executor is calling, not just which frame.

    process() is NOT invoked once per frame in order: live preview
    re-runs the SAME frame after every parameter tweak, the frame slider
    jumps arbitrarily, and only a batch is strictly sequential. Stateful
    steps used to re-derive that from a private _last_index — three
    copies of the same fiddly logic. The executor computes it once, per
    node, and hands it over.
    """
    index: int = 0
    total: int = 1
    committed: bool = False     # a real run (sinks write, state advances)
                                # rather than a preview probe
    is_first: bool = True       # first call since a reset/sequence start
    is_rerun: bool = False      # same index as this step's previous call
    jumped: bool = False        # index moved by anything other than +1


class Params(dict):
    """A step's parameter values. It IS the dict the GUI and save/load
    use — `step.values is step.p` — so there is no wrapper to keep in
    sync, and `**self.p` needs no Mapping shim.

        self.p.sigma                                attribute access
        backend(image, **self.p)                    splat into a callable
        backend(image, **self.p.matching(backend))  ...only what it takes
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"no parameter '{name}' — declared parameters: "
                f"{', '.join(sorted(self)) or '(none)'}") from None

    def __setattr__(self, name, value):
        self[name] = value

    # --- subsets, for backends that don't take every parameter --------
    def only(self, *names) -> dict:
        """Just these: ``backend(**self.p.only('sigma', 'radius'))``."""
        return {n: self[n] for n in names if n in self}

    def without(self, *names) -> dict:
        """Everything except these."""
        return {k: v for k, v in self.items() if k not in names}

    def matching(self, func) -> dict:
        """Only the parameters `func` accepts, matched by name.

        Lets a step declare more parameters than its backend takes (CSV
        options, display-only settings) without the call raising
        TypeError. A backend with **kwargs gets everything.
        """
        import inspect
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return dict(self)
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(self)
        return {k: v for k, v in self.items() if k in sig.parameters}


class ProcessingStep:
    """Base class for every node in a pipeline."""

    NAME: str = "Unnamed Step"
    CATEGORY: str = "General"          # "A/B/C" nests in the palette
    PARAMS: list = []

    N_INPUTS: int = 1
    INPUT_LABELS: tuple = ()           # names for asymmetric input ports
    # Control ports. A step that calls self.control() should declare
    # EMITS_CONTROL so it gets an output to drag from; a step that reads
    # self.inbox should declare ACCEPTS_CONTROL so there is somewhere to
    # drop the edge — sources have no data inputs, so without this an
    # AE -> camera loop would be undrawable.
    EMITS_CONTROL: bool = False
    ACCEPTS_CONTROL: bool = False
    #: What this step IS. One value, not three booleans that could
    #: contradict each other:
    #:   "step"    ordinary processing node
    #:   "source"  generates frames; no data inputs
    #:   "sink"    writes frames out; passes the image through
    #:   "metric"  measures; passes the image through unmodified and
    #:             attaches the value as metadata
    KIND: str = "step"

    # --- set by the executor before every process() call ---------------
    ctx: FrameContext = FrameContext()
    inbox: dict = {}                   # control values from upstream
    _out_meta: dict = None
    _outbox: dict = None

    #: Names a parameter may not use, because attribute lookup would find
    #: the method first and `self.p.<name>` would silently return a bound
    #: method instead of the value. Derived from Params itself, so it
    #: covers dict's methods AND the ones Params adds (only / without /
    #: matching) — a hand-written list would drift the moment Params
    #: gained a helper.
    #:
    #: Note this is about names on Params, NOT on the step: a parameter
    #: called "p", "ctx" or "inbox" is fine, since it is reached as
    #: self.p.p and Params has no such attribute.
    _RESERVED_PARAM_NAMES = frozenset(
        n for n in dir(Params) if not n.startswith("__"))

    KINDS = ("step", "source", "sink", "metric")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.KIND not in ProcessingStep.KINDS:
            raise TypeError(
                f"{cls.__name__}: KIND must be one of "
                f"{', '.join(repr(k) for k in ProcessingStep.KINDS)}"
                f" — got {cls.KIND!r}")

        clashes = sorted({p.name for p in cls.PARAMS}
                         & ProcessingStep._RESERVED_PARAM_NAMES)
        if clashes:
            raise TypeError(
                f"{cls.__name__}: parameter name(s) {clashes} would be "
                f"shadowed by a method of the same name on self.p — "
                f"rename them (e.g. 'keys' -> 'key_list').")

    def __init__(self):
        # One object under two names: `values` is what save/load and the
        # dialog use, `p` is what step code reads.
        self.values = self.p = Params(
            {p.name: p.default for p in self.PARAMS})

    # --- step-facing helpers -------------------------------------------
    def emit(self, **values):
        """Attach metadata to the frame this step is producing. It rides
        along with the frame and is readable downstream as image.meta,
        surviving steps that know nothing about it."""
        if self._out_meta is None:
            self._out_meta = {}
        self._out_meta.update(values)

    def control(self, **values):
        """Send values along this node's outgoing CONTROL edges. They
        arrive in the target's `inbox` on its next execution — the one
        frame of delay is inherent to feedback and is why the edge is
        drawn backwards in the graph."""
        if self._outbox is None:
            self._outbox = {}
        self._outbox.update(values)

    # --- parameters -----------------------------------------------------
    def set_param_values(self, values: dict):
        for k, v in values.items():
            if k in self.values:
                self.values[k] = v

    def get_param_values(self) -> dict:
        return dict(self.values)

    # --- the one method a step must implement ---------------------------
    def process(self, *images):
        raise NotImplementedError

    # --- sequence lifecycle (override only if needed) --------------------
    def frame_count(self) -> int:
        """Sources: how many frames this can produce. 1 = not a sequence."""
        return 1

    # --- lifecycle -------------------------------------------------------
    # Deliberately three hooks, not one. `begin_sequence` used to bundle
    # "drop accumulated history" with "open files and threads", but
    # seeking wants the first and must NOT do the second — otherwise
    # dragging the frame slider would truncate and reopen your output
    # video on every tick.

    def reset_state(self):
        """Drop accumulated history. Cheap, no I/O. Called at the start
        of a run, and whenever the frame index jumps."""

    def open_resources(self, total_frames: int):
        """Open files, start threads. Called once before a committed run
        — never for a preview or a seek."""

    def close_resources(self):
        """Close what open_resources() opened. Always called, including
        on error or cancellation."""


class StatefulStep(ProcessingStep):
    """A step whose state accumulates across frames — temporal filters,
    running statistics, a simulated camera's exposure.

    Implement reset() and advance(); the base deals with the fact that
    frames do not arrive in a tidy 0,1,2,... order:

      * same frame again (live preview after a parameter tweak) -> the
        cached output is returned, so history is not double-fed and the
        preview stays idempotent;
      * index jumped (slider scrub) -> reset(), because the accumulated
        history belongs to a different part of the sequence;
      * batch start -> reset(), so a run is reproducible regardless of
        what previews happened first.
    """

    #: Whether snapshots must deep-copy. Only needed when advance()
    #: mutates arrays IN PLACE — a step that rebinds (``self._prev = new``)
    #: is exact with a shallow copy, which costs a pointer. TemporalNoise's
    #: Welford accumulators (``self._mean += ...``) are the case that needs
    #: True; so does any wrapper around a foreign object it cannot inspect.
    SNAPSHOT_DEEP = False

    #: Attributes that are NOT state: parameters, and the per-call fields
    #: the executor writes immediately before process(). Restoring a stale
    #: ctx or inbox mid-call would be actively wrong.
    _NON_STATE = frozenset({"values", "p", "ctx", "inbox",
                            "_out_meta", "_outbox",
                            "_snapshot", "_cached_out"})

    def __init__(self):
        super().__init__()
        self._snapshot = None
        self._cached_out = None

    def reset(self):
        """Drop accumulated state."""

    def advance(self, *images):
        """Consume one new frame and return the result."""
        raise NotImplementedError

    # --- state snapshots -------------------------------------------------
    def snapshot(self):
        """Capture the state advance() is about to mutate.

        Override with something cheaper if the generic copy is wasteful —
        a step whose whole state is two floats can return a tuple.
        """
        state = {k: v for k, v in self.__dict__.items()
                 if k not in self._NON_STATE}
        return copy.deepcopy(state) if self.SNAPSHOT_DEEP else dict(state)

    def restore(self, snap):
        self.__dict__.update(snap)

    def reset_state(self):
        self._snapshot = None
        self._cached_out = None
        self.reset()

    def process(self, *images):
        ctx = self.ctx
        if ctx.is_first or ctx.jumped:
            # A jump means the accumulated history belongs to a different
            # part of the sequence.
            self.reset_state()
        elif ctx.is_rerun:
            if self._snapshot is not None:
                # Re-running the SAME frame: rewind to the state it
                # started from and advance again, so a parameter changed
                # in between actually shows up. Returning a cached output
                # here (the old behaviour) made live tweaks and
                # frame-mode optimization silently inert on any stateful
                # graph.
                self.restore(self._snapshot)
            elif self._cached_out is not None:
                # Committed run with no snapshot taken: never advance the
                # same frame twice.
                return self._cached_out

        # Snapshots are only useful where a re-run can happen, so a
        # committed run pays nothing for them.
        self._snapshot = None if ctx.committed else self.snapshot()
        self._cached_out = self.advance(*images)
        return self._cached_out


def step_source_location(step_cls) -> tuple:
    """(absolute file, first line) where a step class is defined, or
    ("", 0) if it can't be determined — a class built at runtime by
    step_factory has no file of its own.

    inspect.getsourcefile() follows the class's module, so this keeps
    working when a step moves between modules; hard-coding paths would
    not."""
    import inspect
    try:
        path = inspect.getsourcefile(step_cls) or ""
        line = inspect.getsourcelines(step_cls)[1] if path else 0
    except (TypeError, OSError):
        return "", 0
    return path, line


STEP_REGISTRY: dict[str, type] = {}


def register_step(cls):
    STEP_REGISTRY[cls.__name__] = cls
    return cls
