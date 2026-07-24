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
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


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
    in_sequence: bool = False   # inside begin_sequence()/end_sequence()
    is_first: bool = True       # first call since a reset/sequence start
    is_rerun: bool = False      # same index as this step's previous call
    jumped: bool = False        # index moved by anything other than +1


class Params:
    """A step's parameter values, usable three ways:

        self.p.sigma                    attribute access
        backend(image, **self.p)        splat straight into a callable
        backend(image, **self.p.matching(backend))   ...only what it takes

    It is a Mapping (keys/__getitem__), which is all Python needs for
    ``**`` unpacking — so wrapping an existing backend function is a
    one-liner with no dict rebuilding in between.

    Backed by the same dict the GUI and save/load use, so a value set
    anywhere is visible everywhere with no copying or syncing.
    """
    __slots__ = ("_values",)

    def __init__(self, values: dict):
        object.__setattr__(self, "_values", values)

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(
                f"no parameter '{name}' — declared parameters: "
                f"{', '.join(sorted(self._values)) or '(none)'}") from None

    def __setattr__(self, name, value):
        self._values[name] = value

    # --- Mapping protocol: this is what makes `**self.p` work ----------
    def keys(self):
        return self._values.keys()

    def values(self):
        return self._values.values()

    def items(self):
        return self._values.items()

    def get(self, name, default=None):
        return self._values.get(name, default)

    def __getitem__(self, name):
        return self._values[name]

    def __contains__(self, name):
        return name in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"Params({self._values!r})"

    # --- subsets, for backends that don't take everything --------------
    def only(self, *names) -> dict:
        """Just these parameters: ``backend(**self.p.only('sigma', 'r'))``."""
        return {n: self._values[n] for n in names if n in self._values}

    def without(self, *names) -> dict:
        """Everything except these — e.g. dropping the CSV parameters a
        metric declares but its backend knows nothing about."""
        return {k: v for k, v in self._values.items() if k not in names}

    def matching(self, func) -> dict:
        """Only the parameters `func` actually accepts, matched by name.

        Lets a step declare more parameters than its backend takes (CSV
        options, display-only settings) without the call raising
        TypeError. A backend with **kwargs gets everything.
        """
        import inspect
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return dict(self._values)
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(self._values)
        return {k: v for k, v in self._values.items() if k in sig.parameters}

    def as_dict(self) -> dict:
        return dict(self._values)


def to_float01(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an image array to the pipeline's inter-module contract:
    float32 in [0, 1].

    The executor applies this to every node's ndarray output:
      - unsigned ints -> divided by their dtype max (255 / 65535) —
        deterministic, so uint8/uint16 steps mix freely
      - floats        -> CLIPPED to [0, 1], never rescaled. A step that
        outputs float values outside [0, 1] is violating the contract;
        the executor warns and clips rather than silently guessing a
        scale. Steps working internally in 0-255 float must divide by
        255 themselves (or return uint8, which is scaled exactly).
    """
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


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
    IS_SOURCE: bool = False
    IS_SINK: bool = False
    IS_METRIC: bool = False

    # --- set by the executor before every process() call ---------------
    ctx: FrameContext = FrameContext()
    inbox: dict = {}                   # control values from upstream
    _out_meta: dict = None
    _outbox: dict = None

    def __init__(self):
        self.values: dict = {p.name: p.default for p in self.PARAMS}
        self.p = Params(self.values)

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
        self.on_params_changed(values)

    def get_param_values(self) -> dict:
        return dict(self.values)

    def on_params_changed(self, changed: dict):
        """Called after any parameter changes. Careful: with live editing
        this fires per keystroke, so only invalidate here — defer real
        work to the next process()."""

    # --- the one method a step must implement ---------------------------
    def process(self, *images):
        raise NotImplementedError

    # --- sequence lifecycle (override only if needed) --------------------
    def frame_count(self) -> int:
        """Sources: how many frames this can produce. 1 = not a sequence."""
        return 1

    def begin_sequence(self, total_frames: int):
        """Called once before a batch. Open outputs / reset state here."""

    def end_sequence(self):
        """Called once after a batch, even on error or cancellation."""


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

    def __init__(self):
        super().__init__()
        self._cached_out = None

    def reset(self):
        """Drop accumulated state."""

    def advance(self, *images):
        """Consume one new frame and return the result."""
        raise NotImplementedError

    def begin_sequence(self, total_frames: int):
        self._cached_out = None
        self.reset()

    def process(self, *images):
        ctx = self.ctx
        if ctx.is_rerun and self._cached_out is not None:
            return self._cached_out
        if ctx.is_first or ctx.jumped:
            self.reset()
        self._cached_out = self.advance(*images)
        return self._cached_out


class CachedLoadMixin:
    """Lazily loads a file named by a parameter, reloading only when that
    parameter changes — set_param_values() fires per keystroke while
    typing, so eager loading in a setter would thrash the disk.

        class NUCCorrection(CachedLoadMixin, ProcessingStep):
            LOAD_PARAM = "map_path"
            def load_resource(self, path): return np.load(path)
            def process(self, image):
                return image - self.resource()
    """
    LOAD_PARAM: str = ""

    _cached_key = None
    _cached_value = None

    def load_resource(self, path):
        raise NotImplementedError

    def resource(self):
        path = self.values.get(self.LOAD_PARAM, "")
        if path != self._cached_key:
            self._cached_value = self.load_resource(path) if path else None
            self._cached_key = path
        return self._cached_value

    def invalidate_cache(self):
        self._cached_key = None
        self._cached_value = None


class MetricCSVMixin:
    """Marks a step as a metric AND gives it per-frame CSV dumping.

        class DeltaE(MetricCSVMixin, ProcessingStep):
            NAME = "DeltaE"
            N_INPUTS = 2
            PARAMS = [ParamSpec("method", ...)]      # CSV params appended
            def process(self, reference, test): ...

    IS_METRIC and the two CSV parameters are both set on the subclass, so
    a metric declares neither. Two reasons that is worth the magic:

      * CSV dumping only happens for IS_METRIC nodes, so a class that
        took the mixin but forgot the flag would show CSV options in its
        dialog that silently never wrote anything;
      * setting IS_METRIC directly ON the subclass makes base-class order
        irrelevant — with a plain class attribute,
        `class X(ProcessingStep, MetricCSVMixin)` would find
        ProcessingStep's IS_METRIC = False first and quietly not be a
        metric at all.

    Params are appended at CLASS definition time and are visible in
    cls.PARAMS, unlike the old per-instance injection this replaced —
    that one also fed the params into process() as kwargs, which is what
    broke steps with exact signatures.
    """
    CSV_PARAMS = [
        ParamSpec("dump_csv", "Dump CSV (batch)", "bool", default=False),
        ParamSpec("csv_path", "CSV File", "file", default="",
                  types="CSV files (*.csv);;All files (*)"),
    ]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.IS_METRIC = True
        declared = {p.name for p in getattr(cls, "PARAMS", [])}
        missing = [p for p in MetricCSVMixin.CSV_PARAMS
                   if p.name not in declared]
        if missing:
            cls.PARAMS = list(getattr(cls, "PARAMS", [])) + missing


STEP_REGISTRY: dict[str, type] = {}


def register_step(cls):
    STEP_REGISTRY[cls.__name__] = cls
    return cls
