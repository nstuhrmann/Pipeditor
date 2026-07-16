"""
Base class for all image processing steps.
"""
from dataclasses import dataclass
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


class CachedLoadMixin:
    """
    For steps with one expensive-to-load parameter (typically a file
    path, e.g. a non-uniformity correction map) and other cheap
    parameters. load() runs exactly once per distinct LOAD_PARAM value:

      - on the first process() call after the step is created — which
        includes being restored from a saved pipeline (the cache key
        starts as a never-matching sentinel)
      - again whenever the LOAD_PARAM value actually changes
      - NEVER on runs where only the other parameters changed, and not
        per-frame during a video/stack batch

    Usage:
        class NUCCorrection(CachedLoadMixin, ProcessingStep):
            LOAD_PARAM = "nuc_path"
            PARAMS = [ParamSpec("nuc_path", "NUC Map", "file", default=""),
                      ...cheap params...]

            def load(self, path):
                return parse_nuc_map(path)      # the expensive part

            def process(self, image, **params):
                nuc = self.get_loaded(**params)  # cached unless path changed
                ...
    """
    LOAD_PARAM: str = "path"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_key = object()   # sentinel that never equals a real value
        self._cache_value = None

    def load(self, value):
        """Override: parse/open whatever LOAD_PARAM points to. Raise on
        bad input — the error surfaces with the node's name attached."""
        raise NotImplementedError

    def get_loaded(self, **params):
        key = params.get(self.LOAD_PARAM)
        if key != self._cache_key:
            self._cache_value = self.load(key)
            self._cache_key = key
        return self._cache_value

    def invalidate_cache(self):
        """Force the next get_loaded() to re-run load() — e.g. if the
        file changed on disk while the path stayed the same."""
        self._cache_key = object()


class ProcessingStep:
    """
    Base class for every node in the pipeline.

    Class variables to override:
        NAME        display name shown in the palette and on the node box
        CATEGORY    palette group
        PARAMS      list of ParamSpec — UI is auto-generated from this
        N_INPUTS    number of input ports (default 1)
        IS_SOURCE   True → no input port; process() should ignore the dummy
                    image passed in and load its own data instead
        IS_SINK     True → dark-red header; step writes data but also
                    returns the image unchanged (passthrough)
        IS_METRIC   True → two input ports (A/B), no output port; process()
                    returns a float or str that is displayed on the node

    process() signature:
        N_INPUTS == 1  →  process(self, image: np.ndarray, **params)
        N_INPUTS == 2  →  process(self, image_a: np.ndarray,
                                        image_b: np.ndarray, **params)

    Optional sequence protocol: a step that reads from or writes to a
    video file / image stack (more than one frame) sets
    IS_SEQUENCE_AWARE = True and implements whichever of these it needs;
    everything defaults to a no-op, so ordinary single-image steps are
    entirely unaffected:

        frame_count()             SOURCES: total frames available.
        set_frame_index(i, total) Called immediately before every run()
                                   while IS_SEQUENCE_AWARE — sources use
                                   it to know which frame to load next;
                                   sequence-writing sinks use it to know
                                   the current frame number (e.g. for a
                                   zero-padded output filename).
        begin_sequence(total)     Called once before a batch run starts
                                   (Pipeline.run_sequence only — never
                                   for a plain single-frame run()). Open
                                   persistent resources here (VideoCapture,
                                   VideoWriter, a cached file listing).
        end_sequence()            Called once after a batch run ends,
                                   whether it succeeded, errored, or was
                                   cancelled. Close/flush whatever
                                   begin_sequence() opened.

    A step that only cares whether it's inside an active batch run right
    now (vs. a one-off preview call) tracks that itself via a flag set
    in begin_sequence() / cleared in end_sequence() — see
    sequence_steps.py for the pattern.
    """

    NAME: str = "Unnamed Step"
    CATEGORY: str = "General"
    PARAMS: list[ParamSpec] = []
    N_INPUTS: int = 1
    IS_SOURCE: bool = False
    IS_SINK: bool = False
    IS_METRIC: bool = False
    IS_SEQUENCE_AWARE: bool = False

    def __init__(self):
        self.values: dict[str, Any] = {p.name: p.default for p in self.PARAMS}

    def get_param_values(self) -> dict:
        return dict(self.values)

    def set_param_values(self, values: dict):
        for k, v in values.items():
            if k in self.values:
                self.values[k] = v

    def process(self, *images: np.ndarray, **params):
        raise NotImplementedError(
            f"{self.__class__.__name__}.process() must be implemented"
        )

    def run(self, inputs: list) -> "np.ndarray | float | str":
        """Called by the pipeline executor. inputs is a list of numpy arrays."""
        return self.process(*inputs, **self.values)

    # --- sequence protocol defaults — no-ops for ordinary steps -------
    def frame_count(self) -> int:
        return 1

    def set_frame_index(self, index: int, total_frames: int):
        pass

    def begin_sequence(self, total_frames: int):
        pass

    def end_sequence(self):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.values}>"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
STEP_REGISTRY: dict[str, type] = {}


def register_step(cls):
    STEP_REGISTRY[cls.__name__] = cls
    return cls
