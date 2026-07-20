"""
Steps that carry state across frames (temporal denoise, motion detect,
frame averaging, ...).

The hard part isn't holding the object — it's that process() is NOT
called once per frame in a tidy 0,1,2,... order:

  * Live Update re-runs the SAME frame after every parameter tweak.
    Feeding it to a temporal filter again would poison the history with
    duplicates of one frame.
  * The frame slider jumps arbitrarily (37 -> 12). Whatever history the
    filter accumulated is meaningless for the new position.
  * A batch run is the only context that IS strictly sequential.

_TemporalStep detects which of those three happened by comparing the
frame index against the last one it saw, and keeps the wrapped stateful
object consistent. Subclasses only implement create_processor() and
apply().
"""
import numpy as np

from src.GUI.pipeline_editor.base_step import (
    ProcessingStep, ParamSpec, register_step,
)


class _TemporalStep(ProcessingStep):
    """
    Wraps a stateful per-frame processor (an object built once, then
    called for each new frame).

    Subclasses implement:
        create_processor(**params) -> object
            Build the stateful object. Called once, then again whenever
            any param named in RESET_PARAMS changes.
        apply(processor, image, **params) -> np.ndarray
            Feed one frame and return the result.

    RESET_PARAMS lists the parameters baked into the object at
    construction time (history length, kernel size, ...). Changing one
    rebuilds the processor and drops its history. Params NOT listed are
    assumed to be applied per call, so changing them keeps the history.
    """
    IS_SEQUENCE_AWARE = True
    RESET_PARAMS: tuple = ()

    def __init__(self):
        super().__init__()
        self._proc = None
        self._proc_key = None
        self._last_index = None     # frame index last fed to the processor
        self._last_output = None    # its result, for repeat-frame previews
        self._index = 0
        self._in_sequence = False

    # --- subclass hooks ------------------------------------------------
    def create_processor(self, **params):
        raise NotImplementedError

    def apply(self, processor, image: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError

    # --- sequence protocol ---------------------------------------------
    def begin_sequence(self, total_frames: int):
        # A batch is a fresh temporal run: discard any state left over
        # from previews or a previous batch, so frame 0 always starts
        # from a clean history and batches are reproducible.
        self._in_sequence = True
        self.reset_state()

    def end_sequence(self):
        self._in_sequence = False

    def set_frame_index(self, index: int, total_frames: int):
        self._index = index

    def reset_state(self):
        self._proc = None
        self._proc_key = None
        self._last_index = None
        self._last_output = None

    # --- internals ------------------------------------------------------
    def _processor(self, params: dict):
        key = tuple(params.get(k) for k in self.RESET_PARAMS)
        if self._proc is None or key != self._proc_key:
            self._proc = self.create_processor(**params)
            self._proc_key = key
            # New object => no history, and no valid "last frame".
            self._last_index = None
            self._last_output = None
        return self._proc

    def process(self, image: np.ndarray, **params) -> np.ndarray:
        idx = self._index
        proc = self._processor(params)

        if self._last_index is not None:
            if idx == self._last_index:
                # Same frame again — a live-mode re-run after a param
                # tweak. Feeding it twice would duplicate it in the
                # history, so rebuild from scratch and feed it once:
                # the new parameter value takes effect, and the state
                # stays honest (one frame in, one frame's worth of
                # history) instead of silently corrupted.
                self._proc = self.create_processor(**params)
                self._proc_key = tuple(params.get(k)
                                       for k in self.RESET_PARAMS)
                proc = self._proc
            elif idx != self._last_index + 1:
                # Slider jump / re-ordered access: accumulated history
                # belongs to a different part of the sequence.
                self._proc = self.create_processor(**params)
                self._proc_key = tuple(params.get(k)
                                       for k in self.RESET_PARAMS)
                proc = self._proc

        out = self.apply(proc, image, **params)
        self._last_index = idx
        self._last_output = out
        return out


# ---------------------------------------------------------------------------
# Concrete step — adapt create_processor()/apply() to your class's API
# ---------------------------------------------------------------------------

@register_step
class TemporalDenoise(_TemporalStep):
    NAME     = "Temporal Denoise"
    CATEGORY = "Filter/Temporal"
    # history/strength are constructor arguments of the denoiser, so
    # changing either rebuilds it; add per-call params outside this tuple.
    RESET_PARAMS = ("history", "strength")
    PARAMS = [
        ParamSpec("history", "History Frames", "int", default=8,
                  min_value=1, max_value=64, step=1),
        ParamSpec("strength", "Strength", "float", default=0.5,
                  min_value=0.0, max_value=1.0, step=0.05),
    ]

    def create_processor(self, history: int = 8, strength: float = 0.5,
                         **params):
        # >>> replace with your class <<<
        from src.algorithms.denoise.temporal import TemporalDenoiser
        return TemporalDenoiser(history=history, strength=strength)

    def apply(self, processor, image: np.ndarray, **params) -> np.ndarray:
        # Your object's per-frame entry point.
        return processor(image)
