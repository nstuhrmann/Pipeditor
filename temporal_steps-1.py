"""
Steps that carry state across frames.

Almost everything that used to live here is now in StatefulStep
(base_step.py): the executor supplies a FrameContext saying whether this
is a re-run of the same frame, a jump, or the next frame in sequence, and
StatefulStep turns that into reset()/advance() calls. A temporal step is
therefore just:

    @register_step
    class MyFilter(StatefulStep):
        PARAMS = [...]
        def reset(self):            self.history = []
        def advance(self, image):   ...
"""
import numpy as np

from src.GUI.pipeline_editor.base_step import (
    ParamSpec, StatefulStep, register_step,
)


@register_step
class FrameAverage(StatefulStep):
    """Rolling average of the last N frames — the simplest temporal
    denoiser, and a reference to compare a real one against."""
    NAME = "Frame Average"
    CATEGORY = "Filter/Temporal"
    PARAMS = [ParamSpec("frames", "Frames", "int", default=8,
                        min_value=1, max_value=256, step=1)]

    def reset(self):
        self._buf = []

    def advance(self, image):
        self._buf.append(np.asarray(image, np.float32))
        del self._buf[:-max(1, int(self.p.frames))]
        return np.mean(self._buf, axis=0)


@register_step
class TemporalIIR(StatefulStep):
    """Exponential temporal filter: out = a*in + (1-a)*prev.

    Cheaper than a frame buffer and the usual choice in hardware, but it
    smears motion — which is exactly what you want to be able to measure
    against Frame Average.
    """
    NAME = "Temporal IIR"
    CATEGORY = "Filter/Temporal"
    PARAMS = [ParamSpec("alpha", "Alpha (new frame weight)", "float",
                        default=0.25, min_value=0.01, max_value=1.0,
                        step=0.01, decimals=3)]

    def reset(self):
        self._prev = None

    def advance(self, image):
        a = np.asarray(image, np.float32)
        if self._prev is None or self._prev.shape != a.shape:
            self._prev = a.copy()
            return a
        alpha = float(self.p.alpha)
        self._prev = alpha * a + (1.0 - alpha) * self._prev
        return self._prev
