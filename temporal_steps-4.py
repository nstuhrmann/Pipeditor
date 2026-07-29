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

from src.GUI.pipeline_editor.array_utils import to_luminance
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


@register_step
class GatedTemporalAverage(StatefulStep):
    """Temporal noise averaging that only runs while the scene is flat.

    Two filters, same `frames` parameter n:

      FIR  the true mean of the last n frames. Exact, bounded memory
           (n frames), and its noise reduction is exactly 1/sqrt(n).
      IIR  y = a*x + (1-a)*y_prev with a = 1/n. One frame of memory
           instead of n, but it never forgets — its impulse response has
           an infinite tail, so motion smears further even though the
           steady-state noise reduction is similar (sqrt(a/(2-a)),
           which for n=8 is 0.258 against the FIR's 0.354).

    The gate. Temporal averaging is free noise reduction on a static
    scene and a smearing artefact on a moving one, so it runs only while

        contrast = P(1-p) - P(p)   <   threshold

    with the percentiles taken on luminance. Percentiles rather than
    min/max deliberately: one stuck-hot pixel would peg a min/max
    contrast measure at full scale forever and the filter would never
    engage. p = 0.05 (the default) means P95 - P5.

    Above the threshold the incoming frame passes through untouched.
    What happens to the history then is the interesting choice, and it is
    a parameter:

      update_when_inactive = True (default)
          keep pushing frames into the buffer, so the moment the scene
          settles the filter is immediately warm. Costs: the buffer spans
          the busy period, so the first averaged frame afterwards can
          contain a ghost of it.
      update_when_inactive = False
          freeze the history. No ghosting, but re-engaging restarts from
          a buffer that is n frames stale.

    `hysteresis` widens the threshold for turning OFF only, so a contrast
    hovering at the boundary cannot flip the filter on and off every
    frame — that flicker is very visible, because the noise level pumps
    with it. Default 0 reproduces a plain threshold.

    Emits `contrast`, `averaging_active` and `frames_in_average` as
    metadata, so you can plot the gate against the scene and pick a
    threshold from data rather than by guessing. A Metadata CSV Writer
    picks all three up with no extra wiring.
    """
    NAME = "Gated Temporal Average"
    CATEGORY = "Filter/Temporal"
    PARAMS = [
        ParamSpec("mode", "Filter", "choice", default="FIR",
                  choices=["FIR", "IIR"]),
        ParamSpec("frames", "Frames (n)", "int", default=8,
                  min_value=1, max_value=256, step=1),
        ParamSpec("percentile", "Percentile p", "float", default=0.05,
                  min_value=0.0, max_value=0.5, step=0.01, decimals=4),
        ParamSpec("threshold", "Contrast Threshold", "float", default=0.10,
                  min_value=0.0, max_value=1.0, step=0.01, decimals=4),
        ParamSpec("hysteresis", "Hysteresis", "float", default=0.0,
                  min_value=0.0, max_value=1.0, step=0.005, decimals=4),
        ParamSpec("update_when_inactive", "Keep Filling History While "
                  "Inactive", "bool", default=True),
    ]

    def reset(self):
        self._buf = []          # FIR history
        self._acc = None        # IIR accumulator
        self._active = False    # gate state, for hysteresis

    def _contrast(self, image) -> float:
        lum = to_luminance(np.asarray(image, np.float32))
        p = float(self.p.percentile) * 100.0
        lo, hi = np.percentile(lum, [p, 100.0 - p])
        return float(hi - lo)

    def _gate(self, contrast: float) -> bool:
        """Below threshold -> on. Once on, it takes threshold+hysteresis
        to turn back off."""
        limit = float(self.p.threshold)
        if self._active:
            limit += float(self.p.hysteresis)
        self._active = contrast < limit
        return self._active

    def advance(self, image):
        a = np.asarray(image, np.float32)
        n = max(1, int(self.p.frames))
        contrast = self._contrast(a)
        active = self._gate(contrast)
        feed = active or bool(self.p.update_when_inactive)

        if self.p.mode == "FIR":
            if feed:
                self._buf.append(a)
                del self._buf[:-n]         # keep at most the last n
            elif len(self._buf) > n:
                del self._buf[:-n]         # honour a shrunken n immediately
            out = (np.mean(self._buf, axis=0).astype(np.float32)
                   if active and self._buf else a)
            depth = len(self._buf) if active else 0
        else:
            alpha = 1.0 / n
            if self._acc is None or self._acc.shape != a.shape:
                self._acc = a.copy()       # first frame seeds the filter
            elif feed:
                self._acc = alpha * a + (1.0 - alpha) * self._acc
            out = self._acc.astype(np.float32) if active else a
            # effective window of an exponential filter, for reporting
            depth = n if active else 0

        self.emit(contrast=contrast, averaging_active=bool(active),
                  frames_in_average=int(depth))
        return out
