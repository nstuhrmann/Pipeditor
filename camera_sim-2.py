"""
Camera simulator + auto-exposure controller.

A worked example of the two non-image channels:

    [Camera Simulator] --data--> [Auto Exposure] --data--> [AE Monitor]
            ^                            |
            +--------- control ----------+

  * The camera attaches what it actually used (exposure, gain,
    illumination) to the frame as METADATA, so anything downstream reads
    it from its own input — no ordering rules, and it stays correct even
    if a buffering step sits in between.
  * The AE sends the next exposure back along a CONTROL edge, which the
    camera receives in `self.inbox` on its next execution. The one-frame
    delay is inherent: within a frame the camera runs before the AE that
    measures it, which is also what real hardware does and why `damping`
    exists.

Wire the control edge from Auto Exposure back to Camera Simulator
(right-click the edge -> Control Edge). Without it the camera simply
runs at its parameter settings.
"""
import numpy as np

from src.GUI.pipeline_editor.array_utils import to_luminance
from src.GUI.pipeline_editor.base_step import (
    MetricCSVMixin, ParamSpec, ProcessingStep, StatefulStep, register_step,
)


# X-Rite ColorChecker Classic, sRGB 8-bit reference values, in the
# standard reading order (row 1 = dark skin ... row 4 = white -> black).
_COLORCHECKER_SRGB = [
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67),
    (133, 128, 177), (103, 189, 170),
    (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108),
    (157, 188, 64), (224, 163, 46),
    (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31),
    (187, 86, 149), (8, 133, 161),
    (243, 243, 242), (200, 200, 200), (160, 160, 160), (122, 122, 118),
    (85, 85, 85), (52, 52, 52),
]

_RESOLUTIONS = {
    "320x240": (320, 240),
    "640x480": (640, 480),
    "1280x960": (1280, 960),
}

_BIT_DEPTHS = {"8": 8, "10": 10, "12": 12, "16": 16}


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB EOTF: encoded [0,1] -> linear [0,1]."""
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Inverse sRGB EOTF: linear [0,1] -> encoded [0,1]."""
    c = np.clip(np.asarray(c, dtype=np.float32), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92,
                    1.055 * c ** (1 / 2.4) - 0.055).astype(np.float32)


def render_colorchecker(width: int, height: int) -> np.ndarray:
    """The chart as LINEAR scene reflectance (float32, [0,1]) — what the
    scene sends toward the sensor per unit illumination, before any
    exposure or encoding."""
    img = np.full((height, width, 3), 0.03, np.float32)   # dark surround
    cols, rows = 6, 4
    margin = max(4, int(min(width, height) * 0.07))
    gap = max(2, int(min(width, height) * 0.018))
    pw = (width - 2 * margin - (cols - 1) * gap) // cols
    ph = (height - 2 * margin - (rows - 1) * gap) // rows
    for i, rgb in enumerate(_COLORCHECKER_SRGB):
        row, col = divmod(i, cols)
        x0 = margin + col * (pw + gap)
        y0 = margin + row * (ph + gap)
        lin = srgb_to_linear(np.array(rgb, np.float32) / 255.0)
        img[y0:y0 + ph, x0:x0 + pw] = lin
    return img


@register_step
class CameraSimulator(StatefulStep):
    """Synthetic camera imaging a ColorChecker chart.

    Sensor model, in order: scene reflectance x illumination ->
    photoelectrons (proportional to exposure time), Poisson shot noise,
    Gaussian read noise, analog gain, full-well saturation, ADC
    quantization, optional sRGB encoding. Signal-dependent shot noise is
    what makes this useful for exposure work: under-expose, push with
    gain, and the noise follows.
    """
    NAME = "Camera Simulator"
    CATEGORY = "Camera/Simulation"
    IS_SOURCE = True
    ACCEPTS_CONTROL = True      # exposure/gain arrive from an AE

    FULL_WELL_E = 10000.0
    # A perfect white reflector at illumination 1.0 fills the well in
    # 10 ms at gain 1, so exposure numbers read intuitively.
    _E_PER_MS = FULL_WELL_E / 10.0

    PARAMS = [
        ParamSpec("resolution", "Resolution", "choice", default="640x480",
                  choices=list(_RESOLUTIONS)),
        ParamSpec("num_frames", "Frames", "int", default=60,
                  min_value=1, max_value=10000, step=1),
        ParamSpec("illumination", "Illumination Profile", "choice",
                  default="step",
                  choices=["constant", "step", "ramp", "sine"]),
        ParamSpec("illum_level", "Illumination Level", "float", default=0.5,
                  min_value=0.001, max_value=10.0, step=0.05, decimals=3),
        ParamSpec("exposure_ms", "Exposure Time (ms)", "float", default=5.0,
                  min_value=0.01, max_value=200.0, step=0.5, decimals=3),
        ParamSpec("gain", "Analog Gain", "float", default=1.0,
                  min_value=1.0, max_value=64.0, step=0.5),
        ParamSpec("accept_control", "Accept AE Control", "bool",
                  default=True),
        ParamSpec("read_noise_e", "Read Noise (e-)", "float", default=4.0,
                  min_value=0.0, max_value=200.0, step=0.5),
        ParamSpec("bit_depth", "Bit Depth", "choice", default="12",
                  choices=list(_BIT_DEPTHS)),
        ParamSpec("encoding", "Output Encoding", "choice", default="sRGB",
                  choices=["sRGB", "linear"]),
        ParamSpec("seed", "Noise Seed", "int", default=0,
                  min_value=0, max_value=100000, step=1),
    ]

    def __init__(self):
        super().__init__()
        self._chart = None
        self._chart_key = None
        self._exposure_ms = None
        self._gain = None

    def frame_count(self) -> int:
        return max(1, int(self.p.num_frames))

    def reset(self):
        # A batch (or a slider jump) starts from the parameter values, so
        # a run is reproducible regardless of where previews left it.
        self._exposure_ms = float(self.p.exposure_ms)
        self._gain = float(self.p.gain)

    def _get_chart(self, width: int, height: int) -> np.ndarray:
        if self._chart is None or self._chart_key != (width, height):
            self._chart = render_colorchecker(width, height)
            self._chart_key = (width, height)
        return self._chart

    def _illumination(self, level: float) -> float:
        """Scene illumination for the current frame. The non-constant
        profiles exist to give an AE something to track."""
        t = self.ctx.index / max(1, self.ctx.total - 1)
        profile = self.p.illumination
        if profile == "step":
            if t < 1 / 3:
                return level
            if t < 2 / 3:
                return level * 8.0
            return level * 0.5
        if profile == "ramp":
            return level * float(2.0 ** (-2.0 + 4.0 * t))
        if profile == "sine":
            return level * float(2.0 ** (2.0 * np.sin(2 * np.pi * t)))
        return level

    def advance(self, *_ignored):
        if self._exposure_ms is None:
            self.reset()

        if self.p.accept_control and self.inbox:
            # Control values arrive already scoped to this node, so there
            # is nothing to consume or clear — the executor delivers a
            # fresh inbox each frame.
            self._exposure_ms = float(self.inbox.get("exposure_ms",
                                                     self._exposure_ms))
            self._gain = float(self.inbox.get("gain", self._gain))
        elif not self.p.accept_control:
            self._exposure_ms = float(self.p.exposure_ms)
            self._gain = float(self.p.gain)

        width, height = _RESOLUTIONS.get(self.p.resolution, (640, 480))
        reflectance = self._get_chart(width, height)
        illum = self._illumination(float(self.p.illum_level))

        electrons = (reflectance * illum
                     * self._exposure_ms * self._E_PER_MS)

        # Seeded per FRAME, so re-running the same frame reproduces the
        # identical image rather than shimmering.
        rng = np.random.RandomState(
            (int(self.p.seed) * 1000003 + self.ctx.index) % (2 ** 31))
        electrons = rng.poisson(np.clip(electrons, 0.0, 1e12)).astype(np.float32)
        read_noise = float(self.p.read_noise_e)
        if read_noise > 0:
            electrons += rng.normal(0.0, read_noise,
                                    electrons.shape).astype(np.float32)

        signal = np.clip(electrons * self._gain, 0.0, self.FULL_WELL_E)
        norm = signal / self.FULL_WELL_E
        if self.p.encoding == "sRGB":
            norm = linear_to_srgb(norm)

        levels = (1 << _BIT_DEPTHS.get(self.p.bit_depth, 12)) - 1

        # What the camera actually used rides along with the frame.
        self.emit(exposure_ms=self._exposure_ms,
                  gain=self._gain,
                  illumination=illum,
                  encoding=self.p.encoding,
                  saturated_fraction=float(
                      (signal >= self.FULL_WELL_E * 0.999).mean()))
        return (np.round(norm * levels) / levels).astype(np.float32)


@register_step
class AutoExposure(StatefulStep):
    """Auto-exposure controller. Measures the incoming frame, computes
    the correction in EV, and sends the new exposure back along a control
    edge. Passes the image through unchanged.

    Control law: error_ev = log2(target / measured) in the LINEAR domain
    (the measurement is linearised first if the camera encodes sRGB —
    otherwise loop gain would vary with brightness). The step is damped
    and clamped, then the new exposure product is split between exposure
    time and gain, preferring time (gain amplifies read noise without
    collecting more photons).
    """
    NAME = "Auto Exposure"
    CATEGORY = "Camera/Control"
    EMITS_CONTROL = True        # sends the next exposure back to a camera
    PARAMS = [
        # Expressed in the camera's OUTPUT encoding for convenience, but
        # the loop controls the LINEAR mean (exposure scales linearly, so
        # metering must too). Consequence: the displayed mean of a real
        # scene will NOT equal this number, because mean(encode(x)) !=
        # encode(mean(x)). Judge convergence by AE Monitor's error_ev.
        ParamSpec("target", "Target Level", "float", default=0.45,
                  min_value=0.01, max_value=0.99, step=0.01, decimals=3),
        ParamSpec("metering", "Metering", "choice", default="average",
                  choices=["average", "center-weighted",
                           "highlight-protect"]),
        ParamSpec("damping", "Damping", "float", default=0.6,
                  min_value=0.05, max_value=1.0, step=0.05, decimals=2),
        ParamSpec("max_step_ev", "Max Step (EV)", "float", default=1.5,
                  min_value=0.1, max_value=6.0, step=0.1),
        ParamSpec("min_exposure_ms", "Min Exposure (ms)", "float",
                  default=0.05, min_value=0.01, max_value=100.0,
                  step=0.05, decimals=3),
        ParamSpec("max_exposure_ms", "Max Exposure (ms)", "float",
                  default=33.0, min_value=0.1, max_value=1000.0, step=1.0),
        ParamSpec("max_gain", "Max Gain", "float", default=16.0,
                  min_value=1.0, max_value=64.0, step=1.0),
        ParamSpec("linearize", "Linearize sRGB Input", "bool", default=True),
    ]

    HIGHLIGHT_LIMIT = 0.98      # linear, protect against clipping

    def reset(self):
        self._exposure_ms = 5.0   # fallback if no camera metadata arrives
        self._gain = 1.0
        self._weights = None
        self._weights_shape = None

    def _center_weights(self, shape) -> np.ndarray:
        if self._weights is None or self._weights_shape != shape:
            h, w = shape
            yy = np.linspace(-1.0, 1.0, h)[:, None]
            xx = np.linspace(-1.0, 1.0, w)[None, :]
            self._weights = np.exp(-(xx ** 2 + yy ** 2) / 0.5).astype(np.float32)
            self._weights_shape = shape
        return self._weights

    def advance(self, image):
        meta = getattr(image, "meta", {}) or {}
        # The camera's own report of what it used, carried by the frame
        # being measured — so it is guaranteed to describe THIS image.
        self._exposure_ms = float(meta.get("exposure_ms", self._exposure_ms))
        self._gain = float(meta.get("gain", self._gain))
        encoded = meta.get("encoding", "sRGB") == "sRGB"
        do_linearize = bool(self.p.linearize) and encoded

        lum = to_luminance(np.asarray(image, np.float32)).astype(np.float32)
        lin = srgb_to_linear(lum) if do_linearize else lum
        target = float(self.p.target)
        target_lin = (float(srgb_to_linear(np.array([target]))[0])
                      if do_linearize else target)

        if self.p.metering == "center-weighted":
            w = self._center_weights(lin.shape)
            measured = float((lin * w).sum() / w.sum())
        else:
            measured = float(lin.mean())
        measured = max(measured, 1e-6)

        error_ev = float(np.log2(target_lin / measured))
        max_step = abs(float(self.p.max_step_ev))
        step_ev = float(np.clip(error_ev * float(self.p.damping),
                                -max_step, max_step))

        total = max(self._exposure_ms * self._gain, 1e-6)
        new_total = total * (2.0 ** step_ev)

        if self.p.metering == "highlight-protect":
            p99 = float(np.percentile(lin, 99.0))
            if p99 > 1e-6:
                new_total = min(new_total, total * (self.HIGHLIGHT_LIMIT / p99))

        min_ms = float(self.p.min_exposure_ms)
        max_ms = float(self.p.max_exposure_ms)
        max_gain = float(self.p.max_gain)
        new_total = float(np.clip(new_total, min_ms, max_ms * max_gain))
        new_exposure = float(np.clip(new_total, min_ms, max_ms))
        new_gain = float(np.clip(new_total / new_exposure, 1.0, max_gain))

        # Backward, along the control edge -> camera's next frame.
        self.control(exposure_ms=new_exposure, gain=new_gain)
        # Forward, with this frame -> monitors downstream.
        self.emit(ae_measured=measured, ae_target=target_lin,
                  ae_error_ev=error_ev, ae_applied_ev=step_ev,
                  ae_exposure_ms=new_exposure, ae_gain=new_gain)
        self._exposure_ms, self._gain = new_exposure, new_gain
        return image


@register_step
class AEMonitor(MetricCSVMixin, ProcessingStep):
    """Reports one number from the exposure loop, so convergence can be
    watched on the node, dumped to CSV over a batch, or used as an
    optimizer objective (minimise |error| to tune damping).

    Everything it reports comes from the frame's own metadata, so it is
    correct wherever it sits in the graph."""
    NAME = "AE Monitor"
    CATEGORY = "Metrics/Camera"
    IS_METRIC = True
    _KEYS = {
        "error_ev": "ae_error_ev", "measured": "ae_measured",
        "exposure_ms": "exposure_ms", "gain": "gain",
        "illumination": "illumination",
        "saturated_fraction": "saturated_fraction",
    }
    PARAMS = [ParamSpec("quantity", "Quantity", "choice", default="error_ev",
                        choices=list(_KEYS))] + MetricCSVMixin.CSV_PARAMS

    def process(self, image):
        meta = getattr(image, "meta", {}) or {}
        return float(meta.get(self._KEYS[self.p.quantity], float("nan")))
