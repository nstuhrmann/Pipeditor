"""
Camera simulator + auto-exposure controller.

A worked example of MessageBus closing a control loop across the graph,
and a test rig for exposure algorithms that needs no hardware.

    [Camera Simulator] --> [Auto Exposure] --> [AE Monitor] --> ...
            ^                     |
            |                     |
            +----- message bus ---+

Two topics, deliberately using BOTH bus verbs:

  "camera_control"  AE -> camera. A COMMAND, so the camera pop()s it:
                    exactly one consumer, and acting on it twice would
                    double-apply the correction.
  "camera_state"    camera -> anyone. STATUS, so readers get() it
                    without consuming: the AE needs it to know the
                    current exposure, and the monitor metric reads the
                    same value without stealing it from the AE.
  "ae_status"       AE -> anyone. Status again, for observability.

The loop therefore has a ONE FRAME delay: the camera renders frame N,
the AE measures it and posts a correction, and the camera applies that
correction when it renders frame N+1. That is what real hardware does,
and it is why an over-eager AE oscillates — the `damping` parameter
exists to control exactly that.

Frame-index bookkeeping follows the same rule as the temporal steps: a
live-mode re-run of the SAME frame must not consume another command or
advance the control loop, so previews stay idempotent (identical image
for identical frame index, noise included, because the sensor RNG is
seeded per frame).
"""
import numpy as np

from src.GUI.pipeline_editor.array_utils import to_luminance
from src.GUI.pipeline_editor.base_step import (
    ProcessingStep, ParamSpec, register_step,
)


TOPIC_CONTROL = "camera_control"   # command  -> pop()
TOPIC_STATE = "camera_state"       # status   -> get()
TOPIC_AE = "ae_status"             # status   -> get()


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
class CameraSimulator(ProcessingStep):
    """Synthetic camera imaging a ColorChecker chart.

    Sensor model, in order: scene reflectance x illumination -> photo-
    electrons (proportional to exposure time), Poisson shot noise,
    Gaussian read noise, analog gain, full-well saturation, ADC
    quantization, optional sRGB encoding. Signal-dependent shot noise is
    what makes this useful for exposure work: under-expose and push with
    gain, and the noise follows.

    Exposure is either taken from the parameters (manual) or driven over
    the bus by an Auto Exposure node — see `accept_control`.
    """
    NAME = "Camera Simulator"
    CATEGORY = "Camera/Simulation"
    IS_SOURCE = True
    IS_SEQUENCE_AWARE = True

    FULL_WELL_E = 10000.0
    # Normalization: a perfect white reflector at illumination 1.0 fills
    # the well in 10 ms at gain 1, so exposure values read intuitively.
    _E_PER_MS = FULL_WELL_E / 10.0

    PARAMS = [
        ParamSpec("resolution", "Resolution", "choice", default="640x480",
                  choices=list(_RESOLUTIONS.keys())),
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
                  choices=list(_BIT_DEPTHS.keys())),
        ParamSpec("encoding", "Output Encoding", "choice", default="sRGB",
                  choices=["sRGB", "linear"]),
        ParamSpec("seed", "Noise Seed", "int", default=0,
                  min_value=0, max_value=100000, step=1),
    ]

    def __init__(self):
        super().__init__()
        self._index = 0
        self._total = 1
        self._last_index = None
        self._chart = None
        self._chart_key = None
        self._exposure_ms = None      # None -> take from params on first use
        self._gain = None

    # --- sequence protocol ---------------------------------------------
    def frame_count(self) -> int:
        return max(1, int(self.values.get("num_frames", 1)))

    def set_frame_index(self, index: int, total_frames: int):
        self._index = index
        self._total = max(1, total_frames)

    def begin_sequence(self, total_frames: int):
        # A batch starts from the parameter values, so a run is
        # reproducible regardless of where previews left the loop.
        self._exposure_ms = float(self.values.get("exposure_ms", 5.0))
        self._gain = float(self.values.get("gain", 1.0))
        self._last_index = None

    # --- scene ----------------------------------------------------------
    def _get_chart(self, width: int, height: int) -> np.ndarray:
        if self._chart is None or self._chart_key != (width, height):
            self._chart = render_colorchecker(width, height)
            self._chart_key = (width, height)
        return self._chart

    def _illumination(self, profile: str, level: float) -> float:
        """Scene illumination for the current frame. The non-constant
        profiles exist to give an AE something to track."""
        t = self._index / max(1, self._total - 1)     # 0..1
        if profile == "step":
            # Two abrupt changes: +3 EV, then -1 EV from the base.
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

    # --- sensor ----------------------------------------------------------
    def process(self, image: np.ndarray, resolution: str = "640x480",
                num_frames: int = 60, illumination: str = "step",
                illum_level: float = 0.5, exposure_ms: float = 5.0,
                gain: float = 1.0, accept_control: bool = True,
                read_noise_e: float = 4.0, bit_depth: str = "12",
                encoding: str = "sRGB", seed: int = 0,
                **kwargs) -> np.ndarray:
        idx = self._index
        if self._exposure_ms is None:
            self._exposure_ms, self._gain = float(exposure_ms), float(gain)

        new_frame = idx != self._last_index
        if new_frame and accept_control and self.bus is not None:
            # pop(): a command must be consumed, or the next frame would
            # apply the same correction again.
            cmd = self.bus.pop(TOPIC_CONTROL, None)
            if isinstance(cmd, dict):
                self._exposure_ms = float(cmd.get("exposure_ms",
                                                  self._exposure_ms))
                self._gain = float(cmd.get("gain", self._gain))
        if not accept_control:
            self._exposure_ms, self._gain = float(exposure_ms), float(gain)

        width, height = _RESOLUTIONS.get(resolution, (640, 480))
        reflectance = self._get_chart(width, height)
        illum = self._illumination(illumination, float(illum_level))

        # Photoelectrons before noise.
        electrons = (reflectance * illum
                     * self._exposure_ms * self._E_PER_MS)

        # Seeded per FRAME (not per call), so re-running the same frame
        # in live mode reproduces the identical image.
        rng = np.random.RandomState((int(seed) * 1000003 + idx) % (2 ** 31))
        electrons = rng.poisson(np.clip(electrons, 0.0, 1e12)).astype(np.float32)
        if read_noise_e > 0:
            electrons += rng.normal(0.0, float(read_noise_e),
                                    electrons.shape).astype(np.float32)

        signal = np.clip(electrons * self._gain, 0.0, self.FULL_WELL_E)
        norm = signal / self.FULL_WELL_E                     # linear [0,1]
        if encoding == "sRGB":
            norm = linear_to_srgb(norm)

        levels = (1 << _BIT_DEPTHS.get(bit_depth, 12)) - 1
        out = np.round(norm * levels) / levels

        if self.bus is not None:
            # post()/get(): status, read by the AE and by monitors
            # without any of them consuming it.
            self.bus.post(TOPIC_STATE, {
                "frame": idx,
                "exposure_ms": self._exposure_ms,
                "gain": self._gain,
                "illumination": illum,
                "encoding": encoding,
                "saturated_fraction": float(
                    (signal >= self.FULL_WELL_E * 0.999).mean()),
            })
        self._last_index = idx
        return out.astype(np.float32)


@register_step
class AutoExposure(ProcessingStep):
    """Auto-exposure controller. Measures the incoming frame, computes
    the correction in EV, and commands the camera over the bus. Passes
    the image through unchanged, so it can sit anywhere in the chain.

    Control law: error_ev = log2(target / measured) in the LINEAR domain
    (the measurement is linearized first if the camera encodes sRGB —
    otherwise the loop gain would vary with brightness). The step is
    damped and clamped, then the new exposure product is split between
    exposure time and gain, preferring time (gain amplifies read noise
    without collecting more photons).
    """
    NAME = "Auto Exposure"
    CATEGORY = "Camera/Control"
    PARAMS = [
        # Expressed in the camera's OUTPUT encoding for convenience, but
        # the loop controls the LINEAR mean (exposure scales linearly, so
        # metering must too). Consequence worth knowing: the displayed
        # mean of a real scene will NOT equal this number, because
        # mean(encode(x)) != encode(mean(x)) — 0.45 here settles a
        # ColorChecker at a display mean near 0.38. Judge convergence by
        # AE Monitor's error_ev, not by eyeballing the mean.
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

    def __init__(self):
        super().__init__()
        self._index = 0
        self._last_index = None
        self._exposure_ms = 5.0     # fallback model if no camera state
        self._gain = 1.0
        self._weights = None
        self._weights_shape = None

    def begin_sequence(self, total_frames: int):
        self._last_index = None

    def set_frame_index(self, index: int, total_frames: int):
        self._index = index

    IS_SEQUENCE_AWARE = True

    def _center_weights(self, shape) -> np.ndarray:
        if self._weights is None or self._weights_shape != shape:
            h, w = shape
            yy = np.linspace(-1.0, 1.0, h)[:, None]
            xx = np.linspace(-1.0, 1.0, w)[None, :]
            self._weights = np.exp(-(xx ** 2 + yy ** 2) / 0.5).astype(
                np.float32)
            self._weights_shape = shape
        return self._weights

    def process(self, image: np.ndarray, target: float = 0.45,
                metering: str = "average", damping: float = 0.6,
                max_step_ev: float = 1.5, min_exposure_ms: float = 0.05,
                max_exposure_ms: float = 33.0, max_gain: float = 16.0,
                linearize: bool = True, **kwargs) -> np.ndarray:
        idx = self._index
        if idx == self._last_index:
            # Same frame re-run (live preview): measuring again is fine,
            # but issuing another command would advance the control loop
            # without the scene having advanced.
            return image
        self._last_index = idx

        lum = to_luminance(image).astype(np.float32)

        # Current exposure: from the camera's own status if it is in the
        # graph, else from this node's internal model.
        state = self.bus.get(TOPIC_STATE) if self.bus is not None else None
        if isinstance(state, dict):
            self._exposure_ms = float(state.get("exposure_ms",
                                                self._exposure_ms))
            self._gain = float(state.get("gain", self._gain))
            encoded = state.get("encoding", "sRGB") == "sRGB"
        else:
            encoded = True
        do_linearize = bool(linearize) and encoded

        lin = srgb_to_linear(lum) if do_linearize else lum
        target_lin = (float(srgb_to_linear(np.array([target])) [0])
                      if do_linearize else float(target))

        if metering == "center-weighted":
            w = self._center_weights(lin.shape)
            measured = float((lin * w).sum() / w.sum())
        else:
            measured = float(lin.mean())
        measured = max(measured, 1e-6)

        error_ev = float(np.log2(target_lin / measured))
        step_ev = float(np.clip(error_ev * float(damping),
                                -abs(max_step_ev), abs(max_step_ev)))

        total = max(self._exposure_ms * self._gain, 1e-6)
        new_total = total * (2.0 ** step_ev)

        if metering == "highlight-protect":
            # Predict where the brightest content lands after the change
            # and pull back if it would clip.
            p99 = float(np.percentile(lin, 99.0))
            if p99 > 1e-6:
                cap = total * (self.HIGHLIGHT_LIMIT / p99)
                new_total = min(new_total, cap)

        new_total = float(np.clip(new_total,
                                  min_exposure_ms,
                                  max_exposure_ms * max_gain))
        new_exposure = float(np.clip(new_total, min_exposure_ms,
                                     max_exposure_ms))
        new_gain = float(np.clip(new_total / new_exposure, 1.0, max_gain))

        if self.bus is not None:
            self.bus.post(TOPIC_CONTROL, {          # command -> pop()
                "exposure_ms": new_exposure,
                "gain": new_gain,
            })
            self.bus.post(TOPIC_AE, {               # status  -> get()
                "frame": idx,
                "measured": measured,
                "target": target_lin,
                "error_ev": error_ev,
                "applied_ev": step_ev,
                "exposure_ms": new_exposure,
                "gain": new_gain,
            })
        self._exposure_ms, self._gain = new_exposure, new_gain
        return image


@register_step
class AEMonitor(ProcessingStep):
    """Reports one number from the exposure loop, so convergence can be
    watched on the node, dumped to CSV over a batch, or used as an
    optimizer objective (e.g. minimize |error| to tune damping).

    Reads the status topics with get(), never pop(): a monitor must not
    steal a message the camera or AE still needs."""
    NAME = "AE Monitor"
    CATEGORY = "Metrics/Camera"
    IS_METRIC = True
    PARAMS = [
        ParamSpec("quantity", "Quantity", "choice", default="error_ev",
                  choices=["error_ev", "measured", "exposure_ms", "gain",
                           "illumination", "saturated_fraction"]),
    ]

    def process(self, image: np.ndarray, quantity: str = "error_ev",
                **kwargs) -> float:
        if self.bus is None:
            return float("nan")
        ae = self.bus.get(TOPIC_AE) or {}
        cam = self.bus.get(TOPIC_STATE) or {}
        if quantity in ("illumination", "saturated_fraction"):
            return float(cam.get(quantity, float("nan")))
        if quantity in ("exposure_ms", "gain"):
            return float(cam.get(quantity, ae.get(quantity, float("nan"))))
        return float(ae.get(quantity, float("nan")))
