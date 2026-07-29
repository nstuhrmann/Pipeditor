"""
Visualization, differencing, channel and overlay steps.

All inputs arrive as float32 [0,1] per the pipeline contract; outputs
stay in that range so nothing here trips the executor's clip warning.
"""
import numpy as np
import cv2

from src.GUI.pipeline_editor.array_utils import (
    METRIC_META_PREFIX, match_channels, to_luminance, to_uint8,
)
from src.GUI.pipeline_editor.base_step import (
    ProcessingStep, ParamSpec, register_step,
)


# ---------------------------------------------------------------------------
# False colour
# ---------------------------------------------------------------------------

def _ironbow_lut() -> np.ndarray:
    """Classic thermal 'ironbow' palette: black -> deep blue -> purple ->
    red -> orange -> yellow -> white, as a (256, 3) uint8 RGB LUT."""
    anchors = np.array([
        (0, 0, 0), (0, 0, 80), (60, 0, 120), (120, 0, 140),
        (180, 40, 100), (220, 100, 40), (255, 170, 0),
        (255, 230, 80), (255, 255, 255),
    ], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, len(anchors))
    grid = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(grid, xs, anchors[:, c]) for c in range(3)],
                   axis=1)
    return lut.astype(np.uint8)


_CV2_MAPS = {
    "Jet": cv2.COLORMAP_JET, "Turbo": cv2.COLORMAP_TURBO,
    "Hot": cv2.COLORMAP_HOT, "Viridis": cv2.COLORMAP_VIRIDIS,
    "Inferno": cv2.COLORMAP_INFERNO, "Magma": cv2.COLORMAP_MAGMA,
    "Plasma": cv2.COLORMAP_PLASMA, "Bone": cv2.COLORMAP_BONE,
    "Rainbow": cv2.COLORMAP_RAINBOW,
}


@register_step
class FalseColor(ProcessingStep):
    """Map grayscale intensity to a colour palette — the usual last stage
    of a thermal display chain. Colour input is reduced to luminance."""
    NAME = "False Color"
    CATEGORY = "Color"
    PARAMS = [
        ParamSpec("colormap", "Colormap", "choice", default="Ironbow",
                  choices=["Ironbow"] + sorted(_CV2_MAPS)),
        ParamSpec("invert", "Invert (white-hot / black-hot)", "bool",
                  default=False),
    ]

    def process(self, image):
        g = to_uint8(to_luminance(image))
        if self.p.invert:
            g = 255 - g
        if self.p.colormap == "Ironbow":
            return _ironbow_lut()[g]
        return cv2.cvtColor(cv2.applyColorMap(g, _CV2_MAPS[self.p.colormap]),
                            cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Differencing
# ---------------------------------------------------------------------------

@register_step
class Subtract(ProcessingStep):
    """A - B, scaled by gain. The raw result is signed, which [0,1] can't
    carry, so pick an encoding:

      offset  0.5 + (A-B)*gain/2 — mid-grey means equal, both polarities
              stay visible. The right mode for LOOKING at a difference.
      clip    max(0, (A-B)*gain) — only where A exceeds B.
    """
    NAME = "Subtract"
    CATEGORY = "Compositing"
    N_INPUTS = 2
    INPUT_LABELS = ("A", "B")
    PARAMS = [
        ParamSpec("mode", "Mode", "choice", default="offset",
                  choices=["offset", "clip"]),
        ParamSpec("gain", "Gain", "float", default=1.0,
                  min_value=0.1, max_value=50.0, step=0.5),
    ]

    def process(self, image_a, image_b):
        a, b = match_channels(image_a, image_b)
        diff = (np.asarray(a, np.float32) - np.asarray(b, np.float32)) * self.p.gain
        if self.p.mode == "clip":
            return np.clip(diff, 0.0, 1.0)
        return np.clip(0.5 + diff * 0.5, 0.0, 1.0)


@register_step
class AbsDiff(ProcessingStep):
    """|A - B| * gain — the fastest way to see WHERE two variants differ."""
    NAME = "AbsDiff"
    CATEGORY = "Compositing"
    N_INPUTS = 2
    INPUT_LABELS = ("A", "B")
    PARAMS = [ParamSpec("gain", "Gain", "float", default=1.0,
                        min_value=0.1, max_value=100.0, step=0.5)]

    def process(self, image_a, image_b):
        a, b = match_channels(image_a, image_b)
        return np.clip(np.abs(np.asarray(a, np.float32)
                              - np.asarray(b, np.float32)) * self.p.gain,
                       0.0, 1.0)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

@register_step
class RGBToGray(ProcessingStep):
    NAME = "RGB to Gray"
    CATEGORY = "Color/Channels"
    PARAMS = [ParamSpec("method", "Method", "choice", default="luminance",
                        choices=["luminance", "average"])]

    def process(self, image):
        if image.ndim == 2:
            return image
        if self.p.method == "average":
            return np.asarray(image, np.float32).mean(axis=2)
        return to_luminance(image).astype(np.float32)


@register_step
class ChannelExtract(ProcessingStep):
    NAME = "Channel Extract"
    CATEGORY = "Color/Channels"
    PARAMS = [ParamSpec("channel", "Channel", "choice", default="R",
                        choices=["R", "G", "B"])]

    def process(self, image):
        if image.ndim == 2:
            return image
        return image[..., {"R": 0, "G": 1, "B": 2}[self.p.channel]]


@register_step
class ChannelMerge(ProcessingStep):
    """Assemble RGB from three grayscale inputs; colour inputs are
    reduced to luminance first."""
    NAME = "Channel Merge"
    CATEGORY = "Color/Channels"
    N_INPUTS = 3
    INPUT_LABELS = ("R", "G", "B")

    def process(self, image_r, image_g, image_b):
        planes = [to_luminance(x).astype(np.float32)
                  for x in (image_r, image_g, image_b)]
        h = min(p.shape[0] for p in planes)
        w = min(p.shape[1] for p in planes)
        return np.stack([p[:h, :w] for p in planes], axis=2)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

_CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


@register_step
class AnnotateMetrics(ProcessingStep):
    """Draw every metric value carried by the incoming frame, stacked one
    per line.

    Metrics no longer burn their value into the image themselves — they
    attach it to the frame's metadata. That is what lets several metrics
    coexist: each adds its own entry, and this step lays them all out
    without overlapping. Values appear in execution order.
    """
    NAME = "Annotate Metrics"
    CATEGORY = "Overlay"
    PARAMS = [
        ParamSpec("corner", "Corner", "choice", default="top-left",
                  choices=list(_CORNERS)),
        ParamSpec("scale", "Text Scale", "float", default=1.0,
                  min_value=0.2, max_value=5.0, step=0.1),
        ParamSpec("show_names", "Show Metric Names", "bool", default=True),
        ParamSpec("decimals", "Decimals", "int", default=4,
                  min_value=0, max_value=8, step=1),
    ]

    def _lines(self, meta: dict) -> list:
        lines = []
        for key, value in meta.items():
            if not key.startswith(METRIC_META_PREFIX):
                continue
            name = key[len(METRIC_META_PREFIX):]
            if isinstance(value, float):
                text = f"{value:.{int(self.p.decimals)}f}"
            else:
                text = str(value)
            lines.append(f"{name}: {text}" if self.p.show_names else text)
        return lines

    def process(self, image):
        lines = self._lines(getattr(image, "meta", {}) or {})
        if not lines:
            return image
        img = np.ascontiguousarray(np.asarray(image, np.float32).copy())
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.3, h / 500.0 * float(self.p.scale))
        thick = max(1, int(round(h / 400.0 * float(self.p.scale))))
        row = int(cv2.getTextSize("Ag", font, scale, thick)[0][1] * 1.8) + 4
        fill = 1.0 if img.ndim == 2 else (1.0, 0.85, 0.2)
        outline = 0.0 if img.ndim == 2 else (0.0, 0.0, 0.0)

        widths = [cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines]
        top = self.p.corner.startswith("top")
        left = self.p.corner.endswith("left")
        for i, (text, tw) in enumerate(zip(lines, widths)):
            x = 8 if left else max(4, w - tw - 8)
            y = (row * (i + 1) if top
                 else h - 8 - row * (len(lines) - 1 - i))
            cv2.putText(img, text, (x, y), font, scale, outline,
                        thick + 2, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale, fill,
                        thick, cv2.LINE_AA)
        return np.clip(img, 0.0, 1.0)
