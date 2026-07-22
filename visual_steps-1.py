"""
Visualization / channel steps: false color for thermal data, image
differencing (Subtract, AbsDiff), and channel operations
(RGB->gray, extract, merge).

All inputs arrive as float32 [0,1] per the pipeline contract; outputs
stay in that range (Subtract's signed result is handled explicitly, see
its docstring) so nothing here triggers the executor's clip warning.
"""
import numpy as np
import cv2

from src.GUI.pipeline_editor.base_step import (
    ProcessingStep, ParamSpec, register_step,
)
from src.GUI.pipeline_editor.array_utils import (
    to_luminance, match_channels, to_uint8,
)


# ---------------------------------------------------------------------------
# False color
# ---------------------------------------------------------------------------

def _ironbow_lut() -> np.ndarray:
    """Classic thermal-imaging 'ironbow' palette: black -> deep blue ->
    purple -> red -> orange -> yellow -> white. Built from anchor colors
    interpolated to 256 entries; returned as (256, 3) uint8 RGB."""
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
    "Jet": cv2.COLORMAP_JET,
    "Turbo": cv2.COLORMAP_TURBO,
    "Hot": cv2.COLORMAP_HOT,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Magma": cv2.COLORMAP_MAGMA,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Bone": cv2.COLORMAP_BONE,
    "Rainbow": cv2.COLORMAP_RAINBOW,
}


@register_step
class FalseColor(ProcessingStep):
    """Map grayscale intensity to a color palette — the standard last
    stage of a thermal display chain. Color input is converted to
    luminance first."""
    NAME = "False Color"
    CATEGORY = "Color"
    PARAMS = [
        ParamSpec("colormap", "Colormap", "choice", default="Ironbow",
                  choices=["Ironbow"] + sorted(_CV2_MAPS.keys())),
        ParamSpec("invert", "Invert (white-hot ↔ black-hot)", "bool",
                  default=False),
    ]

    def process(self, image: np.ndarray, colormap: str = "Ironbow",
                invert: bool = False, **kwargs) -> np.ndarray:
        g = to_uint8(to_luminance(image))
        if invert:
            g = 255 - g
        if colormap == "Ironbow":
            return _ironbow_lut()[g]                      # (H,W,3) RGB
        bgr = cv2.applyColorMap(g, _CV2_MAPS[colormap])
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Differencing
# ---------------------------------------------------------------------------

@register_step
class Subtract(ProcessingStep):
    """A − B, scaled by gain. The raw result is signed, which the
    pipeline's [0,1] contract can't carry, so choose how to encode it:

      offset  0.5 + (A−B)·gain/2 — mid-gray means equal, brighter means
              A > B, darker means B > A. The right mode for LOOKING at a
              difference, since negative deviations stay visible.
      clip    max(0, (A−B)·gain) — keeps only where A exceeds B.
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

    def process(self, image_a: np.ndarray, image_b: np.ndarray,
                mode: str = "offset", gain: float = 1.0,
                **kwargs) -> np.ndarray:
        a, b = match_channels(image_a, image_b)
        diff = (a.astype(np.float32) - b.astype(np.float32)) * gain
        if mode == "clip":
            return np.clip(diff, 0.0, 1.0)
        return np.clip(0.5 + diff * 0.5, 0.0, 1.0)


@register_step
class AbsDiff(ProcessingStep):
    """|A − B| · gain — the fastest way to see WHERE two processing
    variants diverge. Crank the gain for small differences."""
    NAME = "AbsDiff"
    CATEGORY = "Compositing"
    N_INPUTS = 2
    INPUT_LABELS = ("A", "B")
    PARAMS = [
        ParamSpec("gain", "Gain", "float", default=1.0,
                  min_value=0.1, max_value=100.0, step=0.5),
    ]

    def process(self, image_a: np.ndarray, image_b: np.ndarray,
                gain: float = 1.0, **kwargs) -> np.ndarray:
        a, b = match_channels(image_a, image_b)
        return np.clip(np.abs(a.astype(np.float32) - b.astype(np.float32))
                       * gain, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Channel operations
# ---------------------------------------------------------------------------

@register_step
class RGBToGray(ProcessingStep):
    NAME = "RGB → Gray"
    CATEGORY = "Color/Channels"
    PARAMS = [
        ParamSpec("method", "Method", "choice", default="luminance",
                  choices=["luminance", "average"]),
    ]

    def process(self, image: np.ndarray, method: str = "luminance",
                **kwargs) -> np.ndarray:
        if image.ndim == 2:
            return image
        if method == "average":
            return image.mean(axis=2).astype(np.float32)
        return to_luminance(image).astype(np.float32)


@register_step
class ChannelExtract(ProcessingStep):
    NAME = "Channel Extract"
    CATEGORY = "Color/Channels"
    PARAMS = [
        ParamSpec("channel", "Channel", "choice", default="R",
                  choices=["R", "G", "B"]),
    ]

    def process(self, image: np.ndarray, channel: str = "R",
                **kwargs) -> np.ndarray:
        if image.ndim == 2:
            return image          # gray input: nothing to extract
        idx = {"R": 0, "G": 1, "B": 2}[channel]
        return image[..., idx]


@register_step
class ChannelMerge(ProcessingStep):
    """Assemble an RGB image from three grayscale inputs. A color input
    on any port is reduced to its luminance first."""
    NAME = "Channel Merge"
    CATEGORY = "Color/Channels"
    N_INPUTS = 3
    INPUT_LABELS = ("R", "G", "B")

    def process(self, image_r: np.ndarray, image_g: np.ndarray,
                image_b: np.ndarray, **kwargs) -> np.ndarray:
        r = to_luminance(image_r).astype(np.float32)
        g = to_luminance(image_g).astype(np.float32)
        b = to_luminance(image_b).astype(np.float32)
        h = min(r.shape[0], g.shape[0], b.shape[0])
        w = min(r.shape[1], g.shape[1], b.shape[1])
        return np.stack([r[:h, :w], g[:h, :w], b[:h, :w]],
                        axis=2).astype(np.float32)
