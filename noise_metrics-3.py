"""
Noise-estimation and sharpness metrics. Values are in the pipeline's
[0,1] intensity units (a temporal sigma of 0.004 = 0.4% of full scale).
"""
import math

import numpy as np
import cv2

from src.GUI.pipeline_editor.array_utils import to_luminance
from src.GUI.pipeline_editor.base_step import (
    MetricCSVMixin, ParamSpec, ProcessingStep, StatefulStep, register_step,
)


def _gray64(image) -> np.ndarray:
    """Luminance in float64 — the extra precision matters for the
    accumulating statistics here (Welford, Laplacian sums)."""
    return to_luminance(np.asarray(image, np.float64))


@register_step
class TemporalNoise(MetricCSVMixin, StatefulStep):
    """Mean per-pixel standard deviation across the frames seen so far
    (Welford — no frame buffer, so any sequence length works). The scene
    must be static for this to mean sensor noise rather than motion."""
    NAME = "Temporal Noise"
    CATEGORY = "Metrics/Noise"

    def reset(self):
        self._n = 0
        self._mean = None
        self._m2 = None

    def advance(self, image):
        a = _gray64(image)
        self._n += 1
        if self._mean is None:
            self._mean = a.copy()
            self._m2 = np.zeros_like(a)
        else:
            d = a - self._mean
            self._mean += d / self._n
            self._m2 += d * (a - self._mean)
        if self._n < 2:
            return 0.0
        return float(np.sqrt(self._m2 / (self._n - 1)).mean())


@register_step
class SpatialNoise(MetricCSVMixin, ProcessingStep):
    """Immerkaer's single-frame estimate: convolve with a mask that
    annihilates locally linear structure, recover sigma from the mean
    absolute response. Texture and strong edges leak in, biasing high —
    trust it on flat scenes, treat it as an upper bound elsewhere."""
    NAME = "Spatial Noise"
    CATEGORY = "Metrics/Noise"

    _MASK = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])

    def process(self, image):
        a = _gray64(image)
        h, w = a.shape
        if h < 3 or w < 3:
            return 0.0
        conv = cv2.filter2D(a, -1, self._MASK,
                            borderType=cv2.BORDER_REPLICATE)
        return float(math.sqrt(math.pi / 2.0)
                     * np.abs(conv[1:-1, 1:-1]).sum()
                     / (6.0 * (w - 2) * (h - 2)))


@register_step
class FPN(MetricCSVMixin, ProcessingStep):
    """Fixed-pattern noise along one axis: collapse to a column (or row)
    mean profile, remove the scene trend with a box smooth, report the
    residual's std — the streaking temporal averaging cannot remove."""
    NAME = "FPN"
    CATEGORY = "Metrics/Noise"
    PARAMS = [
        ParamSpec("axis", "Axis", "choice", default="column",
                  choices=["column", "row"]),
        ParamSpec("detrend", "Detrend Window (px)", "int", default=15,
                  min_value=3, max_value=101, step=2),
    ]

    def process(self, image):
        a = _gray64(image)
        profile = a.mean(axis=0) if self.p.axis == "column" else a.mean(axis=1)
        k = max(3, int(self.p.detrend) | 1)
        if profile.size <= k:
            return float(profile.std())
        padded = np.pad(profile, k // 2, mode="edge")
        trend = np.convolve(padded, np.ones(k) / k, mode="valid")
        return float((profile - trend).std())


@register_step
class Sharpness(MetricCSVMixin, ProcessingStep):
    """Detail/focus measure. Higher is sharper; the absolute number only
    means something when comparing variants of the same scene."""
    NAME = "Sharpness"
    CATEGORY = "Metrics/General"
    PARAMS = [ParamSpec("method", "Method", "choice", default="tenengrad",
                        choices=["tenengrad", "laplacian_var",
                                 "gradient_energy"])]

    def process(self, image):
        a = _gray64(image)
        if self.p.method == "laplacian_var":
            return float(cv2.Laplacian(a, cv2.CV_64F).var())
        gx = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
        if self.p.method == "gradient_energy":
            return float((np.abs(gx) + np.abs(gy)).mean())
        return float((gx * gx + gy * gy).mean())
