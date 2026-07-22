"""
Noise-estimation and sharpness metrics.

All values are in the pipeline's [0,1] intensity units (a temporal sigma
of 0.004 means 0.4% of full scale). Everything is IS_METRIC, so the CSV
checkbox and the optimizer work on all of them out of the box.

  Temporal Noise   per-pixel std across the frames of a batch — measures
                   exactly what temporal denoising removes (NETD-style).
                   Needs a sequence; a single-frame preview shows the
                   running value (0 on the first frame).
  Spatial Noise    Immerkaer's single-frame Laplacian estimator. Fast and
                   parameter-free, but structure/texture leaks into the
                   estimate — trust it on flat-ish scenes, treat it as an
                   upper bound elsewhere.
  FPN              fixed-pattern noise: std of the column (or row) mean
                   profile after detrending, i.e. the strength of the
                   streaking that temporal averaging can NOT remove.
  Sharpness        Tenengrad / Laplacian variance / gradient energy — for
                   quantifying detail retention across tonemapper or
                   denoiser settings.
"""
import math

import numpy as np
import cv2

from src.GUI.pipeline_editor.base_step import (
    ProcessingStep, ParamSpec, register_step,
)


def _to_gray64(image: np.ndarray) -> np.ndarray:
    a = image.astype(np.float64)
    if a.ndim == 3:
        a = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    return a


@register_step
class TemporalNoise(ProcessingStep):
    """Mean per-pixel standard deviation across the frames seen so far
    (Welford's algorithm — no frame buffer, so any sequence length).

    Frame-index bookkeeping mirrors the temporal steps: a live-mode
    re-run of the SAME frame doesn't double-count it, a frame jump or a
    new batch resets the accumulation. The scene must be static for the
    number to mean sensor noise rather than motion."""
    NAME = "Temporal Noise"
    CATEGORY = "Metrics/Noise"
    IS_METRIC = True
    IS_SEQUENCE_AWARE = True

    def __init__(self):
        super().__init__()
        self._reset()
        self._idx = 0

    def _reset(self):
        self._n = 0
        self._mean = None
        self._m2 = None
        self._last_idx = None
        self._last_val = 0.0

    def begin_sequence(self, total_frames: int):
        self._reset()

    def set_frame_index(self, index: int, total_frames: int):
        self._idx = index

    def process(self, image: np.ndarray, **kwargs) -> float:
        idx = self._idx
        if self._last_idx is not None:
            if idx == self._last_idx:
                # Live-mode re-run of the same frame: feeding it again
                # would bias sigma downward. Report the current value.
                return self._last_val
            if idx != self._last_idx + 1:
                self._reset()       # slider jump: history is meaningless

        a = _to_gray64(image)
        self._n += 1
        if self._mean is None:
            self._mean = a.copy()
            self._m2 = np.zeros_like(a)
        else:
            d = a - self._mean
            self._mean += d / self._n
            self._m2 += d * (a - self._mean)

        if self._n > 1:
            sigma = float(np.sqrt(self._m2 / (self._n - 1)).mean())
        else:
            sigma = 0.0
        self._last_idx = idx
        self._last_val = sigma
        return sigma


@register_step
class SpatialNoise(ProcessingStep):
    """Immerkaer's fast single-frame noise estimate: the image is
    convolved with a Laplacian-difference mask that annihilates locally
    linear structure, and sigma is recovered from the mean absolute
    response. Texture and strong edges still leak in, biasing the
    estimate high."""
    NAME = "Spatial Noise"
    CATEGORY = "Metrics/Noise"
    IS_METRIC = True

    _MASK = np.array([[1., -2., 1.],
                      [-2., 4., -2.],
                      [1., -2., 1.]])

    def process(self, image: np.ndarray, **kwargs) -> float:
        a = _to_gray64(image)
        h, w = a.shape
        if h < 3 or w < 3:
            return 0.0
        conv = cv2.filter2D(a, -1, self._MASK,
                            borderType=cv2.BORDER_REPLICATE)
        interior = np.abs(conv[1:-1, 1:-1]).sum()
        return float(math.sqrt(math.pi / 2.0)
                     * interior / (6.0 * (w - 2) * (h - 2)))


@register_step
class FPN(ProcessingStep):
    """Fixed-pattern noise along one axis: collapse the image to a
    column (or row) mean profile, remove the low-frequency scene trend
    with a box smooth, and report the residual's std — the streaking
    amplitude that per-frame averaging cannot remove."""
    NAME = "FPN"
    CATEGORY = "Metrics/Noise"
    IS_METRIC = True
    PARAMS = [
        ParamSpec("axis", "Axis", "choice", default="column",
                  choices=["column", "row"]),
        ParamSpec("detrend", "Detrend Window (px)", "int", default=15,
                  min_value=3, max_value=101, step=2),
    ]

    def process(self, image: np.ndarray, axis: str = "column",
                detrend: int = 15, **kwargs) -> float:
        a = _to_gray64(image)
        profile = a.mean(axis=0) if axis == "column" else a.mean(axis=1)
        k = max(3, int(detrend) | 1)          # odd
        if profile.size <= k:
            return float(profile.std())
        kernel = np.ones(k) / k
        pad = k // 2
        padded = np.pad(profile, pad, mode="edge")
        trend = np.convolve(padded, kernel, mode="valid")
        return float((profile - trend).std())


@register_step
class Sharpness(ProcessingStep):
    """Detail/focus measure. tenengrad = mean squared Sobel gradient
    magnitude (the robust default); laplacian_var = variance of the
    Laplacian (classic autofocus measure); gradient_energy = mean
    absolute gradient. Higher is sharper; the absolute number is only
    meaningful when comparing variants of the same scene."""
    NAME = "Sharpness"
    CATEGORY = "Metrics/General"
    IS_METRIC = True
    PARAMS = [
        ParamSpec("method", "Method", "choice", default="tenengrad",
                  choices=["tenengrad", "laplacian_var",
                           "gradient_energy"]),
    ]

    def process(self, image: np.ndarray, method: str = "tenengrad",
                **kwargs) -> float:
        a = _to_gray64(image)
        if method == "laplacian_var":
            return float(cv2.Laplacian(a, cv2.CV_64F).var())
        gx = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
        if method == "gradient_energy":
            return float((np.abs(gx) + np.abs(gy)).mean())
        return float((gx * gx + gy * gy).mean())      # tenengrad
