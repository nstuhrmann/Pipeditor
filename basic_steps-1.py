"""
Example processing steps and a metric step.
"""
import numpy as np
from base_step import ProcessingStep, ParamSpec, register_step


@register_step
class GaussianBlur(ProcessingStep):
    NAME = "Gaussian Blur"
    CATEGORY = "Filter"
    PARAMS = [
        ParamSpec("sigma", "Sigma", "float",
                  default=2.0, min_value=0.1, max_value=20.0, step=0.1),
    ]

    def process(self, image: np.ndarray, sigma=2.0) -> np.ndarray:
        try:
            from scipy.ndimage import gaussian_filter
            if image.ndim == 3:
                return np.stack(
                    [gaussian_filter(image[..., c], sigma=sigma)
                     for c in range(image.shape[2])], axis=-1)
            return gaussian_filter(image, sigma=sigma)
        except ImportError:
            k = max(1, int(sigma))
            kernel = np.ones((2*k+1, 2*k+1)) / (2*k+1)**2
            return _conv(image, kernel)


@register_step
class Threshold(ProcessingStep):
    NAME = "Threshold"
    CATEGORY = "Segmentation"
    PARAMS = [
        ParamSpec("value",  "Threshold", "int",  default=128,
                  min_value=0, max_value=255, step=1),
        ParamSpec("invert", "Invert",    "bool", default=False),
    ]

    def process(self, image: np.ndarray, value=128, invert=False):
        gray = image if image.ndim == 2 else image.mean(axis=2)
        mask = gray > value
        if invert:
            mask = ~mask
        return (mask.astype(np.uint8) * 255)


@register_step
class Invert(ProcessingStep):
    NAME = "Invert"
    CATEGORY = "Point Operations"
    PARAMS = []

    def process(self, image: np.ndarray) -> np.ndarray:
        mx = 255 if image.dtype == np.uint8 else float(image.max() or 1)
        return mx - image


@register_step
class BrightnessContrast(ProcessingStep):
    NAME = "Brightness / Contrast"
    CATEGORY = "Point Operations"
    PARAMS = [
        ParamSpec("brightness", "Brightness", "int",   default=0,
                  min_value=-255, max_value=255, step=1),
        ParamSpec("contrast",   "Contrast",   "float", default=1.0,
                  min_value=0.1, max_value=3.0, step=0.05),
    ]

    def process(self, image: np.ndarray,
                brightness=0, contrast=1.0) -> np.ndarray:
        out = image.astype(np.float32) * contrast + brightness
        return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Metric step — two inputs, returns a scalar value shown on the node
# ---------------------------------------------------------------------------

@register_step
class MSEMetric(ProcessingStep):
    """
    Mean Squared Error between image A and image B.
    Lower = more similar. Shown as a value on the node.
    """
    NAME      = "MSE"
    CATEGORY  = "Metrics"
    N_INPUTS  = 2
    IS_METRIC = True
    PARAMS    = []

    def process(self, image_a: np.ndarray,
                image_b: np.ndarray) -> float:
        a = image_a.astype(np.float64)
        b = image_b.astype(np.float64)
        if a.shape != b.shape:
            # crop to common size
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]
        return float(np.mean((a - b) ** 2))


@register_step
class PSNRMetric(ProcessingStep):
    """
    Peak Signal-to-Noise Ratio between image A (original) and B (processed).
    Higher = better quality. Shown as a value (dB) on the node.
    """
    NAME      = "PSNR"
    CATEGORY  = "Metrics"
    N_INPUTS  = 2
    IS_METRIC = True
    PARAMS    = [
        ParamSpec("max_value", "Max pixel value", "float",
                  default=255.0, min_value=1.0, max_value=65535.0, step=1.0),
    ]

    def process(self, image_a: np.ndarray,
                image_b: np.ndarray, max_value=255.0) -> str:
        a = image_a.astype(np.float64)
        b = image_b.astype(np.float64)
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]
        mse = np.mean((a - b) ** 2)
        if mse == 0:
            return "∞ dB"
        psnr = 10 * np.log10(max_value ** 2 / mse)
        return f"{psnr:.2f} dB"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _conv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    def ch(c):
        padded = np.pad(c, ((ph, ph), (pw, pw)), mode="reflect")
        wins = sliding_window_view(padded, (kh, kw))
        return np.tensordot(wins, kernel, axes=([2, 3], [0, 1]))

    if image.ndim == 3:
        return np.stack([ch(image[..., c])
                         for c in range(image.shape[2])], axis=-1)
    return ch(image)
