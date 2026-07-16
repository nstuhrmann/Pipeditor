"""
Additional processing steps: Durand tonemap, crop/flip, defect pixel
correction, bilateral/NLM denoise, median filter, SSIM, edge enhancement,
retinex, and unsharp mask.

NOTE on bilateral / nonlocal_means_denoise: you said you already have
these implemented elsewhere. I've wired them up as thin wrappers that
import your functions and forward the step's params as kwargs — adjust
the import path and the PARAMS list below to match your actual function
signatures (I guessed at reasonable, commonly-used parameter names).
If steps/__init__.py can't import this module (e.g. because that import
fails), it'll just print a warning to stderr and skip it rather than
crashing the app — so it's safe to leave as a placeholder until you
point the import at the real module.
"""
import numpy as np
import cv2

from src.GUI.pipeline_editor.base_step import ProcessingStep, ParamSpec, register_step

# Durand reuses the shared cv2-Tonemap-operator plumbing already defined
# for Mantiuk/Drago/Reinhard. Adjust this import if tonemapping.py lives
# in a package (e.g. `from algorithms.tonemapping import _OpenCVTonemapStep`).
from src.algorithms.tonemapping import _OpenCVTonemapStep

# --- your existing implementations -----------------------------------
# Adjust these two imports to wherever bilateral()/nonlocal_means_denoise()
# actually live, and check the PARAMS below match their real kwargs.
from my_filters import bilateral as _bilateral_fn
from my_filters import nonlocal_means_denoise as _nlm_fn


# ---------------------------------------------------------------------------
# Durand tonemap
# ---------------------------------------------------------------------------

@register_step
class Durand(_OpenCVTonemapStep):
    NAME     = "Durand"
    CATEGORY = "Tonemapping"
    CV_FACTORY = "createTonemapDurand"
    PARAMS = [
        ParamSpec("gamma", "Gamma", "float", default=1.0,
                  min_value=0.1, max_value=3.0, step=0.05),
        ParamSpec("contrast", "Contrast", "float", default=4.0,
                  min_value=0.1, max_value=10.0, step=0.1),
        ParamSpec("saturation", "Saturation", "float", default=1.0,
                  min_value=0.0, max_value=2.0, step=0.05),
        ParamSpec("sigma_space", "Sigma Space", "float", default=2.0,
                  min_value=0.1, max_value=10.0, step=0.1),
        ParamSpec("sigma_color", "Sigma Color", "float", default=2.0,
                  min_value=0.1, max_value=10.0, step=0.1),
    ]
    # NOTE: cv2.createTonemapDurand ships in opencv-contrib-python and has
    # been dropped from some recent builds entirely — if this raises
    # AttributeError at runtime, that's a packaging issue, not a bug here.


# ---------------------------------------------------------------------------
# Geometry: Crop / Flip
# ---------------------------------------------------------------------------

@register_step
class Crop(ProcessingStep):
    NAME     = "Crop"
    CATEGORY = "Geometry"
    PARAMS = [
        ParamSpec("x", "X", "int", default=0, min_value=0, max_value=10000, step=1),
        ParamSpec("y", "Y", "int", default=0, min_value=0, max_value=10000, step=1),
        ParamSpec("width", "Width", "int", default=100,
                  min_value=1, max_value=10000, step=1),
        ParamSpec("height", "Height", "int", default=100,
                  min_value=1, max_value=10000, step=1),
    ]

    def process(self, image: np.ndarray, x: int = 0, y: int = 0,
                width: int = 100, height: int = 100) -> np.ndarray:
        h, w = image.shape[:2]
        x0 = max(0, min(x, w))
        y0 = max(0, min(y, h))
        x1 = max(x0, min(x + width, w))
        y1 = max(y0, min(y + height, h))
        return image[y0:y1, x0:x1].copy()


@register_step
class Flip(ProcessingStep):
    NAME     = "Flip"
    CATEGORY = "Geometry"
    PARAMS = [
        ParamSpec("horizontal", "Flip Horizontal", "bool", default=False),
        ParamSpec("vertical", "Flip Vertical", "bool", default=False),
    ]

    def process(self, image: np.ndarray, horizontal: bool = False,
                vertical: bool = False) -> np.ndarray:
        out = image
        if horizontal:
            out = np.fliplr(out)
        if vertical:
            out = np.flipud(out)
        return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Preprocessing: Defect Pixel Correction
# ---------------------------------------------------------------------------

@register_step
class DefectPixelCorrection(ProcessingStep):
    """
    Statistical hot/dead pixel correction: flags pixels that deviate from
    their local median by more than `threshold` × the local median
    absolute deviation, and replaces only those with the local median
    (everything else passes through untouched). Distinct from whatever
    your existing BadPixelCorrection step does with a defect map — this
    one needs no calibration file, just a per-frame statistical pass.
    """
    NAME     = "Defect Pixel Correction"
    CATEGORY = "Preprocessing"
    PARAMS = [
        ParamSpec("kernel_size", "Kernel Size", "int", default=3,
                  min_value=3, max_value=9, step=2),
        ParamSpec("threshold", "Threshold (× MAD)", "float", default=6.0,
                  min_value=1.0, max_value=20.0, step=0.5),
    ]

    def process(self, image: np.ndarray, kernel_size: int = 3,
                threshold: float = 6.0) -> np.ndarray:
        k = kernel_size | 1   # force odd

        def correct_channel(ch: np.ndarray) -> np.ndarray:
            dtype = ch.dtype
            work = ch.astype(np.float32)
            med = cv2.medianBlur(
                work if dtype == np.float32 else ch, k
            ).astype(np.float32)
            deviation = np.abs(work - med)
            mad = np.median(deviation) + 1e-6
            bad = deviation > threshold * mad
            out = np.where(bad, med, work)
            return out.astype(dtype)

        if image.ndim == 2:
            return correct_channel(image)
        return np.stack([correct_channel(image[..., c])
                         for c in range(image.shape[2])], axis=-1)


# ---------------------------------------------------------------------------
# Filter: Bilateral / Non-Local Means (your implementations) / Median
# ---------------------------------------------------------------------------

@register_step
class BilateralDenoise(ProcessingStep):
    NAME     = "Bilateral Filter"
    CATEGORY = "Filter"
    PARAMS = [
        ParamSpec("diameter", "Diameter", "int", default=9,
                  min_value=1, max_value=25, step=2),
        ParamSpec("sigma_color", "Sigma Color", "float", default=75.0,
                  min_value=1.0, max_value=200.0, step=1.0),
        ParamSpec("sigma_space", "Sigma Space", "float", default=75.0,
                  min_value=1.0, max_value=200.0, step=1.0),
    ]

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Forwards straight through to your bilateral() — check the
        # kwarg names above line up with its actual signature.
        return _bilateral_fn(image, **kwargs)


@register_step
class NonLocalMeansDenoise(ProcessingStep):
    NAME     = "Non-Local Means Denoise"
    CATEGORY = "Filter"
    PARAMS = [
        ParamSpec("h", "Filter Strength (h)", "float", default=10.0,
                  min_value=1.0, max_value=50.0, step=1.0),
        ParamSpec("template_window_size", "Template Window", "int", default=7,
                  min_value=3, max_value=21, step=2),
        ParamSpec("search_window_size", "Search Window", "int", default=21,
                  min_value=3, max_value=41, step=2),
    ]

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Forwards straight through to your nonlocal_means_denoise() —
        # check the kwarg names above line up with its actual signature.
        return _nlm_fn(image, **kwargs)


@register_step
class MedianFilter(ProcessingStep):
    NAME     = "Median Filter"
    CATEGORY = "Filter"
    PARAMS = [
        ParamSpec("kernel_size", "Kernel Size", "int", default=3,
                  min_value=3, max_value=25, step=2),
    ]

    def process(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        k = kernel_size | 1   # cv2.medianBlur requires an odd ksize
        img = image if image.dtype == np.uint8 else image.astype(np.float32)
        return cv2.medianBlur(img, k)


# ---------------------------------------------------------------------------
# Metrics: SSIM
# ---------------------------------------------------------------------------

@register_step
class SSIM(ProcessingStep):
    NAME      = "SSIM"
    CATEGORY  = "Metrics"
    N_INPUTS  = 2
    IS_METRIC = True
    PARAMS = [
        ParamSpec("win_size", "Window Size", "int", default=7,
                  min_value=3, max_value=25, step=2),
    ]

    def process(self, image_a: np.ndarray, image_b: np.ndarray,
                win_size: int = 7) -> float:
        a, b = image_a, image_b
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]

        try:
            from skimage.metrics import structural_similarity
            channel_axis = -1 if a.ndim == 3 else None
            score = structural_similarity(
                a, b, win_size=win_size | 1,
                channel_axis=channel_axis,
                data_range=float(max(a.max(), b.max()) - min(a.min(), b.min()) or 1))
            return float(score)
        except ImportError:
            return self._ssim_fallback(a, b, win_size | 1)

    @staticmethod
    def _ssim_fallback(a: np.ndarray, b: np.ndarray, win_size: int) -> float:
        """Plain single-scale SSIM (grayscale, global constants) for when
        scikit-image isn't installed. Averages over channels if color.
        Constants scale with the data range (pipeline contract is
        float [0,1], so L is typically 1.0)."""
        L = float(max(a.max(), b.max()) - min(a.min(), b.min())) or 1.0

        def ssim_gray(x, y):
            x = x.astype(np.float64)
            y = y.astype(np.float64)
            C1 = (0.01 * L) ** 2
            C2 = (0.03 * L) ** 2
            mu_x = cv2.GaussianBlur(x, (win_size, win_size), 1.5)
            mu_y = cv2.GaussianBlur(y, (win_size, win_size), 1.5)
            sigma_x = cv2.GaussianBlur(x * x, (win_size, win_size), 1.5) - mu_x ** 2
            sigma_y = cv2.GaussianBlur(y * y, (win_size, win_size), 1.5) - mu_y ** 2
            sigma_xy = cv2.GaussianBlur(x * y, (win_size, win_size), 1.5) - mu_x * mu_y
            num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
            return float(np.mean(num / den))

        if a.ndim == 2:
            return ssim_gray(a, b)
        return float(np.mean([ssim_gray(a[..., c], b[..., c])
                             for c in range(a.shape[2])]))


# ---------------------------------------------------------------------------
# Compositing: HStack / VStack
# ---------------------------------------------------------------------------

def _match_channels(a: np.ndarray, b: np.ndarray):
    """Promote grayscale to 3-channel if the other input is color, so
    the two can be concatenated."""
    if a.ndim == b.ndim:
        return a, b
    if a.ndim == 2:
        a = np.repeat(a[..., None], b.shape[2], axis=2)
    else:
        b = np.repeat(b[..., None], a.shape[2], axis=2)
    return a, b


def _pad_to(arr: np.ndarray, size: int, axis: int,
            align: str, pad_value: float) -> np.ndarray:
    """Pad `arr` along `axis` up to `size` with pad_value, placing the
    content per `align` (start/center/end)."""
    deficit = size - arr.shape[axis]
    if deficit <= 0:
        return arr
    if align == "start":
        before, after = 0, deficit
    elif align == "end":
        before, after = deficit, 0
    else:   # center
        before = deficit // 2
        after = deficit - before
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (before, after)
    return np.pad(arr, pad, mode="constant", constant_values=pad_value)


class _StackStep(ProcessingStep):
    """Shared implementation for HStack/VStack: concatenates inputs A
    and B along STACK_AXIS, padding the smaller one along the other
    axis. Inputs arrive as float32 [0,1] per the pipeline contract."""
    N_INPUTS = 2
    CATEGORY = "Compositing"
    STACK_AXIS = 1   # 1 = horizontal (side by side), 0 = vertical
    PARAMS = [
        ParamSpec("pad_value", "Padding Value", "float", default=0.0,
                  min_value=0.0, max_value=1.0, step=0.05),
        ParamSpec("align", "Alignment", "choice", default="center",
                  choices=["start", "center", "end"]),
        ParamSpec("gap", "Gap (px)", "int", default=0,
                  min_value=0, max_value=200, step=1),
    ]

    def process(self, image_a: np.ndarray, image_b: np.ndarray,
                pad_value: float = 0.0, align: str = "center",
                gap: int = 0) -> np.ndarray:
        a, b = _match_channels(image_a, image_b)
        pad_axis = 1 - self.STACK_AXIS
        size = max(a.shape[pad_axis], b.shape[pad_axis])
        a = _pad_to(a, size, pad_axis, align, pad_value)
        b = _pad_to(b, size, pad_axis, align, pad_value)
        parts = [a]
        if gap > 0:
            gap_shape = list(a.shape)
            gap_shape[self.STACK_AXIS] = gap
            parts.append(np.full(gap_shape, pad_value, dtype=a.dtype))
        parts.append(b)
        return np.concatenate(parts, axis=self.STACK_AXIS)


@register_step
class HStack(_StackStep):
    """A left of B, heights equalized by padding."""
    NAME = "HStack"
    STACK_AXIS = 1


@register_step
class VStack(_StackStep):
    """A above B, widths equalized by padding."""
    NAME = "VStack"
    STACK_AXIS = 0


# ---------------------------------------------------------------------------
# Tonemapping / enhancement: Edge Enhancement, Retinex, Unsharp Mask
# ---------------------------------------------------------------------------

@register_step
class EdgeEnhancement(ProcessingStep):
    """Laplacian-based edge sharpening: adds back a scaled high-frequency
    (Laplacian) component. Distinct from Unsharp Mask, which uses a
    Gaussian-blur-based high-pass instead."""
    NAME     = "Edge Enhancement"
    CATEGORY = "Tonemapping"
    PARAMS = [
        ParamSpec("strength", "Strength", "float", default=1.0,
                  min_value=0.0, max_value=5.0, step=0.1),
        ParamSpec("kernel_size", "Kernel Size", "int", default=3,
                  min_value=1, max_value=7, step=2),
    ]

    def process(self, image: np.ndarray, strength: float = 1.0,
                kernel_size: int = 3) -> np.ndarray:
        img = image.astype(np.float32)
        lap = cv2.Laplacian(img, cv2.CV_32F, ksize=kernel_size)
        out = img - strength * lap
        return np.clip(out, 0, 255).astype(np.uint8)


@register_step
class UnsharpMask(ProcessingStep):
    """Classic unsharp masking: blur → subtract from original → add back
    scaled by `amount`, only where the difference exceeds `threshold`."""
    NAME     = "Unsharp Mask"
    CATEGORY = "Tonemapping"
    PARAMS = [
        ParamSpec("radius", "Radius (sigma)", "float", default=2.0,
                  min_value=0.1, max_value=20.0, step=0.1),
        ParamSpec("amount", "Amount", "float", default=1.0,
                  min_value=0.0, max_value=5.0, step=0.1),
        ParamSpec("threshold", "Threshold", "float", default=0.0,
                  min_value=0.0, max_value=50.0, step=1.0),
    ]

    def process(self, image: np.ndarray, radius: float = 2.0,
                amount: float = 1.0, threshold: float = 0.0) -> np.ndarray:
        img = image.astype(np.float32)
        blurred = cv2.GaussianBlur(img, (0, 0), radius)
        diff = img - blurred
        if threshold > 0:
            mask = np.abs(diff) >= threshold
            diff = diff * mask
        out = img + amount * diff
        return np.clip(out, 0, 255).astype(np.uint8)


@register_step
class Retinex(ProcessingStep):
    """
    Multi-Scale Retinex (MSR): averages log-domain reflectance estimates
    from several Gaussian surround scales, then rescales to 0-255.
    Operates per-channel (the standard MSR formulation), so color images
    keep their channels independent rather than being routed through
    luminance only.
    """
    NAME     = "Retinex"
    CATEGORY = "Tonemapping"
    PARAMS = [
        ParamSpec("sigma1", "Sigma (scale 1)", "float", default=15.0,
                  min_value=1.0, max_value=100.0, step=1.0),
        ParamSpec("sigma2", "Sigma (scale 2)", "float", default=80.0,
                  min_value=1.0, max_value=200.0, step=1.0),
        ParamSpec("sigma3", "Sigma (scale 3)", "float", default=250.0,
                  min_value=1.0, max_value=400.0, step=1.0),
        ParamSpec("gain", "Gain", "float", default=1.0,
                  min_value=0.1, max_value=5.0, step=0.1),
    ]

    def process(self, image: np.ndarray, sigma1: float = 15.0,
                sigma2: float = 80.0, sigma3: float = 250.0,
                gain: float = 1.0) -> np.ndarray:
        img = image.astype(np.float32) + 1.0   # avoid log(0)

        def msr_channel(ch: np.ndarray) -> np.ndarray:
            log_img = np.log(ch)
            acc = np.zeros_like(ch)
            for sigma in (sigma1, sigma2, sigma3):
                blurred = cv2.GaussianBlur(ch, (0, 0), sigma) + 1e-6
                acc += log_img - np.log(blurred)
            return acc / 3.0

        if img.ndim == 2:
            result = msr_channel(img)
        else:
            result = np.stack(
                [msr_channel(img[..., c]) for c in range(img.shape[2])],
                axis=-1)

        result *= gain
        # Rescale each channel to 0-255 independently (standard MSR output stage)
        out = np.empty_like(result)
        channels = [result] if result.ndim == 2 else \
                   [result[..., c] for c in range(result.shape[2])]
        normed = []
        for ch in channels:
            lo, hi = ch.min(), ch.max()
            span = (hi - lo) or 1.0
            normed.append((ch - lo) / span * 255.0)
        out = normed[0] if result.ndim == 2 else np.stack(normed, axis=-1)
        return np.clip(out, 0, 255).astype(np.uint8)
