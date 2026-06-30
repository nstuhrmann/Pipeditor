"""
Beispiel-Implementierungen von ProcessingStep.

Dies ist bewusst eine kleine Startauswahl. Neue Algorithmen werden
genauso angelegt: von ProcessingStep ableiten, NAME/CATEGORY/PARAMS
setzen, process() implementieren, mit @register_step versehen.

Hier wird absichtlich nur mit numpy gearbeitet (keine Hard-Dependency
auf OpenCV o.ä.), damit die Basis-Klasse unabhängig von einer
bestimmten Bildverarbeitungs-Bibliothek bleibt. Wenn du lieber OpenCV/
scikit-image benutzen willst, einfach in der jeweiligen process()-
Methode importieren und verwenden - die Architektur ist davon unabhängig.
"""
import numpy as np
from base_step import ProcessingStep, ParamSpec, register_step


@register_step
class GaussianBlur(ProcessingStep):
    NAME = "Gaussian Blur"
    CATEGORY = "Filter"
    PARAMS = [
        ParamSpec("sigma", "Sigma", "float", default=2.0, min_value=0.1, max_value=20.0, step=0.1),
    ]

    def process(self, image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        try:
            from scipy.ndimage import gaussian_filter
            if image.ndim == 3:
                return np.stack(
                    [gaussian_filter(image[..., c], sigma=sigma) for c in range(image.shape[2])],
                    axis=-1,
                )
            return gaussian_filter(image, sigma=sigma)
        except ImportError:
            # einfacher Fallback ohne scipy: Box-Blur via gleitendem Mittel
            k = max(1, int(sigma))
            kernel = np.ones((2 * k + 1, 2 * k + 1)) / ((2 * k + 1) ** 2)
            return _convolve2d_same(image, kernel)


@register_step
class Threshold(ProcessingStep):
    NAME = "Threshold"
    CATEGORY = "Segmentierung"
    PARAMS = [
        ParamSpec("value", "Schwellwert", "int", default=128, min_value=0, max_value=255, step=1),
        ParamSpec("invert", "Invertieren", "bool", default=False),
    ]

    def process(self, image: np.ndarray, value: int = 128, invert: bool = False) -> np.ndarray:
        gray = image if image.ndim == 2 else image.mean(axis=2)
        mask = gray > value
        if invert:
            mask = ~mask
        return (mask.astype(np.uint8) * 255)


@register_step
class Invert(ProcessingStep):
    NAME = "Invertieren"
    CATEGORY = "Punktoperationen"
    PARAMS = []

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        max_val = 255 if image.dtype == np.uint8 else float(image.max() or 1)
        return max_val - image


@register_step
class Brightness(ProcessingStep):
    NAME = "Helligkeit/Kontrast"
    CATEGORY = "Punktoperationen"
    PARAMS = [
        ParamSpec("brightness", "Helligkeit", "int", default=0, min_value=-255, max_value=255, step=1),
        ParamSpec("contrast", "Kontrast", "float", default=1.0, min_value=0.1, max_value=3.0, step=0.05),
    ]

    def process(self, image: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
        out = image.astype(np.float32) * contrast + brightness
        return np.clip(out, 0, 255).astype(np.uint8)


def _convolve2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Sehr einfache Fallback-Faltung (nur falls scipy fehlt)."""
    from numpy.lib.stride_tricks import sliding_window_view
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    def conv_channel(ch):
        padded = np.pad(ch, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        windows = sliding_window_view(padded, (kh, kw))
        return np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))

    if image.ndim == 3:
        return np.stack([conv_channel(image[..., c]) for c in range(image.shape[2])], axis=-1)
    return conv_channel(image)
