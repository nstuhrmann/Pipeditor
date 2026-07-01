"""
I/O steps: source nodes (IS_SOURCE=True) and sink nodes (IS_SINK=True).

Source nodes have no input port and load their own image.
Sink nodes receive an image, write it somewhere, and pass it through
unchanged — so they can be placed anywhere in a pipeline, not just at the end.
"""
import os
import numpy as np
from datetime import datetime
from base_step import ProcessingStep, ParamSpec, register_step


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

@register_step
class FileInputStep(ProcessingStep):
    NAME = "Image Source"
    CATEGORY = "Input / Output"
    IS_SOURCE = True
    PARAMS = [
        ParamSpec("path", "File", "file", default=""),
    ]

    def process(self, image: np.ndarray, path: str = "", **kwargs) -> np.ndarray:
        if not path:
            raise ValueError(
                "Image Source: no file selected.\n"
                "Double-click the node and choose a file."
            )
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            return np.array(img)
        except ImportError:
            from PySide6.QtGui import QImage
            qimg = QImage(path).convertToFormat(QImage.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            arr = np.array(ptr).reshape(h, qimg.bytesPerLine())[:, :w * 3].reshape(h, w, 3)
            return arr.copy()


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

@register_step
class DirectoryWriterStep(ProcessingStep):
    """
    Writes each image it receives to a directory.
    Filename: {prefix}_{YYYYMMDD_HHMMSS_ffffff}.{format}
    The image is passed through unchanged so the node can sit anywhere
    in the pipeline (not just at the very end).
    """
    NAME = "Directory Writer"
    CATEGORY = "Input / Output"
    IS_SINK = True
    PARAMS = [
        ParamSpec("output_dir", "Output Directory", "directory", default=""),
        ParamSpec("prefix",     "Filename Prefix",  "str",       default="output"),
        ParamSpec("format",     "Format",           "choice",    default="PNG",
                  choices=["PNG", "JPEG", "TIFF"]),
        ParamSpec("quality",    "JPEG Quality",     "int",       default=95,
                  min_value=1, max_value=100, step=1),
    ]

    def process(self, image: np.ndarray, output_dir: str = "",
                prefix: str = "output", format: str = "PNG",
                quality: int = 95, **kwargs) -> np.ndarray:
        if not output_dir:
            raise ValueError(
                "Directory Writer: no output directory set.\n"
                "Double-click the node and choose a directory."
            )
        os.makedirs(output_dir, exist_ok=True)

        ext = {"PNG": "png", "JPEG": "jpg", "TIFF": "tiff"}.get(format, "png")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.{ext}"
        filepath = os.path.join(output_dir, filename)

        # normalise to uint8
        arr = image
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            a -= a.min()
            if a.max() > 0:
                a = a / a.max() * 255
            arr = a.astype(np.uint8)

        try:
            from PIL import Image as PILImage
            img = PILImage.fromarray(arr)
            save_kwargs = {"quality": quality} if format == "JPEG" else {}
            img.save(filepath, **save_kwargs)
        except ImportError:
            # Fallback: use QImage (PNG only)
            from PySide6.QtGui import QImage
            h, w = arr.shape[:2]
            if arr.ndim == 2:
                qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            else:
                qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
            qimg.copy().save(filepath)

        return image   # passthrough
