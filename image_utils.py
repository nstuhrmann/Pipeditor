"""
Shared numpy <-> Qt image conversion helpers.

This is the single home for these conversions — main.py, node_graphics.py
and image_canvas.py previously each carried their own near-identical copy.
"""
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


def arr_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize any dtype to uint8 via min/max scaling (uint8 passes
    through untouched)."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32) - arr.min()
    mx = a.max()
    if mx > 0:
        a = a / mx * 255
    return a.astype(np.uint8)


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert a (H,W), (H,W,3) or (H,W,4) array of any dtype to a QImage.
    Non-uint8 input is min/max-normalized first. Always returns a deep
    copy, so the source array may be freed or mutated afterwards."""
    arr = np.ascontiguousarray(arr_to_uint8(arr))
    if arr.ndim == 2:
        h, w = arr.shape
        qi = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, c = arr.shape
        if c == 3:
            qi = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
        elif c == 4:
            qi = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
        else:
            qi = QImage(np.ascontiguousarray(arr[..., 0]).data, w, h, w,
                        QImage.Format_Grayscale8)
    return qi.copy()


def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(numpy_to_qimage(arr))


def numpy_to_thumbnail(arr: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    """Convert and scale down to fit (max_w, max_h), aspect preserved."""
    return numpy_to_qpixmap(arr).scaled(
        max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
