"""
Shared numpy <-> Qt image conversion helpers.

This is the single home for these conversions — main.py, node_graphics.py
and image_canvas.py previously each carried their own near-identical copy.
"""
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

from src.GUI.pipeline_editor.array_utils import to_uint8


def arr_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert to uint8 for display/8-bit export.
    - uint8 passes through
    - uint16 is scaled by 1/257 (exact 16->8 bit)
    - floats in [0, 1] (the pipeline's inter-module contract) are scaled
      by 255 directly — NOT min/max stretched, so previews show the data
      as-is instead of silently auto-contrasting every frame
    - anything else falls back to min/max normalization"""
    if arr.dtype in (np.uint8, np.uint16):
        return to_uint8(arr)
    a = arr.astype(np.float32)
    if a.size == 0:
        return a.astype(np.uint8)
    mn, mx = float(a.min()), float(a.max())
    if 0.0 <= mn and mx <= 1.0:
        return to_uint8(a)
    # Out-of-contract data reaching the DISPLAY path only: stretch so
    # something is visible rather than a clipped white/black frame.
    # Step code must use array_utils.to_uint8 instead, which never
    # auto-contrasts.
    span = (mx - mn) or 1.0
    return ((a - mn) / span * 255.0).astype(np.uint8)


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
