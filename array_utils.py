"""
Shared numpy helpers for step code.

Deliberately Qt-free, unlike image_utils.py: step modules run headless
(CLI, tests, the optimizer), so anything they share must not drag in
PySide6. That coupling is exactly why the float->int scaling below had
been copy-pasted into sequence_steps.py and visual_steps.py instead of
being reused from image_utils — which now imports from here too.
"""
import numpy as np


# Rec. 709 luminance weights (sRGB/HD primaries) — the coefficients that
# had been written out inline in four different steps.
_LUMA_R, _LUMA_G, _LUMA_B = 0.2126, 0.7152, 0.0722


def to_luminance(arr: np.ndarray) -> np.ndarray:
    """Collapse a colour image to single-channel Rec.709 luminance.
    Grayscale input is returned unchanged."""
    if arr.ndim == 2:
        return arr
    return (_LUMA_R * arr[..., 0] + _LUMA_G * arr[..., 1]
            + _LUMA_B * arr[..., 2]).astype(arr.dtype
                                            if arr.dtype.kind == "f"
                                            else np.float32)


def match_channels(a: np.ndarray, b: np.ndarray):
    """Promote whichever of the two is grayscale to the other's channel
    count, so they can be stacked, differenced or concatenated."""
    if a.ndim == b.ndim:
        return a, b
    if a.ndim == 2:
        a = np.repeat(a[..., None], b.shape[2], axis=2)
    else:
        b = np.repeat(b[..., None], a.shape[2], axis=2)
    return a, b


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Deterministic conversion to 8-bit.
    - uint8 passes through
    - uint16 scaled by 1/257 (exact 16->8)
    - float is assumed to satisfy the pipeline contract ([0,1]) and is
      clipped and scaled by 255 — never min/max stretched, so nothing
      silently auto-contrasts."""
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr // 257).astype(np.uint8)
    a = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)


def to_uint16(arr: np.ndarray) -> np.ndarray:
    """Deterministic conversion to 16-bit (see to_uint8)."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        return arr.astype(np.uint16) * 257          # exact 8->16
    a = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return (a * 65535.0 + 0.5).astype(np.uint16)
