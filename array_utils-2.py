"""
Shared numpy helpers for step code.

Deliberately Qt-free, unlike image_utils.py: step modules run headless
(CLI, tests, the optimizer), so anything they share must not drag in
PySide6. That coupling is exactly why the float->int scaling below had
been copy-pasted into sequence_steps.py and visual_steps.py instead of
being reused from image_utils — which now imports from here too.
"""
from types import MappingProxyType

import numpy as np


class Frame(np.ndarray):
    """An ndarray carrying per-frame metadata in ``.meta``.

    ONE invariant, which is what keeps this from turning into a pile of
    special cases: **only the executor attaches metadata, and anything
    derived from a Frame carries none.**

    That matters because subclass propagation is both inconsistent and
    useless here. Inconsistent: slicing, arithmetic and np.clip preserve
    the subclass, while cv2 and np.stack silently drop it — so a step
    could never rely on it. Useless: nearly every step calls cv2, and
    the executor re-attaches merged metadata to the output regardless,
    so nothing depends on propagation surviving. Left unchecked it only
    produces surprises — a boolean mask that carries an exposure time, a
    0-d reduction that is a Frame instead of a scalar (and therefore has
    no __round__), an unpickled Frame with no .meta attribute at all.

    Three rules enforce the invariant:

      * ``meta`` defaults to a read-only empty mapping at CLASS level, so
        ``.meta`` can never raise AttributeError however the array came
        into being, and an accidental write fails loudly instead of
        quietly mutating shared state;
      * ``__array_finalize__`` gives derived arrays no metadata at all;
      * ``__array_wrap__`` returns a real scalar for 0-d results, so
        reductions behave exactly as they do on a plain ndarray.

    The result is an object indistinguishable from a plain ndarray in
    every operation, that additionally answers ``.meta`` when the
    executor handed it to you.
    """

    #: Read-only so a stray ``frame.meta["x"] = 1`` on a derived array
    #: raises rather than corrupting the default shared by every Frame.
    meta = MappingProxyType({})

    def __new__(cls, arr, meta=None):
        obj = np.asarray(arr).view(cls)
        if meta:
            obj.meta = dict(meta)
        return obj

    def __array_finalize__(self, obj):
        # Deliberately empty: a view, slice, ufunc result or copy is NOT
        # the frame the executor described, so it inherits no metadata
        # and falls back to the read-only class default.
        pass

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        if out_arr.ndim == 0:
            return out_arr[()]          # scalar, exactly like a plain array
        try:
            return super().__array_wrap__(out_arr, context, return_scalar)
        except TypeError:               # numpy < 2 has no return_scalar
            return super().__array_wrap__(out_arr, context)


# Metadata key prefix under which the executor files metric values, so a
# downstream overlay step can find them all without a nested-dict merge
# rule. Flat keys also merge correctly when two branches join.
METRIC_META_PREFIX = "metric:"


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
