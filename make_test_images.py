"""
Generate 14/16-bit test images for tonemapping work.

    python make_test_images.py --out testimg --bits 14 --align right
    python make_test_images.py --out testimg --bits 14 --align left
    python make_test_images.py --out testimg --bits 16

Everything is generated in float and quantized ONCE, at the end, so the
patterns are identical across bit depths and only the quantization
differs — which is what makes a 14-bit and a 16-bit render of the same
pattern directly comparable.

Bit alignment
-------------
A 14-bit sensor value living in a 16-bit container can sit either way:

  right-aligned  code = value            range 0 .. 16383, LSBs used
  left-aligned   code = value << 2       range 0 .. 65532, low 2 bits zero
                 (a.k.a. MSB-aligned)

It matters because the naive `img / 65535.0` is correct only for
left-aligned data; on right-aligned 14-bit it produces an image 4x too
dark, which is one of the most common ways a thermal pipeline silently
misbehaves. Each image is written with a JSON sidecar recording bits,
alignment and the divisor to use, plus the ground truth for whichever
pattern it is — so a test can assert against numbers instead of eyeballs.

The patterns are LINEAR in scene radiance, like raw sensor data, not
display-encoded. Feed them to a tonemapper as-is.
"""
import argparse
import json
import os

import numpy as np
import cv2

# --------------------------------------------------------------------------
# Patterns. Each returns (float image in [0,1], ground-truth dict).
# --------------------------------------------------------------------------


def linear_ramp(h, w):
    """Full-scale horizontal ramp. The reference for banding: after
    tonemapping, any flat step wider than one input code is quantization
    the operator introduced, not something the source had."""
    img = np.tile(np.linspace(0.0, 1.0, w), (h, 1))
    return img, {"description": "linear 0..1 left-to-right",
                 "monotonic_axis": "x"}


def log_wedge(h, w, n=12, mod=0.004, mode="absolute"):
    """n log-spaced patches, each carrying the SAME faint modulation.

    The most informative pattern here: the patch means give you the
    transfer curve, and whether the modulation survives in each patch
    tells you where the operator spends its output range. A global
    operator keeps it in two or three zones; a good local one keeps it
    in most.

    mode="absolute"  every patch modulated by +-mod of FULL SCALE. The
                     thermal case: a fixed small delta-T regardless of
                     the pedestal it sits on.
    mode="relative"  modulated by +-mod of the PATCH LEVEL, i.e. constant
                     Michelson contrast. The photographic case.

    The ladder is fitted INSIDE the range the modulation leaves free, so
    no patch clips — a clipped patch would silently report a different
    mean and amplitude than the sidecar claims, which is exactly the kind
    of quiet wrongness a test image must not have. With the defaults that
    is about 7 stops; lower `mod` for more."""
    img = np.zeros((h, w), np.float64)
    if mode == "absolute":
        lo, hi = 2.0 * mod, 1.0 - mod
    else:
        lo, hi = 2.0 * mod, 1.0 / (1.0 + mod)
    if hi <= lo:
        raise ValueError("modulation too large for a usable ladder")
    levels, rects = [], []
    pw = w // n
    yy = np.arange(h)[:, None]
    stripe_h = max(2, h // 40)
    for i in range(n):
        # geometric ladder between lo and hi
        level = float(lo * (hi / lo) ** (i / max(1, n - 1)))
        amp = mod if mode == "absolute" else mod * level
        x0, x1 = i * pw, (i + 1) * pw if i < n - 1 else w
        # square modulation, so the amplitude is exact rather than
        # averaged over a sinusoid
        stripes = np.where((yy // stripe_h) % 2 == 0, amp, -amp)
        img[:, x0:x1] = level + stripes
        levels.append(level)
        rects.append([x0, 0, x1 - x0, h])
    span = float(np.log2(levels[-1] / levels[0]))
    return img, {"description": f"{n} log-spaced patches, {span:.1f} stops",
                 "mode": mode,
                 "patch_levels": levels, "patch_rects": rects,
                 "modulation_amplitude": mod,
                 "modulation_is_relative": mode == "relative",
                 "span_stops": span,
                 "stripe_height_px": stripe_h}


def hdr_split(h, w, dark=0.004, bright=0.9, texture=0.15):
    """Dark half and bright half, each carrying identical RELATIVE
    texture. The classic tonemapping failure: a global curve that opens
    up the shadows blows the highlights, and vice versa. Score it by
    comparing recovered texture contrast in the two halves — a good
    operator gets them close."""
    img = np.zeros((h, w), np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    tex = np.sin(xx / 7.0) * np.sin(yy / 9.0)          # +-1
    left = dark * (1.0 + texture * tex[:, : w // 2])
    right = bright * (1.0 + texture * tex[:, w // 2:])
    img[:, : w // 2] = left
    img[:, w // 2:] = right
    return np.clip(img, 0.0, 1.0), {
        "description": "dark|bright split, identical relative texture",
        "dark_level": dark, "bright_level": bright,
        "relative_texture": texture,
        "stops_between": float(np.log2(bright / dark))}


def low_contrast_pedestals(h, w, pedestals=(0.02, 0.1, 0.35, 0.75),
                           amplitude=0.002):
    """Small absolute modulation on several DC levels — the thermal case:
    a fraction-of-a-kelvin signal riding on a large offset. Tests whether
    an operator's local contrast gain depends on the pedestal (most do)."""
    img = np.zeros((h, w), np.float64)
    band = h // len(pedestals)
    xx = np.arange(w)
    for i, dc in enumerate(pedestals):
        y0 = i * band
        y1 = (i + 1) * band if i < len(pedestals) - 1 else h
        img[y0:y1, :] = dc + amplitude * np.sin(2 * np.pi * xx / 64.0)
    return np.clip(img, 0.0, 1.0), {
        "description": "constant-amplitude sine on several pedestals",
        "pedestals": list(pedestals), "amplitude": amplitude,
        "period_px": 64}


def edge_halo(h, w, low=0.05, high=0.8):
    """A single hard vertical edge on flat fields. Local operators
    (Retinex, CLAHE, unsharp) overshoot here; measure the over/undershoot
    in the flat regions adjacent to the edge."""
    img = np.full((h, w), low, np.float64)
    img[:, w // 2:] = high
    return img, {"description": "hard vertical edge, flat either side",
                 "low": low, "high": high, "edge_x": w // 2}


def zone_plate(h, w, level=0.5, contrast=0.4):
    """Radial frequency sweep — spatial frequency rises with radius, so a
    single image shows an operator's response across the whole frequency
    range, and any aliasing or frequency-dependent gain shows as rings."""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / float(min(h, w))
    img = level * (1.0 + contrast * np.sin(r2 * 0.6))
    return np.clip(img, 0.0, 1.0), {
        "description": "zone plate, frequency rises with radius",
        "level": level, "contrast": contrast}


def hot_pixels(h, w, level=0.35, n=12, seed=0):
    """Flat mid-grey field with a few stuck-high and stuck-low pixels.

    Any operator that normalizes by min/max is destroyed by this: one
    stuck pixel sets the range and the whole image collapses to mid-grey.
    Percentile-based stretching survives it. A one-line test that
    separates the two."""
    rng = np.random.RandomState(seed)
    img = np.full((h, w), level, np.float64)
    ys = rng.randint(0, h, n)
    xs = rng.randint(0, w, n)
    coords = []
    for i, (y, x) in enumerate(zip(ys, xs)):
        val = 1.0 if i % 2 == 0 else 0.0
        img[y, x] = val
        coords.append([int(x), int(y), val])
    return img, {"description": "flat field with stuck-at pixels",
                 "level": level, "defects_xy_value": coords}


def noise_bands(h, w, levels=(0.05, 0.25, 0.6), sigma=0.004, seed=0):
    """Flat bands at several levels with a KNOWN noise sigma.

    Contrast stretching amplifies noise by exactly its local slope, so
    measuring sigma per band before and after gives you the operator's
    noise gain as a function of input level — the number that decides
    whether a pretty tonemapper is actually usable."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w), np.float64)
    band = h // len(levels)
    for i, lvl in enumerate(levels):
        y0 = i * band
        y1 = (i + 1) * band if i < len(levels) - 1 else h
        img[y0:y1, :] = lvl + rng.normal(0.0, sigma, (y1 - y0, w))
    return np.clip(img, 0.0, 1.0), {
        "description": "flat bands with known Gaussian sigma",
        "levels": list(levels), "sigma": sigma,
        "band_height": band}


def thermal_scene(h, w, seed=0):
    """A synthetic scene with the shape of a real thermal frame: a warm
    gradient background, a few objects a little above it, one small very
    hot source, and mild noise. Not physically derived — it exists so
    the numeric patterns above have a 'looks like a picture' companion
    to sanity-check against."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = 0.18 + 0.10 * (yy / h) + 0.03 * np.sin(xx / 40.0)
    objects = []
    for (cy, cx, r, lvl) in ((0.35, 0.25, 0.10, 0.42),
                             (0.65, 0.55, 0.14, 0.55),
                             (0.30, 0.75, 0.07, 0.33)):
        m = ((yy - cy * h) ** 2 + (xx - cx * w) ** 2) < (r * min(h, w)) ** 2
        img[m] = lvl
        objects.append({"cx": cx, "cy": cy, "r": r, "level": lvl})
    hot = ((yy - 0.15 * h) ** 2 + (xx - 0.88 * w) ** 2) < (0.02 * min(h, w)) ** 2
    img[hot] = 0.995
    img += rng.normal(0.0, 0.0015, img.shape)
    return np.clip(img, 0.0, 1.0), {
        "description": "synthetic thermal-like scene",
        "objects": objects, "hot_spot_level": 0.995,
        "background_range": [0.18, 0.28]}


PATTERNS = {
    "linear_ramp": linear_ramp,
    "log_wedge": log_wedge,
    "hdr_split": hdr_split,
    "low_contrast_pedestals": low_contrast_pedestals,
    "edge_halo": edge_halo,
    "zone_plate": zone_plate,
    "hot_pixels": hot_pixels,
    "noise_bands": noise_bands,
    "thermal_scene": thermal_scene,
}


# --------------------------------------------------------------------------
# Quantization / packing
# --------------------------------------------------------------------------

def quantize(img: np.ndarray, bits: int, align: str) -> np.ndarray:
    """float [0,1] -> uint16 codes at `bits` precision, packed per `align`.

    Right-aligned keeps the code in 0..2**bits-1. Left-aligned shifts it
    up so the MSB of the sample is the MSB of the container, leaving
    (16 - bits) zero LSBs — the layout most sensor interfaces produce.
    """
    if not 1 <= bits <= 16:
        raise ValueError("bits must be 1..16")
    max_code = (1 << bits) - 1
    codes = np.clip(np.rint(np.clip(img, 0.0, 1.0) * max_code),
                    0, max_code).astype(np.uint16)
    if align == "left":
        return (codes << (16 - bits)).astype(np.uint16)
    if align == "right":
        return codes
    raise ValueError("align must be 'left' or 'right'")


def sidecar(name: str, bits: int, align: str, truth: dict,
            shape) -> dict:
    max_code = (1 << bits) - 1
    full_scale = max_code << (16 - bits) if align == "left" else max_code
    return {
        "pattern": name,
        "height": int(shape[0]), "width": int(shape[1]),
        "bits": bits,
        "alignment": align,
        # Divide the stored uint16 by this to recover [0,1]. Getting this
        # wrong is the bug the alignment distinction exists to expose.
        "full_scale_code": int(full_scale),
        "to_unit_divisor": int(full_scale),
        "lsb_shift": 16 - bits if align == "left" else 0,
        "linear_light": True,
        "ground_truth": truth,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate 14/16-bit tonemapping test images.")
    ap.add_argument("--out", default="testimg", help="output directory")
    ap.add_argument("--bits", type=int, default=14, help="sample depth")
    ap.add_argument("--align", choices=["left", "right"], default="right")
    ap.add_argument("--size", default="512x512", metavar="WxH")
    ap.add_argument("--format", choices=["png", "tiff"], default="png")
    ap.add_argument("--only", default="", metavar="NAMES",
                    help="comma-separated subset of patterns")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    os.makedirs(args.out, exist_ok=True)
    wanted = ([n.strip() for n in args.only.split(",") if n.strip()]
              or list(PATTERNS))

    written = []
    for name in wanted:
        if name not in PATTERNS:
            raise SystemExit(f"unknown pattern '{name}'; "
                             f"choose from {', '.join(PATTERNS)}")
        img, truth = PATTERNS[name](h, w)
        codes = quantize(img, args.bits, args.align)
        stem = f"{name}_{args.bits}bit_{args.align}"
        path = os.path.join(args.out, f"{stem}.{args.format}")
        if not cv2.imwrite(path, codes):
            raise SystemExit(f"could not write {path}")
        meta_path = os.path.join(args.out, f"{stem}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(sidecar(name, args.bits, args.align, truth,
                              codes.shape), f, indent=2)
        written.append((stem, int(codes.min()), int(codes.max())))

    print(f"{len(written)} image(s) -> {args.out}  "
          f"({args.bits}-bit, {args.align}-aligned)")
    for stem, lo, hi in written:
        print(f"  {stem:44s} codes {lo:6d} .. {hi:6d}")


if __name__ == "__main__":
    main()
