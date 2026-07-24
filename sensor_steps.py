"""
Sensor-domain steps: Bayer demosaicing and noise injection.

The noise models mirror what a real sensor produces, so a denoiser can
be exercised against each term independently:

    read / constant   fixed sigma, signal-independent  (amplifier noise)
    shot / percentage  sigma proportional to signal     (photon noise)
    salt and pepper    isolated saturated / dead pixels (defects, bit
                       errors) — the case median filters exist for and
                       Gaussian denoisers handle badly
"""
import numpy as np
import cv2

from src.GUI.pipeline_editor.array_utils import to_luminance, to_uint8
from src.GUI.pipeline_editor.base_step import (
    ProcessingStep, ParamSpec, register_step,
)


# ---------------------------------------------------------------------------
# Demosaic
# ---------------------------------------------------------------------------

_BAYER_PATTERNS = {
    # OpenCV names the conversions after the pattern of the SECOND row's
    # second pixel, which is why the constant names look transposed
    # relative to the sensor pattern people quote. Mapped explicitly so
    # the parameter reads the way a datasheet does.
    "RGGB": cv2.COLOR_BayerBG2RGB,
    "BGGR": cv2.COLOR_BayerRG2RGB,
    "GRBG": cv2.COLOR_BayerGB2RGB,
    "GBRG": cv2.COLOR_BayerGR2RGB,
}

_BAYER_PATTERNS_VNG = {
    "RGGB": cv2.COLOR_BayerBG2RGB_VNG,
    "BGGR": cv2.COLOR_BayerRG2RGB_VNG,
    "GRBG": cv2.COLOR_BayerGB2RGB_VNG,
    "GBRG": cv2.COLOR_BayerGR2RGB_VNG,
}

_BAYER_PATTERNS_EA = {
    "RGGB": cv2.COLOR_BayerBG2RGB_EA,
    "BGGR": cv2.COLOR_BayerRG2RGB_EA,
    "GRBG": cv2.COLOR_BayerGB2RGB_EA,
    "GBRG": cv2.COLOR_BayerGR2RGB_EA,
}


@register_step
class Demosaic(ProcessingStep):
    """Reconstruct RGB from a single-channel Bayer mosaic.

    OpenCV's demosaicers work on integer data, so the float [0,1] frame
    is taken to 8 or 16 bit first — 16 bit by default, because
    demosaicing at 8 bit throws away precision the rest of the pipeline
    still wants (and thermal/raw sources are routinely 12-14 bit).

    Quality: 'bilinear' is fast and soft, 'VNG' resolves detail better,
    'EA' (edge-aware) is usually the best compromise on real edges.

    OpenCV implements VNG for 8-bit only, so selecting VNG forces 8-bit
    working depth. That is not silent: the depth actually used is
    attached to the frame as `demosaic_depth`, so it shows up on the node
    and in --log meta rather than quietly costing you precision.
    """
    NAME = "Demosaic"
    CATEGORY = "Preprocessing"
    PARAMS = [
        ParamSpec("pattern", "Bayer Pattern", "choice", default="RGGB",
                  choices=list(_BAYER_PATTERNS)),
        ParamSpec("quality", "Algorithm", "choice", default="EA",
                  choices=["bilinear", "VNG", "EA"]),
        ParamSpec("depth", "Working Depth", "choice", default="16",
                  choices=["8", "16"]),
    ]

    def process(self, image):
        if image.ndim != 2:
            raise ValueError(
                "Demosaic expects a single-channel Bayer mosaic, got "
                f"{image.ndim} channels — put it before any colour step.")
        pattern = self.p.pattern
        if self.p.quality == "VNG":
            code = _BAYER_PATTERNS_VNG[pattern]
        elif self.p.quality == "EA":
            code = _BAYER_PATTERNS_EA[pattern]
        else:
            code = _BAYER_PATTERNS[pattern]

        # VNG is 8-bit only in OpenCV.
        depth = "8" if self.p.quality == "VNG" else self.p.depth
        if depth == "8":
            mosaic = to_uint8(image)
        else:
            a = np.clip(np.asarray(image, np.float32), 0.0, 1.0)
            mosaic = (a * 65535.0 + 0.5).astype(np.uint16)
        self.emit(demosaic_depth=depth, demosaic_pattern=pattern,
                  demosaic_quality=self.p.quality)
        return cv2.cvtColor(np.ascontiguousarray(mosaic), code)


@register_step
class Mosaic(ProcessingStep):
    """Inverse of Demosaic: sample an RGB image down to a Bayer mosaic.

    Useful for building a test bench — mosaic a clean image, add noise,
    demosaic it again, and measure what the round trip cost.
    """
    NAME = "Mosaic (Bayer)"
    CATEGORY = "Preprocessing"
    PARAMS = [ParamSpec("pattern", "Bayer Pattern", "choice", default="RGGB",
                        choices=list(_BAYER_PATTERNS))]

    # channel index per position in the 2x2 tile, row-major
    _TILES = {
        "RGGB": (0, 1, 1, 2), "BGGR": (2, 1, 1, 0),
        "GRBG": (1, 0, 2, 1), "GBRG": (1, 2, 0, 1),
    }

    def process(self, image):
        if image.ndim == 2:
            return image
        a = np.asarray(image, np.float32)
        h, w = a.shape[:2]
        out = np.empty((h, w), np.float32)
        c00, c01, c10, c11 = self._TILES[self.p.pattern]
        out[0::2, 0::2] = a[0::2, 0::2, c00]
        out[0::2, 1::2] = a[0::2, 1::2, c01]
        out[1::2, 0::2] = a[1::2, 0::2, c10]
        out[1::2, 1::2] = a[1::2, 1::2, c11]
        return out


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

@register_step
class AddNoise(ProcessingStep):
    """Add sensor-like noise.

    The two Gaussian terms are independent and can be combined:

      sigma_const     signal-INDEPENDENT, in intensity units. Read /
                      amplifier noise; dominates in the shadows.
      sigma_percent   signal-DEPENDENT, as a percentage of each pixel's
                      own value. Stands in for photon shot noise, which
                      is why bright areas of a real image are noisier in
                      absolute terms but cleaner in SNR.

    Total per pixel: sigma = sigma_const + (sigma_percent/100) * value.

    `clip` is on by default so the result honours the [0,1] contract;
    turn it off to see how much the noise would have driven outside it.
    """
    NAME = "Add Noise"
    CATEGORY = "Preprocessing/Noise"
    PARAMS = [
        ParamSpec("sigma_const", "Gaussian Sigma (constant)", "float",
                  default=0.01, min_value=0.0, max_value=1.0, step=0.005,
                  decimals=4),
        ParamSpec("sigma_percent", "Gaussian Sigma (% of signal)", "float",
                  default=0.0, min_value=0.0, max_value=100.0, step=0.5,
                  decimals=3),
        ParamSpec("salt_pepper", "Salt & Pepper (% of pixels)", "float",
                  default=0.0, min_value=0.0, max_value=50.0, step=0.1,
                  decimals=3),
        ParamSpec("salt_ratio", "Salt Fraction", "float", default=0.5,
                  min_value=0.0, max_value=1.0, step=0.05),
        ParamSpec("per_channel", "Independent per Channel", "bool",
                  default=True),
        ParamSpec("clip", "Clip to [0,1]", "bool", default=True),
        ParamSpec("seed", "Seed (-1 = random)", "int", default=0,
                  min_value=-1, max_value=1000000, step=1),
    ]

    def process(self, image):
        a = np.asarray(image, np.float32).copy()
        # Seeded per frame, so re-running the same frame in live mode
        # reproduces the identical noise instead of shimmering.
        seed = int(self.p.seed)
        rng = (np.random.RandomState() if seed < 0
               else np.random.RandomState((seed * 1000003
                                           + self.ctx.index) % (2 ** 31)))

        noise_shape = a.shape
        if a.ndim == 3 and not self.p.per_channel:
            noise_shape = a.shape[:2]

        sc = float(self.p.sigma_const)
        sp = float(self.p.sigma_percent) / 100.0
        if sc > 0 or sp > 0:
            unit = rng.normal(0.0, 1.0, noise_shape).astype(np.float32)
            if noise_shape != a.shape:
                unit = unit[..., None]
            sigma = sc + sp * np.abs(a)
            a = a + unit * sigma

        sp_pct = float(self.p.salt_pepper) / 100.0
        if sp_pct > 0:
            mask_shape = a.shape[:2] if a.ndim == 3 and not self.p.per_channel \
                else a.shape
            hit = rng.random_sample(mask_shape) < sp_pct
            salt = rng.random_sample(mask_shape) < float(self.p.salt_ratio)
            if mask_shape != a.shape:
                hit = hit[..., None]
                salt = salt[..., None]
                hit = np.broadcast_to(hit, a.shape)
                salt = np.broadcast_to(salt, a.shape)
            a = np.where(hit, np.where(salt, 1.0, 0.0), a)

        return np.clip(a, 0.0, 1.0) if self.p.clip else a
