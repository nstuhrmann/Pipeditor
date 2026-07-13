#!/usr/bin/env python3
"""
Non-Uniformity Correction (NUC) analysis for thermal imagers.

Given a folder of blackbody frames (temperature and integration time encoded
in the filename, e.g. `mean_10000us_10deg_N128.pgm`), this tool answers:

  1. Residual error of two-point correction over the full temperature range,
     and which pair of calibration temperatures minimizes it.
  2. Residual error of multi-point correction:
       a) per-pixel piecewise-linear with optimal placement of the
          calibration points (exhaustive search where feasible, greedy
          otherwise), as a function of the number of points k;
       b) per-pixel polynomial over ALL calibration temperatures, as a
          function of the polynomial order d = 1 .. n-2. Because fitting and
          evaluating on the same temperatures is in-sample (the highest
          order would trivially be exact everywhere), the polynomial
          residual at each interior temperature is computed LEAVE-ONE-OUT:
          predicted by a fit that excludes that temperature. Endpoints are
          evaluated in-sample (leaving them out would mean extrapolation).
  3. Residual noise expressed as NETD (spatial / FPN NETD), plus temporal
     NETD if repeated frames per condition exist, via the measured
     responsivity dS/dT.

Extended analysis:
  - FPN decomposition of the residual into column stripes, row stripes and
    pixel-random components (in counts and mK).
  - Bad-pixel detection (dead / low-response / high-noise pixels), masked
    from all statistics so they don't dominate the residual.
  - Response linearity (why two-point fails: signal vs. blackbody
    temperature is convex due to Planck radiance + detector nonlinearity).
  - Everything is done per integration time; NUC tables are only valid for
    the tint they were measured at.

Usage:
    python nuc_analysis.py /path/to/blackbody/folder --out ./nuc_results
    python nuc_analysis.py /path/to/folder --tint 10000        # only this tint
    python nuc_analysis.py --synthetic --out ./nuc_results     # self-test

The filename regex (--pattern) must contain named groups `tint` (integration
time) and `temp` (blackbody temperature, deg C); an optional group `navg`
captures the number of averaged frames (e.g. N128). The default matches
names like `mean_10000us_10deg_N128.pgm`. All files sharing (tint, temp) are
treated as repeats and averaged; their frame-to-frame variation gives the
temporal noise estimate (of the stored frames — if those are already
N-frame averages, it is the noise of the averages).

Supported formats: .pgm (8/16-bit), .tif/.tiff, .png, .npy, and headerless
.raw/.bin (specify --raw-shape H W and --raw-dtype, e.g. uint16).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("nuc")


def _progress(i: int, total: int, label: str, every: int | None = None):
    """INFO log every ~10% of a potentially long loop (and at the end)."""
    every = every or max(1, total // 10)
    if (i + 1) % every == 0 or (i + 1) == total:
        log.info("%s: %d/%d", label, i + 1, total)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

DEFAULT_PATTERN = (r".*?(?P<tint>\d+(?:[\.,]\d+)?)\s*us"
                   r".*?(?P<temp>-?\d+(?:[\.,]\d+)?)\s*deg"
                   r"(?:.*?N(?P<navg>\d+))?")
TINT_UNIT = "us"     # label only; grouping/filtering is by the parsed number


def load_frame(path: Path, raw_shape=None, raw_dtype="uint16") -> np.ndarray:
    suf = path.suffix.lower()
    if suf in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path).astype(np.float64)
    if suf == ".npy":
        return np.load(path).astype(np.float64)
    if suf in (".raw", ".bin"):
        if raw_shape is None:
            raise ValueError(f"{path}: raw file needs --raw-shape H W")
        data = np.fromfile(path, dtype=np.dtype(raw_dtype))
        return data.reshape(raw_shape).astype(np.float64)
    # pgm / png / jpg / everything else
    import imageio.v3 as iio
    img = iio.imread(path)
    if img.ndim == 3:  # accidental RGB; take first channel
        img = img[..., 0]
    return img.astype(np.float64)


@dataclass
class Dataset:
    """All frames for one integration time."""
    tint: float                  # in file units (see TINT_UNIT)
    temps: np.ndarray            # sorted unique blackbody temps
    mean_frames: np.ndarray      # (T, H, W) N-weighted frame average per temp
    temporal_std_single: np.ndarray  # (T,) single-frame-equivalent noise (NaN if 1 file)
    temporal_std_mean: np.ndarray    # (T,) noise of the weighted mean frame
    n_frames: dict = field(default_factory=dict)   # temp -> #files
    navg: dict = field(default_factory=dict)       # temp -> sorted set of N from filenames
    bad_mask: np.ndarray = None  # (H, W) bool, True = bad pixel


def scan_folder(folder: Path, pattern: str, raw_shape=None, raw_dtype="uint16",
                only_tint: float | None = None, verbose=True) -> dict[float, Dataset]:
    rx = re.compile(pattern)
    # (tint, temp) -> list of (path, N) where N is the per-file average count
    groups: dict[tuple[float, float], list[tuple[Path, int]]] = {}
    skipped = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        m = rx.search(p.name)
        if not m:
            skipped.append(p.name)
            continue
        temp = float(m.group("temp").replace(",", "."))
        tint = float(m.group("tint").replace(",", "."))
        navg = int(m.group("navg")) if ("navg" in m.groupdict()
                                        and m.group("navg")) else 1
        groups.setdefault((tint, temp), []).append((p, navg))

    log.info("Matched %d files (%d (tint,temp) conditions), skipped %d.",
             sum(len(v) for v in groups.values()), len(groups), len(skipped))
    if skipped[:5]:
        log.info("  e.g. skipped: %s", ", ".join(skipped[:5]))
    if not groups:
        raise SystemExit("No files matched the filename pattern. "
                         "Adjust --pattern (needs named groups 'tint' and 'temp').")

    if only_tint is not None:
        avail = sorted({k[0] for k in groups})
        groups = {k: v for k, v in groups.items()
                  if np.isclose(k[0], only_tint)}
        if not groups:
            raise SystemExit(
                f"--tint {only_tint:g} {TINT_UNIT} not found; available: "
                + ", ".join(f"{t:g}" for t in avail))

    datasets: dict[float, Dataset] = {}
    for tint in sorted({k[0] for k in groups}):
        temps = sorted({k[1] for k in groups if k[0] == tint})
        means, tstd1, tstdm, nfr, nav = [], [], [], {}, {}
        for ti, T in enumerate(temps):
            entries = groups[(tint, T)]
            frames = np.stack([load_frame(p, raw_shape, raw_dtype)
                               for p, _ in entries])
            Ns = np.array([nv for _, nv in entries], dtype=float)
            if len(set(Ns)) > 1:
                log.warning("tint=%g %s, T=%g degC: mixed N per file %s — "
                            "using N-weighted averaging.",
                            tint, TINT_UNIT, T, sorted(set(int(x) for x in Ns)))
            m = len(frames)
            w = Ns / Ns.sum()
            mu = np.tensordot(w, frames, axes=1)             # weighted mean
            means.append(mu)
            if m > 1:
                # Var(frame_i) = sigma1^2 / N_i  =>  single-frame-equivalent:
                # sigma1^2 = sum_i N_i (f_i - mu)^2 / (m - 1)   (per pixel)
                s1 = np.sqrt((Ns[:, None, None]
                              * (frames - mu) ** 2).sum(axis=0) / (m - 1))
                tstd1.append(float(s1.mean()))
                tstdm.append(float(s1.mean()) / np.sqrt(Ns.sum()))
            else:
                tstd1.append(np.nan)
                tstdm.append(np.nan)
            nfr[T] = m
            nav[T] = sorted(set(int(x) for x in Ns))
            _progress(ti, len(temps), f"loading tint={tint:g} {TINT_UNIT}")
        ds = Dataset(tint=tint,
                     temps=np.array(temps, dtype=float),
                     mean_frames=np.stack(means),
                     temporal_std_single=np.array(tstd1),
                     temporal_std_mean=np.array(tstdm),
                     n_frames=nfr, navg=nav)
        ds.bad_mask = detect_bad_pixels(ds)
        datasets[tint] = ds
        H, W = ds.mean_frames.shape[1:]
        allN = sorted({x for v in nav.values() for x in v})
        log.info("tint=%g %s: %d temps (%g..%g degC), frame %dx%d, "
                 "bad pixels: %d (%.3f %%), N_avg per file: %s",
                 tint, TINT_UNIT, len(temps), temps[0], temps[-1], W, H,
                 ds.bad_mask.sum(), 100 * ds.bad_mask.mean(), allN)
    return datasets


# --------------------------------------------------------------------------
# Bad pixels
# --------------------------------------------------------------------------

def detect_bad_pixels(ds: Dataset, gain_sigma=5.0, offset_sigma=5.0) -> np.ndarray:
    """Dead / low-response / wildly deviating pixels via robust z-score of
    the per-pixel gain and offset of a full-range linear fit."""
    lo, hi = ds.mean_frames[0], ds.mean_frames[-1]
    gain = hi - lo                                    # response over full range
    def robust_z(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * 1.4826 + 1e-12
        return (x - med) / mad
    bad = (np.abs(robust_z(gain)) > gain_sigma) | \
          (np.abs(robust_z(lo)) > offset_sigma) | \
          (gain <= 0)
    return bad


def masked(frame: np.ndarray, bad: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.MaskedArray(frame, mask=bad)


# --------------------------------------------------------------------------
# Corrections
# --------------------------------------------------------------------------

def two_point_correct(ds: Dataset, i: int, j: int) -> np.ndarray:
    """Per-pixel gain/offset from calibration temps i, j; returns corrected
    stack (T, H, W) mapped onto the frame-mean signal scale."""
    s1, s2 = ds.mean_frames[i], ds.mean_frames[j]
    m1, m2 = (masked(s1, ds.bad_mask).mean(),
              masked(s2, ds.bad_mask).mean())
    g = (m2 - m1) / (s2 - s1 + 1e-12)                 # per-pixel gain
    o = m1 - g * s1                                   # per-pixel offset
    return g[None] * ds.mean_frames + o[None]


def piecewise_correct(ds: Dataset, idx: list[int]) -> np.ndarray:
    """Piecewise-linear per-pixel correction through the calibration points
    `idx` (indices into ds.temps). Between adjacent points a 2-pt correction
    is applied; outside, the nearest segment is extrapolated. Interpolation
    is done in signal space, keyed by each pixel's own signal (not by the
    known BB temperature), i.e. as a real camera would apply it."""
    idx = sorted(idx)
    cal_sig = ds.mean_frames[idx]                     # (K, H, W)
    cal_ref = np.array([masked(f, ds.bad_mask).mean() for f in cal_sig])
    K = len(idx)
    out = np.empty_like(ds.mean_frames)
    for t, frame in enumerate(ds.mean_frames):
        # segment index per pixel: how many cal signals are below this signal
        seg = np.sum(cal_sig < frame[None], axis=0)   # 0..K
        seg = np.clip(seg, 1, K - 1)                  # extrapolate w/ end segments
        s_lo = np.take_along_axis(cal_sig, (seg - 1)[None], axis=0)[0]
        s_hi = np.take_along_axis(cal_sig, seg[None], axis=0)[0]
        r_lo = cal_ref[seg - 1]
        r_hi = cal_ref[seg]
        g = (r_hi - r_lo) / (s_hi - s_lo + 1e-12)
        out[t] = r_lo + g * (frame - s_lo)
    return out


def poly_correct(ds: Dataset, idx: list[int], degree: int,
                 eval_idx: list[int] | None = None) -> np.ndarray:
    """Per-pixel polynomial correction fitted on the calibration points
    `idx` and evaluated on the frames `eval_idx` (default: all).

    Each pixel gets a least-squares polynomial (signal -> reference signal)
    of the given degree through its K = len(idx) calibration samples;
    degree = K-1 makes it interpolating (exact at the calibration points).

    Signals are normalized per pixel before building the Vandermonde matrix;
    otherwise counts ~1e4 raised to high powers destroy the conditioning.
    A tiny ridge keeps the solve non-singular for degenerate (dead/
    saturated) pixels, which are masked from all statistics anyway.
    """
    idx = sorted(idx)
    K = len(idx)
    d = min(degree, K - 1)
    H, W = ds.mean_frames.shape[1:]
    cal_sig = ds.mean_frames[idx].reshape(K, -1).T          # (P, K)
    cal_ref = np.array([masked(ds.mean_frames[i], ds.bad_mask).mean()
                        for i in idx])                       # (K,)

    # per-pixel normalization of the abscissa
    s0 = cal_sig.mean(axis=1, keepdims=True)                 # (P, 1)
    sc = np.maximum(cal_sig.max(axis=1, keepdims=True)
                    - cal_sig.min(axis=1, keepdims=True), 1e-9) / 2
    x = (cal_sig - s0) / sc                                  # (P, K) in ~[-1, 1]

    V = np.stack([x ** p for p in range(d + 1)], axis=-1)    # (P, K, d+1)
    y = np.broadcast_to(cal_ref, x.shape)[..., None]         # (P, K, 1)
    VtV = V.transpose(0, 2, 1) @ V + 1e-9 * np.eye(d + 1)
    Vty = V.transpose(0, 2, 1) @ y
    coef = np.linalg.solve(VtV, Vty)[..., 0]                 # (P, d+1)

    ev = range(len(ds.temps)) if eval_idx is None else eval_idx
    out = np.full_like(ds.mean_frames, np.nan)
    powers = np.arange(d + 1)
    for t in ev:
        xf = (ds.mean_frames[t].reshape(-1, 1) - s0) / sc    # (P, 1)
        out[t] = (coef * xf ** powers).sum(axis=1).reshape(H, W)
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def residual_nu(corrected: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Spatial std of each corrected frame (counts), bad pixels excluded.
    This is the residual fixed-pattern non-uniformity."""
    return np.array([masked(f, bad).std() for f in corrected])


def fpn_decomposition(frame: np.ndarray, bad: np.ndarray) -> dict:
    """Split residual pattern into column-stripe, row-stripe and pixel-random
    std components (they add in quadrature up to cross terms)."""
    f = masked(frame - masked(frame, bad).mean(), bad)
    col = f.mean(axis=0)                               # (W,) column pattern
    row = (f - col[None, :]).mean(axis=1)              # (H,) row pattern
    pix = f - col[None, :] - row[:, None]
    return {"total": float(f.std()),
            "column": float(col.std()),
            "row": float(row.std()),
            "pixel": float(pix.std())}


def responsivity(ds: Dataset) -> np.ndarray:
    """dS/dT of the frame-mean signal (counts/K) at each BB temperature,
    from central differences (forward/backward at the ends)."""
    s = np.array([masked(f, ds.bad_mask).mean() for f in ds.mean_frames])
    return np.gradient(s, ds.temps)


def to_mK(sigma_counts: np.ndarray, resp: np.ndarray) -> np.ndarray:
    """Counts -> mK via local responsivity."""
    return 1000.0 * sigma_counts / np.maximum(resp, 1e-12)


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def analyze_two_point(ds: Dataset):
    """Sweep all calibration pairs; score = max residual NU (in mK) over all
    measured temperatures. Returns score matrix and the best pair."""
    n = len(ds.temps)
    resp = responsivity(ds)
    score = np.full((n, n), np.nan)
    best = (None, np.inf)
    pairs = list(itertools.combinations(range(n), 2))
    log.info("2-pt sweep: %d calibration pairs", len(pairs))
    for pi, (i, j) in enumerate(pairs):
        res_counts = residual_nu(two_point_correct(ds, i, j), ds.bad_mask)
        res_mk = to_mK(res_counts, resp)
        s = np.nanmax(res_mk)
        score[i, j] = score[j, i] = s
        if s < best[1]:
            best = ((i, j), s)
        _progress(pi, len(pairs), "2-pt sweep")
    return score, best, resp


def analyze_piecewise(ds: Dataset, k_max=None):
    """Piecewise-linear residual vs number of calibration points, with
    optimal point placement.

    For each k: exhaustive search over point subsets if feasible (endpoints
    forced in — extrapolation is always worse than interpolation), greedy
    insertion otherwise. Score = max residual NU (mK) over ALL measured
    temperatures. Calibration points themselves are exact by construction,
    so the max is driven by the temps between points — an honest
    generalization measure as long as the temperature grid is reasonably
    dense."""
    n = len(ds.temps)
    resp = responsivity(ds)
    k_max = k_max or n
    results = {}

    def score_of(idx):
        res = to_mK(residual_nu(piecewise_correct(ds, list(idx)), ds.bad_mask),
                    resp)
        return float(np.nanmax(res)), res

    current = [0, n - 1]
    s, curve = score_of(current)
    results[2] = {"idx": list(current), "score": s, "curve": curve,
                  "method": "endpoints"}

    for k in range(3, min(k_max, n) + 1):
        inner = list(range(1, n - 1))
        n_comb = comb(len(inner), k - 2)
        if n_comb <= 3000:                                   # exhaustive
            log.info("piecewise linear k=%d: exhaustive search over %d subsets",
                     k, n_comb)
            best = (None, np.inf, None)
            for ci, c in enumerate(itertools.combinations(inner, k - 2)):
                idx = [0, *c, n - 1]
                s, curve = score_of(idx)
                if s < best[1]:
                    best = (idx, s, curve)
                _progress(ci, n_comb, f"piecewise linear k={k}")
            results[k] = {"idx": best[0], "score": best[1], "curve": best[2],
                          "method": "exhaustive"}
        else:                                                # greedy
            cand = [i for i in inner if i not in current]
            log.info("piecewise linear k=%d: greedy search over %d candidates "
                     "(%d subsets would exceed the exhaustive limit)",
                     k, len(cand), n_comb)
            best = (None, np.inf, None)
            for ci, c in enumerate(cand):
                idx = sorted(current + [c])
                s, curve = score_of(idx)
                if s < best[1]:
                    best = (idx, s, curve)
                _progress(ci, len(cand), f"piecewise linear k={k} (greedy)")
            results[k] = {"idx": best[0], "score": best[1], "curve": best[2],
                          "method": "greedy"}
            current = best[0]
        if results[k]["idx"] is None:
            break
    return results, resp


def analyze_polynomial(ds: Dataset):
    """Polynomial residual vs order, using ALL calibration temperatures
    (no point selection). Orders d = 1 .. n-2.

    Leave-one-out evaluation: the residual at each interior temperature
    comes from a fit that excludes that temperature (the LOO fit has n-1
    points, hence d <= n-2). The endpoints are evaluated in-sample from the
    full fit, since leaving them out would mean extrapolation. Score = max
    over the full curve. This mirrors the piecewise analysis, where the
    non-calibration temperatures are the held-out ones."""
    n = len(ds.temps)
    resp = responsivity(ds)
    bad = ds.bad_mask
    all_idx = list(range(n))
    results = {}
    orders = range(1, n - 1)                                 # 1 .. n-2
    log.info("polynomial sweep: orders 1..%d, %d LOO fits each",
             n - 2, n - 2)
    for d in orders:
        curve = np.full(n, np.nan)
        full = poly_correct(ds, all_idx, d, eval_idx=[0, n - 1])
        curve[0] = masked(full[0], bad).std()
        curve[-1] = masked(full[-1], bad).std()
        for t in range(1, n - 1):                            # LOO interior
            loo = poly_correct(ds, [i for i in all_idx if i != t], d,
                               eval_idx=[t])
            curve[t] = masked(loo[t], bad).std()
            _progress(t - 1, n - 2, f"poly d={d} LOO", every=max(1, (n - 2) // 4))
        curve_mk = to_mK(curve, resp)
        results[d] = {"curve": curve_mk,
                      "score": float(np.nanmax(curve_mk))}
        log.info("poly d=%d done: max residual %.0f mK", d, results[d]["score"])
    return results, resp


def poly_loo_frame(ds: Dataset, d: int, t: int) -> np.ndarray:
    """Corrected frame at temperature index t, order d, from the same fit
    used in analyze_polynomial (LOO for interior temps, full fit at ends)."""
    n = len(ds.temps)
    idx = list(range(n)) if t in (0, n - 1) else \
        [i for i in range(n) if i != t]
    return poly_correct(ds, idx, d, eval_idx=[t])[t]


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def make_plots(ds: Dataset, out: Path, pair_score, best_pair, pw_results,
               poly_results, resp, maps="best"):
    temps = ds.temps
    n = len(temps)
    tag = f"tint{ds.tint:g}{TINT_UNIT}"
    d_best = min(poly_results, key=lambda d: poly_results[d]["score"])

    # --- response curve + nonlinearity ------------------------------------
    s_mean = np.array([masked(f, ds.bad_mask).mean() for f in ds.mean_frames])
    lin = np.polyval(np.polyfit(temps[[0, -1]], s_mean[[0, -1]], 1), temps)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(temps, s_mean, "o-")
    ax[0].plot(temps, lin, "--", color="gray", label="endpoint line")
    ax[0].set(xlabel="BB temperature (°C)", ylabel="mean signal (counts)",
              title="Response curve")
    ax[0].legend()
    ax[1].plot(temps, s_mean - lin, "o-")
    ax[1].axhline(0, color="gray", lw=0.5)
    ax[1].set(xlabel="BB temperature (°C)", ylabel="counts",
              title="Deviation from endpoint line (nonlinearity)")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_response.png", dpi=130)
    plt.close(fig)

    # --- fit curves: distribution band + sample-pixel fits ------------------
    # Panel A: signal vs T — spatial mean with +-1 sigma and 1..99 percentile
    # bands (the uncorrected non-uniformity), plus three sample pixels.
    # Panel B: the actual correction mappings of those pixels, shown as
    # correction delta (reference - signal) vs signal; the raw mapping is a
    # near-identity diagonal on which both methods would be indistinguishable.
    good = ~ds.bad_mask
    flat = ds.mean_frames.reshape(len(temps), -1)[:, good.ravel()]  # (T, P)
    gain_full = flat[-1] - flat[0]
    order = np.argsort(gain_full)
    sample_p = [order[int(q * (len(order) - 1))] for q in (0.05, 0.5, 0.95)]
    sample_lbl = ["5th %ile gain", "median gain", "95th %ile gain"]
    colors = ["tab:red", "tab:green", "tab:purple"]
    cal_ref_all = np.array([masked(f, ds.bad_mask).mean()
                            for f in ds.mean_frames])

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    mu = flat.mean(axis=1)
    sd = flat.std(axis=1)
    p01, p99 = np.percentile(flat, [1, 99], axis=1)
    ax[0].fill_between(temps, p01, p99, alpha=0.15, color="tab:blue",
                       label="1..99 percentile")
    ax[0].fill_between(temps, mu - sd, mu + sd, alpha=0.35, color="tab:blue",
                       label="mean ± 1σ")
    ax[0].plot(temps, mu, "-", color="tab:blue", label="spatial mean")
    for p, lbl, c in zip(sample_p, sample_lbl, colors):
        ax[0].plot(temps, flat[:, p], ".-", lw=1, color=c, label=lbl)
    ax[0].set(xlabel="BB temperature (°C)", ylabel="signal (counts)",
              title="Raw response: spatial distribution + sample pixels")
    ax[0].legend(fontsize=8)

    pw_idx = pw_results[max(pw_results)]["idx"]
    for p, lbl, c in zip(sample_p, sample_lbl, colors):
        s = flat[:, p]
        # measured calibration samples of this pixel
        ax[1].plot(s, cal_ref_all - s, "o", ms=5, color=c,
                   label=f"{lbl}: samples")
        # piecewise-linear mapping (best k): straight in delta space too,
        # so the polyline through the calibration points IS the fit
        ax[1].plot(s[pw_idx], cal_ref_all[pw_idx] - s[pw_idx], "-",
                   color=c, alpha=0.9)
        # polynomial mapping (best LOO order), full fit, dense evaluation
        x0, xs = s.mean(), max(np.ptp(s) / 2, 1e-9)
        cf = np.polyfit((s - x0) / xs, cal_ref_all, d_best)
        sg = np.linspace(s.min(), s.max(), 200)
        ax[1].plot(sg, np.polyval(cf, (sg - x0) / xs) - sg, "--",
                   color=c, alpha=0.9)
    ax[1].axhline(0, color="gray", lw=0.5)
    ax[1].plot([], [], "-", color="gray",
               label=f"piecewise linear {max(pw_results)}-pt")
    ax[1].plot([], [], "--", color="gray", label=f"poly d={d_best}")
    ax[1].set(xlabel="pixel signal (counts)",
              ylabel="correction Δ = reference − signal (counts)",
              title="Per-pixel correction mappings (sample pixels)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_fit_curves.png", dpi=130)
    plt.close(fig)

    # --- two-point pair score heatmap --------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(pair_score, origin="lower", cmap="viridis")
    ax.set_xticks(range(n), [f"{t:g}" for t in temps], rotation=90)
    ax.set_yticks(range(n), [f"{t:g}" for t in temps])
    ax.set(xlabel="cal temp 2 (°C)", ylabel="cal temp 1 (°C)",
           title="2-pt NUC: worst-case residual NU (mK) vs calibration pair")
    (bi, bj), bs = best_pair
    ax.plot(bj, bi, "r*", ms=16,
            label=f"best: {temps[bi]:g}/{temps[bj]:g} °C → {bs:.0f} mK")
    ax.legend(loc="upper left")
    fig.colorbar(im, label="max residual NU (mK)")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_2pt_pair_score.png", dpi=130)
    plt.close(fig)

    # --- residual vs temperature: 2pt, piecewise, polynomial ---------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    naive = to_mK(residual_nu(two_point_correct(ds, 0, n - 1), ds.bad_mask), resp)
    bestc = to_mK(residual_nu(two_point_correct(ds, bi, bj), ds.bad_mask), resp)
    ax.plot(temps, naive, "o-", label=f"2-pt endpoints ({temps[0]:g}/{temps[-1]:g} °C)")
    ax.plot(temps, bestc, "o-", label=f"2-pt best ({temps[bi]:g}/{temps[bj]:g} °C)")
    for k in sorted(pw_results):
        if k < 3:
            continue
        r = pw_results[k]
        pts = ", ".join(f"{temps[i]:g}" for i in r["idx"])
        ax.plot(temps, r["curve"], ".-", alpha=0.8,
                label=f"piecewise linear {k}-pt [{pts}] °C")
        if k >= 5:
            break
    ax.plot(temps, poly_results[d_best]["curve"], "s--", alpha=0.9,
            label=f"poly d={d_best}, all pts (LOO)")
    if not np.all(np.isnan(ds.temporal_std_mean)):
        ax.plot(temps, to_mK(ds.temporal_std_mean, resp), "k--",
                label="measurement noise floor (mean frames)")
        ax.plot(temps, to_mK(ds.temporal_std_single, resp), "k:",
                label="temporal NETD (single frame)")
    ax.set(xlabel="BB temperature (°C)", ylabel="residual spatial NU (mK)",
           title=f"Residual FPN after NUC, tint={ds.tint:g} {TINT_UNIT}")
    ax.set_yscale("log")
    # Calibration temperatures are exact by construction (residual ~ machine
    # epsilon); mask those points entirely instead of letting them drag the
    # log axis down or draw spikes to the plot edge.
    for ln in ax.get_lines():
        y = np.asarray(ln.get_ydata(), dtype=float).copy()
        y[y < 1e-3] = np.nan                                 # < 1 uK: exact
        ln.set_ydata(y)
    vals = np.concatenate([np.asarray(ln.get_ydata(), dtype=float)
                           for ln in ax.get_lines()])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        ax.set_ylim(0.5 * vals.min(), 2 * vals.max())
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_residual_vs_T.png", dpi=130)
    plt.close(fig)

    # --- fit curves: distribution band + sample-pixel fits ------------------
    # (a) response vs T: mean over pixels, shaded 5-95 % pixel percentile
    #     band, plus raw curves of the sample pixels.
    # (b) correction transfer curves of the same pixels in signal space:
    #     calibration samples (markers), best piecewise-linear map (solid,
    #     through its k selected points) and best-order polynomial map
    #     (dashed, LSQ through all points), both evaluated on a dense grid.
    good = np.argwhere(~ds.bad_mask)
    gain_full = (ds.mean_frames[-1] - ds.mean_frames[0])[tuple(good.T)]
    order = np.argsort(gain_full)
    sample = [tuple(good[order[int(q * (len(order) - 1))]])
              for q in (0.02, 0.25, 0.5, 0.75, 0.98)]        # gain percentiles

    sig = ds.mean_frames.reshape(n, -1)
    p05, p95 = np.percentile(ds.mean_frames[:, ~ds.bad_mask], [5, 95], axis=1)
    ref = np.array([masked(f, ds.bad_mask).mean() for f in ds.mean_frames])
    kb = max(pw_results)
    pw_idx = sorted(pw_results[kb]["idx"])

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    ax[0].fill_between(temps, p05, p95, alpha=0.25,
                       label="pixel distribution (5–95 %)")
    ax[0].plot(temps, ref, "k-", lw=2, label="mean (= reference)")
    colors = plt.cm.tab10(np.linspace(0, 1, len(sample)))
    for (r_, c_), col in zip(sample, colors):
        ax[0].plot(temps, ds.mean_frames[:, r_, c_], ".-", color=col,
                   lw=0.8, ms=4)
    ax[0].set(xlabel="BB temperature (°C)", ylabel="signal (counts)",
              title="Response: mean, distribution, sample pixels "
                    "(gain percentiles 2/25/50/75/98)")
    ax[0].legend(fontsize=8)

    d_best_plot = min(poly_results, key=lambda d: poly_results[d]["score"])
    for (r_, c_), col in zip(sample, colors):
        s_pix = ds.mean_frames[:, r_, c_]                    # raw at all temps
        dense = np.linspace(s_pix.min(), s_pix.max(), 200)
        # piecewise linear through its k selected calibration points
        pw_s = s_pix[pw_idx]
        srt = np.argsort(pw_s)
        ax[1].plot(dense, np.interp(dense, pw_s[srt], ref[pw_idx][srt]),
                   "-", color=col, lw=1.2)
        # polynomial (best LOO order) through all points, normalized abscissa
        s0_, sc_ = s_pix.mean(), max(s_pix.max() - s_pix.min(), 1e-9) / 2
        cf = np.polyfit((s_pix - s0_) / sc_, ref, d_best_plot)
        ax[1].plot(dense, np.polyval(cf, (dense - s0_) / sc_),
                   "--", color=col, lw=1.2)
        ax[1].plot(s_pix, ref, "o", color=col, ms=3.5)
    lo = min(ref.min(), min(ds.mean_frames[:, r_, c_].min() for r_, c_ in sample))
    hi = max(ref.max(), max(ds.mean_frames[:, r_, c_].max() for r_, c_ in sample))
    ax[1].plot([lo, hi], [lo, hi], color="gray", lw=0.6, ls=":",
               label="identity")
    ax[1].plot([], [], "k-", label=f"piecewise linear {kb}-pt")
    ax[1].plot([], [], "k--", label=f"polynomial d={d_best_plot} (all pts)")
    ax[1].plot([], [], "ko", ms=3.5, label="calibration samples")
    ax[1].set(xlabel="raw pixel signal (counts)",
              ylabel="corrected signal (counts)",
              title="Correction transfer curves, sample pixels")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_fit_curves.png", dpi=130)
    plt.close(fig)

    # --- convergence: piecewise vs k AND polynomial vs order ----------------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ks = sorted(pw_results)
    ax.plot(ks, [pw_results[k]["score"] for k in ks], "o-",
            label="piecewise linear (optimal k points)")
    dd = sorted(poly_results)
    # order-d polynomial has d+1 coefficients -> comparable to (d+1)-pt piecewise
    ax.plot([d + 1 for d in dd], [poly_results[d]["score"] for d in dd], "s-",
            label="polynomial, all points, LOO (order+1)")
    if not np.all(np.isnan(ds.temporal_std_mean)):
        ax.axhline(np.nanmedian(to_mK(ds.temporal_std_mean, resp)), color="k",
                   ls="--", label="measurement noise floor (median)")
    ax.set(xlabel="free parameters per pixel (k points / order+1)",
           ylabel="max residual NU (mK)",
           title="NUC convergence: piecewise vs polynomial")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_convergence.png", dpi=130)
    plt.close(fig)

    # --- FPN decomposition + residual map at worst temperature -------------
    def residual_map_figure(frame: np.ndarray, T_label: str, label: str,
                            fname: str, resp_at):
        dec = fpn_decomposition(frame, ds.bad_mask)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        r = masked(frame - masked(frame, ds.bad_mask).mean(), ds.bad_mask)
        v = 3 * r.std()
        im = ax[0].imshow(r, cmap="coolwarm", vmin=-v, vmax=v)
        ax[0].set_title(f"Residual @ {T_label} ({label})")
        fig.colorbar(im, ax=ax[0], label="counts")
        comps = ["total", "column", "row", "pixel"]
        vals_mk = [1000 * dec[c] / resp_at for c in comps]
        ax[1].bar(comps, vals_mk)
        ax[1].set(ylabel="mK", title=f"FPN decomposition ({label})")
        for x, v_ in enumerate(vals_mk):
            ax[1].text(x, v_, f"{v_:.0f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)
        return dec

    decs = {}
    w = int(np.nanargmax(bestc))
    decs["best_2pt"] = (residual_map_figure(
        two_point_correct(ds, bi, bj)[w], f"{temps[w]:g} °C",
        f"best 2-pt {temps[bi]:g}/{temps[bj]:g} °C",
        f"{tag}_fpn_decomposition_2pt.png", resp[w]), w)

    pw_ks = sorted(pw_results) if maps == "all" else [max(pw_results)]
    for k in pw_ks:
        r = pw_results[k]
        w = int(np.nanargmax(r["curve"]))
        pts = "/".join(f"{temps[i]:g}" for i in r["idx"])
        decs[f"piecewise_{k}pt"] = (residual_map_figure(
            piecewise_correct(ds, r["idx"])[w], f"{temps[w]:g} °C",
            f"piecewise linear {k}-pt [{pts}] °C",
            f"{tag}_fpn_decomposition_piecewise_{k}pt.png", resp[w]), w)

    poly_ds = sorted(poly_results) if maps == "all" else [d_best]
    for d in poly_ds:
        w = int(np.nanargmax(poly_results[d]["curve"]))
        decs[f"poly_d{d}"] = (residual_map_figure(
            poly_loo_frame(ds, d, w), f"{temps[w]:g} °C",
            f"poly d={d} (LOO)",
            f"{tag}_fpn_decomposition_poly_d{d}.png", resp[w]), w)

    return decs, d_best


# --------------------------------------------------------------------------
# Synthetic self-test data
# --------------------------------------------------------------------------

def make_synthetic(folder: Path, H=120, W=160, seed=0):
    """Microbolometer-ish synthetic stack: per-pixel gain/offset FPN, column
    stripes, mild per-pixel nonlinearity, shot+read noise, a few bad pixels.
    Written as 16-bit PGM with the default naming scheme."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    temps = np.arange(10, 101, 10.0)                  # 10..100 °C
    tint_us = 10000
    gain = 1 + 0.05 * rng.standard_normal((H, W))
    offs = 300 * rng.standard_normal((H, W))
    col = 40 * rng.standard_normal(W)                 # column stripes in offset
    colg = 0.01 * rng.standard_normal(W)              # column stripes in gain
    nl = 1 + 0.02 * rng.standard_normal((H, W))       # per-pixel nonlinearity
    bad = rng.random((H, W)) < 5e-4
    for T in temps:
        # scene "radiance": convex in T (Planck-like around LWIR)
        L = 5000 + 60 * T + 0.35 * T ** 2
        for rep in range(4):
            sig = (gain + colg[None, :]) * L + offs + col[None, :] \
                  + 0.002 * nl * (L - 8000) ** 2 / 100.0
            sig = sig + rng.normal(0, 8, (H, W))      # temporal noise
            sig[bad] = rng.choice([0, 16383], bad.sum())
            img = np.clip(sig, 0, 65535).astype(">u2")
            name = f"mean_{tint_us}us_{T:g}deg_N128_{rep:02d}.pgm"
            with open(folder / name, "wb") as f:
                f.write(f"P5\n{W} {H}\n65535\n".encode())
                f.write(img.tobytes())
    log.info(f"Wrote synthetic dataset to {folder} ({len(temps)} temps x 4 files)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", type=Path,
                    help="folder with blackbody frames")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help="filename regex with named groups 'tint' and 'temp' "
                         "(optional 'navg'); default matches e.g. "
                         "mean_10000us_10deg_N128.pgm")
    ap.add_argument("--out", type=Path, default=Path("./nuc_results"))
    ap.add_argument("--raw-shape", nargs=2, type=int, default=None,
                    metavar=("H", "W"))
    ap.add_argument("--raw-dtype", default="uint16")
    ap.add_argument("--k-max", type=int, default=8,
                    help="max number of piecewise calibration points")
    ap.add_argument("--tint", type=float, default=None,
                    help=f"analyze only this integration time ({TINT_UNIT}); "
                         "default: all found in the folder")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate & analyze a synthetic dataset (self-test)")
    ap.add_argument("--maps", default="best", choices=["best", "all"],
                    help="residual-map/FPN-decomposition figures: only for "
                         "the best configuration of each method (default), "
                         "or for every piecewise k and every polynomial order")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname).1s %(message)s",
                        datefmt="%H:%M:%S")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        args.folder = args.out / "synthetic_data"
        make_synthetic(args.folder)
    elif args.folder is None:
        ap.error("provide a folder or use --synthetic")

    datasets = scan_folder(args.folder, args.pattern,
                           tuple(args.raw_shape) if args.raw_shape else None,
                           args.raw_dtype, only_tint=args.tint)

    summary = {}
    for tint, ds in datasets.items():
        if len(ds.temps) < 3:
            log.warning(f"tint={tint:g} {TINT_UNIT}: <3 temperatures, skipping.")
            continue
        log.info(f"=== tint = {tint:g} {TINT_UNIT} ===")
        n = len(ds.temps)

        pair_score, best_pair, resp = analyze_two_point(ds)
        (bi, bj), bs = best_pair
        naive = float(pair_score[0, n - 1])
        log.info(f"[Q1] 2-pt with endpoints ({ds.temps[0]:g}/{ds.temps[-1]:g} °C): "
              f"max residual NU = {naive:.0f} mK")
        log.info(f"[Q1] best pair: {ds.temps[bi]:g}/{ds.temps[bj]:g} °C "
              f"→ max residual NU = {bs:.0f} mK "
              f"({100*(1-bs/naive):.0f} % better than endpoints)")

        pw_results, _ = analyze_piecewise(ds, k_max=args.k_max)
        log.info("[Q2] piecewise-linear, optimal point placement:")
        for k in sorted(pw_results):
            r = pw_results[k]
            pts = ", ".join(f"{ds.temps[i]:g}" for i in r["idx"])
            log.info(f"      k={k}: max residual = {r['score']:.0f} mK  "
                  f"@ [{pts}] °C ({r['method']})")

        poly_results, _ = analyze_polynomial(ds)
        log.info("[Q2] polynomial, all points, leave-one-out:")
        for d in sorted(poly_results):
            log.info(f"      d={d}: max residual = "
                  f"{poly_results[d]['score']:.0f} mK")
        d_best_ = min(poly_results, key=lambda d: poly_results[d]["score"])

        curves = {"2pt_endpoints":
                      to_mK(residual_nu(two_point_correct(ds, 0, n - 1),
                                        ds.bad_mask), resp),
                  f"2pt_best_{ds.temps[bi]:g}_{ds.temps[bj]:g}":
                      to_mK(residual_nu(two_point_correct(ds, bi, bj),
                                        ds.bad_mask), resp)}
        for k, r in pw_results.items():
            curves[f"piecewise_{k}pt"] = r["curve"]
        for d, r in poly_results.items():
            curves[f"poly_d{d}_LOO"] = r["curve"]

        rep = {"tint": ds.tint, "tint_unit": TINT_UNIT,
               "temps_C": ds.temps.tolist(),
               "responsivity_counts_per_K": resp.tolist(),
               "spatial_NETD_mK": {name: np.asarray(c).tolist()
                                   for name, c in curves.items()}}
        if not np.all(np.isnan(ds.temporal_std_single)):
            tn1 = to_mK(ds.temporal_std_single, resp)
            tnm = to_mK(ds.temporal_std_mean, resp)
            rep["temporal_NETD_single_frame_mK"] = tn1.tolist()
            rep["noise_floor_mean_frames_mK"] = tnm.tolist()
            log.info(f"[Q3] temporal NETD (single frame): "
                     f"{np.nanmedian(tn1):.0f} mK; measurement noise floor "
                     f"of the averaged frames: {np.nanmedian(tnm):.0f} mK "
                     f"(medians over T)")
        best_curve = curves[f"2pt_best_{ds.temps[bi]:g}_{ds.temps[bj]:g}"]
        kbest = max(pw_results)
        log.info(f"[Q3] spatial (FPN) NETD after best 2-pt: "
              f"median {np.nanmedian(best_curve):.0f} mK, "
              f"max {np.nanmax(best_curve):.0f} mK")
        log.info(f"[Q3] spatial (FPN) NETD after piecewise linear {kbest}-pt: "
              f"median {np.nanmedian(pw_results[kbest]['curve']):.0f} mK, "
              f"max {pw_results[kbest]['score']:.0f} mK")
        log.info(f"[Q3] spatial (FPN) NETD after poly d={d_best_} (LOO): "
              f"median {np.nanmedian(poly_results[d_best_]['curve']):.0f} mK, "
              f"max {poly_results[d_best_]['score']:.0f} mK")

        decs, d_best = make_plots(ds, args.out, pair_score, best_pair,
                                  pw_results, poly_results, resp,
                                  maps=args.maps)
        for name, (dec, w) in decs.items():
            log.info(f"[+]  FPN decomposition @ worst T ({ds.temps[w]:g} °C, "
                  f"{name}, counts): total {dec['total']:.2f} | "
                  f"column {dec['column']:.2f} | row {dec['row']:.2f} | "
                  f"pixel {dec['pixel']:.2f}")

        rep["two_point"] = {
            "pair_score_mK": np.where(np.isnan(pair_score), None,
                                      np.round(pair_score, 1)).tolist(),
            "best_pair_C": [ds.temps[bi], ds.temps[bj]],
            "best_score_mK": bs,
        }
        rep["piecewise"] = {str(k): {"temps_C": [ds.temps[i] for i in r["idx"]],
                                     "max_residual_mK": r["score"],
                                     "search": r["method"]}
                            for k, r in pw_results.items()}
        rep["polynomial_LOO"] = {str(d): {"max_residual_mK": r["score"]}
                                 for d, r in poly_results.items()}
        rep["fpn_decomposition_counts"] = {k: v[0] for k, v in decs.items()}
        summary[f"{tint:g}{TINT_UNIT}"] = rep

    with open(args.out / "nuc_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log.info(f"Plots and nuc_summary.json written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
