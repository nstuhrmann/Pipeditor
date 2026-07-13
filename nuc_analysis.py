#!/usr/bin/env python3
"""
Non-Uniformity Correction (NUC) analysis for thermal imagers.

Given a folder of blackbody frames (temperature and integration time encoded
in the filename), this tool answers:

  1. Residual error of two-point correction over the full temperature range,
     and which pair of calibration temperatures minimizes it.
  2. Residual error of multi-point (piecewise-linear per pixel) correction as
     a function of the number and placement of calibration points, with
     exhaustive search for small k and greedy selection otherwise.
  3. Residual noise expressed as NETD (spatial / FPN NETD), plus temporal
     NETD if repeated frames per condition exist, via the measured
     responsivity dS/dT.

Extended analysis:
  - FPN decomposition of the residual into column stripes, row stripes and
    pixel-random components (in counts and mK).
  - Bad-pixel detection (dead / low-response / high-noise pixels), which are
    masked from all statistics so they don't dominate the residual.
  - Response linearity per pixel (why two-point fails: signal vs. blackbody
    temperature is convex due to Planck radiance + detector nonlinearity).
  - Everything is done per integration time; gain/offset tables are only
    valid for the tint they were measured at.

Usage:
    python nuc_analysis.py /path/to/blackbody/folder \
        --pattern ".*?(?P<temp>\\d+(?:\\.\\d+)?)[cC].*?(?P<tint>\\d+(?:\\.\\d+)?)ms" \
        --out ./nuc_results

    python nuc_analysis.py --synthetic --out ./nuc_results   # self-test

The filename regex must contain named groups `temp` (blackbody temperature,
deg C) and `tint` (integration time, ms). All frames sharing (tint, temp) are
treated as repeats and averaged; the frame-to-frame variation gives the
temporal noise estimate.

Supported formats: .tif/.tiff, .png (16-bit ok), .npy, and headerless .raw/
.bin (specify --raw-shape H W and --raw-dtype, e.g. uint16).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

DEFAULT_PATTERN = r".*?(?P<temp>-?\d+(?:[\.,]\d+)?)\s*[cC].*?(?P<tint>\d+(?:[\.,]\d+)?)\s*ms"


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
    # png / jpg / everything else
    import imageio.v3 as iio
    img = iio.imread(path)
    if img.ndim == 3:  # accidental RGB; take first channel
        img = img[..., 0]
    return img.astype(np.float64)


@dataclass
class Dataset:
    """All frames for one integration time."""
    tint_ms: float
    temps: np.ndarray            # sorted unique blackbody temps [K count = len]
    mean_frames: np.ndarray      # (T, H, W) frame average per temperature
    temporal_std: np.ndarray     # (T,) mean per-pixel temporal std (NaN if 1 frame)
    n_frames: dict = field(default_factory=dict)   # temp -> #repeats
    bad_mask: np.ndarray = None  # (H, W) bool, True = bad pixel


def scan_folder(folder: Path, pattern: str, raw_shape=None, raw_dtype="uint16",
                verbose=True) -> dict[float, Dataset]:
    rx = re.compile(pattern)
    groups: dict[tuple[float, float], list[Path]] = {}
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
        groups.setdefault((tint, temp), []).append(p)

    if verbose:
        print(f"Matched {sum(len(v) for v in groups.values())} files "
              f"({len(groups)} (tint,temp) conditions), skipped {len(skipped)}.")
        if skipped[:5]:
            print("  e.g. skipped:", ", ".join(skipped[:5]))
    if not groups:
        raise SystemExit("No files matched the filename pattern. "
                         "Adjust --pattern (needs named groups 'temp' and 'tint').")

    datasets: dict[float, Dataset] = {}
    for tint in sorted({k[0] for k in groups}):
        temps = sorted({k[1] for k in groups if k[0] == tint})
        means, tstds, nfr = [], [], {}
        for T in temps:
            frames = np.stack([load_frame(p, raw_shape, raw_dtype)
                               for p in groups[(tint, T)]])
            means.append(frames.mean(axis=0))
            tstds.append(frames.std(axis=0, ddof=1).mean() if len(frames) > 1
                         else np.nan)
            nfr[T] = len(frames)
        ds = Dataset(tint_ms=tint,
                     temps=np.array(temps, dtype=float),
                     mean_frames=np.stack(means),
                     temporal_std=np.array(tstds),
                     n_frames=nfr)
        ds.bad_mask = detect_bad_pixels(ds)
        datasets[tint] = ds
        if verbose:
            H, W = ds.mean_frames.shape[1:]
            print(f"tint={tint} ms: {len(temps)} temps "
                  f"({temps[0]}..{temps[-1]} C), frame {W}x{H}, "
                  f"bad pixels: {ds.bad_mask.sum()} "
                  f"({100*ds.bad_mask.mean():.3f} %)")
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


def multi_point_correct(ds: Dataset, idx: list[int]) -> np.ndarray:
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
    # For each pixel, np.interp over its own calibration signals.
    # Vectorized: search segment per (frame, pixel).
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
    measured temperatures. Returns score matrix and per-pair curves for the
    best/naive pairs."""
    n = len(ds.temps)
    resp = responsivity(ds)
    score = np.full((n, n), np.nan)
    best = (None, np.inf)
    for i, j in itertools.combinations(range(n), 2):
        res_counts = residual_nu(two_point_correct(ds, i, j), ds.bad_mask)
        res_mk = to_mK(res_counts, resp)
        s = np.nanmax(res_mk)
        score[i, j] = score[j, i] = s
        if s < best[1]:
            best = ((i, j), s)
    return score, best, resp


def analyze_multi_point(ds: Dataset, k_max=None):
    """Residual vs number of calibration points.

    For each k: exhaustive search over point subsets if feasible (endpoints
    forced in — extrapolation is always worse than interpolation), greedy
    insertion otherwise. Score = max residual NU (mK) over ALL measured
    temperatures. Calibration points themselves are exact by construction, so
    the max is driven by the temps between points — an honest generalization
    measure as long as the temperature grid is reasonably dense."""
    n = len(ds.temps)
    resp = responsivity(ds)
    k_max = k_max or n
    results = {}

    def score_of(idx):
        res = to_mK(residual_nu(multi_point_correct(ds, list(idx)), ds.bad_mask),
                    resp)
        return float(np.nanmax(res)), res

    # k = 2 comes from the exhaustive two-point sweep for consistency
    current = [0, n - 1]
    s, curve = score_of(current)
    results[2] = {"idx": list(current), "score": s, "curve": curve}

    for k in range(3, min(k_max, n) + 1):
        inner = [i for i in range(1, n - 1)]
        n_choose = len(inner)
        if n_choose >= k - 2 and _ncr(n_choose, k - 2) <= 3000:  # exhaustive
            best = (None, np.inf, None)
            for comb in itertools.combinations(inner, k - 2):
                idx = [0, *comb, n - 1]
                s, curve = score_of(idx)
                if s < best[1]:
                    best = (idx, s, curve)
            results[k] = {"idx": best[0], "score": best[1], "curve": best[2],
                          "method": "exhaustive"}
        else:                                                    # greedy
            cand = [i for i in inner if i not in current]
            best = (None, np.inf, None)
            for c in cand:
                idx = sorted(current + [c])
                s, curve = score_of(idx)
                if s < best[1]:
                    best = (idx, s, curve)
            results[k] = {"idx": best[0], "score": best[1], "curve": best[2],
                          "method": "greedy"}
            current = best[0]
        if results[k]["idx"] is None:
            break
    return results, resp


def _ncr(n, r):
    from math import comb
    return comb(n, r)


def netd_report(ds: Dataset, curves: dict, resp: np.ndarray) -> dict:
    """Spatial (FPN) NETD from residual NU, temporal NETD from repeats."""
    rep = {"tint_ms": ds.tint_ms, "temps_C": ds.temps.tolist()}
    rep["responsivity_counts_per_K"] = resp.tolist()
    rep["spatial_NETD_mK"] = {name: c.tolist() for name, c in curves.items()}
    if not np.all(np.isnan(ds.temporal_std)):
        rep["temporal_NETD_mK"] = to_mK(ds.temporal_std, resp).tolist()
    return rep


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def make_plots(ds: Dataset, out: Path, pair_score, best_pair, mp_results,
               resp):
    temps = ds.temps
    n = len(temps)
    tag = f"tint{ds.tint_ms:g}ms"

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

    # --- residual vs temperature: naive/best 2pt + multipoint --------------
    fig, ax = plt.subplots(figsize=(8, 5))
    naive = to_mK(residual_nu(two_point_correct(ds, 0, n - 1), ds.bad_mask), resp)
    bestc = to_mK(residual_nu(two_point_correct(ds, bi, bj), ds.bad_mask), resp)
    ax.plot(temps, naive, "o-", label=f"2-pt endpoints ({temps[0]:g}/{temps[-1]:g} °C)")
    ax.plot(temps, bestc, "o-", label=f"2-pt best ({temps[bi]:g}/{temps[bj]:g} °C)")
    for k in sorted(mp_results):
        if k < 3:
            continue
        r = mp_results[k]
        pts = ", ".join(f"{temps[i]:g}" for i in r["idx"])
        ax.plot(temps, r["curve"], ".-", alpha=0.8,
                label=f"{k}-pt [{pts}] °C")
        if k >= 6:
            break
    if not np.all(np.isnan(ds.temporal_std)):
        ax.plot(temps, to_mK(ds.temporal_std, resp), "k--",
                label="temporal NETD (floor)")
    ax.set(xlabel="BB temperature (°C)", ylabel="residual spatial NU (mK)",
           title=f"Residual FPN after NUC, tint={ds.tint_ms:g} ms")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_residual_vs_T.png", dpi=130)
    plt.close(fig)

    # --- residual vs number of points ---------------------------------------
    ks = sorted(mp_results)
    scores = [mp_results[k]["score"] for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ks, scores, "o-")
    if not np.all(np.isnan(ds.temporal_std)):
        ax.axhline(np.nanmedian(to_mK(ds.temporal_std, resp)), color="k",
                   ls="--", label="temporal NETD (median)")
        ax.legend()
    ax.set(xlabel="number of calibration points", ylabel="max residual NU (mK)",
           title="Multi-point NUC convergence")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_multipoint_convergence.png", dpi=130)
    plt.close(fig)

    # --- FPN decomposition + residual map at worst temperature -------------
    worst = int(np.nanargmax(bestc))
    corr = two_point_correct(ds, bi, bj)[worst]
    dec = fpn_decomposition(corr, ds.bad_mask)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    r = masked(corr - masked(corr, ds.bad_mask).mean(), ds.bad_mask)
    v = 3 * r.std()
    im = ax[0].imshow(r, cmap="coolwarm", vmin=-v, vmax=v)
    ax[0].set_title(f"Residual @ {temps[worst]:g} °C (best 2-pt)")
    fig.colorbar(im, ax=ax[0], label="counts")
    comps = ["total", "column", "row", "pixel"]
    vals_mk = [1000 * dec[c] / resp[worst] for c in comps]
    ax[1].bar(comps, vals_mk)
    ax[1].set(ylabel="mK", title="FPN decomposition of residual")
    for x, v_ in enumerate(vals_mk):
        ax[1].text(x, v_, f"{v_:.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_fpn_decomposition.png", dpi=130)
    plt.close(fig)
    return dec, worst


# --------------------------------------------------------------------------
# Synthetic self-test data
# --------------------------------------------------------------------------

def make_synthetic(folder: Path, H=120, W=160, seed=0):
    """Microbolometer-ish synthetic stack: per-pixel gain/offset FPN, column
    stripes, mild per-pixel nonlinearity, shot+read noise, a few bad pixels."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    temps = np.arange(10, 101, 10.0)                  # 10..100 °C
    tint = 10.0
    gain = 1 + 0.05 * rng.standard_normal((H, W))
    offs = 300 * rng.standard_normal((H, W))
    col = 40 * rng.standard_normal(W)                 # column stripes in offset
    colg = 0.01 * rng.standard_normal(W)              # column stripes in gain
    nl = 1 + 0.02 * rng.standard_normal((H, W))       # per-pixel nonlinearity
    bad = rng.random((H, W)) < 5e-4
    import tifffile
    for T in temps:
        # scene "radiance": convex in T (Planck-like around LWIR)
        L = 5000 + 60 * T + 0.35 * T ** 2
        for rep in range(4):
            sig = (gain + colg[None, :]) * L + offs + col[None, :] \
                  + 0.002 * nl * (L - 8000) ** 2 / 100.0
            sig = sig + rng.normal(0, 8, (H, W))      # temporal noise
            sig[bad] = rng.choice([0, 16383], bad.sum())
            tifffile.imwrite(folder / f"bb_{T:05.1f}C_{tint:g}ms_{rep:02d}.tiff",
                             np.clip(sig, 0, 16383).astype(np.uint16))
    print(f"Wrote synthetic dataset to {folder} ({len(temps)} temps x 4 frames)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", type=Path,
                    help="folder with blackbody frames")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help="filename regex with named groups 'temp' and 'tint'")
    ap.add_argument("--out", type=Path, default=Path("./nuc_results"))
    ap.add_argument("--raw-shape", nargs=2, type=int, default=None,
                    metavar=("H", "W"))
    ap.add_argument("--raw-dtype", default="uint16")
    ap.add_argument("--k-max", type=int, default=8,
                    help="max number of multi-point calibration points")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate & analyze a synthetic dataset (self-test)")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        args.folder = args.out / "synthetic_data"
        make_synthetic(args.folder)
    elif args.folder is None:
        ap.error("provide a folder or use --synthetic")

    datasets = scan_folder(args.folder, args.pattern,
                           tuple(args.raw_shape) if args.raw_shape else None,
                           args.raw_dtype)

    summary = {}
    for tint, ds in datasets.items():
        if len(ds.temps) < 3:
            print(f"tint={tint} ms: <3 temperatures, skipping analysis.")
            continue
        print(f"\n=== tint = {tint:g} ms ===")

        pair_score, best_pair, resp = analyze_two_point(ds)
        (bi, bj), bs = best_pair
        n = len(ds.temps)
        naive = float(pair_score[0, n - 1])
        print(f"[Q1] 2-pt with endpoints ({ds.temps[0]:g}/{ds.temps[-1]:g} °C): "
              f"max residual NU = {naive:.0f} mK")
        print(f"[Q1] best pair: {ds.temps[bi]:g}/{ds.temps[bj]:g} °C "
              f"→ max residual NU = {bs:.0f} mK "
              f"({100*(1-bs/naive):.0f} % better than endpoints)")

        mp_results, _ = analyze_multi_point(ds, k_max=args.k_max)
        print("[Q2] multi-point (piecewise-linear), optimal placement:")
        for k in sorted(mp_results):
            r = mp_results[k]
            pts = ", ".join(f"{ds.temps[i]:g}" for i in r["idx"])
            print(f"      k={k}: max residual = {r['score']:.0f} mK  "
                  f"@ [{pts}] °C ({r.get('method','-')})")

        curves = {"2pt_endpoints":
                      to_mK(residual_nu(two_point_correct(ds, 0, n - 1),
                                        ds.bad_mask), resp),
                  f"2pt_best_{ds.temps[bi]:g}_{ds.temps[bj]:g}":
                      to_mK(residual_nu(two_point_correct(ds, bi, bj),
                                        ds.bad_mask), resp)}
        for k, r in mp_results.items():
            curves[f"{k}pt"] = r["curve"]

        rep = netd_report(ds, curves, resp)
        if "temporal_NETD_mK" in rep:
            tn = np.array(rep["temporal_NETD_mK"])
            print(f"[Q3] temporal NETD: {np.nanmedian(tn):.0f} mK (median over T)")
        best_curve = curves[f"2pt_best_{ds.temps[bi]:g}_{ds.temps[bj]:g}"]
        print(f"[Q3] spatial (FPN) NETD after best 2-pt: "
              f"median {np.nanmedian(best_curve):.0f} mK, "
              f"max {np.nanmax(best_curve):.0f} mK")
        kbest = max(mp_results)
        print(f"[Q3] spatial (FPN) NETD after {kbest}-pt: "
              f"median {np.nanmedian(mp_results[kbest]['curve']):.0f} mK, "
              f"max {mp_results[kbest]['score']:.0f} mK")

        dec, worst = make_plots(ds, args.out, pair_score, best_pair,
                                mp_results, resp)
        print(f"[+]  FPN decomposition @ worst T ({ds.temps[worst]:g} °C, "
              f"best 2-pt, in counts): total {dec['total']:.2f} | "
              f"column {dec['column']:.2f} | row {dec['row']:.2f} | "
              f"pixel {dec['pixel']:.2f}")

        rep["two_point"] = {
            "pair_score_mK": np.where(np.isnan(pair_score), None,
                                      np.round(pair_score, 1)).tolist(),
            "best_pair_C": [ds.temps[bi], ds.temps[bj]],
            "best_score_mK": bs,
        }
        rep["multi_point"] = {str(k): {"temps_C": [ds.temps[i] for i in r["idx"]],
                                       "max_residual_mK": r["score"]}
                              for k, r in mp_results.items()}
        rep["fpn_decomposition_counts"] = dec
        summary[f"{tint:g}ms"] = rep

    with open(args.out / "nuc_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nPlots and nuc_summary.json written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
