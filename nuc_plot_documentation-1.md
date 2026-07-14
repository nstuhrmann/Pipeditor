# NUC Analysis — Plot Documentation

Output of `nuc_analysis.py`. All figures are generated per integration time (filename prefix `tint<value>us_`). Residual and noise values are given in mK, converted from counts via the locally measured responsivity dS/dT (central differences of the frame-mean signal over the blackbody temperatures).

## Noise scale conventions

The tool reports temporal noise on three scales, all derived from the same single-frame-equivalent noise σ₁. σ₁ is estimated per temperature from the repeat files: with m files that are Nᵢ-frame averages, σ₁² = Σᵢ Nᵢ(fᵢ − μ̂)²/(m−1) per pixel, pooled over good pixels in variance domain. Which scale a value uses is determined by what it is compared against:

| Scale | Value | Where it appears |
|---|---|---|
| **Single frame** (N=1) | σ₁ | Dotted "temporal NETD (single frame)" line; `temporal_NETD_single_frame_mK` in JSON. The camera spec number. |
| **Single acquisition** (one file, N=64/128) | σ₁·√((1/N₁+1/N₂)/2) | Everything from the difference analysis: orange bars, the entire diff figure set, diff log lines and JSON (`"scale"` field). |
| **Combined mean** (all frames, ΣN) | σ₁/√ΣN | Dashed "measurement noise floor" line; gray pixel-null bar in the residual decomposition figures; `noise_floor_mean_frames_mK` in JSON. The right reference for anything measured on the averaged frames. |

All *blue* quantities (residual FPN after correction) are measured on the **N-weighted mean frames** and therefore contain the combined-mean noise level as an irreducible floor.

## Response curve (`*_response.png`)

**Left:** frame-mean signal versus blackbody temperature — the radiometric transfer curve at this integration time — with the straight line through the two endpoint temperatures. **Right:** measured curve minus that endpoint line.

Values come directly from the spatial means of the averaged frames (bad pixels masked); no noise reference is shown because at ΣN-frame averaging the curve's uncertainty is negligible against its structure. The deviation shown right is the structure a two-point NUC cannot remove (Planck curvature plus detector nonlinearity); its magnitude over responsivity predicts the 2-pt residual scale, its shape predicts where optimal multi-point calibration temperatures cluster.

## Fit curves (`*_fit_curves.png`)

**Left:** raw uncorrected response — spatial mean signal vs temperature with a ±1σ band and a lighter percentile band showing the pixel-to-pixel distribution (the FPN to be corrected), plus sample pixels from the low/median/high end of the gain distribution. The bands are *spatial* spread across pixels of the mean frames, not temporal noise. Band width relative to mean is the raw FPN; widening with temperature indicates gain FPN, constant width offset FPN.

**Right:** the correction mappings of those sample pixels — measured calibration samples, the piecewise-linear map through its selected points, and the best-order polynomial — drawn so the two fits are distinguishable. Disagreement between the fits concentrates where response curvature is strongest.

## Two-point pair score (`*_2pt_pair_score.png`)

Heatmap over all calibration pairs; each cell is the worst-case residual non-uniformity (mK) over the full range for that pair, the star marks the optimum. Each cell value is the max over temperature of the spatial std of the corrected **mean frames** — so every cell contains the combined-mean noise floor in quadrature. The valley of good pairs typically lies inside the range: interior points balance the bow-shaped nonlinearity error better than the endpoints.

## Residual FPN vs temperature (`*_residual_vs_T.png`)

The central result. Both panels: residual spatial non-uniformity (mK, log scale) versus blackbody temperature, measured as the spatial std of the corrected **mean frames** (bad pixels masked), with gaps at the respective calibration temperatures (exact by construction). Two black reference lines: dashed = measurement noise floor (**combined-mean scale**, σ₁/√ΣN — nothing below it is resolvable with this dataset); dotted = temporal NETD (**single-frame scale**, σ₁ — shown for context, not a bound on the curves).

**Left:** 2-pt (endpoints and best pair) plus the piecewise-linear family for each k with optimal point placement. **Right:** the polynomial family for every order, leave-one-out evaluated, best order highlighted.

Reading: the bow of the 2-pt curves; the piecewise family collapsing onto the floor with growing k; the polynomial arc — low orders miss real structure at the range ends, mid orders hug the floor, high orders blow up near the edges (Runge oscillation amplified by LOO). Polynomial endpoint values are in-sample and optimistically low; curves at ~1.3× the floor correspond to genuine FPN of ~0.8× the floor, since FPN and floor add in quadrature.

## Convergence (`*_convergence.png`)

Worst-case residual (max over all temperatures, mK, measured on the mean frames) versus model complexity, both methods on a common axis of free parameters per pixel (k points / order+1), plus the median **combined-mean** noise floor. Answers "how many calibration points": where the piecewise curve crosses the floor, further points are unverifiable. Piecewise improves monotonically; the LOO polynomial has a minimum at the sensor's effective nonlinearity order and degrades beyond it.

## Residual map + FPN decomposition (`*_fpn_decomposition_*.png`)

Generated for 2-pt endpoints, best 2-pt pair, and (with `--maps best`) the highest piecewise k and best polynomial order — or every k and every order with `--maps all`. Each is evaluated at that correction's own worst temperature.

**Left:** the 2D residual of the corrected **mean frame** as deviation from its spatial mean, in mK (color scale ±3σ, or ±cap with `--cap-mk`). Column stripes, row banding, blotches, or salt-and-pepper point to different physical causes.

**Middle:** the column profile of that residual (mean of each column, mK) with a gray ±1σ band showing what pure white pixel noise would leak into column means (pixel component/√H) — columns outside the band carry real stripe structure.

**Right:** grouped bar chart per component (total, column, row, pixel):

- **Blue (measured):** the sequential decomposition of the residual — column pattern extracted first (std of column means), then row pattern from the remainder, pixel = what is left; total² ≈ column² + row² + pixel². Measured on the mean frame → contains the combined-mean noise floor.
- **Hatched gray (white/temporal-noise level):** for column and row, the leakage of white pixel noise into the stripe estimates (pixel/√H, pixel/√W, self-calibrated from the same frame); for pixel, the measurement noise floor (**combined-mean scale**; omitted if no repeat files exist).
- **Orange (temporal instability, only with `--diff-analysis`):** the same component estimated from the difference frames at this temperature, at **single-acquisition scale** — the direct attribution test: blue ≈ orange means the residual component is explained by temporal instability, blue ≫ orange means static-but-unmodeled structure or drift on timescales longer than the file pair.

Annotation above the blue bar: measured value, and below it the quadrature-corrected genuine FPN √(measured² − null²) — bar heights cannot be subtracted linearly.

## Temporal noise components (`*_diff_noise_components.png`, with `--diff-analysis`)

For each temperature with exactly two files, the difference f₁ − f₂ cancels ALL static structure exactly; what remains is purely non-stationary content. All values in this and the next figure are at **single-acquisition scale** (difference std / √2).

**Left:** column, row, and pixel components of the difference versus temperature (mK, log scale), with the white-noise null curves (pixel/√H, pixel/√W). **Right:** the measured/null ratios; the dashed line at 1 is the white-noise expectation. Ratios ≫ 1 indicate temporally correlated stripe noise. The temperature dependence separates mechanisms: flat excess → column *offset* drift (fixable with a signal-independent column update); excess growing with signal → column *gain* instability (needs a two-reference update or scene-based estimation).

## Difference map (`*_diff_worst_column_map.png`, with `--diff-analysis`)

Shown at the temperature with the largest column excess. **Left:** the difference frame in mK at **single-acquisition scale**. **Middle:** its column profile with the ±1σ white-noise band — reveals whether the excess is broadband or a few rogue columns. **Right:** bar chart with semantic colors: orange = temporal components (total, column, row); solid gray = the pixel component, which is the white-noise reference from which the hatched leakage nulls are derived (hence no null of its own); annotations give the quadrature-corrected correlated part ("corr.").

Note on cross-figure comparison: the gray pixel bar here (single-acquisition scale) and the gray pixel-null bar in the residual figures (combined-mean scale) derive from the same σ₁ but differ by √(ΣN·(1/N₁+1/N₂)/2) — a factor 1.5 for an N=64/128 pair. This is intentional: each gray bar is the noise level relevant to the bar standing next to it.

## Display cap (`--cap-mk`)

Optional, display-only: maps show −cap…+cap, bar charts and residual/score axes 0…cap. Data, logs, and JSON are unaffected. A fixed cap makes maps and bars directly comparable across figures, orders, integration times, and runs; a value of 2–3× the target residual makes "above spec" instantly visible as saturation.
