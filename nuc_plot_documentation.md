# NUC Analysis — Plot Documentation

Output of `nuc_analysis.py`. All figures are generated per integration time (filename prefix `tint<value>us_`). Residual values are given as spatial non-uniformity in mK, converted from counts via the locally measured responsivity dS/dT.

## Response curve (`*_response.png`)

**Left panel:** frame-mean signal versus blackbody temperature — the camera's radiometric transfer curve at this integration time, with the straight line through the two endpoints for reference. **Right panel:** the difference between the measured curve and that endpoint line.

This deviation is the direct cause of two-point residual error: a two-point NUC forces every pixel onto a straight line, so the bow shape shown here — Planck radiance curvature plus detector nonlinearity — is exactly the structure a 2-pt correction cannot remove. Its magnitude in counts, divided by responsivity, predicts the scale of the 2-pt residual curve; its shape (where the curvature concentrates) predicts where optimal multi-point calibration temperatures will cluster.

## Fit curves (`*_fit_curves.png`)

**Left panel:** the raw, uncorrected response of the array — spatial mean signal versus temperature with a shaded ±1σ band and a lighter percentile band showing the pixel-to-pixel distribution, i.e. the non-uniformity the correction must remove, plus a few sample pixels drawn from the low/median/high end of the gain distribution. The width of the band relative to the mean is the raw FPN magnitude; whether the band widens with temperature tells you how much is gain FPN (widening) versus offset FPN (constant width).

**Right panel:** the actual correction mappings of those sample pixels — for each, the measured calibration samples, the piecewise-linear map through its selected points, and the best-order polynomial map, plotted so the two fits are visually distinguishable. Here you see *how* the two methods differ: the polynomial bends smoothly through all samples, the piecewise version follows chords, and any disagreement between them is concentrated where the response curvature is strongest.

## Two-point pair score (`*_2pt_pair_score.png`)

A symmetric heatmap over all calibration-temperature pairs; each cell is the worst-case residual non-uniformity (mK) over the full measured range when that pair is used for the 2-pt correction. The star marks the optimum.

The structure to read: the valley of good pairs typically lies inside the range rather than at the corners — endpoints are intuitive but suboptimal, because interior points balance the bow-shaped error between the middle and the ends (Chebyshev-like placement). The depth of the valley versus the corner value quantifies how much a smarter pair choice buys without any extra calibration effort.

## Residual FPN vs temperature (`*_residual_vs_T.png`)

The central result. Both panels: residual spatial non-uniformity in mK versus blackbody temperature, log scale, with gaps at the respective calibration temperatures (exact by construction) and two black reference lines — the measurement noise floor of the averaged calibration frames (dashed; nothing below it is resolvable with this dataset) and the single-frame temporal NETD (dotted; the camera's temporal noise for context).

**Left:** two-point (endpoints and best pair) and the piecewise-linear family for each k with its optimal points. **Right:** the polynomial family for every order, leave-one-out-evaluated, best order highlighted.

What to look for: the bow of the 2-pt curves; how quickly the piecewise family collapses onto the floor as k grows; and the characteristic polynomial arc — too-low orders miss real structure at the range ends, mid orders hug the floor, high orders blow up near the edges (Runge oscillation amplified by LOO). Endpoint values of the polynomial curves are in-sample and optimistically low, so the fan-out below the floor at the extremes should not be read as physical.

## Convergence (`*_convergence.png`)

The summary of the multi-point question: worst-case residual (max over all temperatures, mK) versus model complexity, with both methods on a common x-axis of free parameters per pixel (k points for piecewise, order+1 for the polynomial), plus the median noise floor.

This plot answers "how many calibration points do I need": read off where the piecewise curve crosses the floor — beyond that, more points are unverifiable effort. The comparison of the two curves shows the methodological result: piecewise improves monotonically, the LOO polynomial has a minimum at the effective nonlinearity order of the sensor and degrades beyond it. The location of that minimum is itself a characterization of the camera.

## Residual map + FPN decomposition (`*_fpn_decomposition_*.png`)

**Left:** the 2D residual image after the named correction, at that correction's own worst temperature, as deviation from the spatial mean in mK (±3σ color scale). This is where the abstract σ numbers become diagnosable: column stripes, row banding, blotches (low-spatial-frequency drift, often thermal gradients across the FPA), or salt-and-pepper (per-pixel residual/noise) each point to different physical causes and different remedies.

**Right:** the quantitative split of the same frame into total, column-stripe, row-stripe, and pixel-random std components in mK. Column and row components are what calibration NUC handles worst and scene-based destriping handles best, so this bar chart tells you whether a downstream column filter is worth adding; if pixel-random dominates and sits near the noise floor, the correction is essentially done.

With `--maps all`, this figure is generated for every piecewise k and every polynomial order, so you can watch which component each added degree of freedom actually removes.
