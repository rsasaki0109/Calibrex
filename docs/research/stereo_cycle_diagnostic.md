# Stereo round-trip and disparity-band diagnostic

## Frozen diagnostic protocol

Use all 12 distinct spent frames from the re-rectification comparison, with its
saved rotations, projection matrices, image remaps and SGBM settings/version.
Recompute left disparity and require exact equality of the restored depth with
its hashed saved array. Compute an independent reverse map by matching horizontally
flipped right/left images with the same positive disparity range, then flip back.
This is an additional image-based diagnostic, not a replacement for SGBM's existing
internal `disp12MaxDiff` check.

For each left match, sample right disparity at rounded `u_left - disparity`.
Absolute forward-minus-reverse disparity <= 1 pixel is consistent; > 1 is
inconsistent; absent/out-of-domain reverse matches are separately missing.
Verify remap support at both ends. No threshold fitting is performed.

Project LiDAR using the dataset reference into the newly rectified left camera,
z-buffer the nearest point per pixel and retain all finite stereo depths in the
existing 0.1–80 m interval. **Do not apply a reference-depth agreement gate or a
2,000-point cap.** Visibility from the right camera is not established by this
left-image z-buffer. Reference uncertainty, timing and occlusions remain factors.

Report all/consistent/inconsistent/reverse-missing populations in fixed disparity
bands [0,8), [8,16), [16,32), [32,64), [64,128) pixels. Depth difference is LiDAR z
minus stereo z; disparity difference is measured minus reference-predicted.
Report counts, medians and absolute-error 90th percentiles, using null for empty
sets. No optimizer, calibration verdict or production frontend is changed.

The diagnostic is committed before examining results. This is post-hoc development
on spent data; even reverse-consistent matches are not independent ground truth.
Low disparity amplifies pixel errors into metre errors, and different populations
must not be called a paired accuracy improvement.


## Results

Driver and protocol were frozen at `ae87b4f0`. All 12 frames completed and every
recomputed forward depth array exactly reproduced its saved re-rectified array,
including invalid pixels. No calibration fit was performed.

Of **183,329** stereo-supported LiDAR projections, **165,917 (90.5%)** pass the
independent 1 px round-trip check, **2,980 (1.6%)** fail it, and **14,432 (7.9%)**
have no valid reverse match. These are pooled points, not independent scenes.
Most removed support is reverse-unavailable rather than explicitly contradictory;
right-image borders/search range and occlusion contribute to missingness.

| Recording | Frame | Round-trip support retained | All median z (m) | Consistent median z (m) | All absolute-z p90 (m) | Consistent absolute-z p90 (m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0005 | frame-38 | 85.1% | -0.089 | -0.083 | 3.43 | 3.05 |
| 0005 | frame-76 | 91.3% | -0.082 | -0.074 | 2.93 | 2.22 |
| 0005 | frame-114 | 87.2% | -0.075 | -0.101 | 1.92 | 1.84 |
| 0002 | frame-93 | 90.0% | -0.047 | -0.038 | 3.50 | 2.92 |
| 0002 | frame-187 | 90.3% | -0.061 | -0.050 | 3.48 | 2.78 |
| 0002 | frame-281 | 90.1% | -0.071 | -0.060 | 3.32 | 2.75 |
| 0009 | frame-111 | 91.4% | -0.037 | -0.033 | 2.38 | 1.94 |
| 0009 | frame-223 | 89.9% | -0.069 | -0.056 | 4.11 | 3.32 |
| 0009 | frame-334 | 91.7% | -0.078 | -0.066 | 2.41 | 2.18 |
| 0011 | frame-58 | 92.9% | -0.057 | -0.046 | 2.93 | 2.68 |
| 0011 | frame-116 | 93.2% | -0.108 | -0.098 | 3.60 | 2.93 |
| 0011 | frame-174 | 92.4% | -0.098 | -0.082 | 3.73 | 3.14 |

Absolute-z p90 decreases in all 12 frames after restricting the population to
round-trip-consistent points; median magnitude decreases in 11/12. Yet every
consistent population still has negative median z residual and negative median
disparity residual (−0.294 to −0.109 px). Consistency therefore does not establish
correct depth. Both directions can agree on an incorrect match or visible surface.

### Disparity dependence among consistent points

The table gives ranges of **per-frame** statistics, not pooled quantiles. Band
counts are pooled across frames; detailed individual counts remain in the report.
These depth residuals use the dataset reference and omit a depth-agreement gate,
unlike earlier surface diagnostics. Large differences can include visibility,
scene motion, timing and reference effects as well as stereo measurement error.

| Disparity band (px) | Count | Range of frame median z (m) | Range of frame absolute-z p90 (m) |
| --- | ---: | ---: | ---: |
| [0, 8) | 6,100 | -6.709 to -3.533 | 9.339 to 42.866 |
| [8, 16) | 26,116 | -1.770 to -0.959 | 3.306 to 11.226 |
| [16, 32) | 55,641 | -0.363 to -0.142 | 0.835 to 2.093 |
| [32, 64) | 73,665 | -0.030 to 0.026 | 0.142 to 0.648 |
| [64, 128) | 4,395 | -0.008 to 0.061 | 0.064 to 5.683 |

The large far-range errors persist even with reverse consistency. Intermediate
32–64 px disparity has much smaller median differences, but high-disparity bands
still contain severe tails in some frames. This does not justify claiming a pure
stereo depth bias or deriving an offset from the reference.

### Decision

Independent reverse consistency is a plausible support-quality filter: it removes
about 9.5% of this projected population and reduces tails. It does **not** solve the
systematic residual by itself, and no downstream benefit has been measured. Keep
all production behavior and prior failed calibration verdicts unchanged.

The next controlled experiment should test the fixed 1 px reverse-consistency
mask on the same eight calibration cases, reporting support and both accepted
and raw outcomes, without tuning disparity cutoffs from these tables. The existing
uncertainty model already scales with depth; these observations warrant checking
its handling of visibility and heavy tails rather than assuming it lacks any
range weighting. Any distance-dependent confidence model needs separate validation.

[Complete diagnostic](../assets/stereo_cycle_20261029/diagnostic.json) and
[schema](../assets/stereo_cycle_20261029/diagnostic.schema.json) preserve hashes,
counts and statistics for all groups and bands. Empty groups have null statistics.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/diagnose_stereo_cycle.py \
  docs/assets/stereo_rerectification_20261029/comparison.json /tmp/stereo-cycle.json
```
