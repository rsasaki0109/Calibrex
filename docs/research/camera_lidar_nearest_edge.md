# Camera–LiDAR: bounded nearest-image-edge distance

This is a post hoc development ablation on the same fourteen reacquired spent
KITTI inputs. It tests whether bounded distance to image ridges reduces the
incorrect matches supported by the original intensity-and-spill objective.
It is not a new SOTA audit, and does not change estimator defaults or thresholds.

## Fixed method

Use the original image downsampling, Gaussian smoothing (sigma 1 pixel), Sobel
gradients and 99th-percentile gradient normalization. Retain gradient ridges by
nonmaximum suppression along the local gradient and a fixed normalized magnitude
threshold of 0.2. Exclude the image border. Compute the Euclidean distance to
the nearest retained ridge and use `exp(-0.5 * (distance / 2)^2)` as the reward,
setting it to zero beyond 6 downscaled pixels. These constants were fixed before
running the experiment, without selecting against reference errors or held scores.

This replaces the complete image response, including its background subtraction;
it is one objective-family ablation, not evidence isolating the contribution of
each ingredient. Bilinear interpolation samples the distance-derived reward field.
There is no orientation penalty, additional baseline restart, or match-count
normalization. All original point weights remain in the denominator, even for
points projecting outside the image or beyond the distance limit.

Nearest-ridge matching can still be many-to-one, can match texture rather than
physical depth boundaries, and does not enforce visibility or temporal identity.
The 6-pixel cutoff is on field samples; bilinear interpolation smooths its cell
boundary. A narrow field may also reduce the convergence basin. These limitations
are reasons to inspect fitted poses as well as detection counts.

Source `a19f1a00` was frozen in an isolated worktree. Both arms retain training-only
initialization, multi-start nuisance refitting and all original search settings.
Four worker processes each run isolated input pairs. Every original arm must
reproduce the previous training-only results within absolute 1e-9 before its
changed arm starts. All fourteen conditions and any failures are retained.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. python -m tools.kitti_camera_lidar_boundary_batch \
  /path/to/capture --output /path/to/new/output --nearest-edge
```

Forty-two correlated controls from two drives are not forty-two independent
observations. Training-fold disagreement is measured against dataset reference
calibration, not ground truth. Worker timings include concurrency effects and
are not a repeated performance benchmark.

## Results and decision

**Do not promote.** All fourteen original-arm parity checks pass within absolute
1e-9 and all paired inputs complete. Detection coverage decreases without
recovering any controls on 0002.

| Diagnostic | Original scalar | Nearest ridge |
| --- | ---: | ---: |
| Detected controls, 0005 | 19/21 | 13/21 |
| Detected controls, 0002 | 0/21 | 0/21 |
| Total detected controls | 19/42 | 13/42 |
| Worst active-axis fold reference disagreement among detected controls | 0.350 deg | 0.246 deg |
| Worst disagreement on the same 11 controls detected by both arms | 0.350 deg | 0.200 deg |
| Target 0002 pitch +3 control t | 0.77397 | -1.76402 |
| Target pitch +3 detected | No | No |
| Sum of profile wall times (concurrent workers) | 177.9 s | 183.7 s |

Eight previously detected controls are lost and two are gained. The smaller
maximum reference disagreement therefore does not establish better overall
accuracy or a superior detector. The shared eleven controls are also a selected
subset, not an independent calibration evaluation.

For the targeted pitch input, all six fitted active-axis reference disagreements
are within 0.052 deg, yet only two held-out folds beat both nuisance-refitted
controls. Its mean objective drop is negative (-0.03365). A small fitted pitch error
can coexist with incorrect poses preferred by the held-out image objective.
The [full-pose diagnostic](camera_lidar_pitch_matches.md) subsequently shows
baseline rotation norms of 0.638–4.717 deg: small pitch error does not establish
a nearly correct full rotation. This is direct evidence that a bounded nearest-image-edge score does
not by itself resolve the ambiguity on this input; it does not establish which
physical surfaces caused it.

Keep the default scalar objective. Before another objective variant, inspect
which image structures the wrong poses match and whether those correspondences
persist across frames. Do not tune the distance cutoff or ridge threshold on
these results as if it were independent validation.

The [complete report](../assets/nearest_edge_20261028/report.json),
[summary](../assets/nearest_edge_20261028/summary.json) and
[hash manifest](../assets/nearest_edge_20261028/manifest.json) retain every input,
arm, fold, image-ridge count, source digest and changed detection flag.
