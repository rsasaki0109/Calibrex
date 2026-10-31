# Camera–LiDAR: preserve the geometry of neighboring matches

This post hoc development experiment adds a direction-and-spacing constraint to
the previous local contour graph. It first screens saved incorrect poses, then
runs the fixed fourteen-input paired optimization experiment if the screen passes.
It uses spent data only and establishes no new SOTA claim.

## Fixed geometric constraint

Keep the prior mutual four-neighbor graph within 0.75 m and the same image ridge
components and six-pixel correspondence limit. For each linked pair, let `dp` be
the displacement between projected points and `dq` the displacement between their
matched ridge pixels. Multiply the same-component agreement by
`exp(-0.5 * ||dq - dp||^2 / (1 + 0.25 * ||dp||)^2)`. Distances are in downscaled
pixels. Average over the fixed neighbors, keeping isolated points neutral and
unknown component matches at zero. Preserve original point weights and the full
score denominator.

The one-pixel floor accommodates pixel quantization; the relative tolerance is
fixed at 25%. These settings were committed before the screen and not tuned after
seeing it. The constraint penalizes collapse, reversed direction and distorted
spacing. It remains invariant to a common translation of all matched points and
cannot alone distinguish identical parallel boundaries. Connected components are
not semantic object identities; 3D neighbors can also cross physical boundaries.

## Fixed-pose screen

Source `4846dcae` re-scores all saved poses in the six pitch folds and eleven
prepared frames, without fitting anything. The original scores reproduce within
absolute 1e-9. The component-only score also reproduces the previous contour
screen within 1e-9, isolating the additional geometric penalty.

| Wrong saved control beats the reference | Nearest ridge | Component only | Component + geometry |
| --- | ---: | ---: | ---: |
| Pooled folds | 5/6 | 5/6 | 1/6 |
| Individual frames | 10/11 | 8/11 | 4/11 |

This passes the prior screen's necessary condition (fewer than five failed folds),
not an adoption gate. It warrants checking whether optimization finds new bad
solutions under the changed objective.

## Paired pose-search experiment

Source `3132716c` freezes a batched implementation with the same penalty. All 24
screen poses match the original per-frame implementation within 5.56e-17 after
pooling. The related 25 tests pass, including collapse/reversal, unknown labels,
isolated support, scalar contribution accounting and the existing profile-search
checks.

Both arms use all fourteen reacquired inputs, training-only initialization and
multi-start nuisance refitting with unchanged search settings. The control arm is
the original scalar objective (19/42 in the earlier experiment), and must reproduce
its prior axis reports within absolute 1e-9. The changed arm uses nearest-ridge
reward plus the geometric graph penalty; extra baseline restarts are disabled.
This is a full objective-family comparison, not an attribution of the difference
from scalar to the geometric term alone. Four isolated workers run the pairs.

Every axis now records all three components and the norm of each training-fold
rotation error, as well as the active-axis error. This addresses the preceding
diagnostic's warning: small pitch error can coexist with large roll/yaw errors.

Reproduce from the frozen source:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. python -m tools.kitti_camera_lidar_boundary_batch \
  /path/to/capture --output /path/to/new/output --match-geometry
```

## Results and decision

**Retain as an experimental candidate; do not promote to default.** All fourteen
original-arm parity checks pass within absolute 1e-9, and all paired inputs
complete. The screen's improvement partly survives pose search, but detection
coverage does not improve and some existing detections are lost.

| Diagnostic | Original scalar | Neighbor geometry |
| --- | ---: | ---: |
| Detected controls, 0005 | 19/21 | 19/21 |
| Detected controls, 0002 | 0/21 | 0/21 |
| Total detected controls | 19/42 | 19/42 |
| Worst active-axis error among detected folds | 0.350 deg | 0.250 deg |
| Worst full rotation error among detected folds | 0.394 deg | 0.296 deg |
| Target pitch +3: worst full rotation error across its six folds | 2.803 deg | 0.354 deg |
| Target pitch +3 control t | 0.77397 | 1.53886 |
| Target pitch +3 detected | No | No |
| Worst full rotation error across all axis folds, including undetected | 13.254 deg | 12.067 deg |
| Sum of profile wall times (concurrent workers) | 172.2 s | 861.6 s |

Two controls are gained and two lost; the unchanged total is not identical
coverage. On 0005 roll +3, pitch and yaw controls are gained; on 0005 pitch +3,
roll and yaw controls are lost. There are seventeen controls detected by both methods. On that common
subset, the maximum active-axis fold error also decreases from 0.350 to 0.250 deg.
These selected-subset diagnostics do not establish an overall accuracy claim.
The large all-fold errors show that the method remains unreliable outside the
reported detected subset. Timing includes concurrency and is not a repeated
benchmark, but the added graph computation has a substantial cost.

For target pitch +3, the fitted baseline rotations are closer to the reference,
but only three held-out folds beat both refitted controls, versus four originally.
The t statistic rises yet remains below the unchanged 2.5 threshold. Better
reference agreement at selected fitted poses does not prove that a displacement
is distinguishable when the other rotation axes can compensate.

Keep this fixed implementation for further development, rather than treating the
screen alone as success or tuning tolerances on its outcomes. The next useful
analysis is the remaining nuisance-refitted counterexamples under the new score,
including the controls that were lost. No fresh evaluation, new SOTA claim,
threshold relaxation or default change occurred.

The [complete paired report](../assets/match_geometry_20261028/report.json),
[summary](../assets/match_geometry_20261028/summary.json),
[fixed-pose screen](../assets/match_geometry_20261028/fixed-pose-screen/report.json)
and [hash manifest](../assets/match_geometry_20261028/manifest.json) retain every
condition, pose error vector, failure and changed detection flag.
