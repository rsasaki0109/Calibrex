# KITTI fixed spatial weights

**Post hoc development; this proposal is not adopted.** The experiment asks
whether reducing concentration in image regions restores the pitch evidence
lost by [nuisance-profiled controls](kitti_camera_lidar_profile_controls.md).
It uses the same two spent drives and fourteen cached inputs. No fresh sensor
data was acquired, and no audit, threshold, default or standing changes.

## Fixed weighting rule

The opt-in `--spatial-balance` assigns each prepared LiDAR point to a 2x2 image
tile at the supplied candidate, using the core objective's scale/half-pixel
coordinates. Each occupied candidate-inside tile receives an equal share of
the frame's original inside weight. Relative point weights within a tile are
preserved. Candidate-outside margin points keep their original weights and
remain available to enter the image under a trial pose. Every point remains;
each frame's total weight is preserved within float32 rounding.

Assignments and weights stay fixed through search. Trial poses do not move
points between weighting groups, and leaving the image still scores zero
against a fixed denominator. Deterministic preprocessing is applied separately
to each training and held-out frame; no score, fitted reference error or
injection label selects a tile or its weight. No sparse-tile cutoff or new
control threshold is tuned. Sparse occupied tiles can consequently amplify
individual points; this rule makes no claim that their evidence is reliable.

```bash
python -m tools.kitti_camera_lidar_profile_probe SAVED_SNAPSHOT \
  --conditional-report PREVIOUS_CONDITIONAL_REPORT.json \
  --multi-start --spatial-balance --output NEW_DIRECTORY
```

The tool records tile point counts and original/balanced weight totals, the
outside support, source hashes and the unchanged optimizer options. The
existing pooled conditional initialization is retained, so these remain local
jackknife diagnostics rather than nested validation folds. Refitted fold poses
can differ from the unbalanced model; input/settings parity is checked, not
baseline-pose parity across different objectives.

## Retained input-assumption failure

The initial source, committed at `50c77d37`, incorrectly required every prepared
point to be inside the candidate image. The existing preparation deliberately
retains an image margin, so **all fourteen inputs failed before any axis probe
completed**. This is an implementation failure, not a measured calibration
result. The original reports, schemas, producer and exact sources remain in
[attempt 1](../assets/kitti_camera_lidar_v3/spatial_weights/attempt_1/report.json).

Source `67ff8c5d` preserves margin support and its weight, handles frames with
no candidate-inside points, and adds regression tests for a margin point that
enters the trial image. It was committed before the retry. The retry uses a
new output directory, pins the failed report's SHA-256, and reuses the exact
same NPZ inputs without downloading data or weakening control gates.

## Retry results and decision

All fourteen retry inputs and 42 axis-control comparisons completed without
errors. The [retry report](../assets/kitti_camera_lidar_v3/spatial_weights/attempt_2/report.json)
pins each previous unbalanced multiple-start report and verifies the same
snapshot, NPZ digest, conditional receipt and optimizer options. Source bytes
and Git HEAD stayed unchanged through each attempt.

| Drive | Unbalanced multiple-start controls detected | Spatially balanced controls detected |
| --- | ---: | ---: |
| 2011_09_26_0005 | 20/21 | 12/21 |
| 2011_09_28_0002 | 0/21 | 0/21 |
| Total | 20/42 | 12/42 |

These are held-out control outcomes, **not native verdicts, observed-axis
coverage or calibration accuracy**. There is no jackknife-std gate in these
counts. Pitch +3 on 0002 remains undetected. On 0005, even baseline yaw loses
its control, and the yaw +3 run's yaw control also fails. The known 0002
roll +3 false control remains rejected, but the unbalanced multiple-start
method already rejected it. Spatial equalization therefore does not solve
the selected detection failure and loses useful control evidence on 0005.
No threshold is relaxed to recover these outcomes, and this weighting rule
is **not recommended or promoted** from these results.

The [publication manifest](../assets/kitti_camera_lidar_v3/spatial_weights/manifest.json)
binds both attempts, all 28 per-input receipts, schemas and exact producer/source
capsules, about 1.05 MB excluding the manifest. Only numerical support/evidence
is published; images, clouds, bags and NPZ payloads remain local. Tests validate
source/input bindings, outside-point preservation, per-frame weight conservation,
equal occupied-tile mass and the failed pitch/yaw controls. The failed first
attempt is retained alongside the retry.

Capture progress stays **2/55 spent drives**. Further evidence-model development
must preserve turn sensitivity rather than count additional abstention as
accuracy. This negative ablation does not justify a new SOTA claim or gates;
the remaining 53 drives, exposure audit and independent frozen preregistration
are still pending.
