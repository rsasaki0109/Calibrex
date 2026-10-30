# Camera-only metric motion passes controlled geometry and most internal checks

Frozen `271b9e75`, evaluated2026-10-30. See
[protocol](stereo_metric_motion_protocol.md),
[plan](../assets/stereo_metric_motion_20261030/plan.json),
[18 controlled cases](../assets/stereo_metric_motion_20261030/evaluation.controls.json),
[all36 real pairs](../assets/stereo_metric_motion_20261030/evaluation.json),
[schema](../assets/stereo_metric_motion_20261030/evaluation.schema.json) and
[verification](../assets/stereo_metric_motion_20261030/verification.json).

All18 known-motion controls pass the frozen<.2deg/<.05m gate. Maximum rotation
error .04315deg, maximum translation error .005561m. These tests establish the
metric scale/frame convention and robustness of PnP on generated correspondences
with depth/pixel noise and outliers; they do not validate the full real image
matching/stereo pipeline against external motion truth.

After controls passed, all36 real pairs were evaluated. **34/36 pass the declared
internal support/cycle checks**. Metric scale comes from camera stereo calibration;
no LiDAR points, target camera/LiDAR transform or OXTS camera-motion conversion
enters estimation. Stereo estimates and cycle checks share measurements, so a
passed check is not independent real-motion accuracy or adoption evidence.

| Recording | Internal support/cycle passes | All pairs retained |
| --- | ---: | ---: |
| 0005 | 6 | 6 |
| 0009 | 6 | 6 |
| 0011 | 6 | 6 |
| 0013 | 4 | 6 |
| 0015 | 6 | 6 |
| 0002 | 6 | 6 |

Failed0013 pairs106→107 and107→108 have inlier ratios below50% in both directions:
298/644 and261/610;267/599 and280/574 respectively. Spatial support and cycle
alone would pass; the ratio gate is retained. No threshold or frame is changed.
All eight previous essential-matrix screen failures are evaluated; stereo-PnP
uses a different measurement model and need not share its cheirality limitations,
especially for little translation. This is not a paired accuracy claim.

Cycle rotation residuals span .000081–.08383deg, median .03353deg; translation
.0000295–.06907m, median .01547m. All36 cycles pass the numeric cycle bounds,
including the two low-ratio failures. A mutually consistent wrong depth scale or
motion bias can remain invisible to forward/reverse agreement.

## Input and implementation checks

All54 unique stereo-frame depth arrays were computed using pinned grayscale SGBM,
re-rectification and restoration. All18 central arrays exactly reproduce the
archived SGBM outputs. Depth paths/hashes are retained in each result and verified;
arrays reside under /workspace/work/stereo-metric-motion-20261030. Both PnP directions,
inlier identities, counts, cells and target-from-source transforms remain in the
report. The image matching is on histogram-equalized images, stereo on raw grayscale.

Three focused tests, repository Ruff and strict documentation build pass. Tests
verify metric translation direction, insufficient/missing-depth failure handling
and blank-image zero support. Integration checks validate schema, all source/input
hashes,54 depth hashes,18 controlled outcomes, center parity and36-case completion.
No core/default changes, new downloads, extrinsic fit, ICP rerun or SOTA claim.

## Next measurable step

Freeze a temporal surface-consistency diagnosis, beginning with independently
specified controlled camera motion and visibility/occlusion handling. On real
frames use only the estimated camera-to-camera transforms, preserving both failed
motion pairs and out-of-view/occluded/unsupported pixels explicitly. Score surface
quality with reference/LiDAR only AFTER selection; never use the target extrinsic
to produce a temporal mask. Independent metric real-motion truth remains absent,
and shared stereo errors must be disclosed. Retain every original surface failure;
temporal self-consistency is an extra check, not proof of correct geometry or
calibration recovery. No real-data temporal filter is run in this stage.
