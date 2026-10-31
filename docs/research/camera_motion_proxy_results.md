# Motion-reference diagnosis does not resolve the three wrong surface regions

Motion audit frozen `7fb15ad9`; subsequent reference-assisted counterfactual frozen
`5bed804c` before its execution,2026-10-30. See
[motion protocol](camera_motion_proxy_protocol.md),
[motion plan](../assets/camera_motion_proxy_20261030/plan.json),
[all36 motion comparisons](../assets/camera_motion_proxy_20261030/evaluation.json),
[motion schema](../assets/camera_motion_proxy_20261030/evaluation.schema.json),
[counterfactual protocol](proxy_temporal_counterfactual_protocol.md),
[counterfactual plan](../assets/proxy_temporal_counterfactual_20261030/plan.json),
[18 reference-assisted diagnoses](../assets/proxy_temporal_counterfactual_20261030/evaluation.json)
and [schema](../assets/proxy_temporal_counterfactual_20261030/evaluation.schema.json).

The three persistent surface failures remain when temporal projection uses the
OXTS/calibration motion proxy instead of the stereoVO estimate. Thus motion
replacement alone is insufficient under this reference model. This does not
prove exact motion, establish true camera/LiDAR calibration or uniquely identify
stereo as the sole error source.

## Motion disagreement and fixed sensitivities

| Quantity across36 pairs | Median | Maximum |
| --- | ---: | ---: |
| Estimated versus nominal proxy rotation | .03515deg | .10140deg |
| Estimated versus nominal proxy translation | .02631m | .13856m |
| Same-index versus camera-time-interpolated proxy rotation | .00780deg | .14515deg |
| Same-index proxy translation change | .02659m | .08386m |
| Common clock shift ±10ms proxy rotation change (72 scenarios) | .00369deg | .02690deg |
| Common clock shift ±10ms proxy translation change | .000667m | .003180m |
| Omit camera/IMU lever-arm translation change | .002017m | .04472m |

Lever-arm removal does not change relative rotation beyond numerical precision.
Same-index substitution differs from a common clock shift: per-frame timestamp
phases can differ, so the larger same-index effect does not contradict the smaller
common-shift result. These fixed variants are sensitivities, not measured uncertainty
bounds or fitted clock corrections. No favorable variant is selected as truth.

For the six pairs around the three failed surfaces, nominal translation disagreement
is .01265–.03875m and rotation .00340–.04216deg. Larger disagreements occur elsewhere,
especially0015 (drive median .10663m translation), but low average disagreement
cannot exclude motion error at individual pixels or target calibration uncertainty.

## Motion-only counterfactual

The counterfactual deliberately uses target-extrinsic-derived camera motion for
EVALUATION ONLY. All rows are labeled reference_assisted. It preserves measured
surface masks, depths,1px temporal threshold and original motion-support flags;
unsupported0013/frame107 remains empty. No reference-derived selector is deployed.

| Persistent failed center | Estimated-motion temporal normal agreement | Proxy-motion counterfactual |
| --- | ---: | ---: |
| 0005/frame38 | 45.24% | 45.20% |
| 0005/frame114 | 76.02% | 76.21% |
| 0009/frame111 | 76.98% | 76.54% |

The overall declared quality/support count changes13/18→14/18 in the counterfactual,
while all three major failures remain below80%. This is diagnostic attribution,
not a deployable quality or calibration gain. The original measurement-only
verdict remains13/18; no baseline or failed frame is relabeled.

## Provenance and limits

OXTS provides an IMU world-pose proxy. Camera2 conversion composes projection
baseline shift, rectification, camera0/Velodyne and Velodyne/IMU calibrations.
Camera2/Velodyne matches the existing reference within1e-8 for all six drives.
The target extrinsic is consumed only in this after-estimation evaluation, so the
proxy cannot be advertised as independent real motion ground truth. Calibration,
OXTS error, interpolation and synchronization are not assigned a validated
covariance by this experiment. All query times are bracketed; none extrapolate.

Four focused tests, repository Ruff and strict documentation build pass. Tests
check integer-nanosecond interpolation, no-extrapolation, source/target convention,
rotational lever arm, metric PnP units and missing-depth failures. Validation checks
all36 motion comparisons,18 counterfactual records, source/input hashes, explicit
reference labels, unchanged baseline populations and retained unsupported flags.
No images downloaded, motion fits rerun, production settings or SOTA claims change.

Next move error attribution to the underlying stereo geometry: inspect image
matching ambiguity and surface reconstruction against the fixed reference regions,
without using reference masks for selection or retuning SGBM on these outcomes.
Review prior frontend experiments before choosing another estimator. The motion
proxy result rules out neither LiDAR-normal error nor calibration/timing effects;
new evidence must distinguish them rather than adding another consistency filter.
