# INS-LiDAR registration-quality development

This is **post hoc development**, following the refuted
[KITTI + Boreas rotation audit](../benchmarks/ins_lidar_rotation_v1.md).
It uses the already-spent first 600 scans of KITTI `2011_09_29_0071`.
No new evaluation data was acquired, no audit gate was changed, and this
experiment establishes no new SOTA standing.

## Question and comparison

The odometry accumulated transforms from all 599 scan registrations, including
two that did not converge. The hand-eye motion builder previously had no way
to exclude relative motions crossing those edges. Could these observations
explain why no rotation axis cleared the frozen 0.1 degree uncertainty ceiling?

Both fits use the **same LiDAR-only odometry trajectory**, computed once, with
all other settings unchanged. The baseline includes every motion; the candidate
excludes a motion if any registration edge between its endpoints did not
converge. It retains later motions whose intervals contain only converged edges.
The two failed edges are zero-based indices **283 and 316** (283 to 284 and
316 to 317). All registration counts, including failures, remain in the result.
The reference extrinsic is used only for post-fit comparison.

## Result

| Quantity | Baseline | Exclude failed edges |
| --- | ---: | ---: |
| Training motions | 115 | 112 |
| Held-out motions | 53 | 52 |
| Estimated rotation axes | 0/3 | 0/3 |
| Roll reported uncertainty (deg) | 0.378174 | 0.377499 |
| Pitch reported uncertainty (deg) | 0.226495 | 0.227188 |
| Yaw reported uncertainty (deg) | 1.225164 | 1.234833 |
| Roll difference to reference (deg) | +0.013378 | -0.037199 |
| Pitch difference to reference (deg) | +0.558456 | +0.543268 |
| Yaw difference to reference (deg) | -1.975854 | -1.850560 |

The baseline reproduces the original three rotation uncertainties and reference
differences exactly. Excluding four crossing motions does not make any rotation
observable. The small changes do not justify adopting this candidate as the
standard method. All three reference differences above concern **unobservable**
axes; they are not new scored accuracy results. The dataset reference is an
estimate, not independent metrology.

Nonconvergence alone does not explain the failure. This comparison does not
establish that the remaining registrations are unbiased or that later
registrations are unaffected by earlier initialization errors. Further work
needs to examine motion errors and stability across time blocks, including
coupling with the clock and reference-scale nuisance parameters.

## API and reproducibility

The experimental policy is available through
`InsLidarRunOptions(exclude_failed_registrations=True)` in both KITTI and Boreas
adapters. Its default is **False**, preserving the existing method. New artifacts
record the chosen policy in `options`; the artifact schema remains v0.1.

`build_sensor_motions(..., registration_valid=mask)` accepts one boolean per
consecutive pose edge. It checks every edge inside a motion, rejects malformed
masks, and retains block boundaries. An all-valid mask produces exactly the
original motions. Omitting the mask preserves trusted-trajectory callers.

The final comparison ran from an isolated checkout of source
`a68644a82137b8bd0e0c7efee3fefd8bab40d455`. All acquired input files were SHA-256
verified against their existing receipt before and after the comparison.
An earlier development comparison at `de6b75a1` is retained locally; it exposed
the need to record the new policy explicitly in the artifact, which the final
source does. Neither comparison changes the original v1 audit.

The evidence contains schema-valid native artifacts, a report and schema, and
the exact comparison source. The shared odometry cache stays outside Git; its
digest is recorded in the report. It avoids repeated scan registration during
later development. Runtime fields include different work in the two branches
of this experiment and **must not be interpreted as a speed comparison**.

- [Development report](../assets/ins_lidar_registration_quality/report.json)
- [Report schema](../assets/ins_lidar_registration_quality/report.schema.json)
- [Baseline artifact](../assets/ins_lidar_registration_quality/baseline.yaml)
- [Filtered artifact](../assets/ins_lidar_registration_quality/filtered.yaml)
- [Comparison source](../assets/ins_lidar_registration_quality/comparison-source.py.txt)

The original audit remains **refuted, 6/8**, and the leaderboard remains seven
audited sensor pairs with three supported pairs. A future claim requires a
new frozen method, new pre-registration and new held-out recordings.
