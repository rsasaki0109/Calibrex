# Disjoint geometric candidate validation protocol

Test frozen [continuation outputs](joint_continuation_results.md) without rerunning
optimization. For each of the five analytic measurement conditions, evaluate six
existing candidate poses (baseline/continuation × two perturbations and truth start)
and two labeled controls: true pose and a fixed wrong pose (+2,-2,+1.5 degrees,
+.4,-.3,+.3 m relative to truth). Total **40 candidate checks**, each with three
frames. True/wrong control construction uses truth, but the validator does not.

Generate 3840 new LiDAR rays per frame using the existing angular lattice shifted
by +.37 horizontal pixels and +.29 vertical pixels. Independently raycast the same
scene at the LiDAR origin. Verify no resulting point coincides with fitting points.
Keep ALL first-hit LiDAR returns; do not use the true pose to remove camera
occlusions, even for previously oracle-filtered fitting conditions. The same
holdout points are used in every condition. Camera depth images remain shared
with fitting: this is held-out sampling, not independent sensors/scenes or unseen
evaluation data. Previous results informed this experiment's design.

The validator receives only depth, candidate matrix, point cloud and depth
representation. Transform points, nearest-pixel project, sample optical z; convert
radial images into optical z using the sampled pixel ray norm. Count behind-camera,
out-of-image and missing-depth points separately. Do not z-buffer away returns or
filter residual outliers/occlusions. Residual = transformed camera z minus measured
optical z; report signed >20 cm and <-20 cm counts, median absolute and absolute
p90 over every supported point.

Declare a provisional pass only if EACH frame has >=100 supported points,
>=75% support of ALL rays, absolute median <=.05 m and absolute p90 <=.20 m.
These are fixed synthetic diagnostic thresholds, not externally calibrated safety
limits, and not inferred extrinsic uncertainty bounds. No after-result tuning.
Missing support fails explicitly. Store all candidates, not just selected successes.
Truth pose metrics label outcomes only AFTER validation. Assess both rejection of
known wrong drift and retention of correct controls. Biased depth may make a true
pose fail consistency, so a rejection does not diagnose extrinsic error uniquely.

Only inspect candidate reports after committing generator/validator/thresholds.
Do not promote to an acceptance gate or real-data claim based on this one scene.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/joint_geometric_validation.py \
  docs/assets/joint_geometric_validation_20261029/plan.json \
  docs/assets/joint_geometric_validation_20261029/evaluation.json
```
