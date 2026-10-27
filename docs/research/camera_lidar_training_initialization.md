# Camera–LiDAR: training-only initialization development

Status on 2026-10-27: **implementation ready for development comparison; real-input
comparison blocked by missing local arrays**. This is a research step toward the
camera–LiDAR claim, not a new SOTA result or an evaluation pre-registration.

## Starting evidence

The [archived-audit replay and input inventory](../assets/sota_development_restart_20261027/report.json)
was generated from committed source `192858e4`. Its accompanying
[schema](../assets/sota_development_restart_20261027/report.schema.json) validates
the report. All twelve archived verdicts and requirement statuses reproduce:
four supported audits across three pairs, eight refuted audits. These are
recomputations from archived scores, not calibration reruns or comparisons
against newly published methods. Existing claim scope and caveats remain binding.

Camera–LiDAR v2 remains **refuted, 8/9 gates**. Its known failure is a confidently
judged axis with 1.503 deg reference disagreement against the frozen 1.5 deg gate.
Neither that gate nor the historical verdict changes. Subsequent development
showed that raising the adoption threshold, spatial weighting and a scan-direction
orientation proxy could lose useful detections. They are not promoted.

The prior [nuisance-profiled controls](../benchmarks/kitti_camera_lidar_profile_controls.md)
also reused an initialization fitted on all frames. Reoptimizing a fold on training
frames alone does not remove that initialization's dependence on held-out frames.
This step removes that dependence before further confidence-rule experiments.

## Implemented comparison

`tools/kitti_camera_lidar_profile_probe.py --training-only-start` initializes each
axis/fold independently:

1. Hold nuisance coordinates at zero relative to the supplied candidate.
2. Evaluate the existing fixed axis-profile grid on that fold's training frames.
   Include the zero candidate first, retaining it on an exact tie.
3. Refine the selected axis using the existing search levels, on training frames.
4. Fit the baseline pose and both signed nuisance-refitted controls using only
   training frames. Score the five fixed poses on the held-out block afterward.

The command cannot also accept `--conditional-report`. Per-fold initial coordinates
are recorded in `training_start_theta`; source and input hashes, unchanged options,
initialization policy and failures remain in the report. Archived/default fold
serialization stays compatible with frozen schemas. Native estimator defaults
and all previous artifacts remain unchanged.

Tests cover a separated two-basin objective, independently initialized folds,
invariance of starts and fitted poses to held-out-objective changes, retention of
failed reports, and legacy evidence schemas. These are implementation tests, not
real-world accuracy evidence. Training-only initialization does not remove temporal
dependence, candidate/preprocessing assumptions, or development-data exposure.

## Fixed next development comparison

Use **all fourteen** already captured inputs: baseline and +1/+3 deg about each
rotation axis on spent KITTI raw `2011_09_26_0005` and `2011_09_28_0002`.
Do not select a subset based on new outcomes. Compare the pooled-start and
training-only-start arms with the same `--multi-start` nuisance search, unchanged
snapshot settings, scalar objective, preprocessing and held-out blocks. Leave
spatial balancing and the orientation proxy off. The only ablation is initialization.

For each input, retain completion/failure, every fold start/pose, signed control
drops, t statistics, detection flags, nuisance-bound hits and elapsed time.
Report all 42 axis controls, per drive and injected axis/amount, including failures.
Paired differences are development diagnostics; 42 correlated controls are not
42 independent recordings. Do not infer accuracy, native judged-axis coverage,
or SOTA support from control counts. Require unchanged old-arm numerical evidence
before interpreting a difference; any mismatch is an investigation outcome.

Example commands after input verification:

```bash
OPENBLAS_NUM_THREADS=1 python -m tools.kitti_camera_lidar_profile_probe SNAPSHOT_DIR \
  --conditional-report CONDITIONAL_REPORT.json --multi-start --output NEW_POOLED_DIR
OPENBLAS_NUM_THREADS=1 python -m tools.kitti_camera_lidar_profile_probe SNAPSHOT_DIR \
  --training-only-start --multi-start --output NEW_TRAINING_DIR
```

Run from the committed source in an isolated worktree and record its identity.
Never edit imported code during a batch. Keep failed attempts and make outputs
separate from inputs. This is development on spent recordings, not a fresh audit.

## Data availability and decision boundary

All fourteen committed snapshot metadata files exist; **zero corresponding
`arrays.npz` files are present beside them** in this workspace. A broader workspace
search also did not find those arrays. The inventory lists each expected hash.
First restore those exact files from the original development cache and let
`load_snapshot` verify hashes and shapes. Do not substitute the bundled two-frame
example or call it the 48-frame experiment. If originals cannot be recovered,
reacquire only these spent recordings and create a separately identified capture;
it cannot be claimed byte-identical to the previous experiment.

No real-input result from this new mode exists yet. Rejecting a known false
control while losing all useful detections is not success. A promising result
must be checked through native per-axis verdicts, with reference disagreement,
1/3 deg detection, judged-axis coverage, failures and runtime reported together.
The all-55 spent-drive denominator remains binding: **2/55 captured, 53 pending**.
Set no new accuracy gate or adoption default from the two-drive pilot.

Only after development is complete should a new frozen claim, scorer, exposure
audit and independent evaluation be registered under the
[SOTA handoff](../sota_handoff.md). Candidate KITTI-360 recordings remain unread
by this work; their exposure audit is incomplete. KITTI-360 alone does not satisfy
the broader research requirement for multiple dataset families and independent
rigs. Dataset calibration remains a reference, not ground truth. A broad SOTA
claim also requires a current, reproducible baseline/literature comparison.

The inventory can be regenerated into a new output directory:

```bash
python -m tools.sota_development_baseline --output NEW_BASELINE_DIRECTORY
```
