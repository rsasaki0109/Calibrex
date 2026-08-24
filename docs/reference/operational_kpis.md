# Operational KPIs and pilot acceptance

This page turns a calibration result into an operational decision. A metric is
only useful for deployment when its input split, provenance, evaluator, and
decision consequence are recorded in the same artifact. The thresholds below
are gate defaults; a program may tighten them, but must record the replacement
thresholds in its pilot configuration rather than changing them after seeing
the result.

## Product and operational KPI matrix

| Product outcome | KPI and acceptance target | Required evidence | Decision consequence |
| --- | --- | --- | --- |
| Capture is usable | 100% of required readiness checks pass; 0 unknown required checks; readiness artifact and source inputs verify by SHA-256 | `koide-readiness` artifact, input manifest, camera/lidar metadata | Block capture or calibration until complete |
| Result is reproducible | Every source, config, dataset, tool/source commit, output, and (when used) image/checkpoint has a declared digest; all available bytes recompute to the declaration | Result/external-run provenance plus verification report | Reject stale or unverifiable artifacts |
| Evaluation is leakage-safe | Training and holdout ID sets are disjoint and their digests match the run; no holdout file is consumed by fitting | Train/holdout manifests and split audit | Reject the candidate and rerun the split |
| Calibration is observable | Koide pilot default: holdout projection ratio ≥ `0.50`, image-edge alignment ≥ `0.20`, depth-edge alignment ≥ `0.20`; all metrics report denominators | Independent holdout benchmark and metric JSON | `PASS` may advance; `WARN` requires review; `FAIL`/`INCONCLUSIVE` blocks |
| Known bad inputs are caught | At least 12 signed falsification controls; default detection fraction ≥ `0.80` | Perturbed-transform control ledger and evaluator output | Block adoption if controls are missing or below target |
| Commercial use is legally safe | Commercial profile uses manual/precomputed initialization; 0 automatic SuperGlue paths or GPL code in the core process | Typed runner config, source/license identity, CI boundary checks | Block commercial execution on a violation |
| Deployment export is safe | Only `PASS`/`ADOPT` results export; Autoware YAML, static TF launch, and manifest all verify; source/result/input digests are present | `AutowareExportArtifact` and its byte-level sidecars | Do not update the vehicle model on any other status |
| Operations can recover | Candidate/baseline comparison and rollback record are produced for each promotion; no in-place overwrite | Calibration CI artifact and lifecycle record | Roll back to the last accepted calibration when observability degrades |
| Decision latency is measurable | Record readiness, external execution, evaluation, review, and export durations for every pilot; establish the fleet-specific time budget from these measurements | Run timing/provenance fields | Escalate a pilot that exceeds the agreed service budget |

These are product-value metrics rather than claims of universal metrology
accuracy. Public-data runs can demonstrate repeatability, holdout behavior, and
failure detection; they do not replace vehicle-specific ground truth or a
deployment smoke test.

## Pilot acceptance path

1. **Declare the pilot.** Pin the Calibrex revision, dataset manifest, camera
   and LiDAR frame names, profile (`commercial` by default), and thresholds.
2. **Run strict readiness.** Require complete camera intrinsics, usable image
   and point inputs, frame binding, time/synchronization evidence, and a
   self-verifying readiness artifact. The artifact status must be exactly
   `ready`, every check must be `pass`, and the required-check unknown count
   must be zero; otherwise external execution is blocked.
3. **Run the external calibrator.** Use the official Koide workflow with a
   fixed source revision and an immutable container image reference
   (`repository@sha256:<64 hex>`). The current repository does not supply an
   image digest or a Docker/Podman engine, so this step is an external
   prerequisite. For commercial use, provide a manual or precomputed initial
   guess; Koide's official documentation excludes SuperGlue from its images
   because of its license constraints.
4. **Import and bind the result.** Supply native `calib.json` with
   `results.T_lidar_camera` and explicit camera/LiDAR frames, or an
   `external-run` artifact. Record the exact command, source revision, image
   digest, input-file inventory/digests, output digest, logs, and license
   boundary. Record training/holdout IDs separately when a learned provider is
   used.
5. **Evaluate independently.** Run the holdout benchmark and signed known-bad
   controls. Keep the candidate, baseline, split manifest, metrics, and
   evaluator provenance together; do not tune on holdout data.
6. **Make the decision.** `PASS` plus the required quality, falsification, and
   provenance gates may become `ADOPT`. `WARN` requires named human review;
   `FAIL`, `INCONCLUSIVE`, `BLOCKED`, missing provenance, or a failed legal
   boundary is not admissible.
7. **Export and smoke-test.** Generate Autoware parent-frame YAML and static TF
   output from the accepted artifact, verify all sidecars, then run the target
   vehicle's TF/perception smoke test before changing the deployed model.
8. **Monitor and roll back.** Retain the accepted artifact as the baseline,
   compare subsequent calibrations through the calibration CI gate, and retain
   a reversible rollback record.

## Current gate state

The checked-in A2D2 Koide pilot is intentionally `BLOCKED`: it has no KITTI
frame-pair manifest and no official native candidate. The recorded external
command is evidence of the required handoff, not evidence that Koide ran. To
close this gate, provide the immutable execution environment, complete
camera/LiDAR inputs and metadata, the native output or external-run artifact,
its byte digests and logs, and an independent holdout/known-bad evaluation.
