# Changelog

## Unreleased

- Switched the public installation path to the verified GitHub Release wheel
  and removed the automatic PyPI publishing workflow following the maintainer
  decision to skip PyPI distribution.

- Added the continuous-time trajectory foundation. `slac.continuous_time_trajectory/v0.1`
  is a typed, schema-valid trajectory contract that declares the interpolation
  model, knot-domain validity (queries outside the domain are rejected, never
  clamped), and clock/capture-time semantics, and converts losslessly to the
  existing piecewise linear-slerp adapter. Analytic SE(3) manifold operations
  (`se3_manifold`) provide right-trivialized exp/log, the SO(3)/SE(3) left
  Jacobians and adjoint, the point-transform Jacobian, and two-knot screw
  interpolation Jacobians, all verified against finite differences. A sparse
  Gauss--Newton/Levenberg--Marquardt fitter (`continuous_time_sparse`)
  assembles the normal equations as a block-banded system in which every
  factor touches at most its two bracketing knots, estimating all knots on the
  manifold from body-frame point measurements and pose measurements (poses
  become four anchored point factors using only the analytic point Jacobian).
  `calibrex trajectory build-contract` and `calibrex trajectory fit` expose the
  workflow, and the fit result validates against
  `slac.continuous_time_trajectory_fit/v0.1` with provenance pinned to both
  input digests.

- Added schema-valid empirical SE(3) uncertainty evidence
  (`slac.empirical_se3_uncertainty/v0.1`). `calibrex camera-lidar
  empirical-uncertainty` refits the native probabilistic multi-frame refiner on
  deterministic contiguous temporal-block subsamples that never split a block
  across the train/holdout boundary, reports tangent-space intervals with
  Sidak-corrected joint family coverage against injected truth, retains weak or
  unobservable directions instead of dropping them, and applies an
  overconfident interval control that must be refuted before the
  PASS/WARN/FAIL policy can certify the intervals. Without a declared reference
  (`--stability-only`) the artifact reports the empirical spread with an
  INCONCLUSIVE policy. Every resample refit is retained as a digest-bound
  schema-valid result, and the artifact validates against its generated schema.

## 0.4.1 - 2026-08-11

- Promoted the no-download KITTI-shaped Camera-LiDAR evidence path into a
  strict five-minute demo. Its deterministic, generator-bound 300x300 fixture
  now detects 16/24 mandatory ±1 deg / ±0.10 m perturbation cases, passes all
  six falsification-policy gates without relaxing thresholds, and remains
  explicitly synthetic rather than a real-sensor accuracy claim. Added direct
  generator parity and strict end-to-end tests plus README commands that
  validate the result and verify its digest-bound provenance bundle.
- Added the Phase 3 provider-neutral probabilistic 2D--3D correspondence
  artifact and an optional Apache-2.0 OpenCV PnP-RANSAC adapter with
  confidence filtering, covariance-aware diagnostics, tests, and complete
  provider/input provenance. The correspondence contract now distinguishes
  dataset-license provenance from the provider's code/model license; ablation
  evidence is marked unverified when the dataset license or source digests
  are undeclared.
- Added a D2D-initialized native multi-frame probabilistic pose refiner with
  covariance/outlier/reliability weighting, robust loss, bounded SE(3)
  correction, disjoint frame holdout, complete candidate traces, and explicit
  uncertainty ablation switches.
- Added explicit D2D candidate-trace initialization for probabilistic
  multi-frame refinement and its paired ablations. The trace must pin the same
  problem digest and frame pair; results record its ID, digest, solver status,
  and hit decision. Runs without a trace declare `problem_initial_transform`
  and are not mislabeled as solved-D2D initialization evidence. The generic
  method ID is now
  `probabilistic_multiframe_refinement/v0.2`; the old v0.1 ID remains accepted
  for artifact compatibility.
- Added a paired probabilistic-refinement ablation benchmark that executes the
  full estimator and three one-factor removals over fixed frame-split seeds,
  retains every schema-valid result, and reports failure-aware paired
  bootstrap comparisons with both input digests. Results and ablations now
  include translation distance and quaternion-geodesic rotation error to the
  problem's digest-pinned reference, making the planned cm/deg gates directly
  auditable alongside held-out reprojection.
- Added a ROS-independent joint camera--LiDAR continuous-time solver boundary
  using per-point firing times, supplied body twists, probabilistic image
  residuals, bounded extrinsic/clock refinement, disjoint holdout evidence,
  and an explicit weak-motion rejection gate.
- Added rolling-shutter row-time deskew plus fixed-clock, no-per-point-time,
  no-rolling-shutter, and no-covariance temporal ablations. The paired
  multi-split benchmark retains every result, failure, source digest, and
  bootstrap comparison. Optional frozen extrinsic/time references now produce
  initial/final rotation, translation, and clock errors in each result and
  paired ablation, making injected-recovery gates directly auditable.
  Continuous-time problems separately declare their dataset license; absent
  license provenance marks the ablation benchmark unverified.
- Added a ROS-independent, provenance-pinned piecewise SE(3) body-trajectory
  adapter with linear translation, shortest-arc quaternion interpolation, and
  strict no-extrapolation behavior. Continuous-time camera--LiDAR refinement
  now uses this non-constant motion model when supplied while retaining the
  constant-twist input as a backward-compatible fallback. A CLI adapter
  attaches existing schema-valid `T_world_body` trajectory artifacts while
  preserving both input digests and frame semantics.
- Added a digest-frozen Camera-LiDAR SOTA audit protocol/result and CLI. It
  evaluates every numerical/integrity gate, dataset-family and independent-rig
  coverage, and can emit only supported, refuted, or incomplete verdicts;
  schema validation prevents an unsupported success label.
- Extended benchmark distributions with p90 and p95 and made mean, median,
  p90, p95, maximum, and failure rate individually addressable by frozen SOTA
  audit requirements, so tail failures cannot be hidden behind a mean. Paired
  bootstrap improvement confidence bounds can also be frozen against an
  explicit reference/candidate method pair, while runtime and peak-memory
  mean/p95 statistics are directly auditable. Evidence whose benchmark
  provenance declares `data_verified: false` is rejected, and result
  validation now enforces the semantics of all three verdicts rather than
  guarding only `supported`. Optional smoke requirements cannot inflate the
  achieved dataset-family or independent-rig coverage counts.
- Added a schema-valid Camera-LiDAR D2D research pipeline with frozen
  Fibonacci-sphere protocols, complete candidate traces, Bull's Eye artifacts,
  an external FeatDepth provider adapter, rectified KITTI problem construction,
  deterministic parallel execution, and a 200/200-hit KITTI raw 0018
  rotation-only reproduction gate.
- Completed the frozen KITTI raw 0018 rotation matrix at `1, 2, 10, 20 deg`
  with 200 deterministic directions per level. The `1, 2, 10 deg` levels
  recovered 200/200; `20 deg` recovered 178/200 with zero computational
  failures, while its `17.0067 deg` p90 and `28.4920 deg` p95 retain the
  direction-dependent capture failures that its `0.2425 deg` median would
  otherwise hide. All new traces and aggregate artifacts are schema- and
  digest-validated.
- Added the KITTI-360 half of the D2D reproduction path: a pinned external
  MiDaS v3.1 provider, official MEI fisheye projection, official
  pose/azimuth-based Velodyne motion compensation with a schema-valid
  provenance manifest, KITTI-360 problem construction, and CLI support. The
  frozen `10 deg` gate completed at 200/200 hits with zero failures (mean
  `0.3812 deg`, maximum `0.4676 deg`), matching the paper's 100% target.
- Completed the frozen KITTI-360 rotation matrix at `1, 2, 10, 20 deg` with
  200 deterministic directions per level. The `1, 2, 10 deg` levels recovered
  200/200; `20 deg` recovered 184/200 with zero computational failures. Its
  `0.3882 deg` median contrasts with a `28.4409 deg` p95 and `45.4505 deg`
  maximum, retaining the direction-dependent capture failures. All new trace,
  benchmark, and Bull's Eye artifacts are schema- and digest-validated.
- Added a digest-pinned A2D2 MiDaS/provider and pre-registered camera-view D2D
  smoke path. The frozen five-direction `10 deg` run completed 5/5 trials but
  failed recovery at 0/5 hits; the two-frame/pre-registration limitation and
  negative result are retained rather than weakening the gate.
- Added digest-checked resumable Camera-LiDAR trial execution plus a bounded
  six-DoF D2D solver, paired Fibonacci rotation/translation protocols,
  schema-valid candidate traces, benchmark aggregation, and CLI commands. The
  first real KITTI-360 `(0.5 deg, 0.5 m)` smoke converged to
  `0.2505 deg / 2.08e-17 m` and passed the frozen hit gate.
- Completed the first full six-DoF matrix cell on KITTI raw 0018 at the frozen
  `(0.5 deg, 0.25 m)` perturbation. All 200 trials converged with zero
  computational failures, but only 84/200 passed the strict
  `<0.5 deg / <0.20 m` accuracy gate. Rotation error was `0.5230 deg` mean,
  `0.5239 deg` median, and `0.7228 deg` p95; translation error was `0.1447 m`
  mean, `0.1421 m` median, and `0.1803 m` p95. The negative 42% hit-rate result,
  all trace identities/digests, and schema-valid aggregate provenance are
  retained without weakening the gate.
- Corrected the UniCalib adapter's default provenance URL to the official WACV
  2026 `han-15/UniCalib` repository and locked it with a unit assertion. Updated
  the SOTA evidence review for TLC-Calib's May 2026 code/data release while
  retaining its non-commercial, external-process-only license boundary.

- Added a digest-locked fixed-frame KITTI raw Camera-LiDAR input artifact and
  recovery benchmark comparing a scalar-reference Pandey I2I optimizer with a
  vectorized, safety-gated deterministic coarse-to-fine path. On the fixed
  official KITTI raw 0005 input, all paired accuracy metrics were identical
  while mean trial runtime improved by 1.719x.
## 0.4.0 - 2026-07-29

- Extended `calibrex doctor` from an environment check into an optional
  dataset-readiness workflow with type inference, quality and degeneracy
  diagnostics, compatible-workflow suggestions, a versioned
  `slac.doctor/v0.1` artifact, generation provenance, and validation support.
- Added local and GitHub-hosted Calibration CI with `calibrex ci`, a composite
  action, Step Summary rendering, enforced falsification and protocol gates,
  digest-bound inputs and custom policies, provenance-bound SVGs, and the
  versioned `slac.calibration_ci/v0.1` decision artifact.
- Added the schema-versioned `slac.external_calibration_run/v0.1` contract for
  digest-bound subprocess, container, precomputed, and imported calibration
  results; migrated the Koide adapter to it and added a ROS-free Kalibr
  camchain importer with explicit frame/time conventions and failure states.
- Added a reproducible full-scale KITTI raw Camera-LiDAR falsification runner
  with a frozen reference/known-bad protocol, selected-frame and raw-input
  SHA-256 provenance, independently verified evidence bundles, an honest
  PASS/FAIL/INCONCLUSIVE decision artifact, and an opt-in official-data test.
- Prepared the public launch surface with a concrete five-minute quickstart,
  an approachable documentation home, a repository social preview, citation
  metadata, and corrected issue links.
- Added guarded PyPI trusted publishing after a GitHub release is published,
  including tag-to-package-version checks and clean wheel smoke tests.
- Replaced the abbreviated license notice with the complete Apache-2.0 text and
  added a distributable `NOTICE`.
- Restored the Calibrex product, Python distribution/package, CLI, and source
  path. Existing `slac.*` schema IDs, `slac_version`, `slac_native`, and
  versioned protocol/policy IDs remain the stable legacy wire namespace.
- Added provenance-bound SVG evidence cards via
  `calibrex render --format evidence-card` and schema-source comparison tables
  via `calibrex compare --format evidence-table`, with deterministic README
  artifacts and drift tests.
- Reworked the README around a concise evidence-first story, generated visual
  summaries, an honest accepted-vs-known-bad comparison, public-data demos, and
  a compact calibration coverage matrix.

## 0.3.0

- Trajectory artifact (`slac.trajectory/v0.1`, schema #18) with ground-truth-free
  quality gates (kinematic health, interpolation clamp fraction, cross-segment
  drift proxy) on motion-compensated online runs.
- Two-pass and native-deskew KISS-ICP odometry modes at rosbag conversion;
  native deskew cut Indoor02 identity selftest translation error ~9.5 cm → ~4.3 cm
  (~55%) with modest rotation regression (~1.2° → ~2.0°); two-pass odometry
  FAILed its own trajectory gate (cross-segment RMSE 0.366 m).
- Anchored temporal time-offset estimation (`time_offset_anchor: initial`),
  dual-mode adapted/anchored provenance, joint extrinsic/temporal separability
  evidence row, and validation-only `inject_time_offset_s` (anchored mode
  tracked +50 ms injection exactly on selftest; adapted mode absorbed it).
- LiDAR-IMU rotation evidence (`lidar_imu` family): `sensor_msgs/Imu` CDR
  decoding, `--imu-topic` bag conversion with OS1 IMU restamp, symmetric
  rotation-rate smoothing (holdout RMSE ~8 deg/s → ~3.6 deg/s, passing 5 deg/s
  gate), per-axis observability, known-bad rotation probes, and supporting-only
  gravity consistency (4.42° on Indoor02).

## 0.2.0

- Pure-Python rosbag2/MCAP reader (sqlite3 + MCAP, CDR PointCloud2 and Odometry).
- Odometry-aware online motion compensation for rosbag2 replays with world-frame
  source maps, odometry-extrapolation gate, and clock-domain restamping.
- KISS-ICP rig-frame odometry path and TIERS Indoor02 real-data validation
  (identity self-consistency control: ~10× extrinsic error reduction with motion
  compensation vs static control).
- Per-point deskew via `sensors.<name>.point_time_field` on rosbag2 online runs.
- Camera-LiDAR projection promoted to ADR-0004-protocol evidence rows with
  family-aware assessment policy gates and KITTI committed-sample demonstration.
- Temporal time-offset perturbation probes and 1D holdout-RMSE estimator on
  motion-compensated online runs (synthetic ±5 ms recovery; honest real-data
  degeneracy reporting).
- Project rename Calibrex → slac (ADR 0006).

## 0.1.0-alpha.1

- Project bootstrap.
- Typed config and result models.
- CLI skeleton with `doctor`, `schema`, `validate`, `verify`, `assess`, `evidence`, `init`, `inspect`, `calibrate`, `evaluate`, `render`, `visualize`, `compare`, `report`, and `export`.
- Frame graph and SE3 utilities.
- Dataset manifest schema, filesystem dataset adapter, timestamp normalization, and MCAP adapter boundary.
- HTML report, report sidecars, and PASS/WARN/FAIL quality aggregation.
- Metric registry, deterministic holdout splitting, and threshold-based metric grading profiles.
- RGB-D Open3D SLAC adapter boundary and example config.
- Backend-neutral graph problem compiler and `slac compile`.
- Public dataset catalog, TUM RGB-D config, KITTI raw config, and direct TUM downloader helper.
- KITTI raw loader and fixed-mounted LiDAR prior factor.
- KITTI `calib_velo_to_cam.txt` importer for fixed-LiDAR initial transforms.
- Autonomous-driving starter profile with Radar and Autoware export surface.
- Public Livox Horizon-Horizon PCD evidence demo for rigidly mounted solid-state LiDAR pairs.
- LiDAR pair evidence metrics, known-bad perturbation cases, holdout point-to-plane summaries, and evidence protocol metadata.
- Machine-readable `summary.json`, `metrics.json`, `observability.json`, `degeneracy.json`, and `evidence.json` report sidecars.
- Machine-readable `comparison.json` artifacts with metric, transform, evidence summary, and evidence protocol compatibility comparisons.
- Static JSON schemas for config, result, comparison, dataset manifest, and report sidecars.
- Bulk schema generation via `slac schema all --output-dir schemas`, with schema drift tests and CI smoke coverage.
- 3D rig viewer artifact for reference, candidate, and estimated extrinsic comparison.
- Falsification assessment framework: `slac assess` applying a policy to evidence, policy artifacts, `--enforce` exit codes, per-DoF known-bad challenge summaries, a fixed support denominator, mandatory challenge verification, and assessment recomputation verification inside evidence bundles.
- Evidence bundle verification: `slac verify`, an `evidence-bundle-verification` schema, a `--require-raw-recomputed` raw recomputation gate, a verified-raw-inputs requirement, and structured verification claims.
- `slac render` and `slac evidence` commands for producing report and evidence artifacts from an existing result without recomputing metrics.
- Protocol and policy JSON schemas, and `compare --enforce-compatible` protocol compatibility enforcement.
- README GIF gallery generation (`generate_calibration_evidence_gif.py --readme-gallery`) with a provenance manifest and schema-validated visual modes.
- Local release smoke helper, wheel smoke coverage in CI, and a draft GitHub release workflow.
- Configurable public-dataset frame sampling: `DatasetConfig.sample_limit` and `slac inspect --sample-limit`.
- N-way `slac report-compare` command with labeled results, protocol-compatibility gating, metric-family rankings, and a `report_comparison` schema (16 schemas total).
- Pure-Python rosbag1 (v2.0) reader with PointCloud2 decoding (`none`/`bz2`, optional `lz4` extra), `slac inspect --type rosbag1`, and the TIERS LidarsCali example config.
- Native LiDAR point-to-plane solver backend (`solver.backend: native_lidar_point_to_plane`) wired into `slac calibrate`, with real rank/condition-number/weak-DoF observability replacing the `uncomputed_alpha_backend` stub and point-to-plane metrics in results.
