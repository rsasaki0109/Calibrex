# ADR 0007: v0.3 SLAC Direction — Trajectory Evidence, Anchored Temporal Estimation, IMU

## Status

Proposed.

## Context

v0.2 (ADR 0005, Accepted) delivered moving-platform support, camera-LiDAR as an
evaluated modality, and temporal offset evidence. The rename to slac (ADR 0006)
made the direction explicit: simultaneous localization and calibration. v0.2's
acceptance evidence left three honest open items:

- The identity self-consistency control recovers a known-identity extrinsic to
  ~9.6 cm / ~1.2° with motion compensation (~10× better than the static
  assumption), but the residual is dominated by LiDAR-odometry drift; per-point
  deskew alone did not tighten it.
- Time-offset probes lose falsification power once the online solver has
  adapted the extrinsic: the holdout-RMSE(δt) curve flattens (time-offset /
  extrinsic partial degeneracy). The estimator honestly reports INCONCLUSIVE,
  but a genuine temporal verdict needs extrinsic anchoring.
- The trajectory itself — half of "simultaneous localization and calibration" —
  is an unevaluated input. Odometry quality silently bounds every
  motion-compensated result, yet no gate, metric, or artifact measures it.

## Decision

v0.3 extends slac along three pillars, in priority order.

### Pillar 1: Trajectory as evaluated evidence

Promote the trajectory from an anonymous input to a first-class, evaluated,
schema-validated artifact:

- A trajectory artifact (poses + provenance: source, frame, interpolation,
  clamp/extrapolation stats) emitted by motion-compensated runs.
- Trajectory quality metrics that do not require ground truth: odometry
  self-consistency (e.g. reverse-replay / leave-segment-out map consistency),
  per-batch pose-interpolation health, and drift proxies; each with recorded
  thresholds and PASS/FAIL/INCONCLUSIVE semantics feeding the run verdict.
- Drift reduction measured, not assumed: a two-pass pipeline option (pass 1
  deskews scans with the raw odometry; pass 2 re-runs LiDAR odometry on
  deskewed scans) evaluated on the Indoor02 identity selftest. Acceptance
  target: identity recovery meaningfully below the v0.2 ~9.6 cm / ~1.2°
  baseline, reported honestly if not achieved.

This is not a SLAM system. It is evidence about the trajectory that calibration
already depends on.

### Pillar 2: Anchored temporal estimation

Resolve the v0.2 temporal degeneracy with an extrinsic-anchored mode:

- `estimate_time_offset` gains an anchored variant: hold the extrinsic fixed
  (at the configured initial or an externally supplied candidate) while
  searching δt, restoring probe power that solver adaptation destroys.
- A joint observability statement for the extrinsic/temporal pair: report when
  the motion profile cannot separate them (the v0.2 flat-curve case) as a
  first-class evidence row, not just a provenance note.
- Acceptance: on Indoor02, anchored probes must detect the injected offsets
  that v0.2 probes missed (synthetic injection on real data), and the anchored
  estimator must bound the Ouster restamp bias or return a justified
  INCONCLUSIVE with the flatness numbers.

### Pillar 3: IMU candidate evaluation (promoted from Deferred)

An honest evaluation layer for externally produced LiDAR-IMU extrinsic
candidates — not IMU calibration itself:

- Rotation-consistency evidence: candidate-rotated IMU angular velocity vs
  LiDAR-odometry-derived angular velocity over the replay window, with
  train/holdout split, known-bad rotation probes, and recorded thresholds —
  the exact ADR 0004 protocol shape.
- Gravity-direction consistency as a second, weaker channel (accelerometer
  low-pass vs world gravity under the odometry trajectory), gated as
  supporting-only evidence.
- Real-data validation on the TIERS Indoor02 bag: the already-downloaded
  source ROS 1 recording carries four IMU streams alongside the LiDARs
  (`/avia/livox/imu` and `/livox/imu` at ~200 Hz, `/os_cloud_node/imu` and
  `/os_cloud_nodee/imu` at ~100 Hz). The current converted rosbag2 subsets do
  not include them, so Pillar 3 extends the conversion tool to carry an IMU
  topic. Same honest-reporting discipline as every v0.2 validation.

## Consequences

Same discipline as v0.2: small PRs, schema-validated configs and artifacts,
provenance everywhere, tests for calibration changes, real-public-data
validation reported honestly including negative results, ROS-independent core.
Pillar order is priority order; Pillars 2 and 3 may proceed in parallel once
the Pillar 1 trajectory artifact exists.

v0.3 does not claim SLAM or SOTA calibration. It claims evidence-grade
evaluation of the trajectory-calibration loop that the slac name promises.

## Deferred

- Full radar extrinsic calibration (`radar_lidar_velocity_consistency` stays
  experimental).
- Camera-LiDAR full-scale PASS demonstration (requires probe-capable full
  KITTI drive data; the committed 76 KB fixture demonstrates the discipline,
  not a PASS).
- Package releases to PyPI and version tags (maintainer decision to skip).
- GitHub Pages deployment (manual, plan-dependent step).
