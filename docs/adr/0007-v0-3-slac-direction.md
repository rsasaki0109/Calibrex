# ADR 0007: v0.3 SLAC Direction — Trajectory Evidence, Anchored Temporal Estimation, IMU

## Status

Accepted (implemented in v0.3; see Acceptance evidence).

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

## Acceptance evidence

All three pillars landed on `main` as small PRs with schema-validatable artifacts,
provenance, tests, and honest public-data reporting (including negative results).

### Pillar 1: Trajectory as evaluated evidence

**PRs:** #38 (trajectory artifact `slac.trajectory/v0.1`, schema #18,
ground-truth-free quality gates), #39 (two-pass and native-deskew KISS-ICP
odometry modes at bag conversion).

**Acceptance measurements:**

- Schema-validated `trajectory.json` beside `timeline.json` on motion-compensated
  online runs; provenance registers `trajectory_path` and graded kinematic,
  interpolation-health, and cross-segment drift-proxy metrics with recorded
  thresholds feeding `trajectory_gate_verdict`.
- Indoor02 KISS-ICP Velodyne→Ouster and Velodyne selftest: 420 poses over
  42.26 s; kinematic and interpolation **PASS**; cross-segment drift proxy
  **PASS** at RMSE **0.143 m** (3799 correspondences, gate 0.30 m) over the
  full odometry track span — independent of the bounded calibration replay budget.
- Native-deskew KISS-ICP odometry at bag conversion (#39) on the identity
  selftest: translation recovery **9.54 cm → 4.34 cm** (~55% reduction) with
  modest rotation regression (**1.17° → 1.96°**); trajectory verdict **PASS**
  (cross-segment RMSE 0.160 m).

**Honest open items:** The ADR-prescribed two-pass odometry mode improved
translation versus the v0.2 baseline (~6.16 cm) but **FAIL**ed its own
trajectory gate (cross-segment RMSE **0.366 m**) — the framework falsified its
own planned approach. **Native deskew is the production choice** for bag
conversion. Rotation recovery did not improve under either deskew variant;
LiDAR-odometry drift still bounds absolute extrinsic accuracy on cross-sensor
runs (~85 cm vs the TIERS GICP seed in run C).

### Pillar 2: Anchored temporal estimation

**PR:** #40 (`time_offset_anchor` dual-mode evaluation, `temporal` evidence
family separability row, validation-only `inject_time_offset_s`).

**Acceptance measurements:**

- Every temporal run records adapted (final extrinsic) and anchored (configured
  initial) sub-blocks, `time_offset_anchor`, and a joint `separability` verdict
  in provenance; reports emit an **Extrinsic/Temporal Separability** evidence
  row.
- Indoor02 selftest with anchored mode and +50 ms validation injection: anchored
  δt̂ moved **−25 ms → −75 ms** — a differential shift of exactly **−50 ms**;
  adapted mode did not track the injection (**−71 ms → −44 ms**, +27 ms shift
  against a −50 ms truth change) — the v0.2 degeneracy directly observed.
- Synthetic moving-rig fixtures recover injected offsets to ±5 ms in anchored
  mode with separability **separable**.

**Honest open items:** The absolute anchored baseline on the shared-clock
selftest stays at **−25 ms**; interpretation (not a measurement): VLP-16
sweep-end header stamp semantics place the effective mean capture ≈50 ms before
the stamp, so a residual-minimizing δt̂ in [0, −50 ms] is consistent with scan-time
semantics rather than estimator error — absolute Ouster restamp bias isolation
remains blocked without tighter stamp modeling. Cross-sensor KISS-ICP anchored
curves stay below the 10 % flatness margin (flatness **0.032 / 0.012**,
separability **degenerate**); ±0.05 s probes still miss on real Indoor02.
Full Pillar 2 acceptance on cross-sensor real data is **not yet met**; differential
injection tracking on selftest is the strongest landed signal.

### Pillar 3: IMU candidate evaluation

**PR:** #41 (`sensor_msgs/Imu` CDR decoding with hand-computed golden vector,
`--imu-topic` bag conversion with OS1 IMU restamp, `lidar_imu` evidence family
in the grade rollup).

**Acceptance measurements:**

- Clock-domain check on the source bag: median `bag_receive_time − header.stamp`
  for `/os_cloud_nodee/imu` is **1645462364.473 s** (same since-boot epoch as
  Ouster points); conversion restamps IMU alongside points.
- Rotation-rate consistency with symmetric smoothing
  (`pose_interval_mean_imu_vs_odometry_log_symmetric_ma2`): holdout RMSE
  **3.56 deg/s** (gate ≤ 5 deg/s), down from ~8 deg/s before the noise-floor fix;
  mean |Δω| **2.64 deg/s**.
- Per-axis observability recorded: z-dominant excitation (p95 |ω_z| ≈ 36 deg/s
  vs |ω_x|, |ω_y| ≈ 12 deg/s); roll −10° and pitch +10° probes detected
  **+34%** and **+40%** RMSE increases; all yaw probes **≈ 0%**.
- Gravity support (supporting-only): angle **4.42°** (warn > 10°).

**Honest open items:** Yaw rotation probes are unfalsifiable under z-dominated
excitation; the +10° yaw known-bad seed is not separated by holdout RMSE
(consistent with axis observability, not metric-power failure). Known-bad
fraction **2 / 12** yields decision boundary **WARN**. The evaluated candidate
uses the documented TIERS GICP seed `R_velo_os1 ≈ R_velo_imu` — an engineering
approximation, not metrology ground truth.
