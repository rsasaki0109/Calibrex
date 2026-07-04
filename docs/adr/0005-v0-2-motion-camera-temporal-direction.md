# ADR 0005: v0.2 Motion, Camera, and Temporal Direction

## Status

Accepted (implemented in v0.2; see Acceptance evidence).

## Context

v0.1 alpha delivered the LiDAR-main evaluation protocol with ADR 0004 gaps
closed: a native point-to-plane rig solver with observability and degeneracy
reporting; holdout plus known-bad-probe evidence gates; online streaming
calibration with multi-batch observation accumulation and a per-batch timeline
rendered in the HTML report; a pure-Python rosbag1 reader for PointCloud2 and
`livox_ros_driver/CustomMsg`; and real-data validation on A2D2, a Livox pair,
and TIERS LidarsCali (solid-state Horizon→Avia and mixed Velodyne→Horizon).

Camera-LiDAR projection metrics remain an experimental overlay. Radar
velocity-consistency is an experimental metric. IMU candidate evaluation is a
roadmap item. Everything in v0.1 assumes a static rig and fixed trajectory.

## Decision

v0.2 extends slac along three pillars, in priority order.

### Pillar 1: Moving-platform support

The binding constraint today is the static-rig / fixed-trajectory assumption.
v0.2 removes it by adding:

- A pure-Python rosbag2/mcap reader using the same dispatch pattern and
  lazy-numpy discipline as the rosbag1 reader.
- Ingestion of odometry and trajectory topics.
- Motion-compensated target batches.
- Trajectory-derived observability, because motion excites degrees of freedom
  a static rig cannot.

Acceptance evidence: an online run on a public moving-platform dataset passing
the same gate discipline as v0.1, with provenance recording the odometry source.

### Pillar 2: Camera-LiDAR as an evaluated modality

Promote the experimental projection and edge-alignment overlay to a first-class
candidate evaluation using the exact ADR 0004 protocol: train/holdout split,
perturbation known-bad probes, PASS/FAIL/INCONCLUSIVE gates with recorded
thresholds, an observability statement, and schema-validated artifacts.

This is not a new SOTA targetless algorithm. It is an honest evaluation layer
for externally produced camera-LiDAR candidates.

### Pillar 3: Temporal calibration evidence

Fold sensor-to-sensor time-offset estimation and validation into online gates.
Foundations already exist: CustomMsg `offset_time`, PointCloud2 stamps, and
replay provenance. Work proceeds in two steps:

1. A time-offset perturbation probe that injects known offsets and shows the
   metric catches them.
2. An estimated-offset report field with its own gate.

Initial implementation (post-run holdout-RMSE probes and 1D estimator on
motion-compensated rosbag2 replays) is in tree; synthetic fixtures recover
injected offsets to ±5 ms when the extrinsic is held at ground truth. Real
Indoor02 cross-sensor runs record honestly that holdout RMSE can be flat once
the online solver has adapted, so large restamp biases are not isolated without
tighter probe margins or extrinsic anchoring — see the tutorial temporal table.

## Consequences

Each pillar lands as small PRs with the established discipline: schema-validatable
configs and artifacts, provenance everywhere, tests for calibration changes,
real-public-data validation reported honestly (including negative results), and
a ROS-independent core with bag and mcap parsed directly.

Pillar order is priority order. Camera-LiDAR and temporal work may proceed in
parallel once the rosbag2 reader lands.

v0.2 does not claim motion-based SOTA calibration. It claims evidence-grade
evaluation of it.

## Deferred

- IMU candidate evaluation.
- Full radar extrinsic calibration (`radar_lidar_velocity_consistency` stays
  experimental).
- Package releases to PyPI and version tags (maintainer decision to skip).
- GitHub Pages deployment (manual, plan-dependent step).

## Acceptance evidence

All three pillars landed on `main` as small PRs with schema-validatable artifacts,
provenance, tests, and honest public-data reporting (including negative results).

### Pillar 1: Moving-platform support

**PRs:** #26 (pure-Python rosbag2/MCAP reader), #27 (odometry-aware motion
compensation), #28 (TIERS Indoor02 real-data validation), #29 (KISS-ICP rig-frame
odometry, odometry-extrapolation gate, clock-domain restamping), #30 (per-point
deskew via `point_time_field`).

**Acceptance measurements:**

- Synthetic moving-rig fixture (`tests/unit/test_rosbag2_online_motion.py`):
  motion compensation recovers injected ground truth to ~0 cm / ~0°; the static
  control lands ~5.72 cm / ~6.12° and fails the online gate.
- TIERS Indoor02 real-data validation (#28) caught two rosbag2 reader bugs that
  synthetic mirror-image tests could not: CDR encapsulation endianness keyed off
  the wrong header byte, and message-mode zstd/lz4 compression declared in
  `metadata.yaml` ignored.
- Identity self-consistency control on Indoor02 (duplicate Velodyne topic,
  KISS-ICP odometry, bounded replay budget in
  `docs/tutorials/public_datasets.md`): motion compensation recovered known
  identity to ~9.6 cm / ~1.2° versus ~98 cm / ~37° for the static control
  (~10× extrinsic error reduction) — the headline acceptance evidence for this
  pillar.
- Per-point deskew (#30): synthetic proof in tree; on the Velodyne selftest
  real sequence deskew did not materially tighten the identity residual (~10.5 cm
  / ~1.5° vs ~9.5 cm / ~1.2° per-message baseline) — residual dominated by
  LiDAR-odometry drift once intra-scan timing is corrected.

**Honest open items:** LiDAR-odometry drift dominates the residual after deskew on
real moving-platform data. Absolute Velodyne→Ouster extrinsic accuracy against the
TIERS GICP seed is not established (~85 cm / ~28° vs ~0.37 m inter-sensor
baseline on run C). MOCAP odometry runs carry an unknown rigid-body alignment
caveat.

### Pillar 2: Camera-LiDAR as an evaluated modality

**PR:** #31.

**Acceptance measurements:**

- Projection/edge-alignment overlay promoted to ADR-0004-protocol evidence rows
  (Candidate Support / Holdout Edge Alignment / Known-Bad Controls /
  Observability Statement / Decision Boundary) with recorded thresholds and
  family-aware assessment policy gates.
- KITTI committed-sample demonstration on the bundled synthetic fixture
  (`examples/public_datasets/kitti_lidar_camera_evidence/`): baseline reference
  extrinsic passes holdout rows but FAILs Known-Bad Controls (0/24 mandatory
  detections on the tiny fixture — discipline working); `+2.0 m` x translation
  with `use_frame_graph_candidate: true` flips Holdout Edge Alignment to FAIL.

**Honest open items:** Camera-LiDAR PASS on the full protocol requires
probe-capable full-scale KITTI frames; the committed 76 KB synthetic fixture
cannot reach falsification power on perturbation probes by design.

### Pillar 3: Temporal calibration evidence

**PR:** #33.

**Acceptance measurements:**

- Post-run time-offset perturbation probes and 1D holdout-RMSE estimator on
  motion-compensated rosbag2 replays; synthetic moving-rig fixtures (frozen
  ground-truth extrinsic) recover injected offsets to ±5 ms.
- Real Indoor02 runs (KISS-ICP Velodyne→Ouster and Velodyne selftest temporal
  configs) record honestly that holdout RMSE is nearly flat once the online
  solver has adapted: probe detection 0/3 and 1/3 respectively, temporal verdict
  **FAIL** (probe gate) — strict no-falsification-power discipline.

**Honest open items:** Temporal probes lack power after solver adaptation without
extrinsic anchoring or tighter probe margins for the measured noise floor; large
restamp biases are not isolated on converged real-data sessions.
