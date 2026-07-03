# ADR 0005: v0.2 Motion, Camera, and Temporal Direction

## Status

Proposed.

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

v0.2 extends Calibrex along three pillars, in priority order.

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
