# ADR 0008: v0.4 Direction — Full-Scale Modality PASS and Absolute Time

## Status

Proposed.

## Context

v0.3 (ADR 0007, Accepted) made the trajectory an evaluated artifact, restored
temporal falsification power via extrinsic anchoring, and added LiDAR-IMU
rotation evidence. Its acceptance evidence left precise open items:

- Camera-LiDAR evidence rows exist and the known-bad flip works, but the
  committed 76 KB KITTI fixture cannot yield a PASS: perturbation probes have
  no power at that scale. A full-scale demonstration is deferred work.
- Radar remains the last experimental modality: `radar_lidar_velocity_consistency`
  computes diagnostics (a nuScenes radar PCD reader and velocity metrics exist)
  but has no ADR 0004 protocol shape — no train/holdout split, no known-bad
  probes, no evidence rows, no gates.
- The anchored temporal estimator tracks injected offsets exactly
  (differential −50 ms on a +50 ms injection) but its ABSOLUTE baseline sits at
  −25 ms on a shared-clock control, consistent with VLP-16 sweep-end stamp
  semantics: "message time" is not "capture time", and the evidence layer has
  no declared convention for the difference.

## Decision

v0.4 extends slac along three pillars, in priority order.

### Pillar 1: Camera-LiDAR full-scale PASS on real KITTI

Download one real KITTI raw drive (the drive the committed fixture samples,
2011_09_26_drive_0005 sync, ~1–2 GB; manual download documented like the TIERS
bags) and demonstrate the full ADR 0004 protocol at scale:

- The dataset-reference candidate must reach a genuine PASS: holdout edge
  alignment, known-bad probes detectable (the fixture's 0/24 becomes a real
  detectable fraction), decision boundary pass, honest observability statement.
- A perturbed candidate must FAIL via the probes, not just via holdout drift.
- Acceptance: both runs' evidence tables recorded in the tutorial with
  unchanged default gates. If real-scale probes still lack power, that finding
  and its analysis are the deliverable — no gate tuning.

### Pillar 2: Radar as an evaluated modality

Promote radar velocity-consistency to the fifth evidence family
(`radar_lidar`), completing the modality roster (lidar pair, camera, temporal,
imu, radar):

- Doppler-velocity consistency of an externally supplied radar extrinsic
  candidate against the ego trajectory: project ego velocity (from the
  odometry/ego-pose track) onto each radar return's line of sight through the
  candidate rotation; residual vs measured Doppler with train/holdout split.
- Known-bad rotation probes (mandatory ±5°/±10° cases) with the established
  detectable-fraction discipline; static-scene fraction and excitation floors
  drive INCONCLUSIVE semantics.
- Real-data validation on nuScenes mini (radar + lidar + ego poses already
  supported by the reader/example assets); honest reporting.

### Pillar 3: Declared capture-time convention (absolute temporal accuracy)

Give the evidence layer an explicit scan reference-time convention so anchored
temporal estimates become absolute, not just differential:

- Per-sensor `capture_time_reference` (e.g. `header` | `mid_sweep` |
  `sweep_start`), applied where message times feed motion compensation and
  temporal evidence; recorded in provenance and the trajectory artifact.
- With per-point time fields available, `mid_sweep` uses the measured offset
  span; without them, it falls back to a declared constant and says so.
- Acceptance: the shared-clock selftest anchored baseline moves from −25 ms to
  ≈ 0 (within ±10 ms) under the declared convention, and the injection
  differential stays exact. The cross-sensor degenerate-separability caveat
  from v0.3 remains out of scope.

## Consequences

Same discipline as v0.2/v0.3: small PRs, schema-validated artifacts, provenance
everywhere, tests for calibration changes, real-public-data validation reported
honestly including negative results, ROS-independent core. Pillar order is
priority order; Pillars 1 and 2 are independent and may proceed in parallel;
Pillar 3 touches shared time semantics and lands alone.

v0.4 does not claim SOTA calibration for any modality. It claims that every
modality slac ships is evaluated at real scale under one falsification
discipline, with time semantics declared rather than implied.

## Deferred

- IMU translation/lever-arm evidence (rotation-only in v0.3).
- Yaw-probe observability under z-dominant excitation (motion-profile
  planning guidance stays a documentation note).
- Native-deskew rotation regression follow-up.
- Package releases to PyPI and version tags (maintainer decision to skip).
- GitHub Pages deployment (manual, plan-dependent step).
