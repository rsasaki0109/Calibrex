# ADR 0010: v0.5 Direction — Reconcile v0.4 and Ship the Operational Trust Platform

## Status

Proposed (2026-09-27). Reconciles ADR 0008, which remains Proposed as a
historical record and is not rewritten.

## Context

ADR 0008 set three v0.4 pillars: full-scale Camera-LiDAR PASS on real KITTI,
Radar as an evaluated modality, and a declared capture-time convention. The
implementation has since moved past that snapshot:

- **Pillar 1 (KITTI full scale).** An opt-in `@pytest.mark.kitti` falsification
  benchmark shows the vendor reference PASS and a declared known-bad FAIL
  without retuned gates. It is not yet published as a maintained benchmark
  artifact.
- **Pillar 2 (Radar).** Radar has a protocol, holdout, signed controls, robust
  ego velocity, trajectory rotation/yaw, lever-arm/time, joint spatiotemporal
  solvers, a nuScenes adapter, `radar_msgs/RadarScan` intake, and a
  radar-service replacement gate. Public nuScenes excitation stays too weak for
  a conclusive full-system PASS. The limit is the data, not missing algorithms.
- **Pillar 3 (capture time).** `capture_time_reference` and per-point capture
  time exist, but TIERS real-data evidence is INCONCLUSIVE. No end-to-end
  acceptance run has shown message time and capture time converging under the
  declared convention.
- **Deferred list.** Native continuous-time IMU lever-arm, clock-offset,
  accelerometer-bias/gravity, and intrinsics factors now exist with synthetic
  holdout and signed controls. PyPI publication remains skipped. Installation
  uses the verified GitHub Release wheel.

Work between 2026-08-21 and 2026-08-25 also added an operational trust layer
that ADR 0008 did not anticipate:

- capture manifests;
- digest-bound raw replay;
- an append-only lifecycle registry;
- multi-LiDAR, camera--IMU, and radar service-replacement gates;
- Autoware promotion and smoke gates;
- a fail-closed Koide real-pilot handoff;
- a required `slac.result.provenance/v0.1` contract.

On 2026-09-27, `calibrex external-run evaluate-camera-imu` added the same
Calibrex-side falsification evidence for Kalibr outputs. The Koide pilot
already applies it to its handoff.

## Decision

1. **Close Pillars 1 and 2 as protocol work.** Their remaining items are
   evidence publication (a maintained KITTI benchmark artifact) and data
   acquisition (Radar captures with stronger excitation). They no longer block
   a release, and new Radar algorithms are not a priority.
2. **Carry Pillar 3 forward unchanged.** The acceptance condition stays as ADR
   0008 wrote it: the shared-clock anchored baseline must fall within ±10 ms
   under the declared convention, and the injection differential must stay
   exact. Gates are not relaxed.
3. **Keep the trust platform release-ready, but do not cut a release yet.**
   The maintainer decided on 2026-09-27 not to release for now. Any future
   release still follows the adoption-track gate: a clean wheel, no schema
   drift, a working quickstart, and a green default test suite. It would be
   distributed as a GitHub Release wheel, not on PyPI.
4. **Treat the external-run contract as proven for Koide and Kalibr.** An
   external tool's own metrics stay non-comparable. Every imported candidate
   still needs Calibrex-side holdout and signed-control evidence before
   adoption. Kalibr lever-arm (translation) evidence stays open.
5. **Favour user-facing paths.** Priorities are a single
   LiDAR-IMU calibration command, lifecycle-registry integration with
   Calibration CI, and the Pillar 3 acceptance demonstration, ahead of new
   solver families.

## Consequences

The development roadmap becomes the dated inventory for v0.5 work, and this ADR
records why the v0.4 pillars close. Large public datasets remain outside the
repository. Opt-in tests read them from environment-variable paths and skip
when the data are absent, so the default suite stays small and offline.

The project rules still apply: a ROS-independent core, no GPL code in
`src/calibrex`, typed public APIs, schema-validatable artifacts, and
provenance on every generated result.
