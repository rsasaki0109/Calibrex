# Radar Service

`calibrex radar-service` is a ROS-independent, read-only v0.1 admission
boundary for replacing an automotive radar serial or mount. It does not run a
solver and does not mutate a lifecycle registry, Autoware package, or sensor
kit. It binds the old/new radar identities, a candidate raw-replay result,
captured `radar_msgs/RadarScan` evidence, and cross-modal radar--LiDAR/camera
evaluation artifacts by SHA-256.

## Gates

An evaluation is `READY` only when every declared digest and self-digest is
valid, the candidate replay is `PASS`/`ADOPT`, replay `holdout`, `known_bad`,
and `observability` gates are `PASS`, all required evidence roles are present,
protocol/capture bindings agree, metric units and splits are valid, and every
changed component is in `mutable_components`. Otherwise it emits `HOLD` with
actionable reasons.

The plan makes both conventions explicit: `doppler_sign_convention` and
`frame_convention`. Evidence that states either convention must match the
plan. The metric budgets cover absolute range bias (m), absolute azimuth bias
(deg), Doppler scale error (ratio), time offset (s), radar--LiDAR association
RMSE (m), minimum FOV overlap ratio, minimum known-bad delta, and minimum
observability rank.

Lifecycle and Autoware references are input-only digest refs. The service
never applies them. Relative refs are rebased when a plan is written to a
different output directory, so the emitted plan remains portable.

## CLI

```bash
calibrex radar-service plan definition.yaml \
  --output /tmp/radar-service/plan/plan.yaml --json
calibrex radar-service evaluate /tmp/radar-service/plan/plan.yaml \
  --output /tmp/radar-service/evaluation/evaluation.yaml --json
calibrex radar-service verify /tmp/radar-service/evaluation/evaluation.yaml \
  --plan /tmp/radar-service/plan/plan.yaml --json
```

The self-contained [synthetic fixture](../../examples/radar_service/synthetic/README.md)
is intentionally synthetic, nonphysical, and makes no accuracy claim. It
uses the existing ROS-independent RadarScan adapter contract; it is only a
schema/gate smoke fixture.
