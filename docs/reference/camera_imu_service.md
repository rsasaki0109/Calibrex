# Camera--IMU service

`camera-imu-service` is a read-only v0.1 admission gate for replacing a
camera/IMU pair. It binds old and new sensor identities, candidate replay,
explicit protocol/capture/holdout/known-bad/observability evidence roles,
metric observations, and a mutable-component allowlist by SHA-256.

```bash
calibrex camera-imu-service plan definition.yaml --output plan.yaml --json
calibrex camera-imu-service evaluate plan.yaml --output evaluation.yaml --json
calibrex camera-imu-service verify evaluation.yaml --plan plan.yaml --json
```

Evaluation is fail-closed. It requires a digest-valid plan and inputs, a
candidate replay with `PASS`/`ADOPT` and `holdout`, `known_bad`, and
`observability` gates all `PASS`, complete evidence roles with matching
protocol/capture bindings, correctly split/unit-tagged metrics within all
budgets, and no changed component outside `mutable_components`. The artifact
is `READY` or `HOLD` with machine-readable reasons. Lifecycle and Autoware
references are read-only; no package or registry mutation is performed.

The synthetic fixture in
[`examples/camera_imu_service/synthetic`](../../examples/camera_imu_service/synthetic/README.md)
is explicitly nonphysical and makes no accuracy claim.
