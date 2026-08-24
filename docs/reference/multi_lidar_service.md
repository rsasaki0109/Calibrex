# Multi-LiDAR service

`multi-lidar-service` is a read-only field-replacement gate for a vehicle's
multi-LiDAR frame graph. It binds the old and new physical sensor identities,
baseline/candidate edge artifacts, the exact mutable incident edge allowlist,
and a candidate raw-replay result by SHA-256.

```bash
calibrex multi-lidar-service plan definition.yaml --output plan.yaml
calibrex multi-lidar-service evaluate plan.yaml --candidate-replay replay-result.json --output evaluation.yaml
calibrex multi-lidar-service verify evaluation.yaml --plan plan.yaml --json
```

Evaluation is fail-closed. It requires a self-digest-valid plan and inputs,
rejects changed edges outside the allowlist, requires candidate replay
`PASS`/`ADOPT` with `holdout`, `known_bad`, and `observability` gates all
`PASS`, and checks candidate graph connectedness and cycle closure budgets.
The result is `READY` or `HOLD` with machine-readable reasons. Lifecycle and
Autoware references may be recorded for provenance, but this service never
applies package or registry mutations.

The synthetic fixture is in
[`examples/multi_lidar_service/synthetic`](../../examples/multi_lidar_service/synthetic/plan.yaml).
