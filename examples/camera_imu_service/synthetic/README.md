# Synthetic camera--IMU service fixture

This fixture exercises the v0.1 camera--IMU replacement gate with a
self-contained candidate replay and metric evidence files. It is deliberately
synthetic, nonphysical, and makes no accuracy claim. The values only exercise
schema validation, digest binding, split/unit checks, and READY/HOLD plumbing.

Run it from the repository root:

```bash
calibrex camera-imu-service plan \
  examples/camera_imu_service/synthetic/plan.yaml \
  --output /tmp/camera-imu-service/plan/plan.yaml --json
calibrex camera-imu-service evaluate \
  /tmp/camera-imu-service/plan/plan.yaml \
  --output /tmp/camera-imu-service/evaluation/evaluation.yaml --json
calibrex camera-imu-service verify \
  /tmp/camera-imu-service/evaluation/evaluation.yaml \
  --plan /tmp/camera-imu-service/plan/plan.yaml --json
```

The plan output and evaluation output intentionally live in different
directories. Relative source references are rebased by the plan command and
remain digest-verifiable from the emitted plan directory.
