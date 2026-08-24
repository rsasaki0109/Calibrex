# Synthetic radar service fixture

This fixture exercises the v0.1 automotive radar serial/mount replacement
gate using captured `radar_msgs/RadarScan`-shaped evidence and cross-modal
radar--LiDAR/camera metric records. It is explicitly synthetic, nonphysical,
and makes no accuracy claim. Values only exercise schema validation, digest
binding, split/unit checks, convention preservation, and READY/HOLD plumbing.

The service is read-only: lifecycle and Autoware references are verified but
never edited. Run from the repository root:

```bash
calibrex radar-service plan \
  examples/radar_service/synthetic/plan.yaml \
  --output /tmp/radar-service/plan/plan.yaml --json
calibrex radar-service evaluate \
  /tmp/radar-service/plan/plan.yaml \
  --output /tmp/radar-service/evaluation/evaluation.yaml --json
calibrex radar-service verify \
  /tmp/radar-service/evaluation/evaluation.yaml \
  --plan /tmp/radar-service/plan/plan.yaml --json
```

The plan and evaluation outputs intentionally use different directories.
Relative source references are rebased by the plan command and remain
digest-verifiable from the emitted artifacts.
