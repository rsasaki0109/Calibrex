# Synthetic raw-capture replay fixture

This fixture is intentionally synthetic. It proves the digest, readiness,
holdout, known-bad, comparison, and CLI plumbing without claiming a physical
LiDAR capture or a real-world calibration accuracy result. The candidate and
control results are precomputed deterministic artifacts bound to
`synthetic-definition.yaml`.

Run:

```text
calibrex replay plan examples/raw_replay/synthetic/synthetic-definition.yaml \
  --output outputs/raw-replay-plan.json
calibrex replay run examples/raw_replay/synthetic/synthetic-definition.yaml \
  --output-dir outputs/raw-replay-synthetic --json
```

No Autoware workspace is modified by this replay.
