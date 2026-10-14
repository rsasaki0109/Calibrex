# Packed CUDA neighbors: complete result validation

Baseline source: `9cf33f996aaaffaf40ab053fc6b3ad1d5f94f46d`. Prototype source:
`081052ee3d0a7bb56544ba14d7a6b946aed34629`. Public GLIM Livox input SHA-256:
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw data is not redistributed.

One complete previous/current pair replays six seconds with four map rounds,
identical point/neighbor/iteration budgets and all 32 signed controls. Calibration,
holdout values, controls and all four full fitted trajectory/covariance/cost
records are exactly equal. Real-data verdicts remain INCONCLUSIVE without
independent extrinsic truth.

The ten-second observer records 23 samples, 13 with
foreign compute processes and 0 with unknown membership. **This is a
shared-GPU result validation, not an isolated speed comparison.** Complete
worker timings remain in attempts.json and measurement.json for audit; they
cannot establish a performance effect. No process names or IDs are published.
Periodic process sampling does not establish continuous exclusivity or reserve
CPU resources. Fresh workers use separate initially empty compilation caches.

The prototype packs target coordinates in the existing leaf order once per
target. Original coordinates and point IDs remain authoritative for normals,
system terms and native repairs. Exact max-heap candidate search sorts its final
neighbors by distance then original point ID. No geometric precision or
validation budgets are reduced. The unsuccessful source-row scheduling
prototype is excluded from both versions.

The index binds two complete workers, frozen sources/manifests, launcher/schema,
device tests, observer source/schema/report and publisher. The CPU audit checks
actual packed rows/bytes, source and result bindings and registration timing.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_packed_neighbors
```
