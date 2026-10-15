# Complete spline evaluation reuse comparison

| Scope | Previous | Shared evaluations | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 92.704 s | 86.305 s | 1.074x |
| Four CPU fits, including covariance | 20.395 s | 13.370 s | 1.525x |

Two alternating fresh-process pairs replay the original public GLIM Livox bag's
first six seconds with four map rounds. This is a descriptive observation on an
unreserved host. The CPU fitter now shares bounded evaluation work within each
fit; both variants run the same adopted CUDA scan-odometry kernels. Whole and
fit ratios have different scopes and are not multiplied.

The previous revision is `28acb5819bc254f1f8c9889c0bd4b03f2deb3c3f` and the current revision
is `6182c4793120a178ce1a5e204a8a0db3833d7ea2`. The input bag SHA-256 is
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. No raw scans or compiler caches are redistributed.

Every worker uses one BLAS thread and a separate initially empty CuPy cache.
First GPU initialization and compilation belong to whole time. The unchanged
dense policy uses 0.03 m voxels, three local-map scans, a 1 nm coordinate grid
and returned-motion grids of 1 micrometre translation and 0.1 microradian
rotation-vector components. Those grids remain off by default.

Every complete physical/evidence field and execution report is exactly equal in
both pairs after excluding provenance timestamps, commands and source revisions.
All four complete fit records in each pair match exactly after excluding wall
time: calibration, covariance, trajectory, costs, status, iterations and dimensions.
The protocol explicitly requires this stronger equality alongside the existing
physical checks. Final verdicts remain **INCONCLUSIVE**; this clip supplies no
independent extrinsic truth.

Whole timing includes startup/imports, decoding, every initial/gyro/feedback/final
odometry pass, association, fitting/covariance, noise updates, evidence, signed
controls, hashing, saves and import/source verification. Fitting remains on CPU.
The publication index binds all four full results, original measurements/logs,
both source capsules/manifests, launcher, schema and publisher.

Audit without a GPU or bag:

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/spline_evaluation_reuse
```

See [implementation and reproduction](../../benchmarks/spline_evaluation_reuse.md).
