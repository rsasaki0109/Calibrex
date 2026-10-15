# Rejected fused CPU scan covariance

An optional Numba candidate fused the target-normal gather, mean and six
distinct FP64 covariance entries while retaining SciPy's neighbour ordering,
native NumPy `eigh` and planarity checks. It avoided the gathered and centered
`(rows, neighbours, 3)` arrays, compiled without fast math or parallel execution,
and reported actual compiled/fallback work and adapter source identity.

The complete public-bag experiment did not demonstrate an improvement, so the
candidate is **not adopted**. Current runtime under `src/calibrex` remains the
native implementation from `d612581`, identical to `7b367723`. The candidate
source remains in commit `2d6e5caef559b4383bf4da44739bc9dc0146e619` and the bound
source capsules; its CLI/API additions are not available on the final branch.

## Complete public-data comparison

| Scope | NumPy covariance | Fused Numba covariance | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker, including first compilation | 165.422 s | 177.744 s | 0.931x |
| Lazy normal preparation, including neighbourhood queries | 59.828 s | 51.136 s | 1.170x |
| Four unchanged fits, including covariance | 14.396 s | 18.932 s | 0.760x |

Two fresh-process pairs alternate NumPy/Numba then Numba/NumPy on an unreserved
host. Both variants use identical frozen sources, eight CPU query workers and
one BLAS thread. Every compiled worker starts with an empty, separate Numba cache;
imports and first compilation belong to complete-worker timing. The fitter is
unchanged, and its wall time also varied. These observations do not establish
a stable slowdown ratio. The local preparation improvement was insufficient to
justify an additional runtime/backend option.

The workload is the original public GLIM Livox bag's first six seconds with four
map rounds, 0.03 m voxels and three local-map scans. Both variants explicitly use
a 1 nm coordinate grid and returned-motion grids of 1 micrometre translation and
0.1 microradian rotation-vector components. These grids remain off by default.
The candidate's tests also compare complete unquantized trajectories and
registration quality exactly, including tied neighborhoods and gap scaling.

The [bound archive](../assets/cpu_fused_scan_covariance/README.md) retains all four
complete results, every full per-round fit, original measurements/logs, both
source capsules/manifests, launcher, schema, publisher and the correctness run
with bound test sources. The CPU-only audit checks declared adapter choices,
source identities, actual work and empty compilation caches before comparing
all physical/evidence fields and fitted arrays exactly. Only independently
validated execution metadata and wall times are excluded.

The candidate's retained correctness run passed 92 tests. Its receipt binds the
source revision, source manifest, exact test sources, launcher and original log.

Both pairs preserve every complete physical/evidence field, including the stored
initial estimate, and every complete fit record exactly. All final verdicts
remain INCONCLUSIVE; this clip supplies no independent extrinsic truth. This
comparison concerns dense CPU odometry and does not establish GPU gains or
performance on default coarse scans.

To reproduce, use the candidate commit above in a separate checkout, install its
`numba` extra, and use a new output directory:

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cpu-cpu \
  --prior-query-workers 8 --current-query-workers 8 \
  --prior-normal-backend numpy --current-normal-backend numba \
  --max-seconds 6 --max-rounds 4 --repeats 2 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-cpu-normal-covariance-pairs
```

The final branch can audit the archive without a GPU, JIT compiler or bag:

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cpu_fused_scan_covariance
```
