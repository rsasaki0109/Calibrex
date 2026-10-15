# Asynchronous CUDA normal preparation

CUDA scan odometry submits a bounded dense target's normal queries and FP64
PCA before CPU source deskew and voxel preprocessing. A pinned host buffer and
completion event let that GPU work run during the CPU work. Registration waits
for completion and applies the existing native tie, eigengap and classification
repairs before reading any normals. No additional CPU thread is created.

The target owns every queued device buffer, pinned allocation and event until
completion. Abandoning a target drains its queued writes. Failed repairs
invalidate affected rows so a retry cannot consume partially prepared normals.
Submission is limited to one 65,536-row chunk and a source-size hint of at least
one tenth of the target; sparse hints and larger maps retain demand-driven
preparation. Heavily cropped sources can still overestimate that hint.

Normal kernels, neighbor budgets, native repair rules, registration stopping,
motion/geometry grids, fitting and evidence evaluation are unchanged. CUDA
execution reports include `normal_schedule=async-v1` and the actual
`gpu_async_normal_batches` counter. CPU remains the default.

## Complete public-data comparison

| Scope | Synchronous preparation | Asynchronous preparation | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 57.791 s | 51.622 s | 1.120x |

Two alternating fresh-process pairs replay the first six seconds of the public
GLIM Livox bag with four map rounds and one BLAS thread. Both use CUDA scan
odometry and CUDA fitter planes. Every worker has its own initially empty CuPy
compiler cache. Whole time includes imports, decoding, initialization and first
compilation, every odometry pass, association, fitting/covariance, noise updates,
holdout, signed controls, hashing, saves and frozen-import verification.

The exact comparison permits only the declared scheduling identity/counter and
run metadata to differ. Every calibration/evidence field, full per-round fit,
trajectory, covariance, iteration count and pre-existing work counter must match
exactly. This verifies preservation of the previous output; the clip has no
independent extrinsic truth and its final verdict remains INCONCLUSIVE.

Host load is unreserved. These observations apply to this dense six-second
workload. They are not multiplied by ratios from other cohorts. The legacy
normal-preparation timer measures host time in submission/completion calls,
including repairs; it excludes CPU work between the two calls and does not time
overlapped GPU work. Its ratio is not a normal-kernel arithmetic speedup.

The [bound archive](../assets/cuda_normal_overlap/README.md) retains all four
full results, per-round fits, logs, source capsules/manifests, launcher/schema,
correctness receipt and schema-export comparison. The audit runs without CUDA.
Raw bag data and compiler caches are not redistributed.

## Reproduce

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit 105e1031206ba6c551d08a1b07742b0307e5f523 \
  --prior-fit-plane-backend cuda --current-fit-plane-backend cuda \
  --require-exact-preparation-results --max-seconds 6 --max-rounds 4 --repeats 2 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-async-normal-overlap-pairs
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_normal_overlap
```

Use committed runtime sources and a new output directory. Keep runtime source,
HEAD, launcher and comparison schema fixed until all workers finish. The dense
settings use three local scans, 3 cm voxels, a 1 nm coordinate grid and returned
motion grids of 1 micrometre translation / 0.1 microradian rotation components.
The grids remain off by default. The existing CUDA odometry option enables
bounded preparation automatically; no public config or result schema changes.
