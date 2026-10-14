# Spatial scheduling of CUDA registration queries

Dense CUDA odometry sources now sort their query-row indices by 63-bit Morton
codes once per registration. Nearby points run together and reuse nearby tree
nodes. The order is retained across all ICP iterations. Sources below 4096 rows
keep their original order to avoid the spatial-sort setup cost.

Each thread still reads its original point and writes transforms, neighbor
distances/indices, repair flags and all 31 system terms to its original row.
The existing reduction consumes those same rows in the same order. Coordinate,
distance, normal and robust-system arithmetic remains FP64 with FMA disabled.
Neighbor selection, uncertainty repairs, pose solves and stopping rules remain
unchanged. Morton codes affect scheduling only, not neighbor precision.

The device builds Morton codes and sorts row indices; CPU supplies six coordinate
extrema. New counters record actual dense-source ordering calls/rows and the
custom code-generation kernel calls. Internal CuPy sorting launches are not
counted as custom kernels. The bounded payload estimate includes the additional
codes and row-index storage under the existing 1 GiB cap.

## Complete public-data comparison

| Scope | Previous CUDA median | Spatial-scheduling median | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 137.106 s | 135.282 s | 1.013x |
| Complete registration | 38.354 s | 36.116 s | 1.062x |
| Nested normal preparation | 14.217 s | 13.658 s | 1.041x |
| Unchanged fits/covariance | 28.471 s | 34.384 s | 0.828x |

Three alternating previous/current pairs replay the first six seconds of the
public GLIM Livox bag with four map rounds and identical CUDA odometry settings.
Registration timing includes ordering setup, normal preparation, all iterations,
reductions and native uncertainty repairs. Normal preparation is nested inside
registration; these component ratios must not be multiplied.

Fresh workers use one BLAS thread and separate initially empty CuPy compilation
caches. Runtime/driver caches may persist; background host load is unreserved.
Whole timing includes imports, decoding, compilation, all odometry passes,
association, fits/covariance and validation. This chat launched no heavy local
tests during the cohort. Another independent CUDA sanitizer workload was
observed on the shared host during the first pair. This is a contended cohort:
the timings cannot establish the isolated performance effect of this change.
The prototype stays a draft pending measurements without competing GPU workloads.

A [monitored repeat](../assets/cuda_source_locality/repeat-v2/README.md) started
with no CUDA compute processes, then detected competing workloads in 32 of 52
ten-second observations. All three quality comparisons remained exactly equal.
That repeat also cannot establish an isolated performance effect. The observer
and complete repeat are source-bound and checked by the parent archive audit.

The [CPU-auditable archive](../assets/cuda_source_locality/README.md) retains all
complete workers, full fitted evidence, exact sources and publication bindings.
Real-data verdicts remain **INCONCLUSIVE** without independent extrinsic truth.
No scan, point, neighbor, iteration or validation budget is reduced.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit 9cf33f996aaaffaf40ab053fc6b3ad1d5f94f46d \
  --max-seconds 6 --max-rounds 4 --repeats 3 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/cuda-source-locality-comparison
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_source_locality
```

CPU is still the default and all coordinate/motion grids remain disabled by
default. Tests check bijective scheduling, small/empty sources, duplicate-code
stability, every fused output and reduced-system bit, exact registration without
motion grids, temporal chaining, native repairs and false publication counters.
