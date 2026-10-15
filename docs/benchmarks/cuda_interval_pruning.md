# Rejected conservative FP32 pruning for CUDA odometry

**Not adopted.** The integrated candidate preserves all outputs, but its final
complete-worker ratio is only 1.038x. The existing asynchronous-normal runtime
is retained. The experimental implementation and its tests are preserved in
the source capsules and the measured revisions below.

The experimental CUDA scan adapter prunes spatial searches using outward FP32 intervals
while retaining the original FP64 candidate distances, neighbor ordering,
normals, residuals and systems. Both box/plane rejection and a candidate-point
prefilter use lower distance bounds. A rejected branch cannot contain a closer
candidate under the retained arithmetic.

Conversions enclose each coordinate or split plane using directed rounding.
Subtraction, nonnegative squared products and sums round downward, with an
additional FP32 margin before comparing against the FP64 candidate threshold.
Nonfinite lower bounds disable rejection. CUDA documents these
[directed conversions](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html)
and [arithmetic intrinsics](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html).
The implementation does not copy external kernel code.

Preparation is limited to 32–65,536 target points with coordinate magnitude
at most 1e6. A relative extent guard keeps tiny translated clouds on the native
path. Each registration iteration also checks source/matrix magnitude. The
native tree arrays and original FP64 kernels remain available for fallback.
Intervals add bounded target storage inside the existing payload policy; this
does not bound total allocator memory or process RSS.

A completion event orders cross-stream consumers and drains preparation on
abandonment. The target also records completion of its initial capture, covering
registration on another stream without normal priming. No CPU thread is added.
The map-registration adapter keeps its existing implementation.

## Complete public-data comparison

| Integrated candidate | Previous whole median | Candidate whole median | Ratio |
| --- | ---: | ---: | ---: |
| Repeated target extent scans | 54.383 s | 63.992 s | 0.850x |
| Reuse existing KD-tree extrema | 53.534 s | 51.576 s | 1.038x |

The first candidate is revision `61921e57`; the revised candidate is
`2249a4f7`. Both compare against `bb131739`, which already includes asynchronous
normal preparation. The revised pair ratios are 1.018x and 1.060x. The extent
reuse removes a duplicate point-cloud reduction; it does not establish a
substantial whole-calibration improvement. Both complete cohorts are retained.

Two alternating fresh-process pairs replay six seconds of the public GLIM Livox
bag with four map rounds, one BLAS thread, CUDA scan odometry and CUDA fitter
planes. Each worker starts with a separate empty CuPy compiler cache. Whole time
includes imports, initialization/compilation, every initial/gyro/feedback/final
odometry pass, association, all fits/covariance, noise updates, holdout/controls,
hashing, saves and frozen-import verification. No warmup is discarded.

The exact gate permits only the source-bound kernel identity, declared pruning
tag/counters and run metadata to differ. Every calibration/evidence field,
full per-round fit, trajectory, covariance, iteration and previous work counter
must match exactly. Actual interval query rows and fused calls are checked
against the total GPU work. CPU-only auditing verifies the source capsules,
executed imports, kernel identities, full fits, quality and aggregate timings.

The [bound archive](../assets/cuda_interval_pruning/README.md) retains all eight
results and full fits, raw logs, source capsules/manifests, launcher/schema,
source-bound regression run and schema comparison. Raw bag data and compiler
caches are not redistributed. Tests cover neighbor membership and distance
bits, ties, coordinate scales, translated clouds, payload boundaries,
cross-stream reads, abandoned trees and native fallback.

Host load is unreserved. These ratios describe this dense six-second workload,
not all point clouds, and are not multiplied by earlier cohorts. The clip has
no independent extrinsic truth and its final verdict remains INCONCLUSIVE.
CPU stays the default. The dense benchmark's coordinate/motion grids remain
off by default, and all 130 public schema/ledger exports are unchanged.

## Reproduce

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit bb131739c0d14d30fc0e1660ce4f2ee587d6d32e \
  --prior-fit-plane-backend cuda --current-fit-plane-backend cuda \
  --require-exact-spatial-results --max-seconds 6 --max-rounds 4 --repeats 2 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-interval-pruning-pairs
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_interval_pruning
```

Reproduction requires the measured candidate revision `2249a4f7` and its frozen
launcher, since the interval path is not enabled in the retained runtime.
Keep runtime source, HEAD, launcher and comparison schema fixed until every
worker finishes. Use a new output directory. The source archive contains the
candidate implementation without a public configuration or schema migration.
