# Batching timed Camera-LiDAR evaluation

Complete prepared-input refinement changes from **38.054701 to
5.140645 s (7.402710x)** compared with the
already optimized [evaluation-reuse implementation](continuous_camera_lidar_evaluation_reuse.md).
The artifact-backed calibration API changes from **35.622119 to
2.203405 s (16.166853x)**. All six complete results
and optimizer histories match exactly. This is automatic CPU batching using
the existing NumPy dependency.

## Identical workload

The prepared problem is byte-identical to the preceding comparison: nine KITTI
scans, all 9,216 selected original points and original matching OXTS velocities.
The same seven training frames/7,168 points and two holdout frames/2,048 points
converge after exactly 341 candidates, under the original 500-candidate bound.
Point times, camera/extrinsics, image means and fixed noise are declared by the
archived generator. **Generated correspondences and point times establish
performance and numerical parity, not independent calibration accuracy or
raw-sensor pipeline speed.** No locked accuracy cohort is opened.

The initial pose/clock, capture order and split, candidate order and ties,
Cauchy loss, confidence, covariance, optical/depth gates, temporal support,
sensitivity probes, step minima and stopping policy remain unchanged.
Results also match the preceding archive's result, excluding only the two
declared top-level run/source provenance fields.

## Ordered array computation

One solve prepares the selected points, means, covariances, confidences and
row times per capture. Each capture retains two contiguous deskew arrays,
keyed by binary clock value and the complete capture policy. Constant body
twist is evaluated in arrays; piecewise trajectories continue through the
existing scalar deskew primitive and share the array projection path.

The array operations retain the scalar core's axis-angle matrix conversion,
all four quaternion normalization stages, both Hamilton products including
zero terms, translation/velocity order and global reduction order. Scalar
`math.sin`/`math.cos` are retained because vector trig can change rounding.
There is no matrix reformulation, lower precision, rounded time or approximate
search. Projection and Mahalanobis errors are batched before the original
ordered Cauchy reduction.

The cache belongs to a single solve. Binary keys retain signed zeros, policy
keys retain units and stamp semantics, and stored point arrays are read-only.
The existing conservative 128 MiB allowance at 4,096 bytes per input point
also covers the capture arrays. Larger problems keep the scalar path.
Public signatures, defaults, result types and core schemas remain unchanged.

| Observed work per complete run | Cached baseline | Batched |
| --- | ---: | ---: |
| Candidate evaluations | 341 | 341 |
| Point motion-compensation computations | 556,032 scalar | 556,032 in 543 batches |
| Point projections, including sensitivity and holdout | 2,462,720 scalar | 2,462,720 in 2,405 batches |

The independent audit replays every capture visit and exact binary clock key
from the complete history. Measured work must equal this replay in every run;
the speedup comes from grouping arithmetic and avoiding per-point Python work.

## Complete fresh-process comparison

Three alternating pairs use identical inputs/dependencies, frozen full core
trees and actual Git identities, with one BLAS thread. The host is unreserved;
all variation is retained and no run is dropped or labeled warmup.
Whole timing includes imports, real Git prechecks, problem parsing/validation,
construction, optimization and work counters, result packaging/saving and
imported-source hashing. Generation, freezing, source compression and audits
are untimed. API timing covers the artifact-backed calibration call.

| Pair (execution order) | Cached whole s | Batched whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (cached, batched) | 41.869765 | 5.188824 | 8.069x |
| 2 (batched, cached) | 38.054701 | 5.140645 | 7.403x |
| 3 (cached, batched) | 37.244182 | 3.387998 | 10.993x |

The **7.402710x** headline is the ratio of three-run medians
within this comparison. It is not multiplied by earlier measurements.
Baseline: `eb8acf735bf5bf56e9b5325c962b8c23c5336553`.
Measured runtime: `a892491de1c7f97cb9ae87f592b4b0bdc868ea1a`.
Only the continuous-time Camera-LiDAR solver changes in the full core manifests.

## Validation and provenance

The implementation and related suite passes 116 cases, including every
coordinate bit, all rotation-matrix quaternion branches, signed zero, cache
eviction, seconds/nanoseconds, rolling shutter in both directions,
constant/piecewise motion, empty/filtered points, scalar memory fallback,
fresh solve ownership and complete optimizer histories with uncertainty
ablations. Ten counter/provenance regressions and fifteen registry checks
passed before measurement. Strict mypy checks all 378 core files.
The archive includes final archive/registry, Ruff, strict docs and unchanged
129-public-schema-plus-ledger verification receipts.

The [complete portable archive](../assets/continuous_camera_lidar_batching/README.md)
contains the input/generation sources, source capsules/manifests, launcher and
schemas, all six complete results/timings/counters/logs and validation receipts.
The audit checks all pins, actual imported source hashes, real full/short Git
identities, original source-point/motion lineage, full histories/options/splits,
observed work and finite positive timings. Only `provenance.created_at` and
`provenance.git_commit` are excluded from cross-run numerical comparison;
the source identity is independently verified. KITTI-derived data retains
its attribution and CC BY-NC-SA 3.0 separately from Apache-2.0 source code.

## Reproduce

Audit without a calibration run or raw download:

```bash
python -m tools.benchmark_continuous_camera_lidar_reuse \
  --audit-only docs/assets/continuous_camera_lidar_batching
```

For a fresh comparison, decompress the original generation inputs into a new directory:

```python
import gzip
from pathlib import Path
archive = Path('docs/assets/continuous_camera_lidar_batching')
inputs = Path('outputs/timed-camera-batch-remeasurement-inputs')
inputs.mkdir(parents=True, exist_ok=False)
for name in ('source-points.json', 'motion-control.json'):
    (inputs / name).write_bytes(gzip.decompress((archive / (name + '.gz')).read_bytes()))
```

```bash
python -m tools.benchmark_continuous_camera_lidar_reuse --ordered-batches \
  --baseline-commit eb8acf735bf5bf56e9b5325c962b8c23c5336553 \
  --problem docs/assets/continuous_camera_lidar_batching/problem.json \
  --input-provenance docs/assets/continuous_camera_lidar_batching/input-provenance.json \
  --control-generator docs/assets/continuous_camera_lidar_batching/control-generator.py.txt \
  --source-points outputs/timed-camera-batch-remeasurement-inputs/source-points.json \
  --motion-control outputs/timed-camera-batch-remeasurement-inputs/motion-control.json \
  --output-dir outputs/timed-camera-batch-new --repetitions 3
```
