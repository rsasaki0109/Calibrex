# Reusing timed Camera-LiDAR evaluation work

Complete prepared-input seven-DoF Camera-LiDAR refinement changes from
**136.523854 to 33.203555 s (4.111724x)**.
The artifact-backed API changes from **134.120710 to 30.923065 s
(4.337239x)**. All six complete results, candidate objectives,
accept/reject decisions and final holdout values match exactly. This is automatic
CPU acceleration of extrinsic-plus-clock refinement with ordinary dependencies.

## Workload and scope

The control reuses all 9,216 selected original KITTI points of the
[probabilistic refinement control](probabilistic_pose_batching.md), with 1,024
points from each of nine scans of `2011_09_30_drive_0018_sync`: 114-116, 229-231,
and 344-346. Original OXTS linear/angular velocities come from the same scan
indices. Image means are generated using a declared pinhole camera and extrinsic,
a 0.02 s clock offset, uniformly declared -0.1 to 0 s point offsets, fixed
0.05 px sinusoidal noise and the original anisotropic covariance/confidence.
The initial pose has a declared rotation/translation perturbation and zero offset.

**Image correspondences and point times are generated execution controls.**
This establishes performance and numerical parity, not independent calibration
accuracy, feature-extraction speed, raw-sensor pipeline speed or a SOTA result.
No locked accuracy cohort is opened. All 9,216 input points remain in both
variants; seven frames/7,168 points train and two frames/2,048 points hold out.
The original 500-evaluation bound, Cauchy loss, confidence and optical gates,
covariance, split, sensitivity test and step minima remain unchanged. Both
variants converge after exactly 341 candidate evaluations.

## Change

Point motion compensation depends on the clock candidate and fixed capture
inputs. Extrinsic-only candidates can share it. One solve now owns a fixed set
of point entries, each with a two-state clock cache. It prepares each original
body-frame point, row exposure time and inverse body/LiDAR transform once, using
the same constant-twist or piecewise-trajectory primitives and operation order.
Binary offset keys retain signed zeros; policy keys retain units and stamp
semantics. No time is rounded or interpolated differently.

Projection also prepares the candidate's two quaternion normalizations once;
point conversion, both Hamilton products and translation addition retain their
original floating-point order. Public APIs, results and schemas are unchanged.
The cache belongs to one solve, holds no cross-solve state, and uses a conservative
128 MiB estimate at 4,096 bytes per input point. Larger inputs retain scalar
deskew. Foreign points or different options also retain scalar behavior.

| Work per complete run | Prior | Reused |
| --- | ---: | ---: |
| Candidate evaluations | 341 | 341 |
| Point projections, including sensitivity and holdout | 2,462,720 | 2,462,720 |
| Actual point motion-compensation computations | 2,462,720 | 556,032 |

## Complete fresh-process comparison

Three alternating pairs use frozen source trees, actual Git identities,
identical dependencies and one BLAS thread. The host is unreserved; all
variation is retained. Whole-process timing includes imports, Git prechecks,
problem parsing/model validation, construction, optimization, work counting,
result packaging/saving and imported-source hashing. Generation, downloading,
freezing, source-input compression and external audits are untimed.

| Pair (execution order) | Prior whole s | Reused whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (prior, reused) | 136.523854 | 35.134018 | 3.886x |
| 2 (reused, prior) | 147.510615 | 33.203555 | 4.443x |
| 3 (prior, reused) | 121.157007 | 26.909230 | 4.502x |

The **4.111724x** headline is the ratio of three-run medians
for this workload. Baseline: `a844dc00354f987da0e103c0dc5bb9610df2eb2a`.
Measured runtime: `0fc27960653a895f4d5c907e8ee65a7e87e7685d`. Only the continuous-time
Camera-LiDAR solver differs in the complete core manifests. No run is discarded
or relabeled as warmup. An initial failed provenance-format check is retained
separately, with its error, source/input snapshots and incomplete attempt.
It contributes no successful timing or fabricated result. The retry changes
only the launcher: both variants' complete core manifests and input bytes are
identical between the failed and final cohorts.

## Validation and provenance

The portable audit checks every pinned source/input/helper/schema, each actual
import and full/abbreviated Git identity, all six complete results, every trace
row, the unchanged initial state/options/split, completion, work counts and
positive finite timers. Only the top-level `provenance.created_at` and
`provenance.git_commit` are excluded from cross-source numerical comparison;
the latter is independently checked against actual measured source identity.

The implementation and related suite passes 68 cases, and eight provenance
regressions cover actual core abbreviations and wrong/missing identities.
Regression coverage includes coordinate bits, signed zero, two-state eviction,
seconds/nanoseconds, rolling shutter in both directions, constant/piecewise
motion, unsupported temporal ranges, memory fallback, fresh capture/option
ownership and complete weighted optimizer traces with uncertainty ablations.
Sixteen archive checks and fifteen schema-registry checks bring the distinct
targeted total to 107. Strict mypy covers all 378 core files; Ruff, strict docs
and all 129 public schemas plus their ledger are checked separately.

The [complete archive](../assets/continuous_camera_lidar_evaluation_reuse/README.md)
contains the schema-valid problem, exact original generation inputs, generator,
complete source capsules/manifests, every result, timings, failure and validation
receipts. Derived data retains KITTI attribution and CC BY-NC-SA 3.0 separately
from Apache-2.0 source code.

## Reproduce

Audit without a calibration run or raw download:

```bash
python -m tools.benchmark_continuous_camera_lidar_reuse \
  --audit-only docs/assets/continuous_camera_lidar_evaluation_reuse
```

For a fresh comparison, decompress the two source JSON inputs into a new
directory (the prepared problem already contains every calibration input):

```python
import gzip
from pathlib import Path
archive = Path('docs/assets/continuous_camera_lidar_evaluation_reuse')
inputs = Path('outputs/timed-camera-remeasurement-inputs')
inputs.mkdir(parents=True, exist_ok=False)
for name in ('source-points.json', 'motion-control.json'):
    (inputs / name).write_bytes(gzip.decompress((archive / (name + '.gz')).read_bytes()))
```

```bash
python -m tools.benchmark_continuous_camera_lidar_reuse \
  --baseline-commit a844dc00354f987da0e103c0dc5bb9610df2eb2a \
  --problem docs/assets/continuous_camera_lidar_evaluation_reuse/problem.json \
  --input-provenance docs/assets/continuous_camera_lidar_evaluation_reuse/input-provenance.json \
  --control-generator docs/assets/continuous_camera_lidar_evaluation_reuse/control-generator.py.txt \
  --source-points outputs/timed-camera-remeasurement-inputs/source-points.json \
  --motion-control outputs/timed-camera-remeasurement-inputs/motion-control.json \
  --output-dir outputs/timed-camera-new --repetitions 3
```
