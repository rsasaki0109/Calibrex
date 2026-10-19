# Resident CUDA probabilistic refinement

`refine-probabilistic-multiframe` and `benchmark-probabilistic-ablation` accept
`--evaluation-backend cuda`. The default remains `numpy`. The CUDA extra is
optional; the NumPy path does not import CuPy.

```bash
calibrex camera-lidar benchmark-probabilistic-ablation correspondence.json problem.json \
  --evaluation-backend cuda --split-seeds 0,1 \
  --result-dir outputs/ablation/results \
  --definition-output outputs/ablation/definition.json \
  --output outputs/ablation/benchmark.json --json
```

The adapter fuses point transforms, camera projection and covariance errors in
FP64 with fused multiply-add disabled. NumPy still computes robust logarithms
and reductions in native order. Up to 12 independent search candidates share
one kernel launch and download; bounded chunks preserve candidate order and
partial evaluation limits. Candidate generation, stopping, confidence
thresholds, initializer-preserving acceptance and disjoint holdout use the same
solver. Execution selection is separate from the locked estimator options.

An ablation call validates its two inputs once and rechecks their SHA256 before
and after every fit. Its private models and CUDA snapshots live only for that
call. Immutable geometry can serve different frame splits and uncertainty
weights; device inputs are reused when their complete geometry, camera arrays
and row layout match. Every result records its own call, fallback and transfer
counts (including kernel launches and pose uploads), runtime versions, kernel
digest and arithmetic policy in
`provenance.execution`. Existing CPU result serialization is unchanged.

The runtime bounds its host snapshot estimate and retained device cache to
128 MiB each. Per-evaluator packing/scratch has a separate conservative
128 MiB estimate. Tiny inputs, unsupported distortion (including nonzero fifth
radial coefficient), missing required optical parameters and budget limits use
native evaluation and report fallback calls. Extreme distortion radii re-enter
the native path to retain its overflow errors. An explicitly requested CUDA
runtime reports missing dependencies/devices rather than silently replacing
that runtime with NumPy.

## Measurement protocol

The reproducible complete-worker comparison is:

```bash
python tools/benchmark_probabilistic_refinement.py \
  --build-kitti-fixture data/public/kitti_raw_icp/2011_09_30_drive_0018_sync \
  --points-per-frame 16384 --output-dir outputs/projected-control
python tools/benchmark_probabilistic_refinement_ablation.py \
  --correspondence outputs/projected-control/correspondence.json \
  --problem outputs/projected-control/problem.json \
  --output outputs/resident-probabilistic-comparison
```

The fixture selects 16,384 points from each of nine already-used public KITTI
scans (147,456 correspondences). Image means are generated projections with
fixed noise and declared camera models. This supports execution/parity claims;
it supplies no independent image correspondence or calibration accuracy truth.
No new SOTA data or frozen protocol is consumed.

The comparison runs all four uncertainty settings for each of two holdout
seeds, retaining every result and every evaluation. Six fresh workers execute
prior NumPy, current NumPy, CUDA, CUDA, current NumPy, prior NumPy. BLAS threads
are fixed to one; each CUDA worker starts with a fresh CuPy compilation cache
and the driver cache disabled. The host is unreserved. The timed API includes
input loading, all eight fits, result saving and benchmark aggregation; whole
worker time also includes imports, artifact validation and frozen-source checks.

The predeclared gate requires at least 2x whole-worker improvement against both
prior NumPy and current NumPy with validated-input reuse. Every result field,
complete trace, acceptance decision, split, uncertainty option, failure and
quality confidence interval must match exactly. Only separately checked
execution lineage, timestamps, source commit, durations and bound output hashes
are normalized. Each output hash is verified before normalization. Frozen
source manifests and imported modules are verified in every worker.

Small-input pilots retained exact results but did not establish a sufficient
GPU advantage: the 9,216-point default 20-fit ablation measured 1.94x against
prior NumPy and 1.66x against input-reusing NumPy. Shared point-result caching
favored CPU execution; four CPU processes measured 1.45x, and a fused compiled
CPU pilot measured 1.86x including cold compilation. The initial large-input
CUDA pilot measured 2.19x against prior NumPy but only 1.84x against input reuse.
These cohorts remain separate from the resident-runtime comparison; none is a
speed claim for the complete raw-sensor calibration stack.

## Measured resident runtime

The final frozen-source cohort passes both declared gates:

| Implementation | Whole-worker median |
| --- | ---: |
| Prior NumPy | 144.719 s |
| NumPy with validated-input reuse | 123.017 s |
| Resident CUDA with pose batches | 42.552 s |

This is 3.401x against the prior implementation
and 2.891x against the input-reusing CPU
control. Both CUDA workers retain all 2,568 logical evaluations with 236 kernel
launches and zero CPU fallbacks. All 48 complete artifacts, traces, acceptance
and quality aggregates match exactly under the declared normalization. This
measurement concerns eight probabilistic fits; it does not time feature
extraction, D2D initialization or the full raw-sensor calibration stack.

The [indexed complete evidence archive](../assets/probabilistic_cuda_refinement/README.md)
includes every worker, source snapshot and earlier pilot summary, with a
CPU-only audit and independently verified Git source bindings. The two
whole-worker observations are retained for each implementation; no observations
are pooled across earlier pilots or removed as warmup.
