# Overlapping fresh input verification on public KITTI

The ordinary six-DoF estimate still reads and verifies every external file on
every call. A single invocation-owned worker verifies the provider checkpoint
while the main thread loads verified observations and runs estimation. No result
is constructed or returned until the checkpoint check succeeds. Large-file
SHA256 reads also reuse one bounded buffer instead of allocating a new chunk
for each read.

In the final six-case comparison, complete API-call medians improve
**1.52–2.63x against the previous sequential input path**, using the same CUDA
v0.4 solver in all three methods. All 54 complete numerical payloads match their
historical NumPy oracles exactly. This is an input-handling improvement; solve
timing differences below reflect scheduling and host variation, not a new solver.
The earlier overlap-only experiment had inconsistent results, including a
regression, and is retained in full below.

## Workload and verification contract

Inputs are the frozen public KITTI raw `2011_09_30_drive_0018_sync/image_02`
images, original Velodyne scans, and FeatDepth provider SHA256
`7e3ef953a360a7c01ed0cdc3cf947db97fb0aff5912ced805d80bebe210d3360`.
The provider references a 661,073,985-byte checkpoint with SHA256
`559f168c762d01af7c745b32ced8930a12423a0e9b62af791f7d894bd902af27`.
The [candidate batching evidence](../cuda_candidates/README.md) binds the same
six problems, full NumPy oracles, 25 frames and 3,130,279 points. Twenty frames
are used for fitting and five predetermined frames for holdout. Priors, search
bounds, histogram ranges, evaluation budget and sample precision are unchanged.

Problem and provider mappings are parsed and hashed from the same byte reads.
Images, depths, uncertainty and LiDAR inputs are checked before calculation;
decoded arrays are bound to their checked bytes. Only the checkpoint verification
is deferred. A missing file, digest mismatch or worker I/O failure rejects the
invocation before a standard result exists. The worker is joined on both success
and exceptions. There is no persistent verification cache, mtime shortcut or
implicit input snapshot. Explicit immutable snapshots retain their existing
semantics and are not used in these measurements.

The streaming path uses one 16 MiB private reusable buffer for files at least
64 MiB, chosen using the opened file descriptor's size. This size check selects
the buffer strategy; every file byte is still hashed. Smaller files retain the
previous 1 MiB read loop. Directory manifest ordering and hash contents are
unchanged. Buffer memory is additional to solver memory and released after each
file; each active large-file hash owns a separate buffer.

## Final measurements

The measured source is commit
`fa4e1447ae4a85a5980e62876de692a7197bece4`; the sequential reference is
`9180b4c610f8a32e2b3ba328458cf530594e136f`. Four archived `baseline.*.txt`
files are exact Git blobs for its provenance helper, depth verifier, problem
loader and estimate pipeline. The benchmark loads those pinned implementations
and retains the same current CUDA backend and solver across methods.

Each case runs each method three times. The order rotates so that each method
occupies each position once per case. Each invocation constructs a fresh backend
and solver. There is no separate warmup; process, compilation and OS disk caches
persist. The Windows host is shared with other user work. The environment is
Python 3.12.10, NumPy 2.3.4, CuPy 14.2.0 and an NVIDIA CUDA GPU. GPU model names
and local paths are anonymized in public evidence.

`wall_seconds` measures the complete Python API call, including fresh file
verification, backend creation, calculation, final verification wait and result
construction. JSON serialization and benchmark aggregation are excluded. These
are three-invocation medians in seconds; ratios divide the two medians.

| Case | Previous sequential | Buffered sequential | Buffered + overlap | Previous / overlap | Overlap range |
| --- | ---: | ---: | ---: | ---: | ---: |
| x-positive | 5.321880 | 3.648726 | 2.023574 | 2.63x | 1.882389–5.363936 |
| x-negative | 3.476044 | 3.364524 | 1.981081 | 1.75x | 1.953275–5.493819 |
| y-positive | 3.470794 | 3.386998 | 1.934413 | 1.79x | 1.823408–2.012585 |
| y-negative | 3.443484 | 3.297081 | 2.011058 | 1.71x | 1.866951–2.136426 |
| z-positive | 3.475433 | 3.505248 | 1.890131 | 1.84x | 1.887665–1.949097 |
| z-negative | 3.422643 | 3.289801 | 2.246340 | 1.52x | 1.909919–3.423038 |

The buffer-only method has mixed, generally modest gains. The larger observed
benefit comes from overlapping verification with loading and calculation. These
timings do not isolate a buffer effect from host variation. For example,
x-positive's previous sequential invocations span 4.342–7.640 s, and x-negative's
buffered sequential invocations span 3.321–7.364 s. No outlier is discarded.

| Case | Previous pipeline | Buffered + overlap pipeline | Previous solve | Buffered + overlap solve |
| --- | ---: | ---: | ---: | ---: |
| x-positive | 5.252313 | 1.974678 | 1.032151 | 0.575257 |
| x-negative | 3.420141 | 1.922944 | 0.915357 | 0.800137 |
| y-positive | 3.418824 | 1.882026 | 0.780320 | 0.673356 |
| y-negative | 3.390227 | 1.954515 | 0.767404 | 0.648538 |
| z-positive | 3.413233 | 1.835502 | 0.833837 | 0.675290 |
| z-negative | 3.370177 | 2.194437 | 0.748572 | 0.715563 |

Pipeline time sums four nonoverlapping accounting phases. The
`input_verification_and_loading` phase includes loading before calculation plus
the final checkpoint completion wait. The background checkpoint duration in
`input_verification` is diagnostic and overlaps other phases; it must not be
added again. Its final-run case medians span 1.833–2.192 s, with final completion
wait medians of 0.295–0.568 s. Pipeline timing stops after that wait; API-call
timing additionally includes result construction and thread cleanup.

## Earlier experiment retained

Commit `284cad9776f5b96144a9a7a43d298cd2893ab2a4` added checkpoint overlap
without the large-file buffer change. Its 36 trials also preserve full numerical
parity, but shared-host timings do not support a consistent speedup. Both
[superseded definition](superseded.definition.json) and its
[derived aggregate](superseded.benchmark.json) retain every measurement,
including y-positive's regression.

| Case | Previous API call median | Overlap-only API call median | Ratio |
| --- | ---: | ---: | ---: |
| x-positive | 8.618415 | 7.441654 | 1.16x |
| x-negative | 8.267151 | 7.057600 | 1.17x |
| y-positive | 5.006685 | 7.834248 | 0.64x |
| y-negative | 8.724494 | 5.182705 | 1.68x |
| z-positive | 5.572217 | 3.270520 | 1.70x |
| z-negative | 6.603033 | 5.695785 | 1.16x |

The final run measures a changed implementation with an additional serial
buffered control. It is not an independent accuracy trial or evidence that
buffering alone explains the difference between runs. Both experiments, their
source commits and all invocation timings remain available. Three repetitions
on one sequence and a shared host do not establish statistical confidence or
a portable speed guarantee; bootstrap fields are schema summaries only.

## Reproduction and evidence

After preparing the same public data and frozen six-case sweep:

```bash
python tools/benchmark_camera_lidar_input_overlap.py \
  --sweep-root outputs/public-kitti-six-dof-sweep-cuda-v1 \
  --output-dir outputs/public-kitti-input-overlap-v2 --repeats 3
```

The output directory must not already exist, preserving previous experiments.
For ordinary Python estimates, `overlap_input_verification=False` selects
sequential verification with the current buffer strategy. The ordinary CLI
estimate uses checkpoint overlap automatically when a checkpoint is referenced;
providers without a checkpoint remain sequential.

[comparison.definition.json](comparison.definition.json) retains all 54 final
trials, their complete API/pipeline/solve times, phase and verification
diagnostics, original output digests and frozen numerical payload digests.
[comparison.benchmark.json](comparison.benchmark.json) is derived from it using
the standard aggregator. Eighteen `*.result.json` files contain the first
repetition of every final method/case, with full traces and holdout records. The
first-repetition publication rule was fixed before the final run; representatives
are not selected by speed. All 90 complete raw payloads across both experiments
were checked against their bound historical CPU oracles.

The archived `publication-launcher.py.txt` records normalization, validation and
digest binding. Published-byte hashes are distinct from original raw-result,
configuration, input and source digests. Canonical Git blob hashes are recorded
separately from measured Windows working-tree hashes. `${WORKSPACE_ROOT}`,
`${DATASET_ROOT}` and `${PUBLIC_SWEEP_ROOT}` replace local prefixes; original
input digests remain unchanged. Existing full problem/oracle evidence is bound
through `../cuda_candidates/` references.

Numerical equality preserves the existing outcome, including its limitations.
The learned disparity is not metric depth, training overlap with public KITTI
cannot be ruled out, LiDAR scans are not motion compensated, and independent
observability/covariance have not been established. Quality remains `warn`;
y-positive retains its approximately 0.527-degree rotation error above the
existing 0.5-degree hit threshold. No new calibration-accuracy claim is made.
