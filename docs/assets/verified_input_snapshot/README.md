# Verified input reuse on public KITTI data

Repeated six-DoF estimates previously verified the frozen provider, checkpoint,
images, depth maps and LiDAR scans on every call. An explicit caller-owned
`CameraLidarSixDofInputSnapshot` now verifies once, loads arrays from digest-bound
bytes, and retains immutable depth/point/range buffers. Each estimate checks the
problem artifact SHA-256 again and creates a fresh backend and solver. No global
cache, modification-time shortcut, saved optimizer state, or persistent device
workspace is introduced. Default estimates continue to verify external files.

Public demos/sweeps opt in with `--reuse-verified-inputs`. Their benchmark
pipeline times charge initial verification/loading/capture exactly once, divided
equally across every planned attempt. Each sweep case/process owns a separate
snapshot. Python result phase timings exclude this separately recorded initial
preparation; callers must include it once when measuring their complete session.

Snapshots intentionally keep the original values after external files change or
disappear. Provenance states `external_files_reverified=false`,
`problem_artifact_reverified=true`, the captured input digests and preparation
cost. New snapshots perform full verification. Loaded arrays are parsed from the
same bytes whose digests were checked; changes between provider verification
and array loading are rejected. Immutable byte owners also prevent re-enabling
array writes. Invocation metadata is deep-copied for independent CPU callers.
Report preview images retain their own digest check before copying.

## Measured workload and timing scope

KITTI raw `2011_09_30_drive_0018_sync`, 25 uniform provider frames,
3,130,279 points, 1226x370 rectified pinhole depth maps. Frozen FeatDepth inverse
depth is retained in its declared convention. Twenty training frames resolve
histogram ranges; five prespecified holdout frames never enter fitting or range
selection. Every invocation starts from the same controlled x-positive prior,
with the original +/-2 degree and +/-1 metre bounds, 800-evaluation cap, and
stopping rules. All twenty timed invocations converge after 229 evaluations and
exactly match the complete bound CPU numerical payload.

The measured implementation is
`9cdba2408bc5acc65367f04f92ae73a8dbd66fdf`. Both methods use ordered batched
CUDA and prepare a fresh projector on every invocation. There are ten calls per
method, alternating fresh-first on odd repetitions and snapshot-first on even
ones. The snapshot is created immediately before its first invocation, and its
full **2.796825200 s** preparation cost is included once via **0.279682520 s**
per snapshot trial. It owns **190,892,928 bytes (182.0 MiB)** of host arrays;
GPU input caches, allocator pools and scratch memory are additional. The snapshot
is released by its caller when the session ends.

An absent CuPy disk-cache directory was selected before process startup. The
first fresh-file call includes CUDA/NVRTC initialization. No invocation is
discarded and no separate warmup is used; subsequent compiled caches persist
inside the process. The i7-9750H / NVIDIA CUDA GPU 6 GiB / 32 GiB Windows host was
shared with unrelated user work. Fresh-file calls 3-5 show substantial input
and solve variation. These observations do not establish statistical confidence
or a portable latency guarantee. The solver implementation is unchanged, so
the difference between solve medians is not a claimed solver optimization.

| Repetition | Fresh pipeline (s) | Snapshot pipeline incl. preparation share (s) | Fresh input phase (s) | Snapshot input phase (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.855330 | 1.667508 | 2.648187 | 0.115664 |
| 2 | 3.964495 | 1.584451 | 2.545330 | 0.076885 |
| 3 | 9.224800 | 2.549019 | 5.723374 | 0.160472 |
| 4 | 10.566499 | 2.924146 | 7.951313 | 0.093818 |
| 5 | 10.566351 | 1.684310 | 8.335694 | 0.093301 |
| 6 | 4.389073 | 1.777228 | 2.857232 | 0.085674 |
| 7 | 4.245244 | 1.611139 | 2.796689 | 0.090566 |
| 8 | 4.192822 | 1.616200 | 2.634912 | 0.081623 |
| 9 | 4.420476 | 1.603052 | 3.010954 | 0.077149 |
| 10 | 4.098708 | 1.645776 | 2.630890 | 0.078602 |
| Median | 4.404774 | 1.656642 | 2.826960 | 0.088120 |

Pipeline median ratio: **2.659x**, including initial capture. Ten-call pipeline
phase totals: **60.523798 / 18.662828 s (3.243x)**. Pipeline ranges are
**3.964495-10.566499 / 1.584451-2.924146 s**. These phases include input handling,
training-range resolution, backend preparation, search and initial/final
holdout evaluation; result construction, serialization, problem building and
report rendering are outside them. A separate measured API-call wall total
includes result construction and the same one-time capture charge:
**61.755560 / 19.317953 s (3.197x)**, excluding result serialization.

The per-invocation input phase median drops from **2.826960 to 0.088120 s**;
this input-only ratio excludes initial capture, which is already included in
the pipeline and wall totals above. Every CUDA call retains 231 histogram
downloads, 37,674,720 downloaded histogram bytes and zero evaluation input
uploads. Backend/device reuse is not part of this optimization.

## Numerical validation and retained artifacts

- [Raw comparison definition](comparison.definition.json) retains all twenty
  timed attempts, phases, invocation wall times, original result hashes, capture
  charges, alternating order and measured code/launcher hashes.
- [Native comparison aggregation](comparison.benchmark.json) is generated from
  that definition without changing the trial matrix.
- [Fresh-file representative](fresh.result.json) and
  [snapshot representative](snapshot.result.json) retain the complete first-run
  search histories, poses, metrics and initial/final holdout evaluations.
- [Six-axis validation definition](axis-validation.definition.json) and
  [aggregation](axis-validation.benchmark.json) bind twelve new CUDA snapshot
  estimates (two per signed axis) and six imported historical CPU numerical
  oracles. Timings are omitted from the axis validation.

Both timed methods have the complete numerical payload SHA-256
`fdf4d36e8cbe9b8a88a5cb765f4c387f67fa2021df8e51271ce461bb4f9e3160`.
The frozen problem SHA-256 is
`1daf5e6b5da531ca873d0c3b4abab6141fd0f0d49d9937c48cb265851d748f18`;
provider SHA-256 is
`7e3ef953a360a7c01ed0cdc3cf947db97fb0aff5912ced805d80bebe210d3360`.
Snapshot identity is
`0134996a8dbcbb2c6fe881dcd20b847b29075aff71d267dc15c1be87a4e4ac16`.
The checkpoint SHA-256 remains
`559f168c762d01af7c745b32ced8930a12423a0e9b62af791f7d894bd902af27`.

Axis evaluation counts remain x+229, x-313, y+277, y-241, z+265, z-241.
Complete payload equality includes acceptance decisions, candidate parameters,
skipped frames, output poses, metrics and holdout records. Imported CPU timings
are not used for speed ratios. Dataset calibration is never the optimizer's
target. Convergence and parity preserve existing results but do not establish
accuracy; quality stays `warn`, covariance and observability remain unknown,
raw scans are not motion-compensated, and possible provider-training overlap
with KITTI limits independent evaluation.

Raw datasets, weights and original outputs stay external. Published JSON uses
`${WORKSPACE_ROOT}` / `${DATASET_ROOT}` placeholders, without altering the
numerical payload. Trial output hashes bind original external bytes;
representative hashes bind the normalized published files separately. Measured
source hashes bind actual worktree bytes; `committed:*` source hashes bind
canonical LF Git blobs. All six published JSON artifacts pass their registered
schemas, use LF, and retain native aggregation equality.

Related tests cover immutable buffers, metadata isolation, concurrent CPU calls,
same-size/same-mtime external changes, deleted inputs, wrong paths, problem
digest changes, verification-to-load changes, initial-cost accounting, CLI
reuse and process-local sweep snapshots. The initial focused run passes 77 tests.
Pinned strict mypy (347 modules), Ruff and schema compatibility pass.

The initial full suite passes 1,854 tests with 21 skips and six failures,
all matching the previously reproduced unchanged-main Windows baseline:
focal ratio tolerance, cache home selection, committed browser sample bytes,
SQLite `?` filename, report path redaction and README hero source hashes.
The later HTML clarification displays the shared preparation cost and uses the
same benchmark pipeline totals in individual rows and summaries; it does not
change the measured estimation files. The final related run passes 78 tests,
including new initial-result, holdout and histogram settings with two CPU frame
workers. Headless Chromium verifies desktop/mobile rendering, visible initial
cost accounting, all nine overlays and local evidence links without browser
errors; the desktop layout is visually inspected.

For public-demo usage and timing semantics, see the
[tutorial](../../tutorials/public_six_dof_demo.md). To repeat the paired latency
workload, construct one snapshot of the frozen problem, alternate ten fresh and
ten snapshot CUDA estimates using the same five holdout IDs, retain every result,
and add one tenth of its preparation time to each snapshot pipeline/API duration.
