# Public Livox geometry reuse evidence

The full fresh-process comparison against `c6fd98b7b5af73be41fa01b5e4938773f4696fb1`
measures **40.684825 → 10.689548 s (3.81x)**.
This is a direct cumulative comparison, without multiplying earlier ratios.
The separate incremental comparison against PR #152 (`d9d156fc2aa8dc8f3afc2a00a3c10d32e3edb0f3`)
measures **12.171752 → 10.727784 s (1.13x)**;
its API subset improves **1.27x**. The additional complete-process gain is modest.
All 20 final formal result payloads are exactly equal after the explicit volatile-field normalization.

## Implementation and scope

Measured implementation: `6cbd03776a4d55ef0af4c51877a8902a706a2a65`. One context belongs to one calibration API
invocation. Inspection, candidate plus all 24 controls and the native registration
adapter share decoded inputs. Equivalent sampled-file sets reuse invariant
statistics. Source voxel sums and prepared plane arrays are reused. Point lists
returned to the adapter are copies; prepared arrays are read-only. A new run reads
fresh inputs. There is no process-wide geometry cache.

Compact extents use a direct position table capped at **8 MiB per index**. This
capture uses 679 source map voxels across 13,224 positions (105,792 table bytes).
Larger extents retain sparse integer search, huge extent products use structured
keys, and unrepresentable coordinates retain the scalar path. Full voxel keys
still reject clipped/out-of-range aliases. Sparse neighborhoods calculate distances
only for occupied candidates, retaining the same 27-slot order and first tie.
Dense neighborhoods keep the regular array path. Source accumulation order,
Hamilton transform operations, nearest-normal rejection, gates, support
denominators and residual arithmetic remain unchanged. Query chunks remain bounded
at 4,096 × 27 associations. Avoiding unused per-point accumulator construction
also preserves the same sums and dictionary insertion order.

The complete pipeline decodes the two PCD captures once each, rather than six
decodes, and constructs the plane index once rather than in each of 25 queries.
All finite input points (23,921 source / 23,666 target) still enter candidate
evaluation; the unchanged registration config caps each ICP cloud at 500
centroids. Both native ICP backends, all their diagnostics/probes, Open3D GICP,
NDT availability, quality evaluation and all saves remain included.

The baseline loader substitutes all **five changed modules**: `data/livox.py`,
`data/inspect.py`, `evaluation/lidar.py`, `pipelines/calibrate.py` and
`solvers/native_registration_comparison_solver.py`. Numerical ICP solvers and
artifact/report writers are shared. Each worker records resolved module paths;
the parent checks those bindings and freezes HEAD, measured source/config/input
bytes. Source archives are exact Git blobs; original working-byte hashes and
committed-byte hashes are recorded separately. `run.git_commit` identifies the
shared checkout, while method provenance identifies the baseline substitutions.

Workers start fresh interpreters and call public `run_calibration`. Wall includes
startup/imports, config loading, inspection, reads, evaluation, registration and
result/report/artifact saves, plus worker measurement JSON writes. Calibrex CLI
argument dispatch and parent audit/aggregation are excluded. API time starts at
`run_calibration` entry and ends after its artifact writes; it is a subset of wall.

## Final direct comparison

| Repeat | Baseline wall (s) | Candidate wall (s) | Baseline API (s) | Candidate API (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 39.056933 | 8.373868 | 36.329608 | 6.066320 |
| 2 | 39.319624 | 9.269625 | 36.887002 | 6.885840 |
| 3 | 52.157569 | 11.938388 | 49.151507 | 8.719089 |
| 4 | 61.321662 | 11.145085 | 58.221077 | 8.253034 |
| 5 | 40.684825 | 10.689548 | 37.227998 | 7.787415 |
| Median | 40.684825 | 10.689548 | 37.227998 | 7.787415 |

Ranges: baseline wall 39.056933–61.321662 s; batched wall 8.373868–11.938388 s. No outliers are removed.

## Final incremental comparison against PR #152

| Repeat | Baseline wall (s) | Candidate wall (s) | Baseline API (s) | Candidate API (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 16.004299 | 11.134646 | 12.434858 | 8.110550 |
| 2 | 11.964188 | 13.425734 | 9.354368 | 9.761655 |
| 3 | 12.407923 | 10.727784 | 9.698756 | 7.638738 |
| 4 | 12.171752 | 8.911244 | 9.767738 | 6.470955 |
| 5 | 10.953229 | 8.677460 | 8.557300 | 6.353184 |
| Median | 12.171752 | 10.727784 | 9.698756 | 7.638738 |

Ranges: baseline wall 10.953229–16.004299 s; batched wall 8.677460–13.425734 s. These timings are kept separate from the cumulative
comparison. The extra wall gain is approximately 13%;
the API subset gain is approximately 27%.

## Retained intermediate and interrupted measurements

The [intermediate evidence](intermediate/pipeline.definition.json) retains ten
complete workers from `81f62fc7c97bc2fff544109771b5d9e98eb8adbf`. It used two baseline substitutions and
reused geometry only within candidate/control evaluation. Median wall was
13.313453 → 10.413110 s.
No intermediate timing enters either final comparison.

A subsequent attempt was interrupted by the user before its parent completed.
Both API workers saved complete results. The separate `interrupted.pipeline.*`
protocol retains their measured API durations and result/artifact hashes, with
`data_verified=false` for the absent completed parent verification. Full subprocess
wall durations and final guards are unavailable, so these timings do not enter
any formal median. Their canonical payloads equal the 20 final formal payloads.

This directory retains **30 formal full-process workers plus two completed API
workers from the interrupted attempt**. Definitions and aggregates validate
against standard benchmark schemas; each aggregate is recomputed from its own
published definition. Only run ID, timestamps, destination paths and Windows
path separators are normalized for result parity. All numerical metrics, reason
strings, diagnostics, controls and quality grades remain present. Publication
normalizes private workspace prefixes only in commands and metadata.

## Input and limitations

Inputs are the official [Livox calibration sample](https://github.com/Livox-SDK/Livox_automatic_calibration),
downloaded with `tools/download_public_dataset.py livox_horizon_horizon_pcd_sample`.
File hashes are `19b3864cbee76b785d207fcb3fe48f6f506d142d8233f842d0a87fe4330fe407`
and `61a74eb8b8ff4eeba4ca0ff824a4046ae220e947e8156f27d93da452538f5b6f`.
The target's 23,786 declared records include 120 nonfinite positions excluded by
the unchanged population policy. Point clouds are not redistributed here.

The existing final quality remains **fail**, and the registration adapter remains
**inconclusive**. Benchmark success means complete execution and parity, without
a ground-truth accuracy claim or independent temporal validation for this pair.
Five repeats per comparison reverse method order every other repeat. No separate
warmup is used; OS caches persist, earlier experiments ran on the same host, and
unrelated user work continues. `OMP_NUM_THREADS=1` is set only for each worker,
equally for both methods. Host settings and user processes are untouched.
Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and Open3D 0.19.0. These descriptive
ranges do not establish confidence intervals or a portable speed guarantee.
No GPU is used by this change. Prior benchmark ratios must not be multiplied.

## Reproduction

Use the measured implementation commit, materialize the official data, and run
each comparison into a new directory:

```bash
python tools/download_public_dataset.py livox_horizon_horizon_pcd_sample
python tools/benchmark_livox_evaluation.py --output-dir outputs/livox-geometry-original \
  --baseline-commit c6fd98b7b5af73be41fa01b5e4938773f4696fb1 --repetitions 5 --omp-threads 1
python tools/benchmark_livox_evaluation.py --output-dir outputs/livox-geometry-incremental \
  --baseline-commit d9d156fc2aa8dc8f3afc2a00a3c10d32e3edb0f3 --repetitions 5 --omp-threads 1
```

Local verification covers 163 relevant tests plus 44 CLI/lifecycle tests (one
optional CLI test skipped), strict typing of 349 source files, Ruff, exact result
parity, memory bounds, holes/clipped keys, dense/sparse neighborhoods, malformed
inputs, new-run replacement, point-list isolation and unchanged sample semantics.
