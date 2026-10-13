# Parallel CPU LiDAR summaries and finite differences

Large native immutable CPU-backend batches share three additional helpers across the existing
three candidate workers. Each summary or finite-difference request computes two
disjoint row blocks. Summary accumulation and robust normal equations retain
their original serial order; fastmath remains disabled. CUDA finite differences
and queries keep their existing kernels. The captured CUDA summary experiment
remains disabled by default. Extra helpers are restricted to the CPU backend:
the helper-enabled CUDA experiment had a 0.865x ratio of whole medians and is retired.

The CPU finite-difference context retains only its last exact immutable batch
within a 64 MiB cache. Keys include observation identity, all thirteen transforms
and all six step values. Every returned residual/Jacobian is writable and independent
of the read-only cache. A shared context serializes requests; failed requests
join helpers before releasing outputs. Mutable inputs bypass caching and parallelism.

The helper pool is separate from the candidate pool to avoid nested-pool deadlock.
It is created only for eligible native compiled solves with at least 32,768
global targets, a matching per-split sampling budget and at least six logical CPUs.
Each factor separately checks at least 32,768 immutable rows. Summary scratch is
bounded at 16 MiB per request; difference scratch uses two columns, bounded at
32 MiB per request. Result arrays and caches are additional allocations.
Small/default sampling preserves its previous execution route. No schema changes
or new configuration switches are introduced. Execution identity notes report
actual successful summary/parallel-difference calls and cache hits.

## Final whole-process observations

The primary CPU comparison uses three alternating pairs; a CUDA policy smoke uses
one pair. Default sampling retains five pairs from the preceding experimental source
with unchanged small-batch dispatch. These use fresh interpreters and persistent
compiler caches. CPU scopes used a shared directory with separate compiled-code
namespaces; CUDA scopes used separate method directories. Wall includes startup, initial compilation,
replay, every candidate/refinement, transfers, diagnostics, hashing and saves.
No cold or slow observation is discarded; OS caches persist. CPU and CUDA cohorts
are separate comparisons against the same preceding source and are not pooled.
Another project benchmark was observed during the initial CPU dense cohort.
Its processes were left alone. After it ended, the CPU comparison was repeated
with three fresh pairs and a new compiler cache directory. The initial CPU
cohort is retained separately, including every slower observation. Ambient
snapshots are retained; these are local observations rather than portable
confidence estimates. The repeated CPU cohort is the primary CPU comparison.

CPU candidate initial inventories contained only baseline-prefixed compiled files.
The candidate added its own files and left every baseline file unchanged. The
auditor verifies those hashes for all CPU cohorts, proving the observed separate
code namespaces. The original protocol wording about separate CPU method
directories was inaccurate and is retained with this correction. The final harness
creates separate directories explicitly; that directory fix was not formally retimed.

| Workload | Baseline whole median | Candidate whole median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|
| cpu-dense | 88.192810 s | 50.415180 s | 1.749x | 1.728x |
| cuda-policy-smoke | 46.185255 s | 51.526152 s | 0.896x | 0.896x |
| default | 5.601387 s | 5.556325 s | 1.008x | 0.980x |

| Workload | Pair | Order | Baseline whole wall | Candidate whole wall |
|---|---:|---|---:|---:|
| cpu-dense | 1 | baseline, candidate | 96.386516 s | 60.088088 s |
| cpu-dense | 2 | candidate, baseline | 86.419340 s | 50.004198 s |
| cpu-dense | 3 | baseline, candidate | 88.192810 s | 50.415180 s |
| cuda-policy-smoke | 1 | baseline, candidate | 46.185255 s | 51.526152 s |
| default | 1 | baseline, candidate | 13.078563 s | 13.188131 s |
| default | 2 | candidate, baseline | 5.342286 s | 5.556325 s |
| default | 3 | baseline, candidate | 5.306797 s | 5.415222 s |
| default | 4 | candidate, baseline | 5.601387 s | 5.392596 s |
| default | 5 | baseline, candidate | 6.010569 s | 6.289709 s |
| cpu-dense-initial | 1 | baseline, candidate | 158.756539 s | 108.669489 s |
| cpu-dense-initial | 2 | candidate, baseline | 152.339609 s | 78.728266 s |
| cpu-dense-initial | 3 | baseline, candidate | 129.920303 s | 69.603435 s |
| dense-experimental | 1 | baseline, candidate | 74.296698 s | 65.452187 s |
| dense-experimental | 2 | candidate, baseline | 42.270747 s | 50.028804 s |
| dense-experimental | 3 | baseline, candidate | 43.256966 s | 32.811758 s |

The CUDA policy smoke is one pair and does not establish a GPU speedup.
All 44 complete cohort saved/API results match within their respective cohorts
after path separator normalization
and exclusion of strictly validated optional execution identity notes. The dense
split retains 768,000 train and 192,000 holdout targets. Complete histories,
diagnostics and saved YAML bytes are retained. The duplicated public GLIM sensor
topic supplies execution parity; it does not establish independent two-sensor
calibration accuracy. Ratios cannot be multiplied by preceding published gains.

## Actual work, measured separately from formal timing

Independent final-policy CPU and CUDA dense workers exactly match their respective
first formal candidate results. They preserve native public hooks and candidate
execution guards. Call sums can nest and overlap and include instrumentation;
they are not critical-path percentages or formal phase speedups. Instrumentation
retains contexts and is not evidence of production peak memory.

| Counter | CPU dense | CUDA dense |
|---|---:|---:|
| `cpu_summary_calls` | 154 | 154 |
| `cpu_summary_rows` | 76,674,615 | 76,674,615 |
| `parallel_summary_calls` | 154 | 0 |
| `parallel_summary_rows` | 76,674,615 | 0 |
| `scalar_summary_calls` | 0 | 154 |
| `scalar_summary_rows` | 0 | 76,674,615 |
| `cpu_residual_calls` | 53 | 53 |
| `cpu_normal_calls` | 129 | 129 |
| `initial_rank_calls` | 26 | 26 |
| `initial_rank_compiled` | 26 | 26 |
| `discarded_residual_lists_avoided` | 102 | 102 |
| `discarded_residual_rows_avoided` | 59,724,347 | 59,724,347 |
| `gpu_payload_uploads` | 0 | 26 |
| `gpu_payload_bytes` | 0 | 2,066,730,664 |
| `compiled_packing_calls` | 52 | 52 |
| `compiled_packing_rows` | 24,960,000 | 24,960,000 |
| `cpu_distinct_centroid_tie_rows` | 307,670 | 307,670 |
| `cpu_parallel_difference_calls` | 56 | 0 |
| `cpu_difference_calls` | 56 | 0 |
| `cpu_difference_cache_hits` | 73 | 0 |
| `cpu_difference_rows` | 32,768,988 | 0 |
| `max_cpu_difference_cache_bytes` | 33,398,792 | 0 |
| `gpu_difference_calls` | 0 | 56 |
| `gpu_difference_cache_hits` | 0 | 73 |
| `gpu_difference_rows` | 0 | 32,768,988 |
| `max_gpu_difference_cache_bytes` | 0 | 33,398,792 |
| `compiled_replay_calls` | 120 | 120 |
| `compiled_replay_rows` | 960,000 | 960,000 |
| `gpu_query_calls` | — | 52 |
| `gpu_uniform_rows` | — | 24,960,000 |
| `gpu_adaptive_rows` | — | 24,960,000 |
| `gpu_query_upload_bytes` | — | 1,198,080,000 |

The CPU context executes 56 finite-difference misses in parallel and reuses
73 exact cached batches, preserving all 129 logical normal-equation calls.
The CPU-only diagnostic imports no CuPy. CUDA retains its existing 56 difference
launches, 73 cache hits and query work. The final policy retains scalar CPU summaries
on the CUDA backend. It launches no parallel CPU summaries for CUDA.

## Preliminary and unsuccessful attempts

The earlier candidate only parallelized summaries and cached CPU differences.
It reached about 1.47x CPU dense and 1.05x CUDA dense ratios of medians.
Its default cohort was stopped during the third candidate after five completed
workers to add parallel finite differences. Completed cohorts, all five workers,
the partial log, exact source snapshot, stdout/stderr and stop record are retained.
These observations are separate from the final candidate.

| Complete preliminary cohort | Baseline whole median | Candidate whole median | Ratio |
|---|---:|---:|---:|
| cpu-dense-initial | 152.339609 s | 78.728266 s | 1.935x |
| dense-experimental | 43.256966 s | 50.028804 s | 0.865x |
| preliminary/cpu-dense | 85.425054 s | 58.101687 s | 1.470x |
| preliminary/dense | 31.795531 s | 30.189281 s | 1.053x |
| attempts/scope-smoke | 12.691429 s | 12.344791 s | 1.028x |

Seven synthetic kernel-only probes retain every trial, input seed/digest,
result-bit checks, launchers and baseline source bytes. They include indexed
residuals, CUDA split differences, vector residuals, SIMD residuals, unit-norm
shortcuts, NumPy summaries and threaded summaries. Rejected/slower variants
are included; kernel timings are not whole-pipeline speedups.

The first native multithread phase profile failed because multiple active
profilers conflicted. The succeeding selected-call profile preserves all
native candidates and a complete calibration result. Its cumulative call
times overlap across threads. A CLI scope validation failure produced no
trial. An earlier ten-module CPU baseline completed one saved worker before
compiled-replay binding validation failed; its model is retained, but no
parent whole wall was persisted. No whole timing or missing result is invented.
The new `cpu-full` scope archives all fourteen relevant modules and strictly
requires two explicit CPU backend overrides plus CPU-only load bindings.

## Provenance and reproduction

Measured final source: `0337890f461c5cd7f1fe05cc4b3e5d0bf5ea6971`.
Helper-enabled experimental source: `35f075c54f053e61a3d1532b70f13b5f5b8932c2`.
Control source: `aa19ec5d4bf01e1167e70f48332e6e13d51263c0`.
Preliminary source: `33858bd2ac853d30cf31645651de4be6e38b4c8c`.
First attempted source: `3d16f57051da55dad232e3d3ffea1c49b7137c6a`.

Final benchmark cache-directory correction: `f15c0da6750db611b7f73ca90d37886e93f3d6f8`.
This correction was tested separately; calibration runtime sources remain identical
to the final measured source. The correction was not formally retimed.

Run the frozen `launchers/run-parallel-summary-v4.py.txt` as a Python file in
a checkout of the helper-enabled experimental commit, with verified public inputs.
It runs the original three formal cohorts, then the two instrumented workers. Copy the
diagnostic launcher to its original private name as referenced by that launcher.
`launchers/run-parallel-summary-confirm-v1.py.txt` runs the final CPU cohort,
CUDA policy smoke and final diagnostics on the final measured commit, with new
recorded compiler cache layouts. Diagnostic launcher names must match their references.
Final measured sources differ from the experiment only in requiring the CPU
backend to create the extra pool. The auditor proves that exact source difference;
CPU dispatch and small/default sampling remain unchanged.
Source/input guards check all twenty measured modules, configs and input bytes.
The publication index binds each retained file. Source archives preserve original
checkout line endings. Only execution-location metadata is made portable;
full API results and saved YAML remain unchanged. No raw sensor data is included.

Validation: 316 distinct targeted tests passed, Ruff passed, mypy passed over 356 source
files, and all 126 generated schemas are byte-identical to the preceding source.
Tests cover exact residual/Jacobian bits, reversed arrays, thresholds and budgets,
both robust policies, three concurrent callers, cache invalidation/ownership,
exception cleanup, complete optimizer/clock histories and strict identity parsing.
