# Fuse continuous-time LiDAR observation construction

The optional CPU adapter combines native raw-normal scoring, plane selection and
ordered gathers for batches of at least 8,192 rows. Two Numba kernels release the
GIL without an inner thread pool. They retain FP64 operation order with fastmath
disabled. Exact score ties at distinct centroids retain Python `math.dist`; equal
centroids retain the first uniform plane, including signed zeros. Output columns
use independent read-only NRT allocations. Small batches and unsupported kernels
keep NumPy construction. Public query and observation hooks retain their guards.

This change accelerates CPU construction around the existing GPU queries and
finite differences. GPU workload coverage does not increase.

## Complete fresh-process comparisons

Each workload has three alternating pairs with independent initially empty
persistent Numba/CuPy caches. All cold, slow and first runs are retained.
Wall includes startup, imports, compilation, replay, transfers, all clock
candidates and refinements, diagnostics, hashes, YAML and measurement saves.
Parent audits are outside wall; OS caches persist. Workloads and cohorts are
reported separately, without pooling or multiplying speedups.

| Comparison / workload | Baseline median | Candidate median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|
| Previous GPU / 960,000 captured points | 53.362673 s | 50.009781 s | 1.067x | 1.100x |
| Previous GPU / 48,000 captured points | 10.123346 s | 9.228069 s | 1.097x | 1.061x |
| Previous GPU / default sampling | 5.450336 s | 5.448372 s | 1.000x | 0.992x |
| Packed CPU / 960,000 captured points | 136.841544 s | 45.582851 s | 3.002x | 2.930x |

The incremental dense improvement over the preceding GPU implementation is
modest. This is not evidence of a large new whole-pipeline speedup. The
separate CPU control describes the cumulative hybrid implementation and
must not be multiplied by earlier results.

| Workload | Pair | Order | Baseline whole wall | Candidate whole wall |
|---|---:|---|---:|---:|
| dense | 1 | baseline, candidate | 53.362673 s | 57.517236 s |
| dense | 2 | candidate, baseline | 49.359184 s | 44.867407 s |
| dense | 3 | baseline, candidate | 57.592006 s | 50.009781 s |
| full | 1 | baseline, candidate | 17.513913 s | 17.596075 s |
| full | 2 | candidate, baseline | 8.637609 s | 8.144351 s |
| full | 3 | baseline, candidate | 10.123346 s | 9.228069 s |
| default | 1 | baseline, candidate | 13.378151 s | 14.417411 s |
| default | 2 | candidate, baseline | 5.450336 s | 5.386229 s |
| default | 3 | baseline, candidate | 5.405486 s | 5.448372 s |
| cpu-dense | 1 | baseline, candidate | 140.224374 s | 48.579472 s |
| cpu-dense | 2 | candidate, baseline | 136.841544 s | 37.436309 s |
| cpu-dense | 3 | baseline, candidate | 133.544087 s | 45.582851 s |

All 24 formal results match exactly after normalizing path separators and
excluding only strictly validated execution identity notes. Full API results,
YAML artifacts, optimization histories, diagnostics and logical counters are
retained. Dense controls use 768,000 train and 192,000 holdout targets; the
final result has rank 6 and status `max_iterations`.

GPU/GPU comparisons archive fourteen preceding modules. The separate CPU
control archives ten CPU modules; the packed builder is shared with the GPU
position callback and compiled packing callback disabled by the archived
CPU solver. Physical settings, samples and writers are unchanged.

## Observed work, outside the timing cohorts

| Pre-change hybrid construction | Rows |
|---|---:|
| `build_calls` | 52 |
| `rows` | 24,960,000 |
| `both_present` | 12,878,352 |
| `score_ties` | 783,186 |
| `same_centroid_ties` | 475,516 |
| `different_centroid_ties` | 307,670 |

| Current execution | Count |
|---|---:|
| `gpu_linearization_calls` | 56 |
| `gpu_linearization_requests` | 129 |
| `initial_rank_calls` | 26 |
| `initial_rank_compiled` | 26 |
| `gpu_linearization_cache_hits` | 73 |
| `max_per_context_cache_bytes` | 33,398,792 |
| `gpu_linearization_rows` | 32,768,988 |
| `gpu_residual_calls` | 0 |
| `gpu_residual_rows` | 0 |
| `gpu_payload_uploads` | 26 |
| `gpu_payload_bytes` | 2,066,730,664 |
| `cpu_linearization_calls` | 0 |
| `cpu_residual_calls` | 207 |
| `cpu_normal_calls` | 129 |
| `cpu_uniform_query_calls` | 0 |
| `cpu_adaptive_query_calls` | 0 |
| `cpu_adaptive_repaired_rows` | 0 |
| `compiled_packing_calls` | 52 |
| `compiled_packing_rows` | 24,960,000 |
| `cpu_distinct_centroid_tie_rows` | 307,670 |
| `compiled_replay_calls` | 120 |
| `compiled_replay_rows` | 960,000 |
| `gpu_plane_query_calls` | 52 |
| `gpu_uniform_query_rows` | 24,960,000 |
| `gpu_adaptive_query_rows` | 24,960,000 |
| `gpu_map_upload_bytes` | 105,635 |
| `gpu_query_upload_bytes` | 1,198,080,000 |

| Instrumented phase | Sum of call durations |
|---|---:|
| `CudaPlaneQueries.query` | 6.008207 s |
| `compiled_packing` | 2.841975 s |
| `_PackedObservationBuilder.build` | 9.341024 s |
| `cpu_residual_calls` | 13.457525 s |
| `gpu_linearization` | 9.948843 s |
| `cpu_normal_calls` | 2.564417 s |

These call sums overlap across candidates and nest inside other phases;
they are not whole-wall percentages. GPU queries, finite differences, cache
hits and ordered CPU normal equations remain unchanged. CPU replay, standalone
residuals and native adaptive boundary/tie repairs remain on CPU.

## Preserved diagnostics and limitations

The first pre-change diagnostic saved a complete result but its historical
comparison failed because a nested diagnostic provenance also contained a
different git commit. Its result, source snapshot, launcher and failure log
are retained under `profiles/v1`. The second diagnostic excludes historical
git commit fields for that comparison and confirms all numerical values;
full saved provenance is retained under `profiles/v2`.

The public GLIM sensor is copied into two topics as an execution parity
control. This does not establish independent two-sensor calibration accuracy.
Local descriptive timings vary; three pairs are not portable confidence
estimates. No peak-memory reduction is quantified.

Source commit: `55e8f6217e39164bbc4ff8b717e6596b16c33455`.
All measured sources, baseline modules, configs, launchers, complete models
and portable location metadata are bound by `publication-index.json`.
Source archives preserve measured checkout bytes. Only module/file locations
are rewritten; full result values and saved YAML bytes remain unchanged.
