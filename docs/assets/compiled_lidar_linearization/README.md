# Keeping continuous-time linearization in numeric arrays

The measured baseline is `46e8df3a85ba6ce6706b5b83f3791595665198aa`
(the preceding compiled-query version). The candidate source is
`e847ea4e6de4bd07895930f33519b943fdcd455e`. The compiled benchmark scope
archives ten modules, including the factor, optimizer, MAD filter and optional
Numba adapter. It verifies every loaded module filename and all source bytes.

The optional CPU backend keeps the same thirteen numerical finite-difference
evaluations in numeric arrays, avoiding repeated Python list transposition and
list-to-array conversion. Ordered normal-equation kernels retain every row,
left-weighted product, gradient and Hessian addition. The optimizer consumes
the packed weighted system directly; public linearization still returns its
existing residual/Jacobian/Hessian/gradient fields. Small/nonfinite inputs and
custom factor methods retain existing behavior. Observation or transform
replacement invalidates or refreshes the corresponding captures.

Captured native plane geometry skips temporary match records and unused
distance fields. Batched hybrid selection retains the scalar three-product
residual formula and first-candidate ordering. Exact residual ties still use
scalar `math.dist` for the centroid-distance key. Unsupported geometry and
extreme/nonfinite scores retain scalar selection. Normal-less nearest voxels
continue to reject their uniform candidate.

Observation preparation reuses the exact public-constructor normalization of
immutable plane normals, retaining at most 1,024 plane captures and identity
references for the first 65,536 native point tuples. Mutable/unsupported fields,
changed constructors and uncaptured points use the public constructor.
Prepared pose rows become native floats in 4,096-row conversion blocks.

Repeated timestamp batches reuse immutable prepared poses and numeric rows in
a solve-scoped LRU with a conservative 128 MiB retained-payload budget. The
full timestamp tuple and exact base-transform bytes key each entry. Changed
points and extrinsics always run the ordered world-coordinate kernel again.
Every cache hit updates the same interpolation/clamp/extrapolation diagnostics;
a stricter extrapolation tolerance replays the original first-failure prefix.
Unsupported diagnostic overrides keep the per-query path.

Adaptive tree queries accept or reject an unambiguous nearest plane in bulk
only when its conservative roundoff interval lies entirely inside or outside
the gate. Near ties, gate boundaries, tiny distances and unsupported geometry
retain the ordered scalar `math.dist` repair.

MAD filtering reuses the factor and its packed arrays instead of constructing
and packing a second factor. Partition-based medians retain the original
odd/even formulas. Stable sorting preserves the original support-floor ties
and final observation order. Nonfinite or non-native residuals retain the
statistics/scalar path. Retained numeric rows preserve original indices and
read-only storage; public MAD results and thresholds are unchanged.

The Python default and saved schemas remain unchanged. The optional
`continuous_time_query_backend: numba_cpu` execution note identifies adapter
v0.2, Numba version, `fastmath=false` and `cache=true`. The arithmetic order,
sampling, temporal holdout, robust weights, damping, acceptance rules and
iteration budgets remain the same.

## Complete-process measurements

The verified [AIST GLIM Livox recording](https://zenodo.org/records/6864654)
and supplied trajectory use the existing tutorial conversion. A copied point
cloud is the second topic, providing an identity execution control without
an independent two-sensor extrinsic or clock reference.

<!-- measurements:start -->

In the archived method IDs, `sparse` identifies the implementation
with dense linearization, captured geometry and prepared MAD arrays.

### Default sampling budgets

| Metric | Baseline median (s) | Compiled median (s) | Ratio |
|---|---:|---:|---:|
| Complete process | 11.419431 | 8.323002 | 1.37× |
| Full evaluation API | 8.706023 | 5.755083 | 1.51× |
| Clock profile + extrinsic refinement | 7.744084 | 4.860563 | 1.59× |
| Adaptive nearest-plane calls | 0.537481 | 0.186777 | 2.88× |
| Worker imports | 2.112278 | 2.023576 | 1.04× |
| Worker CPU, all threads | 12.421875 | 9.296875 | 1.34× |

| Pair | Order | Baseline wall (s) | Compiled wall (s) |
|---|---|---:|---:|
| 1 | baseline → compiled | 27.766257 | 16.547838 |
| 2 | compiled → baseline | 11.419431 | 8.248455 |
| 3 | baseline → compiled | 11.060818 | 8.323002 |
| 4 | compiled → baseline | 15.412262 | 10.106261 |
| 5 | baseline → compiled | 10.543867 | 7.578790 |

Baseline wall range: 10.543867–27.766257 s.

Compiled wall range: 7.578790–16.547838 s.

First candidate, own empty compiled cache: 16.547838 s (included).

| Calculation count | Baseline per run | Compiled per run |
|---|---:|---:|
| Uniform normal fallback | 883 | 883 |
| Interior pose interpolation | 199,892 | 99,391 |

### 120 target frames, all captured points

| Metric | Baseline median (s) | Compiled median (s) | Ratio |
|---|---:|---:|---:|
| Complete process | 76.194248 | 32.280461 | 2.36× |
| Full evaluation API | 73.106335 | 29.003228 | 2.52× |
| Clock profile + extrinsic refinement | 70.402356 | 25.939339 | 2.71× |
| Adaptive nearest-plane calls | 4.436920 | 1.322672 | 3.35× |
| Worker imports | 2.263285 | 2.458281 | 0.92× |
| Worker CPU, all threads | 75.984375 | 33.265625 | 2.28× |

| Pair | Order | Baseline wall (s) | Compiled wall (s) |
|---|---|---:|---:|
| 1 | baseline → compiled | 72.863296 | 65.425788 |
| 2 | compiled → baseline | 70.815200 | 30.007821 |
| 3 | baseline → compiled | 76.194248 | 35.921352 |
| 4 | compiled → baseline | 87.984057 | 32.280461 |
| 5 | baseline → compiled | 78.327067 | 27.016137 |

Baseline wall range: 70.815200–87.984057 s.

Compiled wall range: 27.016137–65.425788 s.

First candidate, own empty compiled cache: 65.425788 s (included).

| Calculation count | Baseline per run | Compiled per run |
|---|---:|---:|
| Uniform normal fallback | 883 | 883 |
| Interior pose interpolation | 1,296,914 | 434,232 |

### Prior candidate: default budgets

| Metric | Baseline median (s) | Compiled median (s) | Ratio |
|---|---:|---:|---:|
| Complete process | 17.005656 | 16.292120 | 1.04× |
| Full evaluation API | 14.301607 | 10.530992 | 1.36× |
| Clock profile + extrinsic refinement | 13.518006 | 8.443214 | 1.60× |
| Adaptive nearest-plane calls | 0.657815 | 0.750400 | 0.88× |
| Worker imports | 2.835184 | 2.931566 | 0.97× |

| Pair | Order | Baseline wall (s) | Compiled wall (s) |
|---|---|---:|---:|
| 1 | baseline → compiled | 17.005656 | 16.883971 |
| 2 | compiled → baseline | 29.870457 | 17.501927 |
| 3 | baseline → compiled | 24.260037 | 16.292120 |
| 4 | compiled → baseline | 14.962965 | 11.326237 |
| 5 | baseline → compiled | 14.460176 | 9.043197 |

Baseline wall range: 14.460176–29.870457 s.

Compiled wall range: 9.043197–17.501927 s.

First candidate, own empty compiled cache: 16.883971 s (included).

| Calculation count | Baseline per run | Compiled per run |
|---|---:|---:|
| Uniform normal fallback | 883 | 883 |
| Interior pose interpolation | 199,892 | 199,892 |

### Prior candidate: 120 target frames

| Metric | Baseline median (s) | Compiled median (s) | Ratio |
|---|---:|---:|---:|
| Complete process | 56.427418 | 52.678920 | 1.07× |
| Full evaluation API | 52.966413 | 50.036244 | 1.06× |
| Clock profile + extrinsic refinement | 49.127861 | 45.401763 | 1.08× |
| Adaptive nearest-plane calls | 2.988062 | 5.972435 | 0.50× |
| Worker imports | 2.293762 | 2.045346 | 1.12× |

| Pair | Order | Baseline wall (s) | Compiled wall (s) |
|---|---|---:|---:|
| 1 | baseline → compiled | 83.970250 | 61.860810 |
| 2 | compiled → baseline | 75.395782 | 60.961356 |
| 3 | baseline → compiled | 56.427418 | 28.086238 |
| 4 | compiled → baseline | 51.360295 | 27.899039 |
| 5 | baseline → compiled | 51.661806 | 52.678920 |

Baseline wall range: 51.360295–83.970250 s.

Compiled wall range: 27.899039–61.860810 s.

First candidate, own empty compiled cache: 61.860810 s (included).

| Calculation count | Baseline per run | Compiled per run |
|---|---:|---:|
| Uniform normal fallback | 883 | 883 |
| Interior pose interpolation | 1,296,914 | 1,296,914 |

<!-- measurements:end -->

Each configuration contains five paired fresh interpreters. Order reverses
every other pair. Each method has its own initially empty compiled cache;
all first compilation, imports and later cache loading are included. Metadata
records the cache file hashes before and after every run. All timings remain,
without warmup, outlier exclusions or changed point budgets.

Wall includes interpreter startup, imports, replay, every clock candidate and
extrinsic refinement, holdout diagnostics, all input hashes, YAML output and
measurement save. Parent source/input audits and CLI presentation are outside
that timer. Phase timers and counters wrap existing calls, with their overhead
included. Optional instrumentation selection reads the config during startup,
also inside wall time. These are local descriptive observations with persistent
OS caches and unrelated activity; they are not portable confidence estimates.
The worker CPU diagnostic sums all worker threads from entry through the
saved-result hash. It excludes interpreter startup and measurement-JSON
serialization; complete-process wall time includes both. CPU and wall times
are reported separately, without normalizing or replacing observed timings.

The unchanged default configuration captures 3,000 source records and 36 target
frames with 400 points each, selecting 6,000 train and all 3,200 holdout targets.
The unchanged full-capture configuration uses 120 frames and all 38,400 train /
9,600 latest holdout targets. Both keep the existing MAD/Huber settings,
initial estimates, temporal partitions and bounded optimization. No additional
subsampling or approximate nearest search is introduced.

The default result retains rank 6, six clock rounds, 4,773 train / 2,526 holdout
correspondences, -0.04 s selected offset and holdout RMSE 0.05078387183780445 m.
The full-capture result retains rank 6, eight clock rounds, 29,181 train / 3,451
holdout correspondences, -0.025 s selected offset and holdout RMSE
0.0922906037408748 m. Its status remains `max_iterations`; the acceleration
preserves that bounded outcome and does not claim convergence or an absolute
clock reference. Holdout never chooses the solution.

All 20 current and 20 prior API results are checked against their actual
schema-valid saved YAML, raw file hashes and full canonical payload hashes.
Within each separately measured configuration/source cohort,
every value and provenance field matches except the exact declared optional
execution identity note (adapter v0.1 versus v0.2). Full unfiltered payloads and
hashes remain in the published trials. The comparison preserves every other
note, candidate score, transform, offset, rejection, diagnostic, reason and
status. Ordered point-index/capture-timestamp/XYZ hashes bind the unchanged
disjoint train and holdout samples. Ratios compare these same complete runs;
earlier gains and profiler times are not multiplied into them.

## Validation and provenance

Local validation: 464 relevant tests passed, one optional integration skipped;
Ruff and strict mypy passed; all 126 exported schemas remain byte-identical.
Tests cover float-bit residuals, numerical Jacobians, weighted/unweighted
normal equations, complete solver histories and diagnostics, hybrid ties,
missing candidates, frozen field captures, cache bounds, invalidation, custom
methods, MAD thresholds and support-floor ordering. Complete public-data
results retain all quality judgments and provenance.

The four definitions and recomputed aggregates retain all trials and canonical
payloads. `publication-index.json` binds exact Git source/config archives,
definitions, aggregates, conversion lineage and the publication launcher.
Original measurement hashes and working source hashes remain in the index.
Archives use LF; publication checks that only line endings differ from working
bytes. Private executable/workspace prefixes become placeholders. Compiled
binaries stay in local runtime caches and are not published.

The preceding candidate `09c346935c610ed32517f31772e4be46090b5817` is retained
separately in `prior_default_budget.*` and `prior_full_capture.*`. Its five-pair
wall medians were 17.005656 → 16.292120 s (1.04×) and
56.427418 → 52.678920 s (1.07×), respectively. Those measurements motivated
the additional bounded pose reuse and bulk adaptive gates. They remain full
separate cohorts with their original payloads, source revision and timings;
they are not discarded or pooled with the current candidate. Changed source
archives are named `prior.*`, with the same baseline and public inputs.
The recorded Git revisions in result and train-diagnostic provenance differ
between the prior and current cohorts and are preserved. The launcher verifies
those two exact expected revisions separately when checking cross-cohort
numerical parity; strict within-cohort comparisons exclude only the declared
execution identity note.

Official inputs and conversion lineage match the preceding fixture. The raw
bag MD5 is `fb17a6d4e0bc1765e276f0dcca8a5063` (613,350,674 bytes) and the
GT archive MD5 is `ea25f7703ec5d076ba91dba63666d92c` (140,570 bytes).
`conversion-provenance.json` binds the trajectory, converted database/metadata
and original clock-normalization command. AIST Zenodo record 6864654 licenses
the data under CC-BY-4.0.

## Reproduce

Prepare the recording with the
[verified public-data instructions](../adaptive_plane_search/README.md) and
install `python -m pip install -e ".[numba]"`. At the measured candidate commit,
keep source, config and inputs fixed, and use a new output folder per cohort:

```bash
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope compiled \
  --baseline-commit 46e8df3a85ba6ce6706b5b83f3791595665198aa \
  --config examples/public_datasets/glim_versatile/online_continuous_time_compiled_queries_config.yaml \
  --numba-cache-policy persistent \
  --repetitions 5 --output-dir outputs/glim-linearization-default
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope compiled \
  --baseline-commit 46e8df3a85ba6ce6706b5b83f3791595665198aa \
  --config examples/public_datasets/glim_versatile/online_continuous_time_compiled_full_capture_config.yaml \
  --numba-cache-policy persistent \
  --repetitions 5 --output-dir outputs/glim-linearization-full
```

`--numba-cache-policy cold` gives every process a separate empty cache; report
that alternative protocol separately. The ordinary calibration command applies
the acceleration with the existing optional-backend configuration:

```bash
calibrex continuous-time-lidar-pair \
  examples/public_datasets/glim_versatile/online_continuous_time_compiled_full_capture_config.yaml \
  --output outputs/glim-linearization.yaml
```
