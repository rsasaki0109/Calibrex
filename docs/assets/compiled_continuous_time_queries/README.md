# Exact compiled continuous-time queries on public Livox data

The measured baseline is `f93502cf447c0c0a0eb06d80660ea51fa6bbe3ca`,
including the preceding geometry/interpolation reuse. The measured candidate
is `16e45b0a6a7253439c27526af9db36c8a6072388`. Seven runtime modules are
substituted together; every loaded filename and source hash is verified.
The additional optional Numba adapter is bound in the measured source archive.

The implementation captures adaptive-plane centroids in the original level
and lexical cell order. Exact SciPy cKDTree queries find possible nearest
centroids in batches. Scalar `math.dist` applies the inclusive gate and
resolves nearby ties in the original order. Capturing a tree requires every
centroid to be inside its declared voxel cell and every cell at a level to
have the same size. Under those conditions every centroid within the gate
also lies inside the original per-level search box. Manual maps, unsupported
geometry, very small distances and small batches use the scalar query.
No approximate search or additional downsampling is used.

The optional `continuous_time_query_backend: numba_cpu` setting fuses
timestamp interval lookup, interpolation, pose composition, world coordinates
and repeated optimizer residuals into CPU kernels. Hamilton products,
normalization counts, scalar addition/product order and shortest-arc slerp
are retained. Kernels use `fastmath=False`, `cache=True` and `nogil=True`.
The factor retains its numerical finite differences, robust weights, normal
equations, acceptance rules and observation-array invalidation.

Trajectory capture requires native immutable sample poses, strictly increasing
timestamps and a supported integer range. Unsupported trajectories, replaced
samples/timestamps or an instance interpolation override retain scalar handling.
Interpolation/clamp/extrapolation diagnostics count the same visited prefix,
including early rejection. The Python backend remains the default; requesting
Numba without the optional dependency raises an actionable error. Actual kernel
execution and its Numba version are recorded in existing provenance notes,
without changing the saved result schema.

## Complete-process measurements

Both cohorts use the verified [AIST public Livox recording](https://zenodo.org/records/6864654),
its supplied trajectory and the existing tutorial conversion. The copied
point-cloud topic is an identity execution control; these data provide no
independent two-sensor extrinsic or clock reference.

<!-- measurements:start -->

In the archived method IDs, `sparse` identifies the implementation
with exact adaptive tree queries and optional compiled CPU kernels.

### Default sampling budgets

| Metric | Baseline median (s) | Compiled median (s) | Ratio |
|---|---:|---:|---:|
| Complete process | 26.255426 | 15.642958 | 1.68× |
| Full evaluation API | 24.539626 | 12.440388 | 1.97× |
| Clock profile + extrinsic refinement | 23.486181 | 11.192398 | 2.10× |
| Adaptive nearest-plane calls | 4.183876 | 0.703498 | 5.95× |
| Worker imports | 1.903144 | 2.470318 | 0.77× |

| Pair | Order | Baseline wall (s) | Compiled wall (s) |
|---|---|---:|---:|
| 1 | baseline → compiled | 25.878120 | 15.866192 |
| 2 | compiled → baseline | 26.212697 | 10.614904 |
| 3 | baseline → compiled | 27.624444 | 15.642958 |
| 4 | compiled → baseline | 28.135323 | 18.035339 |
| 5 | baseline → compiled | 26.255426 | 15.348786 |

Baseline wall range: 25.878120–28.135323 s.

Compiled wall range: 10.614904–18.035339 s.

First candidate, empty compiled cache: 15.866192 s (included).

| Calculation count | Baseline per run | Compiled per run |
|---|---:|---:|
| Uniform normal fallback | 883 | 883 |
| Interior pose interpolation | 199,892 | 199,892 |

### 120 target frames, all captured points

| Metric | Baseline median (s) | Compiled median (s) | Ratio |
|---|---:|---:|---:|
| Complete process | 154.500906 | 78.868734 | 1.96× |
| Full evaluation API | 152.804193 | 74.014702 | 2.06× |
| Clock profile + extrinsic refinement | 150.784643 | 70.151344 | 2.15× |
| Adaptive nearest-plane calls | 22.298625 | 4.648079 | 4.80× |
| Worker imports | 1.620610 | 2.625096 | 0.62× |

| Pair | Order | Baseline wall (s) | Compiled wall (s) |
|---|---|---:|---:|
| 1 | baseline → compiled | 137.603824 | 62.528738 |
| 2 | compiled → baseline | 125.725489 | 56.714489 |
| 3 | baseline → compiled | 154.500906 | 81.832990 |
| 4 | compiled → baseline | 167.653343 | 123.214457 |
| 5 | baseline → compiled | 206.469119 | 78.868734 |

Baseline wall range: 125.725489–206.469119 s.

Compiled wall range: 56.714489–123.214457 s.

First candidate, empty compiled cache: 62.528738 s (included).

| Calculation count | Baseline per run | Compiled per run |
|---|---:|---:|
| Uniform normal fallback | 883 | 883 |
| Interior pose interpolation | 1,296,914 | 1,296,914 |

<!-- measurements:end -->

Each cohort contains five paired fresh interpreters, reversing order every
other pair. All timings are retained. There is no warmup, removed outlier or
discarded initial compilation. Each cohort starts with a new empty compiled
cache and retains it across its subsequent fresh interpreters. Metadata stores
the cache file names and SHA-256 hashes before and after each trial. Compiled
binaries are local runtime caches and are not published.

Whole-process wall includes interpreter startup, imports, replay, all clock
candidates, extrinsic refinements, holdout diagnostics, every input hash, YAML
output and worker measurement save. The optional backend's import, compilation
and cache loading are inside this wall time. Parent provenance audits and CLI
presentation are outside it. Phase timers wrap existing calls and include their
wrapper overhead. The worker also reads the config to select optional kernel
instrumentation; that setup is included in wall time.

The default-budget cohort retains the preceding capture: 3,000 source records,
36 target frames with 400 captured points each, 28 train/eight latest holdout
frames, a 6,000-point train cap and all 3,200 holdout points. Its configuration
differs from the preceding default-budget example only in project metadata and
the optional backend flag.

The full-capture cohort uses 120 target frames with 400 points each: all 38,400
train and 9,600 latest holdout points, with the same 3,000 captured source
records. Setting `continuous_time_max_target_points_per_split: 0` means no
additional solver subsampling. Baseline and candidate run this same heavier
configuration. Both cohorts keep the same within-cohort initial estimate,
MAD filtering, Huber loss, sampling identities, eight-round clock limit and
eight-iteration extrinsic limit. Holdout never selects the solution.

The default cohort retains 184,000 adaptive queries, rank 6, six clock rounds,
4,773 train/2,526 holdout correspondences and holdout RMSE
0.05078387183780445 m. The full-capture cohort retains 1,248,000 queries,
rank 6, eight clock rounds, 29,181 train/3,451 holdout correspondences and
holdout RMSE 0.0922906037408748 m. Its status remains `max_iterations`:
the acceleration preserves the existing bounded solver outcome and does not
claim convergence. The selected offsets (-0.04 s and -0.025 s respectively)
are profile outputs, not verified clock references.

Every one of the 20 API results is checked against its actual schema-valid
saved YAML, file hash and full canonical payload hash. Within each cohort,
all result values and provenance match across methods except the exact optional
execution identity note. The comparison recognizes only the documented identity
format, and keeps all other notes, revisions, candidate scores, poses, offsets,
rejections, train diagnostics, holdout evidence, statuses and reasons. Full
unfiltered payloads and their hashes remain in the published trials. Ordered
sampled point-index/capture-timestamp/XYZ hashes bind the distinct temporal
partitions and are verified on every run.

These are local descriptive observations on one public recording, with
persistent OS caches and unrelated activity. Ratios compare medians of these
same complete runs. They do not multiply earlier gains or combine profiler
times with whole-process measurements. Small jobs may amortize compilation
poorly; the first candidate trial and every later timing are visible above.

An earlier full-capture cohort stopped during its third pair when the parent
process ended. Its five completed saved/API results, phase timings, file hashes
and four rounded console wall observations remain in `interrupted-cohort.json`.
Its post-run source/input guard did not complete. This incomplete cohort is
preserved separately; the formal full-capture table uses a new five-pair run
with a fresh empty compiler cache and completed provenance guards.

## Validation and provenance

Local validation: 326 relevant tests passed, one optional integration skipped;
Ruff and strict mypy passed; all 126 exported schemas are byte-identical.
Tests compare pose/world/residual float bits, numerical Jacobians, Hessians
and gradients; complete clock-profile results and all trajectory diagnostics;
nearest identities, ties, inclusive gates and scalar fallbacks. They also
protect full artifact retention and the narrowly specified identity comparison.
The measured source commit's CI passed 2,241 tests (56 skips) on Python 3.12,
2,240 tests (57 skips) with ROS sourced, and 56 tests in the Numba adapter job.
All Python-version, adapter and schema-compatibility jobs completed successfully.

The two definitions and their recomputed aggregates retain all trials and full
canonical payloads. `publication-index.json` binds the source archives,
definitions, aggregates, conversion lineage and publication launcher, and
retains original measurement hashes and working source hashes. Source archives
are exact Git blobs with LF. Working files may use CRLF; publication checks
that only line endings differ. Private executable/workspace prefixes become
placeholders; numerical payloads are preserved.

Official inputs and conversion lineage are the same as the preceding fixture.
`conversion-provenance.json` records raw bag, GT archive, trajectory and converted
database/metadata hashes and the exact clock-normalization command. The raw
bag's official MD5 is `fb17a6d4e0bc1765e276f0dcca8a5063` (613,350,674 bytes);
the GT archive's is `ea25f7703ec5d076ba91dba63666d92c` (140,570 bytes).
Source: AIST Zenodo record 6864654, CC-BY-4.0.

## Reproduce

Prepare the recording using the
[checksum-verified public-data instructions](../adaptive_plane_search/README.md).
Install the optional CPU compiler:

```bash
python -m pip install -e ".[numba]"
```

At the measured source commit, keep source, config and inputs fixed throughout
measurement. Use a new output directory for each cohort:

```bash
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope runtime \
  --baseline-commit f93502cf447c0c0a0eb06d80660ea51fa6bbe3ca \
  --config examples/public_datasets/glim_versatile/online_continuous_time_compiled_queries_config.yaml \
  --numba-cache-policy persistent \
  --repetitions 5 --output-dir outputs/glim-compiled-default
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope runtime \
  --baseline-commit f93502cf447c0c0a0eb06d80660ea51fa6bbe3ca \
  --config examples/public_datasets/glim_versatile/online_continuous_time_compiled_full_capture_config.yaml \
  --numba-cache-policy persistent \
  --repetitions 5 --output-dir outputs/glim-compiled-full
```

Use `--numba-cache-policy cold` to retain a separate empty compiled cache for
every process; this is a different timing protocol and must be reported
separately. The ordinary command uses the requested backend and records its
actual execution identity:

```bash
calibrex continuous-time-lidar-pair \
  examples/public_datasets/glim_versatile/online_continuous_time_compiled_full_capture_config.yaml \
  --output outputs/glim-continuous-time-compiled.yaml
```
