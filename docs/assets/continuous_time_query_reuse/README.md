# Reusing exact continuous-time geometry on public Livox data

The measured baseline is `3f1e1f412d5d373493395e53b2636e079184e457`, including
the preceding sparse adaptive-plane acceleration. The measured implementation
is `3ab47025860140dfb529e5cf39ef97c6821c1254`. Seven runtime modules are
substituted together; the benchmark records and verifies every loaded filename.

The implementation captures uniform centroids/normals once per solve, reuses
the checked packed plane index, batches moving world-coordinate transforms in
the original Hamilton-product/normalization order, and reuses immutable
odometry poses and scalar slerp interval constants. Query counters still count
every call. Independent input hashing runs concurrently with replay/optimization;
config and recording file identities, sizes and modification/change times are
checked before/after the operation. Changed inputs invalidate the result.

Adaptive occupied-neighbor tuples retain their original order and use a shared
2 MiB estimated storage budget across levels. Geometry lookup retains its
existing combined 8 MiB array budget. Pose and composed-pose LRUs each retain at
most 8,192 entries; prepared interpolation retains at most 1,024 intervals.
Adaptive and odometry query caches support parallel readers and pickling.
All caches are scoped to their owning map/track/solve, and new geometry captures
observe edits to the original mutable plane accumulators.

## Complete-process measurements

Both cohorts use the same verified [AIST public Livox recording](https://zenodo.org/records/6864654),
its supplied trajectory, and the existing tutorial conversion. The second
topic duplicates the first sensor as an identity execution control. These data
provide no independent two-sensor extrinsic or clock reference.

<!-- measurements:start -->

In the archived method IDs, `sparse` identifies the implementation
with the added geometry, interpolation and query reuse.

### Existing bounded sampling

| Metric | Baseline median (s) | Reuse median (s) | Ratio |
|---|---:|---:|---:|
| Complete process | 15.542231 | 8.755586 | 1.78× |
| Full evaluation API | 13.646167 | 6.638731 | 2.06× |
| Clock profile + extrinsic refinement | 11.125639 | 5.711680 | 1.95× |
| Adaptive nearest-plane calls | 1.484466 | 0.959705 | 1.55× |
| Worker imports | 1.430621 | 1.517954 | 0.94× |

| Pair | Order | Baseline wall (s) | Reuse wall (s) |
|---|---|---:|---:|
| 1 | baseline → reuse | 28.019063 | 8.755586 |
| 2 | reuse → baseline | 15.327547 | 8.423320 |
| 3 | baseline → reuse | 14.965334 | 9.253689 |
| 4 | reuse → baseline | 16.822853 | 8.961019 |
| 5 | baseline → reuse | 15.542231 | 7.400178 |

Baseline wall range: 14.965334–28.019063 s.

Reuse wall range: 7.400178–9.253689 s.

| Calculation count | Baseline per run | Reuse per run |
|---|---:|---:|
| Uniform normal fallback | 220,413 | 697 |
| Interior pose interpolation | 79,422 | 46,582 |

### Solver default sampling budgets

| Metric | Baseline median (s) | Reuse median (s) | Ratio |
|---|---:|---:|---:|
| Complete process | 32.152753 | 17.195876 | 1.87× |
| Full evaluation API | 30.450080 | 15.456899 | 1.97× |
| Clock profile + extrinsic refinement | 28.335724 | 14.636664 | 1.94× |
| Adaptive nearest-plane calls | 4.416391 | 2.640364 | 1.67× |
| Worker imports | 1.201084 | 1.261734 | 0.95× |

| Pair | Order | Baseline wall (s) | Reuse wall (s) |
|---|---|---:|---:|
| 1 | baseline → reuse | 39.428962 | 18.826137 |
| 2 | reuse → baseline | 31.313682 | 16.350723 |
| 3 | baseline → reuse | 31.224687 | 17.755333 |
| 4 | reuse → baseline | 32.179855 | 17.195876 |
| 5 | baseline → reuse | 32.152753 | 15.585302 |

Baseline wall range: 31.224687–39.428962 s.

Reuse wall range: 15.585302–18.826137 s.

| Calculation count | Baseline per run | Reuse per run |
|---|---:|---:|
| Uniform normal fallback | 690,620 | 883 |
| Interior pose interpolation | 200,091 | 199,892 |

<!-- measurements:end -->

Each cohort contains five paired fresh interpreters, reversing order every other
pair. All timings remain, without warmup or excluded outliers. Whole-process
wall includes startup/imports, complete replay, all clock candidates and
extrinsic refinements, holdout diagnostics, every input hash, YAML output and
measurement save. Parent audits and CLI presentation are outside that timer.
Phase timers and calculation counters wrap existing calls, without extra
optimization, input reads or warmup. Timing wrapper/counter overhead is included.

The unchanged small configuration selects 2,000 captured source records and
1,200 target points per split. The additional configuration uses the solver's
default 6,000-record/6,000-target sampling caps with the same capture limits:
all 3,000 captured source records, 6,000 of 11,200 train targets and all 3,200
holdout targets. Both keep 28 train and eight latest holdout frames, MAD
filtering, Huber loss, the same initial estimate and fixed-extrinsic iteration
budget. Holdout never selects the solution.

The small cohort retains 62,400 adaptive queries, eight clock rounds, rank 6,
894 train/876 holdout correspondences and holdout RMSE 0.053133620013224026 m.
The default-budget cohort retains 184,000 queries, six clock rounds, rank 6,
4,773 train/2,526 holdout correspondences and holdout RMSE 0.05078387183780445 m.
The selected -0.04 s offset is the unchanged profile output, not a verified
clock reference. Different sampling budgets yield different result payloads;
parity is required within each cohort.

All 20 API results and actual saved schema-valid YAML artifacts match in full
within their cohorts, including every candidate score, selected pose/offset,
rejection count, train diagnostics, holdout evidence, reasons and provenance.
Only path separators are normalized; the shared measured checkout revision
remains. Ordered sampled index/capture-timestamp/XYZ hashes bind the distinct
temporal partitions and are checked in every process.

Timings describe one public recording on one machine, with persistent OS caches
and unrelated activity. Ratios use these same complete runs and do not multiply
earlier gains or mix profiler times into runtime comparisons.

## Validation and provenance

Local validation: 330 relevant tests passed, three optional data integrations
skipped; Ruff and strict mypy passed; all 125 exported schemas remain
byte-identical. Tests compare float bits against scalar geometry, nearest
identities/ties/gates, moving poses across chunk boundaries, shortest-arc slerp
and clamp branches, immutable captures, cache eviction/invalidation, query
counters, parallel readers and pickling. Hash-overlap tests prove concurrent
operation and reject modified/added/removed inputs or changed configuration.
Source-commit CI completed 2,220 tests (55 skips) on Python 3.12 and 2,219
tests (56 skips) with ROS sourced; adapter and schema-compatibility checks passed.

The two definitions and their recomputed aggregates retain all trials and full
canonical result payloads. `publication-index.json` binds the published files,
original measurement hashes and working source hashes. Exact baseline/measured
Git archives use LF; Windows working files can have CRLF. Only line endings
differ, as checked before publication. Private executable/workspace prefixes
are replaced with placeholders; numerical payloads remain unchanged.

Official inputs and the measured conversion lineage are the same as the
preceding sparse-query fixture. `conversion-provenance.json` records raw bag,
GT archive, trajectory and converted database/metadata digests and the exact
clock-normalization command. The raw bag's official MD5 is
`fb17a6d4e0bc1765e276f0dcca8a5063` (613,350,674 bytes); the GT archive's is
`ea25f7703ec5d076ba91dba63666d92c` (140,570 bytes). Source: AIST Zenodo record
6864654, CC-BY-4.0.

## Reproduce

Prepare the public recording using the
[checksum-verified sparse-query instructions](../adaptive_plane_search/README.md).
Keep source, config and inputs fixed throughout measurement, and use a new
output directory for each cohort:

```bash
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope runtime \
  --baseline-commit 3f1e1f412d5d373493395e53b2636e079184e457 \
  --repetitions 5 --output-dir outputs/glim-query-reuse-bounded
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope runtime \
  --baseline-commit 3f1e1f412d5d373493395e53b2636e079184e457 \
  --config examples/public_datasets/glim_versatile/online_continuous_time_default_budget_config.yaml \
  --repetitions 5 --output-dir outputs/glim-query-reuse-default
```

The ordinary continuous-time command applies the same acceleration:

```bash
calibrex continuous-time-lidar-pair \
  examples/public_datasets/glim_versatile/online_continuous_time_default_budget_config.yaml \
  --output outputs/glim-continuous-time-default.yaml
```
