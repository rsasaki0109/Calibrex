# Public Livox packed evaluation evidence

Measured source: `ddee58110943558dddc86e07638d98de8cf9cc26`. This change keeps the existing algorithm and shares
contiguous coordinates between voxel and plane checks. Occupied neighbors are
prepared once in scalar offset order. Unbuffered voxel additions preserve input
order, and the resulting dictionary preserves first occurrence of each key.

Against PR #154 (`3e1d9b31c35ad884b9a556427cb0729a5071184e`), the full candidate
plus 24-control evaluation call measures **3.238162 -> 0.953632 s
(3.40x)**. This is a phase inside the same complete run,
excluding prior inspection and subsequent solvers, input-manifest hashing and saves.
It includes all work performed by `livox_pair_evidence_from_dataset`; inspection
may already have populated the invocation-scoped input/statistics cache for both methods.
No extra evaluation or warmup is added for timing.

Whole-process medians are **8.592928 -> 6.201291 s
(1.39x)**; the API subset is **6.168758 -> 3.911791 s
(1.58x)**. The larger phase gain must not be presented as the
whole-process gain. A separate direct cumulative comparison to the version before
Livox acceleration measures **54.025577 -> 9.269612 s
(5.83x)**. Ratios from different PRs are not multiplied.

## Incremental comparison

| Repeat | baseline subprocess_wall_seconds (s) | batched subprocess_wall_seconds (s) | baseline pipeline_seconds (s) | batched pipeline_seconds (s) | baseline livox_evidence_seconds (s) | batched livox_evidence_seconds (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 11.090379 | 7.653201 | 8.339141 | 4.985308 | 4.392274 | 1.340202 |
| 2 | 8.603423 | 6.192636 | 6.168758 | 3.876601 | 3.238162 | 0.942792 |
| 3 | 8.478290 | 6.215716 | 6.112893 | 3.925004 | 3.134286 | 1.047149 |
| 4 | 8.592928 | 6.201291 | 6.304928 | 3.911791 | 3.416829 | 0.939531 |
| 5 | 8.367022 | 6.179157 | 6.052314 | 3.864506 | 3.155061 | 0.953632 |
| Median | 8.592928 | 6.201291 | 6.168758 | 3.911791 | 3.238162 | 0.953632 |

## Direct cumulative comparison

| Repeat | baseline subprocess_wall_seconds (s) | batched subprocess_wall_seconds (s) | baseline pipeline_seconds (s) | batched pipeline_seconds (s) | baseline livox_evidence_seconds (s) | batched livox_evidence_seconds (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 54.025577 | 8.316664 | 49.665008 | 4.635557 | 44.197954 | 1.150935 |
| 2 | 63.654208 | 8.108413 | 59.231271 | 5.162571 | 53.595254 | 1.295437 |
| 3 | 66.710272 | 11.238523 | 61.696251 | 7.409764 | 55.489454 | 1.268679 |
| 4 | 36.251009 | 13.131612 | 33.139072 | 8.433906 | 30.023464 | 1.825199 |
| 5 | 35.793597 | 9.269612 | 33.503588 | 6.305183 | 30.053843 | 2.096733 |
| Median | 54.025577 | 9.269612 | 49.665008 | 6.305183 | 44.197954 | 1.295437 |

## Algorithm and limits

The public capture has 679 source planes. The CSR neighborhood grid has 17,640
rows and 18,333 occupied entries, at most 23 per row. The original direct table
and CSR lookup arrays together retain 393,632 bytes; their combined budget is
8 MiB per prepared plane index. This cap covers lookup storage, not point arrays,
reference geometry or preparation temporaries. Queries use chunks of at most
4,096 points and 27 associations per point. Larger extents retain the previous
sparse/structured/scalar paths. Missing cells, clipped aliases, nearest ties,
nearest planes without normals, NaN/infinite gates and arbitrary-size scalar
voxel keys keep the original behavior.

Captured coordinate arrays are read-only. Only the latest transformed array per
file is retained, shared by the two checks; transform identity preserves signed
zero distinctions. A fresh invocation observes replaced input bytes. Public
point lists are independent copies. For sufficiently large safe inputs, voxel
sums use checked integer encoding and `numpy.add.at`, not a pairwise reduction.
First-seen key order is retained so shared-key traversal and centroid RMSE stay
exact. Invalid or unrepresentable inputs retain the scalar exception path.
No new dependency or GPU backend is introduced.

All 20 final saved results and complete API payloads are equal after normalizing
run ID, timestamps, output destinations and Windows separators. Every numerical
metric, all 24 controls, diagnostic, reason and quality grade is retained. Quality
remains `fail`, and registration remains `inconclusive`; successful benchmark
execution is not a ground-truth calibration accuracy claim. The
[intermediate comparison](intermediate/pipeline.definition.json) retains ten
earlier workers from `11180abfc6d17e6a4a7cec4d89c11232900624af` without phase timing. Its numerical source
is the same, but its benchmark tool and recorded Git identity differ. Intermediate
timings are excluded from final medians, and result parity is checked separately
within that cohort.

## Protocol and provenance

Each comparison uses five fresh-process repetitions per method, reversing order
every other repeat, with no separate warmup or outlier exclusion. All timings and
ranges are retained in the definitions and aggregates. OS caches persist and
unrelated host activity is not controlled; timing variation is visible in the
tables. Descriptive ratios are not confidence intervals or portable guarantees.
Only child `OMP_NUM_THREADS=1` is set, equally for both methods.

Wall includes interpreter/import startup, config loading, inspection, complete
inputs, candidate plus 24 controls, native point-to-point/point-to-plane ICP,
Open3D GICP, NDT availability and result/report/artifact saves. Workers call the
calibration Python API; CLI argument dispatch is excluded. Measurement JSON
writes are included in wall, while parent validation and aggregation are excluded.
Only the five pinned modules listed in each definition are substituted for the
baseline; numerical ICP solvers and writers are shared. Loaded module paths,
source/config/input bytes and the Git identity are frozen and checked.

Definitions and aggregates are schema-validatable. All three aggregates recompute
exactly, and all 30 saved YAML results are reloaded and compared to their complete
API payloads. Every saved artifact hash is verified. Source archives are exact
Git bytes; committed-byte and working-byte hashes are recorded separately.
Private workspace prefixes are normalized only in commands and metadata;
canonical result strings and original input/output hashes remain unchanged.
Publication resolves the method's short commit to its full Git ID; canonical
results retain the original short identity.

Inputs are the [official Livox sample](https://github.com/Livox-SDK/Livox_automatic_calibration):
23,921 finite source points and 23,666 finite target points. Target header count
23,786 includes 120 nonfinite positions excluded by the unchanged reader policy.
Their SHA-256 values are
`19b3864cbee76b785d207fcb3fe48f6f506d142d8233f842d0a87fe4330fe407` and
`61a74eb8b8ff4eeba4ca0ff824a4046ae220e947e8156f27d93da452538f5b6f`.
Point clouds are not redistributed. This is one pair, without independent
temporal or ground-truth validation. Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1,
Open3D 0.19.0.

## Reproduction

Use the measured commit, download the pinned public data and run into new folders:

```bash
python tools/download_public_dataset.py livox_horizon_horizon_pcd_sample
python tools/benchmark_livox_evaluation.py --output-dir outputs/packed-incremental \
  --baseline-commit 3e1d9b31c35ad884b9a556427cb0729a5071184e --repetitions 5 --omp-threads 1
python tools/benchmark_livox_evaluation.py --output-dir outputs/packed-original \
  --baseline-commit c6fd98b7b5af73be41fa01b5e4938773f4696fb1 --repetitions 5 --omp-threads 1
```

Local verification: 251 distinct tests passed and two environment-dependent tests
skipped (official KITTI sequence not configured; Windows symlink creation unavailable).
Ruff and strict mypy on 349 source files passed. Tests cover exact residual bits,
lookup budgets, dense/sparse neighborhoods, ordered sums, first-seen keys,
nonassociativity, signed zeros, invalid inputs, fresh reads and bounded query storage.
