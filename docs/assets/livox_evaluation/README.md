# Public Livox full-pipeline acceleration evidence

The measured change reuses decoded records, invariant cloud statistics and source
planes within one candidate/probe evaluation, then discards that context. It
also batches the same 27 voxel neighbors and both Hamilton products of the
coordinate transform. Checked sparse integer encodings replace structured-key
comparisons; complete voxel coordinates are still checked on every match. Huge
extent products use structured lookup, and no dense grid is allocated.
No process-wide cache, reduced input population, fewer
probes, changed gate or approximate nearest-neighbor search is used.

Full fresh-process wall medians are **53.535174 →
14.244674 s (3.76x)**. All ten complete
normalized result payloads are exactly equal, including all 24 known-bad controls,
native ICP diagnostics, Open3D GICP results, metrics, reason strings and quality
grades. The final quality remains **fail** and native comparison remains
**inconclusive**; successful benchmark trials mean execution and parity.

## Scope and source binding

Implementation commit: `8c760c7d0859133934a26c26c1f788f95aa2ffaa`. Baseline Livox modules: `c6fd98b7b5af73be41fa01b5e4938773f4696fb1`.
The baseline loader substitutes only `data/livox.py` and `evaluation/lidar.py`
using their archived exact Git bytes. Everything else, including both native
ICP solvers, optional Open3D adapter, pipeline and writers, uses the same
implementation checkout. Each worker records its resolved module filenames;
the parent checks those bindings and freezes source/config/input bytes and HEAD.
`run.git_commit` in the complete result identifies that shared checkout, while
benchmark method provenance identifies the two substituted baseline modules.

Wall includes fresh interpreter and imports, config loading, dataset inspection,
all PCD reads, the candidate and 24 rotational/translational controls, native
point/plane ICP with all diagnostics, Open3D GICP, NDT availability check, quality
evaluation, result YAML, HTML/Markdown/JSON report and artifact saves. It also
includes the worker's measurement JSON write. Parent validation/comparison and
benchmark aggregation are outside wall. `pipeline_seconds` starts at the public
`run_calibration` API and ends after its artifact writes; it is a subset of wall.
The unchanged config caps registration inputs at 500 voxel centroids; candidate
evaluation still uses every finite raw point (23,921 source / 23,666 target).
Fresh workers call the public calibration Python API. Calibrex CLI argument
dispatch is excluded; the worker's measurement JSON write is included in wall.

The official [Livox calibration sample](https://github.com/Livox-SDK/Livox_automatic_calibration)
contains two PCD captures. Inputs are downloaded by
`tools/download_public_dataset.py livox_horizon_horizon_pcd_sample` and remain
external. Their file SHA256 values are
`19b3864cbee76b785d207fcb3fe48f6f506d142d8233f842d0a87fe4330fe407` and
`61a74eb8b8ff4eeba4ca0ff824a4046ae220e947e8156f27d93da452538f5b6f`.
The target header declares 23,786 records; the unchanged reader excludes 120
nonfinite positions. This single pair is limited geometry evidence, without
independent temporal calibration validation or ground-truth recovery claims.

## Repetitions

Five repetitions use fresh subprocesses, reverse method order every other
repetition, and have no separate warmup. OS caches persist. Unrelated user work
continues on the shared Windows host. `OMP_NUM_THREADS=1` is set only in child
environments for both methods to fix the Open3D parallel reduction setting;
host settings and user processes are untouched. Python 3.12.10, NumPy 2.3.4,
SciPy 1.17.1 and Open3D 0.19.0; no GPU is used by this change.

| Repeat | Baseline wall (s) | Batched wall (s) | Baseline pipeline (s) | Batched pipeline (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 53.535174 | 12.577426 | 50.853829 | 10.140637 |
| 2 | 48.473362 | 11.892567 | 46.004257 | 9.657413 |
| 3 | 54.858395 | 16.256048 | 52.124112 | 13.082981 |
| 4 | 77.499665 | 21.934240 | 73.788006 | 18.552510 |
| 5 | 52.687805 | 14.244674 | 49.503751 | 11.610806 |
| Median | 53.535174 | 14.244674 | 50.853829 | 11.610806 |

Pipeline medians improve **4.38x**. Wall ranges are
48.473362–77.499665 s baseline and
11.892567–21.934240 s batched. Pipeline ranges
are 46.004257–73.788006 and
9.657413–18.552510 s. No outliers are dropped.
Ratios describe these repeated captures and shared-host timings, without a
portable speed guarantee or confidence claim. Bootstrap fields are schema
summaries only. Prior ICP-only gains must not be multiplied by these ratios.

## Numerical contract and retained evidence

The same lexicographic 27-neighbor order resolves ties. A nearest voxel without
a normal still rejects a correspondence. Residual multiply/add/sqrt ordering,
Hamilton product ordering, input-order sums, population denominators and scalar
exception behavior are preserved. Query chunks materialize at most 4,096 × 27
associations. Very large keys and unsupported coordinates use the original
scalar path. Every evaluation invocation constructs a new context and observes
fresh inputs; reused geometry is confined to that invocation.

`pipeline.definition.json` and `pipeline.benchmark.json` retain all ten timings,
all complete result strings and their canonical output SHA256 values. Raw result
file and every written artifact's digests are separate and remain recorded.
Parity normalizes only run ID, creation/generation timestamps, destination paths
and Windows path separators. All numeric and diagnostic fields remain present.
The four JSON reports validate against standard benchmark schemas, and both
aggregates are recomputed exactly from their published definitions.

`pilot.pipeline.*` retains the two earlier full-pipeline trials: 61.149350 s
baseline and 26.448565 s candidate, with exact within-pilot result parity. That
uncommitted candidate preceded batched coordinate transforms and is not bound to
the final implementation commit. No pilot timing enters the final medians.
`pilot-import-failure.log.txt` retains an earlier launcher import-order failure;
it produced no calibration trial. The corrected loader uses normal package
initialization order. Source/archive digests and the publication/profiling
launchers remain available. Published measured/baseline sources are exact Git blobs; original
working-byte hashes and original report hashes are recorded separately.

A baseline cProfile investigation measured 174.301498 s with profiling overhead;
166.427 s lay in candidate/control evaluation. Its nested cumulative costs
included 47.052 s neighborhood residuals, 37.463 s PCD decoding, 27.730 s repeated
geometry coverage and 27.679 s repeated coordinate transforms. These are
overlapping profiling costs, not formal wall timings or additional speedups.
The profiling launcher and baseline sources bind that investigation; its
instrumented timing does not enter any benchmark median.

## Reproduction

```bash
python tools/download_public_dataset.py livox_horizon_horizon_pcd_sample
python tools/benchmark_livox_evaluation.py \
  --output-dir outputs/public-livox-evaluation-v2 --repetitions 5 --omp-threads 1
```

The output directory must be new or empty. Public tests cover exact residual and
transform bits, ties, normal-less nearest voxels, neighbor bounds, chunk bounds,
malformed/extra files, large keys, original exceptions, all 24 controls and fresh
reads across invocations. The external checkpoint-loading experiments are not
part of this implementation or its measured gain.

The [intermediate structured-lookup evidence](structured_lookup/README.md)
retains all ten executions of commit `41455b1de39394891c4d659af553e12b663261e7`,
which measured 82.582993 → 35.090965 s wall medians (2.35x) before integer
lookup. Its source archives and timings remain separate. No intermediate or
pilot timing enters the final medians. There are 22 distinct recorded calibration
executions: ten final, ten intermediate and two pilot; the pilot reports are
duplicated in both evidence folders for completeness.
