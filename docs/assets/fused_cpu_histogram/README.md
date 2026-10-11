# Fused CPU histograms on public KITTI raw

Six schema-valid JSON files preserve timing and numerical validation of the
fused `numba_cpu` backend measured at commit `f00c54b`
(`f00c54b9cd7ce32818ee72f6a96398814f99d5c6`). The public input is
`2011_09_30_drive_0018_sync/image_02`, original Velodyne scans, and frozen
FeatDepth provider SHA256
`7e3ef953a360a7c01ed0cdc3cf947db97fb0aff5912ced805d80bebe210d3360`.
The same 25 frames contain 3,130,279 points: 20 fit and five holdout, fixed
before optimization and histogram-range selection. The declared prior adds
local x rotation 0.5 degrees and camera-frame x translation 0.5 m to the
dataset reference. Search bounds, steps, and the 800-evaluation budget use
defaults. The reference is never the search target.

Projection, rounded-pixel validation, stable nearest-point selection, and
histogram accumulation run in a compiled float64 loop. Frozen feature bins
avoid repeated searches, and thread-local scratch inspects and clears only
pixels touched by the current scan. The candidate uses two persistent frame
threads with ordered reduction. No subsampling, approximate arithmetic, or
search-budget reduction is used.

Each method was invoked three times from the same problem bytes and prior.
Serial NumPy runs first on repetitions one and three; the candidate runs first
on repetition two. An absent `NUMBA_CACHE_DIR` was selected before the first
candidate. There is no separate warm-up: its first solve includes JIT compilation,
and subsequent candidates reuse the compiled cache in the same process.

| Repetition | Execution order | Serial solve (s) | Fused solve (s) | Serial pipeline (s) | Fused pipeline (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Serial, candidate | 53.927852 | 5.139488 | 56.559027 | 8.200420 |
| 2 | Candidate, serial | 100.623447 | 3.624189 | 103.568001 | 6.388714 |
| 3 | Serial, candidate | 67.077730 | 3.509601 | 70.886101 | 6.499727 |
| Median | | 67.077730 | 3.624189 | 70.886101 | 6.499727 |

The ratio of solve medians is **18.5083x**; the ratio of pipeline phase medians
is **10.9060x**. The first pair is **10.4928x** for solve, including cold JIT
compilation. Pipeline phases include input verification/loading, immutable
feature preparation and binning, solve, holdout evaluation, and pool shutdown.
Shared demo setup, result serialization, and HTML rendering are excluded.
Measurements ran on an i7-9750H, Windows, Python 3.12.10, NumPy 2.3.4, and
Numba 0.65.1, without isolating other running work or changing numerical library
thread settings. The serial range, 53.928–100.623 s, shows substantial variation.
Three repetitions on one workload do not establish a portable speed guarantee
or statistical confidence. Bootstrap fields are schema summaries and must not
be interpreted as confidence here.

All six timed invocations converged after exactly 229 evaluations. Complete
solver records, metrics, final poses, skipped frames, and initial/final holdout
evaluations match exactly, including the earlier serial/CUDA x-positive trace.
The canonical complete numerical payload SHA256 is
`fdf4d36e8cbe9b8a88a5cb765f4c387f67fa2021df8e51271ce461bb4f9e3160`.
Final reference differences are 0.390923 degrees / 0.160078 m and holdout MI is
1.293495 nat. These results preserve the existing calibration outcome; they do
not establish accuracy. Quality remains `warn`, covariance and observability
are unknown, raw scans are not motion compensated, and FeatDepth may have KITTI
training overlap.

Separately, the fused candidate was run on the six previously frozen signed-axis
problems and compared against each original CPU result's full numerical payload.

| Case | Evaluations | Complete numerical equality | Solver status |
| --- | ---: | --- | --- |
| x-positive | 229 | Exact | Converged |
| x-negative | 313 | Exact | Converged |
| y-positive | 277 | Exact | Converged |
| y-negative | 241 | Exact | Converged |
| z-positive | 265 | Exact | Converged |
| z-negative | 241 | Exact | Converged |

`axis-validation.definition.json` and `axis-validation.benchmark.json` retain
these six candidates and their six imported historical CPU numerical oracles.
Timing is omitted from this matrix: it validates numerical equality and is not
a paired speed measurement. All twelve source result hashes and six problem
hashes are bound. The validation launcher saved all six standard results before
a metadata error interrupted aggregation; a separate bound aggregator read
those unchanged bytes and supplied the required metric interpretations.

`comparison.definition.json` and `comparison.benchmark.json` retain every
timed invocation, individual timings, inputs/configuration, source identity,
and original external standard-result SHA256 bindings. `serial.result.json`
and `fused.result.json` retain the first pair's full traces and holdout records.
Every subsequent full numerical payload was checked against this pair, and
its shared digest is recorded in each timed benchmark trial. The adapter,
measurement launcher, and evidence publisher SHA256 values are also bound.
Dataset/workspace paths are replaced with `${DATASET_ROOT}` and
`${WORKSPACE_ROOT}`. Original output hashes bind original external bytes;
published representative files have separate hashes in `source_artifacts` and
retain their original-result hash under `run.provenance.publication`.
Raw data, depth weights, generated HTML, and images remain outside Git.

The generated local report was checked at desktop/mobile widths, including
expanded provenance, nine decoded images, and nine SVG overlays. Original
measurement artifacts remain in `outputs/public-kitti-six-dof-fused-cpu-v1`;
axis results remain in `outputs/public-kitti-six-dof-fused-cpu-axis-validation-v1`.

Reproduce with an optional dependency and a previously absent cache directory:

```bash
pip install 'calibrex[numba]'
NUMBA_CACHE_DIR=/path/to/new-empty-cache calibrex demo kitti-six-dof \
  /path/to/2011_09_30_drive_0018_sync /path/to/featdepth-provider.json \
  --candidate-backend numba_cpu --candidate-frame-workers 2 --repetitions 3 \
  --output-dir outputs/kitti-six-dof-fused-cpu --json
```

In PowerShell, set `$env:NUMBA_CACHE_DIR` before invoking the same command.
The fixed provider, split, prior, search settings, and first invocation must
be retained for a comparable measurement.
