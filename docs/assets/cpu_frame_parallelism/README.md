# CPU frame parallelism on public KITTI raw

Four schema-valid JSON files preserve evidence from serial NumPy and a
two-thread NumPy candidate on `2011_09_30_drive_0018_sync/image_02`, original
Velodyne scans, and frozen FeatDepth provider SHA256
`7e3ef953a360a7c01ed0cdc3cf947db97fb0aff5912ced805d80bebe210d3360`.
The same 25 frames contain 3,130,279 points: 20 fit and five holdout, fixed before
optimization and histogram-range selection. The declared prior adds local
x rotation 0.5 degrees and camera-frame x translation 0.5 m to the dataset
reference. Search bounds, steps, and the 800-evaluation budget use defaults.
The reference is never the search target.

Each method was invoked three times from the same problem bytes and prior.
Serial runs first on repetitions one and three; the candidate runs first on
repetition two. There is no separate warm-up and every invocation is retained.
The candidate reuses one two-thread pool across search and initial/final
holdout evaluation. Frame calculations run concurrently and results are
reduced in input order. Thread startup is included in solve time.

| Repetition | Execution order | Serial solve (s) | Two-thread solve (s) |
| --- | --- | ---: | ---: |
| 1 | Serial, candidate | 64.057565 | 60.098134 |
| 2 | Candidate, serial | 72.149458 | 54.548826 |
| 3 | Serial, candidate | 70.177615 | 42.316660 |
| Median | | 70.177615 | 54.548826 |
| Range | | 64.057565–72.149458 | 42.316660–60.098134 |

The ratio of solve medians is **1.2865x**. Pipeline phase medians are
73.583689 / 57.475266 s (**1.2803x**), including input verification/loading,
backend preparation, solve, and holdout/pool shutdown. Shared demo setup,
result serialization, and HTML rendering are excluded. Measurements ran on
an i7-9750H, Windows, Python 3.12.10, and NumPy 2.3.4, without isolating other
running work or changing numerical library thread settings. Variation is
visible; three repetitions on one workload do not establish a portable
speed guarantee or statistical confidence. The artifact's bootstrap fields
are schema summaries and must not be interpreted as confidence here.

All six runs converged after exactly 229 evaluations. Complete solver
records, metrics, final poses, skipped frames, and initial/final holdout
evaluations match exactly, including the earlier serial/CUDA x-positive
trace. The canonical complete numerical payload SHA256 is
`fdf4d36e8cbe9b8a88a5cb765f4c387f67fa2021df8e51271ce461bb4f9e3160`.
Final reference differences are 0.390923 degrees / 0.160078 m and holdout
MI is 1.293495 nat. Convergence and CPU parity do not prove calibration accuracy:
quality remains `warn`, covariance and observability are unknown, raw scans
are not motion compensated, and FeatDepth may have KITTI training overlap.

`comparison.definition.json` and `comparison.benchmark.json` retain all six
attempts, per-invocation timings, source/inputs/configuration, and original
external standard-result SHA256 bindings. `serial.result.json` and
`parallel.result.json` retain the first pair's complete traces and holdout
records; subsequent complete numerical payloads were checked against this
pair, with the shared payload digest recorded in every benchmark trial.
Dataset/workspace paths are replaced with `${DATASET_ROOT}` and
`${WORKSPACE_ROOT}`. Original result hashes bind original external bytes;
published representative files have separate hashes in `source_artifacts`
and retain their original-result hash under `run.provenance.publication`.
Raw data, depth weights, generated HTML, and images remain outside Git.

The measured implementation is commit `5809c30`. After measurement the
report's CSS was updated to wrap long checkpoint hashes on mobile; the
original HTML and all original numerical JSON remain in
`outputs/public-kitti-six-dof-cpu-frames-v1`. HTML was regenerated only for
presentation and checked at desktop/mobile widths, including expanded
provenance, nine decoded images, and nine SVG overlays.

Reproduce with:

```bash
calibrex demo kitti-six-dof \
  /path/to/2011_09_30_drive_0018_sync /path/to/featdepth-provider.json \
  --candidate-backend numpy --candidate-frame-workers 2 --repetitions 3 \
  --output-dir outputs/kitti-six-dof-cpu-frames --json
```
