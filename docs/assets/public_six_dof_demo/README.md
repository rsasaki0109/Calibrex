# Public KITTI raw six-DoF demonstration evidence

These seven schema-valid JSON artifacts retain numerical evidence from
`calibrex demo kitti-six-dof` on public KITTI raw
`2011_09_30_drive_0018_sync/image_02`, original Velodyne scans, and the frozen
FeatDepth provider SHA256
`7e3ef953a360a7c01ed0cdc3cf947db97fb0aff5912ced805d80bebe210d3360`.
The full provider contains 25 uniformly selected frames (3,130,279 points);
every fifth selected frame is holdout, leaving 20 fit / 5 holdout. The declared
initial pose adds local x rotation 0.5° and camera-frame x translation 0.5 m
to the dataset reference. Default search bounds, steps, and 800-evaluation
budget were fixed before executing either backend.

`comparison.definition.json` / `comparison.benchmark.json` describe the previous
single-invocation comparison. `numpy.result.json` / `cuda.result.json` retain all 229
evaluations, effective settings, frame IDs, initial/output/reference poses,
holdout records, timing phases, and backend identities. Both runs converged.
All solver records (apart from backend identity), metrics, final poses, and
initial/final holdout evaluations match exactly.

| Evidence | Initial | Final (both backends) |
| --- | ---: | ---: |
| Difference to reference rotation | 0.500000° | 0.390923° |
| Difference to reference translation | 0.500000 m | 0.160078 m |
| Fit MI | 1.064769 nat | 1.234290 nat |
| Holdout MI | 1.085693 nat | 1.293495 nat |

The final invocation ran on an i7-9750H / GTX 1660 Ti 6 GiB, Windows, Python
3.12.10, NumPy 2.3.4, CuPy 14.2.0, driver 596.36, with the local CUDA 12.8
toolkit. The default 256 MiB cache retained all 25 frames (190,892,928 bytes),
with zero large input uploads during evaluation.

| Final invocation | Verify/load | Prepare | Solve | Holdout | Pipeline total |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumPy | 5.813 s | <0.001 s | 54.824 s | 0.114 s | 60.751 s |
| CUDA | 4.377 s | 4.275 s | 6.734 s | 0.013 s | 15.398 s |

The final solve-time ratio is 8.14x; the pipeline-time ratio is 3.95x. These
are single invocations, including lazy CUDA compilation, rather than a warm
or repeated latency benchmark. The HTML overlays were rendered locally at
1440 px and 390 px widths: all nine images decoded, all nine SVGs rendered,
and the page had no horizontal overflow (tables scroll within their containers).

`first-invocation.definition.json` preserves the earlier development invocation
with the same numerical settings and split. Its initial-provenance annotation
was later clarified to identify a controlled perturbation rather than inherit
dataset-provided evidence. Numerical outputs and all 229 evaluations match the
final invocation. Its CPU/CUDA solve times were 49.445 / 10.794 s (4.58x), and
pipeline times 51.941 / 16.859 s (3.08x). Preserve both observations; this
variation does not support a portable speed guarantee or confidence interval.

## Three repetitions per backend

`repeat3.definition.json` / `repeat3.benchmark.json` record a subsequent
`--repetitions 3` comparison on the same hardware, public frames, frozen provider,
initial perturbation, and solver settings. All six invocations converged in 229
evaluations. Their complete numerical traces, final poses, metrics, and holdout
records match each other and the earlier published standard results exactly.
The generated shared problem SHA256 is
`dc29f3ea7d307ba4809cd28584e627140f1f4c86c659292e75fdfa99ae9ce16f`.

| Execution order | Repetition | Backend | Solve | Pipeline total |
| ---: | ---: | --- | ---: | ---: |
| 1 | 1 | NumPy | 74.306 s | 76.865 s |
| 2 | 1 | CUDA | 3.302 s | 6.349 s |
| 3 | 2 | CUDA | 3.611 s | 7.652 s |
| 4 | 2 | NumPy | 75.150 s | 77.958 s |
| 5 | 3 | NumPy | 51.270 s | 54.110 s |
| 6 | 3 | CUDA | 8.685 s | 17.739 s |

| Backend | Successful / attempted | Median solve | Solve range | Median pipeline | Pipeline range |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumPy | 3 / 3 | 74.306 s | 51.270–75.150 s | 76.865 s | 54.110–77.958 s |
| CUDA | 3 / 3 | 3.611 s | 3.302–8.685 s | 7.652 s | 6.349–17.739 s |

The ratio of solve medians is 20.58x; the ratio of measured pipeline medians is
10.04x. Every invocation starts from the shared declared prior with fresh input
verification/loading and projector preparation. All invocations, including the
first, are included; there is no separate warm-up. Process-wide kernel caches
persist between invocations. Pipeline total sums load/verification, preparation,
solve, and holdout; shared demo setup and report rendering are excluded.
These observed ranges expose substantial timing variation. Repeated timings on
the same frames do not establish independent recovery accuracy, confidence
intervals, or a portable speed guarantee. Bootstrap fields inherited from the
benchmark format are not statistical confidence for this demonstration.

All six original standard results remain local, with each output hash bound in
the repeated benchmark. Published repeated artifacts normalize paths and retain
original local definition/benchmark hashes; their output hashes refer to the
original external result bytes, as noted in each trial. They do not refer to the
two earlier normalized standard results. Measurements used the development
worktree based on `c0e74cb`; generator-source SHA256 values identify the modified
code rather than implying it was already present in that base commit.
The repeated HTML was also rendered at
1440 px and 390 px: all nine images and SVGs rendered without page overflow.

Paths in these published artifacts are normalized to `${DATASET_ROOT}` and
`${WORKSPACE_ROOT}`. Numerical values, original input/configuration hashes,
and timing values are unchanged. Standard-result provenance retains each
original local result SHA256 under `source_artifacts.local_result_before_publication`.
The final benchmark's output hashes bind the published result bytes; trial
notes retain the original local output hashes. Its source-artifact map also
retains the original local benchmark hash. Problem, provider, and image hashes
refer to the original external files; they are not redistributed here. The
first definition's output hashes refer to its original local results.

No raw images, scans, depth arrays, checkpoints, or generated HTML are committed.
Reproduce the report with the [public-data demo instructions](../../tutorials/public_six_dof_demo.md).
The local original comparison also passed schema validation. Generated schemas
remain byte-identical to the committed schemas.

This demonstrates execution and numerical equivalence on one public sequence,
not independent calibration accuracy. Dataset calibration is reference
evidence, FeatDepth training overlap with this drive is possible, and result
quality remains `warn` with unknown covariance and observability. No holdout
acceptance threshold or provider tuning was used.
