# Public KITTI raw six-DoF demonstration evidence

These five schema-valid JSON artifacts retain numerical evidence from
`calibrex demo kitti-six-dof` on public KITTI raw
`2011_09_30_drive_0018_sync/image_02`, original Velodyne scans, and the frozen
FeatDepth provider SHA256
`7e3ef953a360a7c01ed0cdc3cf947db97fb0aff5912ced805d80bebe210d3360`.
The full provider contains 25 uniformly selected frames (3,130,279 points);
every fifth selected frame is holdout, leaving 20 fit / 5 holdout. The declared
initial pose adds local x rotation 0.5° and camera-frame x translation 0.5 m
to the dataset reference. Default search bounds, steps, and 800-evaluation
budget were fixed before executing either backend.

`comparison.definition.json` / `comparison.benchmark.json` describe the final
implementation run. `numpy.result.json` / `cuda.result.json` retain all 229
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
