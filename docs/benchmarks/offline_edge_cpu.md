# Parallel offline camera–LiDAR preparation

The pooled offline edge estimator can prepare independent image responses and
LiDAR depth discontinuities concurrently. Select a CPU budget through its Python
API; the default is one worker, and accepted budgets are integers from 1 to 32.
This adds no dependency.

```python
from calibrex.evaluation.camera_lidar_edge import (
    EdgeAlignmentOptions,
    estimate_from_frames,
)

result = estimate_from_frames(
    frames, camera, candidate,
    options=EdgeAlignmentOptions(preparation_workers=8),
    provenance=provenance,
)
```

`frames`, `camera`, `candidate` and `provenance` are the caller's synchronized
inputs, camera model, initial transform and input provenance. The same options
can be passed to `run_ros2_camera_lidar_edge`. ROS remains outside the core.
The execution budget is recorded in the artifact's existing `options` mapping;
registered result schemas are unchanged.

Workers compute the original image and depth-edge operations. The caller still
selects frames, rejects repeated views, projects candidate support and reports
progress in original input order. The pending queue is bounded by the worker
count rather than recording length. Retained preparation results scale with
that count and per-frame array sizes; caller-owned input frames and accepted
output frames retain their existing memory costs. Worker and caller failures
propagate, outstanding queued work is cancelled, and running workers are joined.
The single-worker and empty-input paths create no executor.

## Complete public-data comparison

The [complete indexed evidence](../assets/offline_edge_preparation/README.md)
contains three fresh processes per variant, with fixed order
serial/parallel/parallel/serial/serial/parallel and one BLAS thread. Every
observation is retained; the host is unreserved and no warmup is removed.
The predeclared gate is 2x on `estimate_from_frames`, including preparation,
the main fit, profile starts, all jackknife fits and held-out axis controls.

| Scope, median of three processes | One worker | Eight workers | Ratio |
| --- | ---: | ---: | ---: |
| Frame preparation | 5.203 s | 1.341 s | 3.881x |
| Complete estimation API | 6.543 s | 2.522 s | 2.595x |
| Complete fresh measurement worker | 13.822 s | 9.156 s | 1.510x |

The last scope additionally includes Python startup, imports, source and input
verification, PNG decoding and point loading, saves, and exit. The 2.595x ratio
applies to the estimation API, without those additional stages.

All six results preserve every serialized numerical bit after removing only
the explicitly selected worker count from `options`. All 48 prepared frames,
42,506 points, dropped-frame lists and 429 objective evaluations agree. The
complete comparisons include the pose, uncertainties, every held-out control,
reported observability, policy verdict and reasons. Timing wrappers only count
calls and bracket the unchanged preparation and objective functions.

The source is commit `73b155344a85b1caa6e659639be8e6bdf2de6abd`.
The inputs are the previously spent KITTI raw
`2011_09_30_drive_0018_sync` pairs, indices 15 through 485 at stride 10.
The existing acquisition receipt pins all images, scans and calibration files.
Decoded luminance is rounded to uint8. Frame times use `index / 10` for this
execution control, and the official rectified camera-02/Velodyne transform
initializes it. This is performance development without new accuracy or SOTA
evidence. These observations provide no portable latency guarantee and are
separate from the online grid measurements.

An initial independent six-process cohort completed every worker, but its
original Git audit failed because Windows Git archive output used CRLF source
text while Git comparison normalized it. Its original records, raw source,
failure diagnostic and post hoc corrected audit remain in `initial/`. Its API
ratio is 2.147x. The formal cohort above was run separately after correcting the
text comparison and source freezing; the two cohorts are not pooled.

## Reproduction and replay

Commit code and tools before running. Reuse the pinned fixture in place:

```bash
python tools/benchmark_offline_edge_preparation.py \
  --fixture /path/to/existing/kitti-subset --receipt /path/to/fetch-48.json \
  --output outputs/offline-edge-comparison
python tools/benchmark_offline_edge_preparation.py \
  --audit outputs/offline-edge-comparison --verify-git
```

The driver requires 20 GiB free, limits evidence to 64 MiB and saves every
launch, failure and timeout. Raw images and scans are not copied. The CPU-only
auditor validates artifact schemas, every worker file, imported source hashes,
complete quality, preparation hashes, logical work, order and timing boundaries.
Optional Git verification binds source, tools and the result schema to the
original commit. Archived Python is read as evidence, never executed.
