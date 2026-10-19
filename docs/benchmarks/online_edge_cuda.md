# CUDA online camera–LiDAR edge evaluation

The Levinson–Thrun online solver accepts an explicit CUDA backend for its
729-pose rolling grid. Image edge preparation also accepts CUDA. NumPy remains
the default, and default solver imports and execution do not load CuPy.

```python
from calibrex.solvers.levinson_thrun_online_solver import (
    LevinsonThrunOnlineSolver,
    levinson_edge_response,
)

edge = levinson_edge_response(luminance, evaluation_backend="cuda")
# Construct synchronized LevinsonOnlineFrame objects with these edge images.
result = LevinsonThrunOnlineSolver(evaluation_backend="cuda").solve(
    frames, initial_transform, options
)
execution = result.as_dict()["execution"]
```

Install the optional `cuda` extra to use this path. The native A2D2 adapter
accepts `evaluation_backend: cuda` in its existing
`lidar_camera_depth_edge_alignment` factor options; it applies the selection to
both edge preparation and grid evaluation. A2D2's fisheye grids retain native
evaluation. Its two-frame sample remains inconclusive, with no independent
extrinsic accuracy or deployment claim.

`prepare_levinson_online_frames(items, loader, workers=3, evaluation_backend="cuda")`
accepts a loader returning `LevinsonOnlineImage` objects. Independent CPU loads
run concurrently while CUDA initializes on the caller; subsequent CUDA work
stays on that caller. Output frame order is preserved. The A2D2 factor exposes
this through `preparation_workers`, defaulting to one. Loaders must support
concurrent calls when more than one worker is requested.

## Computation and decisions

The GPU keeps edge images, point coordinates and weights resident across
rolling windows. A single kernel transforms points, projects them, interpolates
the edge response, and reduces each candidate's raw score. Only a raw score,
an absolute error enclosure and a boundary flag return for each frame/pose.
There is no transformed-point or per-point-score download.

FP64 arithmetic has fused multiply-add disabled. The GPU reduction can differ
from native NumPy arithmetic. A conservative enclosure accounts for coordinate,
projection, interpolation and reduction errors. Coordinate slack starts at
`2**-40 * (1 + 4*sum(abs(point)) + sum(abs(translation)))`; projected-coordinate
bounds and the image's maximum response bound interpolation changes. Depth and
outer-image boundary ambiguities trigger native evaluation.
An absolute error floor also covers subnormal score arithmetic.
Projection shares one rounded FP64 reciprocal per point. The enclosure uses
an upper bound on inverse depth; ill-conditioned depths re-enter native scoring.

Every interval that can change the strict comparison with the native center
is resolved on CPU. Every possible winner is also evaluated natively. The
original candidate order, first-index ties, worsening count, selected pose and
update decision are preserved. Centers, final train/holdout evaluations,
known-bad controls and numerical curvature retain native evaluation. Tests
cover boundary points, near ties, constant/zero images, both camera axes,
strided inputs, tiny grid steps and drift sequences.

Image preparation performs the same FP64 neighbor differences, bounded
spilloff steps and final weighted sum on GPU, retaining native float bits in
the tested cases. It downloads the final image once. This selection is
independent of the solver constructor when using the Python API.

The shared CPU beam assignment uses binary search over sorted beam centers
instead of allocating a point-by-beam distance matrix. Adjacent distances and
first-index ties retain native arithmetic. Unusual center spacing, order or
numeric ranges retain dense `argmin`.

## Ownership and provenance

Each CUDA solve captures private copies of its inputs. Later mutations of
caller arrays cannot change the captured run. The retained device snapshot
cache is limited to 64 MiB; owned grid buffers have a conservative 128 MiB
budget. Image preparation separately bounds its six image buffers to 128 MiB.
These limits cover owned arrays, rather than the CUDA context or CuPy allocator
pool. Private host copies scale with input size.

Tiny inputs, fisheye projection, extreme numeric ranges and device-buffer
limits use native scoring and report the resulting native frame/pose counts.
Explicit CUDA requests report unavailable dependencies/devices and execution
errors. They do not silently replace a failing runtime.

`execution` records software versions, the grid-kernel SHA256, a SHA256 over
all captured frames, policy, native ambiguity/winner checks, uploads,
readbacks and owned-buffer estimates. These counters describe grid evaluation.
The benchmark records the image-kernel SHA256 separately. CPU result
serialization and the registered config/result schemas remain unchanged.

## Complete-worker measurement

```bash
python tools/benchmark_online_edge_cuda.py \
  --fixture /path/to/kitti-subset \
  --receipt /path/to/fetch-48.json \
  --output outputs/online-edge-comparison
python tools/benchmark_online_edge_cuda.py \
  --audit outputs/online-edge-comparison --verify-git
```

The fixed input is 48 image/cloud pairs from the already-spent KITTI raw
`2011_09_30_drive_0018_sync` recording, indices 15 through 485 at stride 10.
The subset contains those 96 files, both full timestamp files and three official
calibration files. The receipt pins the source URL, archive size, ETag and
every file's SHA256. Raw images and scans remain outside the evidence output.
Image downsampling is four, spilloff radius is 32, and up to 2,000 LiDAR
discontinuity points are retained per frame. The default nine-frame window
produces 30 rolling steps with 38 train and 10 shadow-holdout frames.
The dataset calibration initializes the execution control; it supplies no new
accuracy evidence or SOTA evaluation.

Commit source and tools before measuring. The driver freezes the committed
Python source tree and toolchain, then runs NumPy, fused CPU, CUDA, CUDA,
fused CPU, NumPy in fresh processes. The compiled CPU control fuses pinhole
projection/interpolation while retaining native transforms and reductions.
Its noncenter candidates compute raw scores directly, omitting normalization
and support statistics that cannot affect grid decisions.
Both CPU controls receive the same beam-search improvement as CUDA.
All variants use three preparation workers. CPU edge preparation runs within
those workers; CUDA initialization overlaps CPU loading and CUDA edge
preparation runs on the caller thread.

Whole-worker time includes imports, fixture hashing/loading, preparation,
first-use compilation, all tracking and final diagnostics, result saving and
artifact validation. BLAS uses one thread. CUDA workers use separate empty
compilation caches with the driver cache disabled. The host is unreserved.
All attempts are retained, without warmup removal or automatic retries.

The gate requires at least 2x whole-worker median improvement against both
CPU controls. All prepared frame bits and every complete result field must
match. Only separately validated execution metadata is removed for result
comparison. The CPU-only audit recomputes ratios and validates result/config
schemas, source/import hashes, all outputs and execution contracts. Optional
Git verification checks both source and tools against the frozen commit.
Measurements from earlier development screens are kept separate.

For local development, `--working-copy` freezes the current source and tools
with normalized Python line endings and complete SHA256 manifests. Such
records explicitly use `source_kind: working-copy`, `source_commit: null`
and a separate base Git revision. They support the same content, parity and
timing audit; `--verify-git` rejects them. This mode does not create a commit or
claim a committed-source measurement.

## Measured working-copy cohort

The [indexed complete evidence](../assets/online_edge_cuda/README.md) passes
both 2x gates in six counterbalanced fresh workers:

| Implementation | Whole-worker median |
| --- | ---: |
| NumPy, three preparation workers | 62.852 s |
| Compiled raw-score CPU, three preparation workers | 31.042 s |
| CUDA with overlapped initialization/preparation | 14.036 s |

This is 4.478x against NumPy and 2.212x against the compiled CPU control.
All prepared-frame bits and complete traces, poses, probabilities, controls
and curvature match. Both CUDA workers retain all 21,840 noncenter poses
across 30 windows, with 270 projection/reduction launches and 4,717,440
score-readback bytes. Native ambiguity/finalist checks remain enabled.
The source is explicitly recorded as an uncommitted content snapshot.
