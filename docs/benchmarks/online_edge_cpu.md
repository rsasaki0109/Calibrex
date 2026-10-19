# Compiled CPU online camera–LiDAR evaluation

The online tracker accepts `evaluation_backend="numba"` for CPU-only execution.
Install the optional `numba` extra. NumPy remains the default and does not import
Numba or CuPy. Explicit accelerator requests report dependency and execution
errors.

```python
from calibrex.solvers.levinson_thrun_online_solver import LevinsonThrunOnlineSolver
from calibrex.solvers.levinson_online_preparation import prepare_levinson_online_frames

frames = prepare_levinson_online_frames(
    items, loader, workers=3, evaluation_backend="numba"
)
result = LevinsonThrunOnlineSolver(evaluation_backend="numba").solve(
    frames, initial_transform, options
)
```

The existing A2D2 factor also accepts `evaluation_backend: numba` and
`preparation_workers: 3`. Its fisheye projection retains native scoring and the
two-frame public sample remains inconclusive.

## Arithmetic and scope

Pinhole interpolation runs in an ordered FP64 loop, without fast math. Native
NumPy transforms, dot products, frame sums and support reductions retain their
original order. Noncenter grid candidates omit only unused support and
normalization statistics. Centers, final train/holdout, signed controls and
curvature retain complete evaluation. Unsupported projections use the native
projector.

Image spilloff uses horizontal and vertical maximum filters instead of stacking
nine shifted images. Every radius still applies exactly one multiplication by
gamma before updating the spread. It retains the native FP64 operation sequence.
NumPy image operations release the GIL, allowing independent preparation workers
to compute concurrently while preserving frame order. This image path needs no
image JIT compilation. Each objective also prepares its rotation and translation
once, reusing those exact values across frames.

No global solver functions are patched. Compiled code uses `cache=False`, and
the runtime retains no input/result array cache. Temporary arrays scale with
one frame's points or image; grid buffers have a separate bound below. As for
NumPy execution, callers must leave input
arrays unchanged during a synchronous solve. Optional execution metadata records
software versions, kernel identity and all grid/full evaluation counts. Default
CPU serialization and registered schemas retain their existing definitions.

## Complete-worker comparison

```bash
python tools/benchmark_online_edge_cpu.py \
  --fixture /path/to/existing/kitti-subset \
  --receipt /path/to/fetch-48.json \
  --output outputs/online-edge-cpu-comparison
python tools/benchmark_online_edge_cpu.py \
  --audit outputs/online-edge-cpu-comparison --verify-git
```

Commit source and tools before measuring. The driver freezes those bytes and
launches six fresh workers in NumPy, compiled benchmark control, production
Numba, Numba, compiled control, NumPy order. Both compiled controls retain native
transforms and dot products; the earlier benchmark control uses NumPy image
preparation. All variants use the same three-worker preparation budget, all
48 already-spent real KITTI image/cloud pairs, all 30 rolling windows, holdout,
controls and curvature. The dataset calibration initializes an execution control
and supplies no new accuracy evidence.

Timing includes imports, input verification/loading, first compilation,
preparation, all calibration/diagnostics, saves and schema/source validation.
BLAS uses one thread; the host is unreserved. No warmup is removed and all
attempts are retained. The adoption gate is 2x against the production NumPy path;
the ratio against the stronger compiled benchmark control is reported separately.
All prepared-frame bits and complete quality results must match exactly.

Raw input files are read in place, with no download or dataset copy. A run needs
20 GiB of free space before launching and limits its evidence folder to 64 MiB.
It freezes one source tree shared by the workers. CPU-only audits validate
schemas, hashes, full work and results without importing optional accelerators
or executing archived Python.

## Maintained CPU backend measurement

Six fresh workers from committed source
`bb90e12f2fa2b811a10ad92dfe222aaffef7831f` measured:

| Implementation | Whole-worker median |
| --- | ---: |
| Production NumPy, three preparation workers | 51.174 s |
| Compiled benchmark control, three preparation workers | 23.452 s |
| Production Numba, three preparation workers | 23.724 s |

The maintained CPU path is **2.157x against production NumPy** and **0.989x
against the compiled benchmark control**, reaching that control's performance
level through the maintained Python and A2D2 APIs. All prepared-frame bits and
all six complete quality payloads match. Native default serialization is
unchanged. The six observations come from an unreserved host, with two per
implementation, and establish no portable latency guarantee.

The [indexed archives](../assets/online_edge_cpu/README.md) retain all six
workers, full results/configs, source and tool snapshots, schemas, receipt and
logs. The first image-JIT cohort remains separate: its 55.588 s NumPy, 26.107 s
compiled-control and 26.036 s production medians yielded 2.135x and 1.003x.
These cohorts are neither pooled nor multiplied. The selected runtime avoids
image JIT compilation and shares per-objective pose preparation; the measurements
support maintained CPU-backend adoption at the compiled control's performance
level.

## Rotation sharing in the maintained CPU grid

Runtime `v0.2` shares bit-identical native rotation matrices within each raw
candidate call. A regular 728-candidate grid needs 27 rotated point clouds per
frame. Ordered chunks of at most 64 poses interpolate in up to four GIL-releasing
workers, capped by the reported CPU count. Each candidate retains the native
NumPy dot product and each frame is added in its original order. Full statistics,
centers, final train/holdout, controls and curvature use the same compiled kernel
with their existing native transforms and reductions.

Numeric grid buffers are limited to 64 MiB, including rotated points, one native
matrix-product temporary, pose metadata and all active interpolation chunks.
Buffers are released before preparing the next frame. This is a buffer bound,
not a limit on process memory or JIT compiler memory. Oversized frames and
unsupported projections use single-pose evaluation. Calls with fewer than 64
or more than 729 poses, or with entirely distinct rotations, also retain the
single-pose path. Worker exceptions propagate and all workers are shut down.
Execution counts continue to record every logically evaluated point and pose.

The [complete indexed evidence](../assets/online_grid_reuse/README.md) compares
the maintained Numba implementation at `b7d6203fa3024a78b6693e7536329b75d2b1dc6e`
with `c2e694e106ebdfec290538b7fd018f45c17ead7f`. Both use the typed PNG array reader,
the same preparation, all 48 existing KITTI pairs, all 30 updates, 21,958
objectives, 395,476,000 logical point rows, holdout controls and curvature.

| Timing scope | Previous CPU | Shared rotations | Ratio |
| --- | ---: | ---: | ---: |
| Complete `solve` API, including first projection JIT and diagnostics | 22.019 s | 5.965 s | **3.692x** |
| Per-worker median of all 30 online updates | 0.681927 s | 0.126579 s | **5.387x** |
| First online update, including projection JIT | 2.420839 s | 2.167978 s | 1.117x |
| Sum of all 30 online updates | 21.913 s | 5.836 s | 3.755x |
| Complete fresh worker, including imports, preparation, saves and validation | 32.604 s | 16.220 s | 2.010x |

Each cell is the median of three fresh workers. Update medians retain the first
update; no warmup is removed. The first update benefits less because compilation
dominates it. Timing wrappers only bracket the original functions. All six
complete quality payloads match as canonical JSON bytes, including signed zeros,
and all prepared-frame snapshots match. The host was unreserved, with one BLAS
thread. These are performance diagnostics on already-spent data, without a new
accuracy claim or portable latency guarantee.

The predeclared gate was 2x for the complete `solve` API against the maintained
CPU control. The cold-worker result is reported separately. Earlier experimental
cohorts keep their original verdicts; their results are not pooled with this
cohort or multiplied by these ratios.

```bash
python tools/benchmark_online_grid_reuse.py \
  --baseline b7d6203fa3024a78b6693e7536329b75d2b1dc6e \
  --fixture /path/to/existing/kitti-subset --receipt /path/to/fetch-48.json \
  --output outputs/online-grid-reuse-comparison
python tools/benchmark_online_grid_reuse.py \
  --audit outputs/online-grid-reuse-comparison --verify-git
python tools/audit_online_grid_reuse_archive.py \
  docs/assets/online_grid_reuse/complete-evidence.tar.gz --verify-git
```

Commit source and timing tools before running. The driver freezes both maintained
source trees and one common toolchain, fixes baseline/candidate/candidate/baseline/
baseline/candidate order, and retains every attempt and log. It reuses inputs in
place, requires 20 GiB free, and limits evidence to 64 MiB. The archive auditor
checks safe paths, schemas, every indexed byte, source identities, complete work,
timing boundaries and parity without executing archived Python or importing
Calibrex, Numba or CuPy.
