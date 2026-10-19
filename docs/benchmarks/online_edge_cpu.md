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
one frame's points or image. As for NumPy execution, callers must leave input
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
