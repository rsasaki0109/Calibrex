# Bounded CUDA candidate batches on public KITTI

The new CUDA v0.4 implementation reduces complete six-DoF **solve time by
2.18–2.47x against the already accelerated CUDA v0.3 implementation** on the
six frozen signed-axis problems below. This compares implementations of the
same GPU execution path. Every one of the 36 timed results matches its frozen
NumPy oracle exactly, including all candidate scores, search traces, poses,
skipped frames, evaluation counts and initial/final holdout records.

The measured implementation is commit
`feaffa0ac283d4312fb2f9e2c031e86303272d42`; the reference is
`2017b3a3bb2742430170bff8ef842d7be9e5d9a7`. The three archived `baseline.*.txt`
files are exact Git blobs of the reference evaluator, projector and solver.
The benchmark binds measured Windows working-tree SHA256 values separately
from canonical Git blob SHA256 values. The measured code was frozen during
both final timing runs; subsequent documentation and GPU-label anonymization
do not alter the solver.

## Workload and timing contract

Inputs are public KITTI raw `2011_09_30_drive_0018_sync/image_02`, original
Velodyne scans, and the existing frozen FeatDepth provider SHA256
`7e3ef953a360a7c01ed0cdc3cf947db97fb0aff5912ced805d80bebe210d3360`.
These are the same inputs as the [six-axis sweep](../public_six_dof_sweep/README.md),
including 25 frames, 1226 x 370 image grids, and 3,130,279 points. Twenty fit
frames contain 2,503,637 points. Five predetermined holdout frames never enter
optimization or histogram-range selection. Each prior adds +/-0.5 degrees of
local rotation and +/-0.5 m of camera-frame translation along the named axis.
Bounds remain +/-2 degrees / +/-1 m, with the original steps and 800-evaluation
cap. No point subsampling, lower precision, or smaller search budget is used.

Each case runs each method three times, alternating method order. Every
invocation constructs a fresh CUDA backend and solver. There is no separate
warmup; existing compilation caches, the process runtime and OS file caches
persist. The Windows host is shared with other user work. Environment:
i7-9750H, NVIDIA CUDA GPU with 6 GiB device memory, Python 3.12.10,
NumPy 2.3.4, CuPy 14.2.0, CUDA runtime 12.9, NVRTC 12.8. Device model names
are anonymized in public metadata; original raw-result digests remain intact.
Three repetitions on one sequence do not establish statistical confidence
or a portable speed guarantee. Bootstrap fields are schema summaries only.

`solve_seconds` includes conservative GPU filtering, compact input gathering,
scratch preparation, all objective evaluations and optimization. Pipeline
time sums input handling, CUDA backend preparation, solve, and holdout/shutdown
phases. Serialization, benchmark aggregation and report rendering are outside
these recorded phases.

The six-case run uses an explicitly verified input snapshot per case. Capture
and verification take **2.460–2.773 s per case**, charged once before repeated
use; capture is excluded from the per-invocation times below. Each invocation
reverifies the problem artifact, and uses the captured external input bytes.
Adding capture to a single invocation changes its total speedup. A snapshot
is reused only at the caller's explicit request; ordinary CLI estimates still
verify external files on each invocation.

| Case | Evaluations | Existing CUDA solve (s) | Candidate CUDA solve (s) | Ratio |
| --- | ---: | ---: | ---: | ---: |
| x-positive | 229 | 1.225680 | 0.533842 | 2.30x |
| x-negative | 313 | 1.671952 | 0.740060 | 2.26x |
| y-positive | 277 | 1.515732 | 0.639080 | 2.37x |
| y-negative | 241 | 1.321933 | 0.605046 | 2.18x |
| z-positive | 265 | 1.582589 | 0.639499 | 2.47x |
| z-negative | 241 | 1.301664 | 0.560588 | 2.32x |

All entries are three-invocation medians. Ratios divide the two medians.

| Case | Existing CUDA pipeline (s) | Candidate CUDA pipeline (s) | Ratio | Separate capture (s) |
| --- | ---: | ---: | ---: | ---: |
| x-positive | 1.384056 | 0.647821 | 2.14x | 2.713804 |
| x-negative | 1.777530 | 0.849358 | 2.09x | 2.460149 |
| y-positive | 1.616103 | 0.741295 | 2.18x | 2.657702 |
| y-negative | 1.429702 | 0.722039 | 1.98x | 2.517882 |
| z-positive | 1.708915 | 0.754156 | 2.27x | 2.772603 |
| z-negative | 1.418651 | 0.673354 | 2.11x | 2.727566 |

A separate three-repetition x-positive run uses **fresh external-file
verification/loading every invocation**, with no input snapshot. Solve medians
are **1.414336 / 0.728821 s (1.94x)** for existing/candidate CUDA. Pipeline
medians are **4.007975 / 3.421207 s (1.17x)**. All six complete results also
match the original NumPy oracle. Loading and verification remain the main
cost of this short fresh-file workload. Do not use the snapshot pipeline
ratios as fresh-file or end-to-end depth-generation speedups.

## Where the work is removed

Up to four independent candidates share launches and a histogram download
across up to 32 fit frames. Candidate order, winner ties and the original
search decision sequence remain intact. Positive histogram counts are grouped
for NumPy entropy reductions without padding zeros into the original reduction
order. Each candidate slot retains private scratch; a separate reset clears
only pixels hit by that slot's previous projection. Sparse visibility and
histogram kernels visit points rather than the entire image. Warp aggregation
combines identical integer histogram increments where supported.

For pinhole frames, one GPU mask conservatively excludes only points that
cannot enter the half-pixel image frustum anywhere within the declared search
bounds. Retained points keep input order and original LiDAR ranges. Across
the six cases, **785,490–873,746 of 2,503,637 fit points** remain. Other camera
models use full clouds. Different frames and poses outside the prepared bounds
use the ordinary full-input path.

In x-positive, histogram download calls fall from **231 to 60** while bytes
remain exactly **37,674,720**. No histogram samples are discarded. Ordinary
cached inputs occupy 190,892,928 bytes; compact additional inputs occupy
25,135,680–27,959,872 bytes across cases, within the shared 256 MiB input
cache budget. Private scratch peaks at 630,906,240–636,554,624 bytes, below
the separate 1 GiB limit. CuPy allocation-pool reservations, temporary filter
arrays, outputs and other GPU users are additional. Insufficient cache,
scratch capacity, allocation failure, or more than 32 fit frames falls back
to ordered CUDA frame batches. Runtime failures still report an error.

## Evidence and reproduction

`sweep.definition.json` retains all 36 timed trials; `fresh.definition.json`
retains all six fresh-file trials. Corresponding benchmark JSONs are derived
from these definitions. Separately charged snapshot captures and original
timing-summary digests are recorded in their provenance/limitations.
`superseded-*` definitions and aggregates preserve earlier measurements before
the caller-device-context restoration fix; they are not the final timings.
Every attempt is retained, including the earlier baseline timing outlier.

The published full results are repetition 1, chosen before inspecting timings:
two methods per case, six historical NumPy oracles, and two fresh-file results.
The six problems are included as metadata; inputs and weights remain external.
All 34 public JSONs validate against existing schemas. Published bytes have
separate SHA256 bindings from original raw inputs, results and sources. Paths
and GPU model labels are normalized; complete numerical payloads are unchanged.
`publication-launcher.py.txt` archives the publication/check procedure. Its
`CALIBREX_PUBLIC_SWEEP_ROOT` environment variable points to the local six-case
input directory.

Normal estimates automatically use this path:

```bash
calibrex camera-lidar estimate-six-dof problem.json \
  --projection-backend cuda --frame-workers 1 --output result.json
```

After preparing the frozen six-case inputs, reproduce the paired measurement:

```bash
python tools/benchmark_d2d_cuda_candidates.py \
  --sweep-root outputs/public-kitti-six-dof-sweep-cuda-v1 \
  --output-dir outputs/cuda-candidates --repeats 3
```

Use `--input-mode fresh_files` for repeated external-file verification. To
measure one case, use `--problem`, `--oracle-result` and `--protocol-template`
instead of `--sweep-root`. The reference code is loaded from its pinned Git
commit, so that commit must be available locally. See the
[GPU tutorial](../../tutorials/gpu_acceleration.md#bounded-cuda-candidate-batches)
for implementation limits and memory controls.

Exact parity preserves the existing calibration outcome, including the
y-positive rotation error of about 0.527 degrees from a 0.5-degree prior error.
It does not establish independent accuracy. Quality stays `warn`; covariance
and observability are unknown, scans are not motion compensated, and learned
disparity is not metric depth. FeatDepth's KITTI training overlap is possible.
