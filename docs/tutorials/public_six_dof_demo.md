# KITTI raw CPU/GPU calibration demo

Run a Camera–LiDAR six-DoF comparison using public KITTI raw data and a frozen
depth-provider artifact. No personal recording or rosbag conversion is needed.
One command verifies inputs, selects 25 frames, estimates on 20, evaluates the
other 5, and writes an offline HTML report with image overlays.

## Prepare public inputs

Get a synced, rectified drive and date-level calibration files from the
[official KITTI raw dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php).
This example uses `2011_09_30_drive_0018_sync`, `image_02`, and original
Velodyne scans. Follow the dataset's download and license terms. Keep this layout:

```text
2011_09_30/
  calib_cam_to_cam.txt
  calib_velo_to_cam.txt
  2011_09_30_drive_0018_sync/
    image_02/data/*.png
    velodyne_points/data/*.bin
    velodyne_points/timestamps.txt
```

Supply an existing `slac.depth_provider/v0.1` manifest with at least 25 frames
from that drive. Image, depth, and checkpoint hashes are verified. This demo
requires a rectified pinhole camera and depth returned to the source image grid:
identical source/output dimensions, unit scales, and zero crop offsets.

To generate a provider once, use the existing external FeatDepth runner from a
checkout and a separate environment with the pinned FeatDepth dependencies:

```bash
PYTHONPATH=src /path/to/featdepth-env/bin/python \
  tools/run_featdepth_provider.py /path/to/2011_09_30_drive_0018_sync \
  --featdepth-repository /path/to/FeatDepth \
  --checkpoint /path/to/fm_depth.pth --frame-count 25 \
  --output-directory /path/to/depth \
  --artifact-output /path/to/featdepth-provider.json
```

See the [Camera–LiDAR input lock](../research/camera_lidar_sota_plan.md) for the
repository/checkpoint identity and reproduction limits. The demo does not
download data, infer new depth, or refine the provider on holdout frames. Raw
data, images, and weights remain external to the repository.

## Run the comparison

Install the [optional CUDA backend](gpu_acceleration.md#install), then run:

```bash
calibrex demo kitti-six-dof \
  /path/to/2011_09_30_drive_0018_sync /path/to/featdepth-provider.json \
  --output-dir outputs/kitti-six-dof-demo --json
```

The candidate defaults to `cuda`; the reference always uses NumPy CPU. Without
CUDA, explicitly choose `--candidate-backend numpy` for a CPU replay, or
`numba_cpu` for the optional CPU accelerator. An unavailable CUDA backend
produces a failed trial, without substituting a CPU candidate.

`--cuda-cache-mib N` requires CUDA and configures its input cache.
`--max-evaluations N` sets the shared search budget (default 800, minimum 13).
`--rotation-offset-deg X Y Z` and `--translation-offset-m X Y Z` declare the
controlled initial perturbation: defaults `(0.5, 0, 0)` degrees and `(0.5, 0, 0)`
metres. Rotation uses local Euler angles; translation is added in the camera
frame. Search bounds are ±2 degrees and ±1 metre per axis around this initial
pose. Record changed settings before comparison; do not tune on holdout results.

Open `outputs/kitti-six-dof-demo/report.html` locally. Images and links are
relative, so move the entire directory together. Use a new or empty output
directory for every run; existing evidence is preserved.

## Read the evidence

- Exactly 25 endpoint-inclusive uniform positions are selected from numeric-
  sorted provider frames. If it contains 25, all are used. Every fifth selected
  frame is holdout (20 fit / 5 holdout), fixed before optimization and histogram
  range selection. This is a sample of the supplied provider, not necessarily
  all frames of the drive.
- Both runs receive the same perturbed initial pose. Dataset calibration creates
  this controlled prior and supplies reference differences; it never becomes
  the optimizer's search target. The prior is not an independent calibration.
- Initial/final reference differences and fit/holdout MI are reported separately.
  Convergence describes optimizer termination. Standard-result quality remains
  `warn`: covariance and observability are unknown.
- Exact numerical parity checks the full search trace, final poses, metrics,
  status, and initial/final holdout evaluations. Backend identities and timing
  remain in individual results. A mismatch is a failed comparison.
- Load/verification, preparation, solve, and holdout times are separate. CUDA
  solve time includes lazy compilation. One invocation per backend is
  descriptive, not a repeated latency benchmark or confidence claim. The ratio
  appears only for a converged matching pair.
- Three prespecified holdout frames show initial and output pose overlays using
  NumPy visibility semantics. At most 1,200 finite depth pairs are sampled for
  display; optimization uses full scans. Colours use a fixed 0–80 m LiDAR range.
- Learned-depth overlap, depth errors, and synchronization affect the objective.
  FeatDepth training overlap with drive 0018 is recorded as possible. Dataset
  calibration is reference evidence rather than independent metrology.

Schema-valid outputs are `source.problem.json`, `demo.problem.json`,
`numpy.result.json`, the candidate result, `comparison.definition.json`, and
`comparison.benchmark.json`. The benchmark binds results, provider, problem,
preview images, and generator sources by SHA256. Standard results retain
effective settings, train/holdout IDs, backend identity, transfer counters,
input hashes, and the full trace. HTML links to all available artifacts.

Exit code `0` requires two converged runs, evaluable holdout, and exact parity.
Code `2` retains partial/failed comparisons and reasons in HTML and JSON. Bad
inputs are rejected before the comparison starts.

## Local public-data evidence

On the original public drive 0018 frames, both backends converged in 229
evaluations with exact numerical parity. Reference rotation difference changed
from 0.5000° to 0.3909°, translation from 0.5000 m to 0.1601 m, and holdout MI
from 1.0857 to 1.2935 nat. The final GTX 1660 Ti invocation took 54.824 s CPU /
6.734 s CUDA for solving, and 60.751 / 15.398 s for the measured pipeline phases.
The earlier invocation's solve-time ratio was 4.58x versus 8.14x for the final
invocation, illustrating timing variation.

[Published schema-valid evidence and limitations](../assets/public_six_dof_demo/README.md)
retain both observations and full numerical traces from the final run. This is
one development drive and perturbation, rather than a multi-dataset recovery
evaluation or a portable speed guarantee.
