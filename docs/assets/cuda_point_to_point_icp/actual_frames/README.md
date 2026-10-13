# Actual-frame LiDAR ICP CUDA evaluation

These cohorts use distinct consecutive Velodyne frames from the public
[KITTI Raw dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php),
sequence 2011_09_30_drive_0018_sync. No copied or synthetic target cloud is
used. Source scans contain roughly 125,000 points. Official OXTS navigation
and the daily IMU-to-Velodyne calibration provide an initialization, followed
by a declared 3 cm/2 cm/1.5 cm translation and small rotation perturbation.
The fixture retains exact raw-file, navigation, calibration and source hashes.

The primary comparison uses default 50-iteration ICP with complete spatial
holdout, local rematching and six initial-value diagnostics. Breadth and small
cohorts use a predefined 20-iteration cap. Both methods retain automatic bounded
diagnostic parallelism; all workers set and verify one internal BLAS thread.
The primary has three alternating pairs; other cohorts have one smoke pair.

| Cohort | Iteration cap | Native whole median | CUDA whole median | Ratio | Status |
|---|---:|---:|---:|---:|---|
| pilot | 20 | 63.468311 s | 29.026270 s | 2.187x | max_iterations |
| primary-115 | 50 | 55.983115 s | 25.303228 s | 2.212x | converged |
| breadth-230 | 20 | 29.587284 s | 15.634474 s | 1.892x | max_iterations |
| breadth-345 | 20 | 28.698942 s | 15.961634 s | 1.798x | max_iterations |
| small-115 | 20 | 2.798039 s | 3.527018 s | 0.793x | max_iterations |

The primary whole-worker ratio is **2.212x**; complete-solver ratio is **2.390x**.
All primary solves converge. Complete native result bytes match exactly across
CPU/CUDA and repetitions: pose, history, splits, correspondences, holdout,
rematching diagnostics, six start trials and symmetry flags. The other cohorts
also retain exact complete results within each cohort. All fourteen completed
workers are saved, including the 20-iteration pilot that reached its cap.

| Cohort | Pair | Order | Native whole wall | CUDA whole wall |
|---|---:|---|---:|---:|
| pilot | 1 | native, cuda | 63.468311 s | 29.026270 s |
| primary-115 | 1 | native, cuda | 55.983115 s | 25.303228 s |
| primary-115 | 2 | cuda, native | 55.384290 s | 25.332281 s |
| primary-115 | 3 | native, cuda | 57.209044 s | 25.228562 s |
| breadth-230 | 1 | native, cuda | 29.587284 s | 15.634474 s |
| breadth-345 | 1 | native, cuda | 28.698942 s | 15.961634 s |
| small-115 | 1 | native, cuda | 2.798039 s | 3.527018 s |

| Cohort | Initial holdout RMSE | Final holdout RMSE | Navigation translation difference |
|---|---:|---:|---:|
| pilot | 0.153941 m | 0.142109 m | 0.032132 m |
| primary-115 | 0.153941 m | 0.142109 m | 0.032134 m |
| breadth-230 | 0.147279 m | 0.142171 m | 0.009685 m |
| breadth-345 | 0.180983 m | 0.176710 m | 0.027798 m |
| small-115 | 1.331548 m | 1.135372 m | 0.439500 m |

Holdout scores are the native fresh-target-rematching scores on the spatial
source split. The navigation pose is used for initialization and consistency
checking; these raw rotating scans are not deskewed. Navigation differences
are not independently established calibration accuracy. This evaluates
registration between time-separated frames of the same LiDAR.

The 768-point cohort launches zero GPU spatial queries and uses native fallback.
Cold CUDA import/initialization makes its whole worker slower. Use the native
solver for small clouds. Breadth smoke ratios are local observations and are
not median-confidence estimates; their iteration caps remain explicitly reported.

Every CUDA worker starts with a fresh empty disk cache. Whole time includes
startup, input verification, point preparation, solve, all diagnostics, source
guards, initial holdout evaluation and saves. Solver time includes cold CUDA
import, compilation, transfers and native diagnostics, but excludes input
preparation and the additional initial-holdout evaluation. No cold or slow
trial is discarded. OS caches persist. Cohort speed ratios are never pooled
or multiplied.

The protocol was saved before formal execution. The primary uses source commit
`0a30e10021124f52e7afcb44e5b5cf00ba8475c7`. All measured files and fixtures remained
unchanged throughout each worker. Source snapshots preserve measured bytes and
normalized Git identity. Identical result bytes are stored once per digest,
with a reference from every worker. Only local path metadata in the acquisition
manifest was made portable. Raw pointcloud files remain external public inputs.

Reproduce with the existing official range downloader and optional CUDA extra:

```shell
python -m pip install -e ".[cuda]" threadpoolctl
python -m tools.download_kitti_raw_lidar_windows 2011_09_30_drive_0018_sync --center-frame-ids 115,230,345 --neighbor-radius 1 --output-root data/public/kitti_icp --manifest-output outputs/kitti-selection.json
# Put the official daily calib_imu_to_velo.txt in data/public/kitti_icp.
python -m tools.prepare_kitti_icp_pairs --data-root data/public/kitti_icp --calibration data/public/kitti_icp/calib_imu_to_velo.txt --source-frames 115 --max-iterations 50 --output outputs/kitti-pairs
python tools/benchmark_cuda_point_to_point_icp.py --pair-fixture outputs/kitti-pairs/0000000115-0000000116.json --data-root data/public/kitti_icp --output outputs/kitti-real-icp --repeats 3
python tools/audit_actual_icp_archive.py
```

The launcher applies its thread budget before Python/NumPy starts and records
actual loaded BLAS thread counts. It does not change settings in an existing
application process. The earlier failures with unbounded nested threading
remain documented in the parent archive; their cause remains unresolved.
