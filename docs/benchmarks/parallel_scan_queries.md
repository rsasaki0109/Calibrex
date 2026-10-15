# Parallel CPU scan queries

`ScanOdometryOptions(query_workers=8)` runs independent nearest-neighbor query
rows on eight CPU workers. This covers scan correspondence searches and both
complete and lazy target-normal neighborhoods. SciPy retains the same query
arithmetic, tie ordering and distance gates for each row. Normal estimation,
robust systems, motion updates and convergence checks retain their existing
implementation. The default is one query worker.

The joint bag CLI records the requested budget in its command provenance and
serialized odometry options:

```bash
calibrex imu-lidar joint data/public/glim_versatile/livox.bag \
  --lidar-topic /livox/points --imu-topic /livox/imu --acceleration-unit g \
  --dataset-family glim_versatile --dataset-license "upstream AIST GLIM dataset terms" \
  --odometry-backend cpu --odometry-query-workers 8 \
  --max-seconds 6 --max-rounds 4 --output outputs/joint-parallel-cpu.json
```

Choose a worker budget for the available CPU. Extra threads are most useful for
dense scans; small query batches can spend more time starting threads. This
option controls native CPU searches and adds no dependency to the default core.

## Complete public-data pilot

| Scope | One CPU query worker | Eight CPU query workers | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 313.401 s | 178.759 s | 1.753x |
| Four fits, including covariance | 18.110 s | 17.913 s | 1.011x |

One fresh-process pair uses the same frozen sources with one versus eight CPU
query workers. This is a preliminary fixed-order observation on an unreserved
host. It does not establish a stable ratio or compare CPU threading with GPU
execution. CPU fitting is unchanged; its timing variation belongs to whole time.
Every initial/gyro/feedback pass and the final odometry segment are included.

The [bound archive](../assets/cpu_parallel_scan_queries/README.md) retains the full
results, every fitted trajectory/covariance, original measurements and logs,
input hash, both source capsules/manifests and the exact comparison launcher.
Its CPU-only audit verifies physical results, decisions, common geometry,
declared worker counts, executed source bytes and recomputed medians. The public
clip supplies no independent extrinsic truth; its final verdict is INCONCLUSIVE.

Both workers use one BLAS thread, 0.03 m voxels, three local-map scans, a 1 nm
coordinate grid and returned-motion grids of 1 micrometre translation and
0.1 microradian rotation-vector components. The coordinate and motion grids are
explicit benchmark settings and remain disabled by default.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cpu-cpu \
  --prior-query-workers 1 --current-query-workers 8 \
  --max-seconds 6 --max-rounds 4 --repeats 1 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-cpu-query-workers
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cpu_parallel_scan_queries
```

Use a new output directory and committed runtime sources. Both worker variants
capture the same working-tree bytes, including their original line endings.
