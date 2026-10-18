# CPU map changes through complete bag calibration

Five later scheduling/search prototypes were [screened and rejected](lidar_execution_screen.md).
The measurements below retain their original runtime and scope.

The reached-normal cache and eight CPU query workers improve complete fresh
bag workers by **1.249x at 0.2 m map voxels** and **1.687x at 0.03 m map voxels**
on this public execution control. All six paired eight-worker processes are
faster. These whole-worker improvements are smaller than the map-evaluation
improvements: decoding, reference odometry, startup and verification remain.

This extends the earlier [KITTI map-only measurements](cpu_map_normal_cache.md)
to both bag streams, common CUDA reference odometry and complete calibration
evidence. It introduces no runtime or calibration-policy change. The two
densities are separate workloads; their ratios and earlier KITTI ratios are
not pooled or multiplied.

## Complete fresh-worker results

| Map voxel size | Previous CPU map, 1 worker | Reached normals, 1 worker | Reached normals, 8 workers | Previous / 8 workers |
| --- | ---: | ---: | ---: | ---: |
| 0.2 m, median | 17.646 s | 15.158 s | 14.132 s | **1.249x** |
| 0.03 m, median | 32.449 s | 24.793 s | 19.234 s | **1.687x** |

Every worker includes process startup, imports, fresh CUDA compilation, both
bag streams, fixed target noise, all 60 reference scans, odometry, map
preparation, training and four jackknife fits, full covariance, held-out scoring,
all six axis controls, saved artifacts, source/input verification and hashes of
prepared arrays. Parent aggregation and source freezing are outside wall time.
Each CUDA worker starts with an independent empty compiler cache; the driver
cache is disabled. One BLAS thread is used and the host is unreserved.

All original observations are retained:

| Density | Triad | Previous CPU map (s) | Reached, 1 worker (s) | Reached, 8 workers (s) |
| --- | --- | ---: | ---: | ---: |
| 0.2 m | 1 | 17.646 | 13.939 | 14.052 |
| 0.2 m | 2 | 18.711 | 17.251 | 15.742 |
| 0.2 m | 3 | 15.264 | 15.158 | 14.132 |
| 0.03 m | 1 | 31.748 | 22.963 | 17.893 |
| 0.03 m | 2 | 32.838 | 24.793 | 20.788 |
| 0.03 m | 3 | 32.449 | 29.316 | 19.234 |

The first coarse eight-worker run is slower than reached normals with one
worker. Its observation is included. Eight workers are an explicit budget,
not a universally faster default; one worker remains the production default.

## Where the time remains

| Density | Stage | Previous CPU map | Reached, 1 worker | Reached, 8 workers |
| --- | --- | ---: | ---: | ---: |
| 0.2 m | Collection: decoding, odometry and map preparation | 8.649 s | 8.273 s | 7.579 s |
| 0.2 m | Complete map evaluation | 2.370 s | 1.281 s | 1.073 s |
| 0.03 m | Collection: decoding, odometry and map preparation | 8.176 s | 7.392 s | 7.499 s |
| 0.03 m | Complete map evaluation | 17.182 s | 10.343 s | 5.706 s |

These are medians of nested API stages. Collection arithmetic is unchanged;
its variation belongs to whole-process time. Fit timing is included in
evaluation and must not be added again. Stage medians do not sum to median
whole time, which additionally includes imports, saves, hashing and verification.
The next large whole-worker improvement must also reduce the remaining front
end or startup work; the map-only ratio cannot supply that improvement.

## Frozen workload and quality

All arms replay six seconds of the spent public GLIM Livox bag. Reference and
target use `/livox/points`; target-only Gaussian noise has sigma 0.01 m and
seed 20261018. This copied-sensor control supplies no independent two-LiDAR
accuracy, temporal validation or SOTA evidence. All saved policy statuses are
`pass`, which does not change that limitation.

Map windows contain eleven reference scans and target sampling retains every
fifth scan. The dense 0.03 m odometry, three-scan odometry map, distance and
planarity gates, Huber loss and iteration limits are common to all arms. Every
arm explicitly uses the same 1 nm odometry geometry grid, 1 micrometre returned
translation grid and 0.1 microradian rotation-vector grid. Map preprocessing
uses the two separately declared voxel sizes; all other options are common.

Normal work falls from 386,587 to 109,365 rows in the coarse case, and from
2,179,563 to 724,383 rows in the dense case. No input point, correspondence
query, covariance or evidence call is removed to achieve this saving.

All 18 workers and 90 complete fits are retained. Sixty paired full fits,
complete artifacts, covariance, every control and all prepared pose/map/target
array identities match every float bit after narrowly declared run metadata
and query-worker options are normalized. The audit also checks executed source,
native CUDA odometry kernel identity, input and compiler-cache bindings,
nonoverlapping timing scopes and every original median. This is execution
agreement, not a new accuracy claim.

## Reproduction

```bash
python -m tools.benchmark_cpu_map_bag \
  --bag data/public/glim_versatile/livox.bag --topic /livox/points \
  --point-time-field t --dataset-family glim_versatile \
  --dataset-license "upstream AIST GLIM dataset terms" \
  --baseline-commit 96871f64588a122e33e01c713e6710f0f0f36d71 \
  --current-commit 917748b10d5eff9bb077b332a018eafdb33e9fcd \
  --odometry-backend cuda --query-workers 8 --repeats 3 \
  --max-seconds 6 --map-voxel-sizes-m 0.2 0.03 \
  --output-dir outputs/new-cpu-map-bag-comparison

python -m tools.audit_cpu_map_bag docs/assets/cpu_map_bag_validation --check
```

Use a new output directory. The reusable driver freezes committed runtime
sources, rotates variant order, alternates density order, validates its report
schemas and retains every worker, including failures. The
[bound archive](../assets/cpu_map_bag_validation/README.md) includes all raw
measurements, complete results and fits, protocols, helpers, logs, source
manifests and a CPU-only audit. Shared prior/lazy source capsules are reused;
the small current delta adds only bag-option provenance. Raw scans and prepared
coordinate arrays are not redistributed.
