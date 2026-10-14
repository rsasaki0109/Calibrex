# Fused CUDA map calibration and evidence

The CUDA LiDAR–LiDAR map backend now combines tree compaction, uncertainty
flags, native repair scatters, source-ordered residual packing, Jacobians and
robust products in one compiled module. It retains the previous FP64 spatial
query, covariance/eigensolve, BLAS reduction and numerical repair policies.
Public APIs and core result/config schemas are unchanged; CPU execution remains
ROS-independent and imports no CuPy.

The improvement targets **first-use compilation cost**. Three alternating
previous/current fresh-process pairs with separate initially empty CuPy caches
produce:

| Scope | Previous median | Fused median | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete fresh worker | 17.288 s | 10.043 s | 1.721x |
| Complete calibration and evidence API | 12.450 s | 5.265 s | 2.365x |
| Nested NVRTC compilation | 9.319 s | 2.459 s | 3.789x |
| NVRTC programs per worker | 63 | 17 | — |

The API includes map preparation, all fits, covariance, holdout, signed controls
and jackknife. Worker timing additionally includes process imports, raw input
reads, hashes, saved artifacts and frozen-import verification. Compilation is
nested; these ratios must not be multiplied. See the
[bound evidence archive](../assets/cuda_map_fusion/README.md).

After each variant's own prime, three complete warm API calls have medians of
1.514 s → 1.430 s (1.059x), with zero new compilations. This fixed-order,
unreserved check is descriptive. The cold improvement is **not** a claim of
2.365x sustained acceleration with an already populated cache.

All three cold evaluation pairs match exactly, including transforms, covariance,
signed controls, holdout, jackknife and policy decisions. Prime/warm checks also
match exactly. Each worker preserves 762,118 map rows and 9,147,116 query rows;
the new path records 276 fused map kernel calls, 8,384,998 residual input rows
and 3,838,813 normal-system rows. No sample, neighbor, iteration or validation
budget is reduced. No implicit grid or neighbor specialization is added.

The input is public KITTI raw sequence `2011_09_30_drive_0018_sync`, scans 115,
230 and 345, used as a copied-sensor identity execution control with fixed
0.01 m noise and generated poses. This evaluates execution and evidence, not
independent two-sensor extrinsic accuracy or reference-odometry generation.
The raw scans are not redistributed.

The shared device tree is also used by gyro-deskewed odometry. A separate public
GLIM quality run retains exact equality for all four complete fitted
trajectories/covariances/costs, holdout and 32 signed controls. It is a quality
check only; its timings are not compared with the earlier saved baseline.

No foreign or unknown compute processes appeared in 11 ten-second observations
of the cold cohort. Sampling does not guarantee continuous GPU exclusivity or
reserve CPU resources. Workers use one BLAS/OMP/MKL thread each. Published
records omit process IDs, process names and hardware models.

Audit all bindings and complete evidence without a GPU or raw scans:

```bash
python -m tools.audit_cuda_map_pipeline docs/assets/cuda_map_fusion
```

The measured runtime commit is `211f112`; the prior runtime is `1f45f4f`.
Frozen launchers, sources, schemas, device parity tests and measurements are
included in the archive.
