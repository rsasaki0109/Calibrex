# Direct previous-CUDA check

A separate, source-bound previous/current CUDA pair ran the same complete
six-second workload and numerical policy. Previous CUDA took 154.593 s;
current CUDA took 132.839 s (1.164x).
Both fresh workers started with isolated empty CuPy compilation caches.
All result and full per-round quality checks pass. This single pair is a
preliminary incremental observation, without repeated-trial stability claims;
it is not pooled with the three CPU/CUDA pairs or multiplied by their ratio.
Host background load was not reserved.

The baseline source is `1edc540881be9740db552b729e0e3d27cc0e4b1e` and current source is
`86d9b3d9aa4416cc2e6534a1d14971964e2df1b2`. Complete workers, sources, imports, kernel
identities and the launcher are bound by the local publication index. The
parent archive audit validates this pair as well.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_spatial_pruning/previous_cuda
```
