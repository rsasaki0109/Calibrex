# Complete public GLIM spatial-pruning comparison

Three alternating CPU/CUDA pairs run the same committed source revision
`86d9b3d9aa4416cc2e6534a1d14971964e2df1b2`. The original public Livox bag SHA-256 is
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw data is not redistributed.

| Scope | CPU median | CUDA median | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 293.774 s | 118.996 s | 2.469x |
| Four CPU fits, including covariance | 19.980 s | 20.945 s | 0.954x |

| Pair | Method | Whole seconds | Fit seconds |
| --- | --- | ---: | ---: |
| 0 | CPU | 322.236859 | 19.979838 |
| 0 | CUDA | 117.227281 | 18.501644 |
| 1 | CUDA | 118.995790 | 20.945206 |
| 1 | CPU | 293.774264 | 18.564928 |
| 2 | CPU | 256.721956 | 24.392535 |
| 2 | CUDA | 146.503050 | 22.532274 |

All three comparisons pass: maximum rotation difference 0 deg, lever-arm difference 0 m and clock difference 0 s. Per-round covariance, trajectories, costs and all evidence decisions also pass the existing unchanged audit tolerances.

The first CUDA worker records 8 odometry passes,
112,029,956 GPU query rows, 4,777
fused systems, 448 validated bulk captures and
0 public-node fallbacks. Native spatial,
normal, stopping and motion-grid repairs are retained and counted.

The first six seconds feed four map rounds and the full evaluation pipeline.
Both methods use one BLAS thread and the same explicit dense policy: 0.03 m
voxels, three local-map scans, a 1 nm coordinate grid and motion grids of
1 micrometre translation and 0.1 microradian rotation-vector components.
Whole timing includes imports, decoding, GPU setup and compilation, every
odometry pass, association, fits/covariance, evidence, hashes, saves and
verified imports. Each GPU worker starts with an isolated empty CuPy
compilation cache; driver/runtime caches may persist. No slow trial is excluded.

All six results remain **INCONCLUSIVE**. This clip has no supplied independent
extrinsic truth. The CPU fitter is unchanged; its timing ratio is not a GPU
fitting speedup. The whole-worker ratio applies to this declared dense workload.
CPU remains the default. Earlier cohort ratios are not multiplied.

The publication index binds every file except itself. Complete results and
all fit arrays, exact source archives/manifests, original logs, executed import
and kernel hashes, launcher, schema and publisher are retained. The audit uses
ordinary CPU dependencies without requiring a bag or GPU.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_spatial_pruning
```

See [implementation and reproduction](../../benchmarks/cuda_spatial_pruning.md).

## Direct previous-CUDA check

A separate, source-bound previous/current CUDA pair ran the same complete
six-second workload and numerical policy. Previous CUDA took 154.593 s;
current CUDA took 132.839 s (1.164x).
Both fresh workers started with isolated empty CuPy compilation caches.
All result and full per-round quality checks pass. This single pair is a
preliminary incremental observation, without repeated-trial stability claims;
it is not pooled with the three CPU/CUDA pairs or multiplied by their ratio.
Host background load was not reserved.

[Bound direct comparison](previous_cuda/README.md).
