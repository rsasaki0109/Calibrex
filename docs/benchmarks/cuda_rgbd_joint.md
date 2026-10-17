# Optional CUDA execution for native RGB-D graphs

On a **27,544-correspondence public TUM XYZ graph**, CUDA reduces the complete
optimizer from **10.699 → 4.408 seconds (2.427x)**. Fresh worker latency,
including imports, preparation, GPU initialization, compilation and saving,
is **19.659 → 14.484 seconds (1.357x)**. These are medians of three alternating
pairs using the same committed runtime and different execution backends.

All complete results, frontend records and numerical histories agree exactly.
Both backends retain the same 20-iteration `max_iterations` outcome. The input
is the already spent TUM `freiburg1_xyz` development sequence; this comparison
makes no convergence, accuracy or SOTA claim.

The original smaller multi-window example does **not** establish a speedup:
its recorded medians are **34.535 → 38.144 seconds (0.905x)**. Its native batches
stay on CPU, without loading the CUDA adapter. NumPy remains the default.

## Enable the optional backend

Install the existing `calibrex[cuda]` extra and add this option inside
`pipeline.factors.tum_rgbd_joint_point_to_plane.options`:

```yaml
joint_evaluation_backend: cuda
```

The default is `numpy`. CUDA initialization starts only after a supported native
plan reaches 4,096 point factors. Once initialized, the same solve can reuse
the session for smaller evaluations. Custom or unsupported factors retain their
existing scalar path, and device plans exceeding the cache budget use NumPy.
The ordered CUDA sums support CPython 3.10, 3.11 and 3.12.

The selection applies to native solves, ablations, observability and diagnostic
work inside each window. The core remains independent of ROS. No required
dependency or public schema changes; CPU result provenance also retains its
previous option representation.

## Implementation

The optional adapter keeps immutable geometry and two calibrated-point states
on the GPU. It evaluates field and pose perturbations in FP64 kernels, returning
the same residuals and Jacobian to the existing CPU robust weighting, Schur
solver and observability calculations. Scalar priors and regularizers retain
their native evaluators. Family diagnostics reuse validated native residual
dimensions rather than calculating every point residual again.
The NumPy path also reuses this validated metadata for owned native factors.
Unowned and unsupported factors keep their callback-based dimension checks.

The kernels preserve the host's ordered lattice sum, including CPython 3.12
compensation, and the original two quaternion products. The host performs both
quaternion normalizations. Fused multiply-add is disabled. No samples,
correspondences, probes or iterations are removed.

Device geometry and calibrated-point caches are local to one synchronous
execution scope, contain at most two plans, and have a 64 MiB budget including
a declared 4 MiB scratch reserve. Column output chunks contain at most 131,072
elements. This comparison's reported owned-data bound is 10,364,160 bytes and
its largest column chunk has 91,296 elements. CUDA driver and CuPy allocator
memory are outside that owned-data bound. Returning or raising releases the
adapter's plan references.

## Dense graph: every attempt

Both variants use runtime `e3f39bbfc7fb07df730d9f7bebf41e3fa70c14c0`, three
map frames and eight query frames starting at paired index 180. The maximums
are 50,000 map points, 20,000 query points and 4,000 correspondences per query
frame. The actual map contains 78,855 records and 645 voxels. Both backends
receive 27,544 data factors and ten prior/regularization factors, with one
BLAS/OpenMP thread.

| Fresh worker | Preparation seconds | Optimizer seconds | Whole worker seconds |
| --- | ---: | ---: | ---: |
| 01-cuda | 6.341 | 4.408 | 14.704 |
| 02-numpy | 5.144 | 10.237 | 18.997 |
| 03-numpy | 5.451 | 10.699 | 19.764 |
| 04-cuda | 5.889 | 4.670 | 14.484 |
| 05-cuda | 5.476 | 4.296 | 13.562 |
| 06-numpy | 5.235 | 10.758 | 19.659 |

Each worker has a fresh CuPy compiler cache; persistent CUDA driver caching is
disabled. The optimizer timer includes optional-library loading, GPU context
startup, compilation, transfers and trace hashing. Whole-worker time also
includes imports, public-input hash verification, decoding, graph preparation
and evidence saving. Preparation uses the existing CPU path; its timing
variation is not a GPU preparation claim. The host was unreserved and these
ratios do not provide a hardware-independent guarantee.

All 13 actual input files are verified against the existing public inventory.
The 24 frontend events retain all eleven images' pixel hashes, all sampled
points, ordered map geometry, complete correspondence records, factor metadata
and parameter blocks. Each worker also retains 21 full residual/Jacobian and
81 residual evaluations, including array shapes, input states and ordered
factor identities. All six complete payloads also match the previous published
runtime's payload, including all 60 known-bad probes.

Each CUDA worker records 21 linearizations, 606 adapter evaluations and 1,090
kernel launches without a budget fallback. The adapter evaluation count includes
prior-only parameter probes; it does not replace the graph-level 21/81 trace
counts above. The NumPy workers do not import CuPy or use a CUDA adapter.

## Original multi-window example

The full original workload retains three native window processes, 24 solves,
168 Jacobian evaluations, 1,624 residual evaluations and 492 fixed-population
probes in every run. All calibration results and frontend, pair, reassociation,
holdout, curvature and multistart histories agree. Only execution metadata and
its configuration fingerprints are normalized for comparison.

| Fresh worker | Requested backend | Complete suite seconds |
| --- | --- | ---: |
| 01-current | cuda | 38.424 |
| 02-prior | numpy | 33.487 |
| 03-prior | numpy | 35.583 |
| 04-current | cuda | 37.864 |

Every parent and child process in this screen leaves the CUDA adapter unloaded.
The 4,096-point initialization threshold avoids GPU startup for this example,
but the observed elapsed times still do not support a small-workload speedup.
The [full screen](../assets/cuda_rgbd_joint/full-suite-screen.tar.gz) and its
[summary](../assets/cuda_rgbd_joint/suite-summary.json) are retained separately
from the dense graph comparison.

## Evidence and validation

The [archive inventory](../assets/cuda_rgbd_joint/index.json) binds every raw
file by SHA-256 and size. It includes the
[configurations](../assets/cuda_rgbd_joint/config-cuda.yaml),
[all dense attempts](../assets/cuda_rgbd_joint/attempts.json),
[summary](../assets/cuda_rgbd_joint/summary.json),
[provenance](../assets/cuda_rgbd_joint/provenance.json), source capsules,
[record schemas](../assets/cuda_rgbd_joint/record-schemas.json) and
[validation](../assets/cuda_rgbd_joint/validation.json).

The [development records](../assets/cuda_rgbd_joint/development-screens.tar.gz)
preserve every earlier screen, including the rejected NumPy/compiled-CPU
variants, a quaternion-rounding mismatch, an unavailable-header compilation
failure, an overly strict telemetry check, and the slower small-workload CUDA
trial before the initialization threshold. Those screens provide no headline
performance claim. Their exact source capsules and prototype launchers are
included; the final comparison uses the committed implementation directly.

```powershell
python tools/audit_cuda_rgbd_joint.py docs/assets/cuda_rgbd_joint
# From a checkout of the measured runtime, with the downloaded public TUM sequence:
Copy-Item docs/assets/cuda_rgbd_joint/launcher.py.txt .tmp-calibrex-cuda-rgbd-measurement-v3.py
python .tmp-calibrex-cuda-rgbd-measurement-v3.py
```

The launcher refuses to reuse `outputs/cuda-rgbd-joint-pairs-v3`. Keep earlier
results and choose a new output path in the copied launcher for another
experiment. The saved launcher hash identifies that exact measurement code.
