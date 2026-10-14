# Fixed-neighbor CUDA queries: rejected after cold measurement

Three alternating fresh-process comparisons of complete public KITTI map
evaluation give whole medians of **20.702 s → 23.979 s (0.863x)**. API medians
are **15.661 s → 19.231 s (0.814x)**. The prototype is rejected and runtime code
is restored to the previous implementation. The
[complete archive](../assets/cuda_fixed_neighbors/README.md) retains the measured
code, device tests, six workers and process observations.

The prototype fixes the neighbor count at compile time and expands candidate
updates into constant slots. It preserves FP64 distance arithmetic, distance/ID
ordering, traversal and native repairs. Tests cover all 1–17 neighbor counts,
ties, missing neighbors and extreme scales. All three complete evaluations,
including transforms, covariance, signed controls, holdout, jackknife and
decisions, match exactly. Each worker retains 9,147,116 device query rows.

An earlier fixed-order pilot reused a shared compilation cache and appeared
much faster because the previous worker ran first. It does not establish an
acceleration effect. Alternating warm query kernels also cannot establish whole
calibration speed. The cold full-worker comparison takes precedence.

The workload uses raw KITTI sequence `2011_09_30_drive_0018_sync`, scans 115, 230
and 345, as a copied-sensor identity execution control with fixed noise and
generated poses. It includes preparation, compilation, fits, holdout, signed
controls, jackknife, hashes and saved validation. Reference odometry and
independent two-sensor extrinsic accuracy are outside this control's scope.
There is no reduction in sample, neighbor, fit or validation budgets.

The observer found no foreign or unknown compute processes in 17 ten-second
samples. Sampling does not establish continuous GPU exclusivity or reserve CPU
resources. Workers use one BLAS/OMP/MKL thread and independent empty CuPy caches.
No hardware model, process IDs or raw scans are published.

A separate cold profile of the same complete API evaluation records 63 NVRTC
compilations taking 10.808 s within 14.517 s of profiled execution. It uses the
generic query path and retains the complete measurement and source bindings.
This is a diagnostic of initialization cost, not another speed comparison.
The next useful target is reducing generated kernel count across preparation,
residual and system arithmetic; faster individual queries leave substantial
initialization work intact.

Audit without a GPU or the raw scans:

```bash
python -m tools.audit_cuda_fixed_neighbors docs/assets/cuda_fixed_neighbors
```

Reproducing the rejected prototype requires clean commit
`1bb392b` and the raw inputs identified by the protocol. The active runtime
contains the previous generic query implementation.
