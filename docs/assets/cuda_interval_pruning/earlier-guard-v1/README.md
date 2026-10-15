# Rejected conservative spatial interval candidate

**Not adopted.** Output equality does not establish a useful whole-worker gain.

| Scope | Previous CUDA | Interval candidate | Ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 54.383 s | 63.992 s | 0.850x |

| Repeat | Variant | Whole seconds |
| --- | --- | ---: |
| 0 | prior | 55.160675 |
| 0 | current | 65.272271 |
| 1 | current | 62.711611 |
| 1 | prior | 53.605573 |

This first version performs repeated CPU target-extent reductions. It is retained as measured, before reusing the existing cKDTree extrema.

Two alternating fresh-process pairs replay six seconds of the public GLIM Livox
bag with four map rounds, one BLAS thread, CUDA scan odometry and CUDA fitter
planes. Each worker has a separate initially empty CuPy compiler cache. Whole
time includes imports, setup/compilation, every odometry pass, association,
fits/covariance, noise updates, holdout/controls, hashes, saves and import checks.
No warmup or worker is discarded. Host load is unreserved.

Baseline `bb131739c0d14d30fc0e1660ce4f2ee587d6d32e`; candidate `61921e5785876f9ee47ff030f034f11a357f25c2`.
Bag SHA-256 `f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. The unchanged dense policy is 3 cm voxels,
three local scans, a 1 nm coordinate grid and returned-motion grids of
1 micrometre / 0.1 microradian. CPU and disabled grids remain the defaults.

The candidate encloses boxes, split planes and points in outward FP32 intervals,
using downward-rounded lower distance bounds to reject work. Actual candidate
distances, tie ordering, normals, residuals/systems and native repairs stay FP64.
Native arrays/kernels remain for bounded-path fallbacks; events order reads and
drain abandoned writes. No extra CPU thread is introduced.

Every calibration/evidence field, complete fit, trajectory, covariance,
iteration and previous work counter matches exactly. The exact gate allows only
declared source-bound kernel identities/pruning counters and run metadata to
differ. Actual interval work is positive in every measured pass and bounded by
total GPU work. The clip has no independent extrinsic truth and its verdict
remains INCONCLUSIVE. Ratios are not multiplied across cohorts.

All full results/fits/logs, deterministic source capsules/manifests, launcher,
schema and publisher are SHA-256 bound. The audit runs without CUDA. Raw bag
data and compiler caches are not redistributed. The retained runtime keeps the
previous asynchronous-normal implementation; these interval kernels are experimental.
