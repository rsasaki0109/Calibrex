# Rejected conservative spatial interval candidate

**Not adopted.** Output equality does not establish a useful whole-worker gain.

| Scope | Previous CUDA | Interval candidate | Ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 53.534 s | 51.576 s | 1.038x |

| Repeat | Variant | Whole seconds |
| --- | --- | ---: |
| 0 | prior | 55.165947 |
| 0 | current | 54.201855 |
| 1 | current | 48.950571 |
| 1 | prior | 51.901612 |

The revised source reuses cKDTree extrema instead of scanning all target coordinates. Its source-bound regression run passes 136 tests, and all 130 public schema/ledger exports are byte-identical. The [earlier integrated version](earlier-guard-v1/README.md) retains its slower full cohort and is recursively audited.

Two alternating fresh-process pairs replay six seconds of the public GLIM Livox
bag with four map rounds, one BLAS thread, CUDA scan odometry and CUDA fitter
planes. Each worker has a separate initially empty CuPy compiler cache. Whole
time includes imports, setup/compilation, every odometry pass, association,
fits/covariance, noise updates, holdout/controls, hashes, saves and import checks.
No warmup or worker is discarded. Host load is unreserved.

Baseline `bb131739c0d14d30fc0e1660ce4f2ee587d6d32e`; candidate `2249a4f760b6c2d5c3226cca7f1b461754adc310`.
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
