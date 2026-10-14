# Rejected packed CUDA-neighbor prototype

The prototype does not establish useful whole-pipeline acceleration. Whole
medians increase by 3.3%; complete registration is nearly unchanged. The
prototype is rejected and its runtime changes are restored to the previous
version. The frozen measured sources and device tests remain in this archive.

| Scope | Previous median | Prototype median | Observed ratio |
| --- | ---: | ---: | ---: |
| Whole worker | 62.250 s | 64.313 s | 0.968x |
| Complete registration | 17.290 s | 17.024 s | 1.016x |
| Nested normal preparation | 6.764 s | 5.822 s | 1.162x |
| Spatial tree construction | 3.897 s | 4.252 s | 0.916x |
| Unchanged fit/covariance | 13.784 s | 13.504 s | 1.021x |

Three alternating previous/current pairs keep the same public six-second GLIM
workload and four map rounds. All 12 full fitted trajectories/covariances/costs,
holdout values and all 32 signed controls per pair are exactly equal.

No foreign or unknown CUDA compute process was detected in 41
ten-second observations. The observer would terminate only the owned cohort on
interference. Periodic sampling does not establish continuous GPU exclusivity,
detect all graphics activity or reserve CPU resources. Whole timing includes
fresh-process imports, decoding, compilation, all odometry, fits and validation;
workers use separate initially empty compilation caches. Normal preparation is
nested inside registration and component ratios must not be multiplied.

Source manifests match the first validation pair and the bound archives are
reused byte-for-byte. The measured current commit is
`7216d17c12203cd94152117cd1f0eda87360bcf8`. All worker and observer/source bindings are
CPU-auditable. Verdicts remain INCONCLUSIVE without independent extrinsic truth.
No point, neighbor, iteration or validation budget is reduced.
