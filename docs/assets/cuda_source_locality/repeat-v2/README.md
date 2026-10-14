# Monitored repeat of CUDA source scheduling

This repeat keeps exactly the same runtime source bytes and workload as the
parent comparison. Source manifests match; the existing bound source archives
are reused byte-for-byte. The measured current commit includes publication
changes only.

The observer sampled CUDA process membership every ten seconds: 52
samples, 32 with foreign GPU processes, and 0 with unknown process
membership. The cohort started clear, but competing workloads appeared during
execution. **This is not an uncontended speed measurement.** No process names
or process IDs are published. This periodic check also does not reserve CPU
resources or detect all graphics activity.

All three complete comparisons retain exact calibration, holdout and decision
equality. Full fitted trajectories, covariance, costs and 32 controls are kept.
Verdicts remain INCONCLUSIVE without independent extrinsic truth.

Whole medians are 79.980 s and
75.516 s (observed ratio
1.059x). These are retained timings, not an isolated
speedup claim.

The index binds six complete workers, source archives/manifests, frozen launcher,
schemas, tests, observer source/report and publisher. The report binds the
observer and launcher hashes. The parent audit also checks this nested repeat.
