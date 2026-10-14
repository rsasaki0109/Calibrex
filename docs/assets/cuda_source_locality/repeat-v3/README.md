# Monitored repeat of CUDA source scheduling

This repeat keeps exactly the same runtime source bytes and workload as the
parent comparison. Source manifests match; the existing bound source archives
are reused byte-for-byte. The measured current commit includes publication
changes only.

The observer sampled CUDA process membership every ten seconds: 42
samples, 0 with foreign GPU processes, and 0 with unknown process
membership. No foreign or unknown compute processes were detected. The observer
would terminate only the owned cohort if interference appeared. Periodic sampling
does not prove continuous GPU exclusivity. No process names
or process IDs are published. This periodic check also does not reserve CPU
resources or detect all graphics activity.

All three complete comparisons retain exact calibration, holdout and decision
equality. Full fitted trajectories, covariance, costs and 32 controls are kept.
Verdicts remain INCONCLUSIVE without independent extrinsic truth.

Whole medians are 64.765 s and
64.220 s (observed ratio
1.008x). These are retained timings, not an isolated
speedup claim. The 1.008x whole ratio does not establish a meaningful performance
benefit; registration medians increase from 17.903 s to 19.216 s (0.932x).
The source-ordering prototype should not be adopted as an effective acceleration
on the basis of this workload.

The index binds six complete workers, source archives/manifests, frozen launcher,
schemas, tests, observer source/report and publisher. The report binds the
observer and launcher hashes. The parent audit also checks this nested repeat.
