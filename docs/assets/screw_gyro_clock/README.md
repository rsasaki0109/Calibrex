# Complete gyro clock development evidence

`complete-development.tar.gz` contains 880 indexed files in 3,908,344 bytes.
It retains both complete 402-module source capsules, all three prior/current
problems, options, fits and native outcomes, current final normal models,
schemas, logs, regression source and original producers. The nine-state,
63-factor comparison retains every residual and derivative block and performs
zero optimizer steps.

The [receipt](receipt.json) and [schema](receipt.schema.json) bind the original
archive, index, source commits and native artifacts. The
[summary](summary.json) and [schema](summary.schema.json) retain complete input
equality, every reported comparison, initial test/command/lint failures and
remaining limitations. Baseline outcomes are reused from the previous stored
batch cohort; both source capsules establish their binding to the baseline.

| Seed | Current status / iterations | Complete native result |
| --- | --- | --- |
| 20261010 | converged / 20 | [JSON](seed-20261010.json) |
| 20261011 | **max_iterations / 32** | [JSON](seed-20261011.json) |
| 20261012 | converged / 19 | [JSON](seed-20261012.json) |

All three quality policies pass and all 32 signed controls are detected per
case. These are spent development outcomes, without a speed, independent
accuracy or general convergence claim. Raw final gradients remain above the
configured stopping threshold. The middle case's iteration-limit result is
retained, without reruns or retuned gates.

See the [full comparison, boundary convention and CLI replay](../../benchmarks/screw_gyro_clock.md).
To replay an original producer, copy `generator.py.txt` from the archive to
`outputs/gyro-clock-development-v1/run.py` at the numerical candidate, then use
a fresh output directory and its declared source argument. The archive's
factor producer similarly records its original working path and hashes. It
does not need a GPU or an original recording.
