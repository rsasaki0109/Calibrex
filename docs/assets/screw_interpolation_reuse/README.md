# Complete screw interpolation reuse evidence

`complete-evidence.tar.gz` contains 907 indexed files in
4,308,915 bytes. `first-attempt.tar.gz` retains the initial
missed-gate cohort: 907 indexed files in
4,308,600 bytes. Each archive includes both complete
401-module source capsules, frozen tools, all twelve full input/result/fit/timing
records and logs, derived public GLIM geometry, record schemas and the archive producer.

The [final inventory](inventory.json), [first-attempt inventory](first-attempt-inventory.json),
[final summary](summary.json), [initial summary](first-attempt-summary.json),
[development record](development.json) and its [schema](development.schema.json)
bind the original bytes. The baseline is `297acf16310500c1a6644a584e15e7b9ec28cf27`;
the final candidate is `c017754233f22bdd921f0b6f76a9d333bbc1f686`. The first candidate
is `2e5da540f1b81b22d8dff900bb1201289fa4a8f7`.

The initial geometry API ratio was 1.900x and missed the unchanged 2x gate.
After sharing exponentials, replacing general matrix block dispatch and dropping
discarded geometric RMSE derivatives, the final paired API ratios are
2.395x for synthetic joint recovery and
2.765x for public geometry.
All complete inputs and results agree exactly within every pair in both cohorts.
All joint cases retain their policy passes and 32 detected controls; all geometry
fits converge in 15 iterations. See the
[complete scopes, raw ranges and startup measurements](../../benchmarks/screw_interpolation_reuse.md).

CPU-only replay, without a GPU or original recording:

```bash
python tools/audit_screw_residual_archive.py docs/assets/screw_interpolation_reuse/complete-evidence.tar.gz
python tools/audit_screw_residual_archive.py docs/assets/screw_interpolation_reuse/first-attempt.tar.gz
```

The first command exits 0. The second validates its complete records and exits 2
because its geometry gate was missed. The auditor never executes archived code.
Both archives retain the development checks, the initial test/cache/packaging and
profiling-launcher failures, their corrections, and one exact diagnostic fit profile.
Profiler times overlap and contain instrumentation overhead; they are not paired
performance observations. The 62 maintained reuse checks and 210-check targeted run
bind to the saved test sources. These are already-spent development fixtures;
assigned GLIM timestamps and capture-local planes do not describe a physical
recording trajectory. No accuracy, SOTA or fixed-budget reliability claim is made.
