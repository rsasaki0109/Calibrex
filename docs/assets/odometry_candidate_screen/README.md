# Odometry candidate screening

All ten candidate strategies remain experimental. No runtime change is adopted.
The best complete target-construction/registration screen is only 1.158x; no
whole-bag improvement is inferred from these small fixed-input observations.

| Candidate / scope | Previous | Candidate | Observed ratio |
| --- | ---: | ---: | ---: |
| Stable GPU voxel reduction, raw scans | 0.022896 s | 0.021742 s | 1.053x |
| Stable GPU voxel reduction, processed scans | 0.016450 s | 0.020034 s | 0.821x |
| Fixed k=11 query | 0.392656 s | 0.387144 s | 1.014x |
| Unrolled neighbor insertion | 0.392656 s | 0.375619 s | 1.045x |
| Leaf-packed reference coordinates | 0.392656 s | 0.394731 s | 0.995x |
| GPU radix tree with lazy native repairs | 0.457124 s | 0.394617 s | 1.158x |
| Tree-ordered normal queries | 0.377905 s | 0.380340 s | 0.994x |
| Morton-ordered source queries | 0.377905 s | 0.385769 s | 0.980x |
| CPU-tree-ordered source queries | 0.377905 s | 0.400486 s | 0.944x |
| Guarded GPU batches of four iterations | 0.360379 s | 0.371897 s | 0.969x |
| Guarded GPU batches of eight iterations | 0.360379 s | 0.353789 s | 1.019x |

Five separate cohorts retain all 744 observations and their own baselines.
They are not pooled. Four cohorts rotate the candidate order over five repeats
of eight captured public GLIM registration inputs. One first-case compilation
warmup per variant is outside their timing. Scope includes construction,
normal preparation, native repairs and registration. The voxel cohort retains
seven rounds over eight original raw scans and eight processed inputs; its
six-round retained-cache medians exclude round zero explicitly, with round zero
retained. Its first candidate round takes about 2.935 seconds, including
on-demand operation compilation; module setup is outside this pilot's timing.
There is no fresh-process/cold performance claim and no complete-bag candidate
evaluation. One BLAS thread is used; host CPU/GPU activity is unreserved.

All registration decisions, iterations and correspondence counts match. Layout,
radix and ordering pilots retain exactly equal numeric registration fields;
batched updates use abs 1e-9 / rel 1e-6 for transforms/RMSE/eigenvalues. All
batched fixed-case runs report zero full-path fallbacks. This does not validate
their unexercised fallback/boundary cases. The voxel launcher asserts coordinate
bit equality and preserves point counts; coordinate payloads are not archived,
so its CPU-only audit checks counts/timings/source bindings, not the coordinates.

The CPU-only audit recomputes every timing aggregate and checks all observations,
retained registration fields/work, source capsules, launcher/protocol hashes,
scope and input identities. Raw scans and compiler caches are not redistributed.
The launchers are diagnostic scripts with fixed repository-relative inputs and
output directories; they are frozen as text and are not production APIs.

The separately retained cProfile run diagnoses the current pipeline. Its
instrumentation and eight fixture captures add overhead; its roughly 88 second
worker profile is not compared with uninstrumented bag benchmarks. Runtime
source is `3b006a4be3babcf2b0e5c537993105b7d1c68b46` and bag SHA-256 is
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. This clip supplies no independent calibration truth.

```bash
python -m tools.audit_odometry_candidate_screen docs/assets/odometry_candidate_screen
```
