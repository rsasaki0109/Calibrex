# Further odometry acceleration candidates

Ten candidate strategies were screened after the optional CUDA spline fitter.
All remain experimental: the strongest complete target-construction/registration
pilot is 1.158x, and no whole-bag candidate speedup is established. Production
runtime remains identical to `3b006a4be3babcf2b0e5c537993105b7d1c68b46`.

| Candidate | Observed ratio | Timed scope |
| --- | ---: | --- |
| Stable GPU voxel reductions | 1.053x raw / 0.821x processed | Eight scans of each kind |
| Fixed k=11 spatial queries | 1.014x | Eight target constructions and registrations |
| Unrolled neighbor insertion | 1.045x | Same complete registration scope |
| Leaf-packed reference coordinates | 0.995x | Same complete registration scope |
| GPU radix tree and lazy native repairs | 1.158x | Same complete registration scope |
| Tree-ordered normal queries | 0.994x | Same complete registration scope |
| Morton-ordered source queries | 0.980x | Same complete registration scope |
| CPU-tree-ordered source queries | 0.944x | Same complete registration scope |
| Guarded batches of four GPU updates | 0.969x | Same complete registration scope |
| Guarded batches of eight GPU updates | 1.019x | Same complete registration scope |

The [complete archive](../assets/odometry_candidate_screen/README.md) retains
five separate cohorts, all 744 observations, all source/launcher snapshots and
input identities. Four cohorts use five rotated-order repeats over eight
captured public GLIM registration inputs. Each variant receives one explicit
first-case compilation warmup outside timing. The voxel pilot retains all seven
rounds; its warm medians use rounds one through six, with the 2.935 second first
candidate round retained separately. Module setup is outside that pilot's scope.
These screens do not measure fresh-process startup or complete bag calibration.
Their baselines differ, and their ratios are not pooled or multiplied. Host
CPU/GPU activity is unreserved; one BLAS thread is used.

The voxel launcher asserts identical coordinate bits and point counts, but only
counts and timings are retained for CPU audit. Other cohorts retain every
registration result and work counter. Layout, tree and ordering variants keep
all numeric registration fields equal; batched updates use abs 1e-9 / rel 1e-6
for transforms, RMSE and the Hessian eigenvalue. Decisions, iterations and
correspondence counts match. The batched screen exercises no full-path fallback,
so it does not establish fallback/boundary correctness or production readiness.

The GPU radix pilot illustrates the need to measure the complete operation:
median construction time falls from approximately 0.248 to 0.092 seconds over
eight targets, while subsequent registration gets slower. The complete ratio
is only 1.158x. Query layout, order and device-side update batches also fail to
establish a useful combined improvement on these fixed inputs.

## Current pipeline diagnosis

A separate source-bound cProfile run replays six seconds of the original public
bag, using the adopted CUDA spline fitter and four map rounds. It identifies
work spread across target construction, preprocessing/local maps, registration,
normal preparation and gyro rotations. These scopes overlap. The profile has
instrumentation and fixture-capture overhead, and is not compared with ordinary
whole-worker benchmark times. The clip provides no independent extrinsic truth.

The next substantial improvement needs a pipeline design that reduces combined
preparation and registration cost. Existing complete-bag physical/evidence
checks remain the acceptance gate; small component ratios do not establish
that benefit.

```bash
python -m tools.audit_odometry_candidate_screen docs/assets/odometry_candidate_screen
```

The CPU-only audit checks source capsules, launchers, input/protocol bindings,
all observations, registration fields/work, timing medians and declared scopes.
Frozen diagnostic scripts have repository-relative input/output paths and are
archived as text. They introduce no production API or optional dependency.
