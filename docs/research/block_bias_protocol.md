# Separated scalar-bias and pose updates

Compare a fixed alternating-block allocation against archived joint TRF on all
54 [analytic-sampler cases](analytic_sampling_results.md): 36 perturbed trials and
18 truth-start controls. Same analytic sampler, original residual/saturation,
soft-L1(.1 m), populations, initial poses, pose/bias bounds and post-fit sensitivity.
Do not change pixel sampling or use injected bias to initialize the solver.

Initialize pose delta and bias to zero. Up to eight cycles:

1. Hold pose fixed. Minimize the measured scalar bias objective on [-.5,.5] with
   bounded Brent search, maxiter 40, xatol 1e-6. Accept only actual cost decrease.
2. Hold current bias fixed. Run six-DoF TRF from current pose coordinates using
   the original global bounds/scales, two-point Jacobian, soft-L1 and 1e-8
   tolerances; max_nfev=min(20,floor(remaining_calls/7)). Accept only cost decrease.
3. Converge only when both stages succeed, full-cycle decrease <=1e-8*max(1,old_cost)
   AND absolute bias change <=1e-6. Retain original per-frame support and local
   identifiability checks. A cap is not convergence and does not pass recovery.

One shared hard ceiling of 800 counts initial objective, ALL scalar trials and
ALL pose finite-difference evaluations. Start no cycle with <48 remaining calls.
No hidden/free replay, reference grid lookup, added starts or outcome-based retries.
Record every cycle's stage costs, acceptance, status, delta and cumulative calls.
Post-fit support/sensitivity accounting remains separate as in the baseline.
Scalar objective may be nonconvex; bounded Brent is not a global-optimality claim.

Joint success retains original convergence, rotation <.5 deg, translation <.20 m,
bias error <.05 m and local rank gates. Report pose-only accuracy separately.
Explicitly report all six negative-bias truth-start failures, four old budget
failures, any regressions and full denominator, not only targeted successes.
Archived baseline is pinned, not rerun or timed as a simultaneous benchmark.
This changes solver/allocation only, not a validated production method or SOTA claim.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/evaluate_block_bias.py docs/assets/block_bias_20261029/plan.json \
  docs/assets/block_bias_20261029/evaluation.json
```
