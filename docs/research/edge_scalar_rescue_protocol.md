# Scalar rescue with image-only depth sampling

Pre-registered 2026-10-30 before evaluation. Combine the previously frozen
image-only gated sampler with the previously frozen scalar rescue rule, changing
only the rescue module's residual/sensitivity/joint-solver imports. Compare to
the pinned 63-case image-only sampler result, not the oracle experiment.

The baseline joint fit runs unchanged and is preserved in every result. Only
accepted baseline fits with at least 43 remaining optimizer calls are eligible.
At fixed pose evaluate current bias and clipped ±.01 m bias probes. Trigger if
cost decreases by more than 1e-8*max(1,current cost). Bounded scalar search uses
[-.5,.5] m, maxiter 40 and xatol 1e-6. Adopt only on scalar success, measured
improvement exceeding the same threshold and unchanged robust identifiability.
Otherwise retain baseline. No labels, injected bias, oracle sampler, pose
refinement or retuning of the .2 m sampling threshold enters the rescue.

Total 800 optimizer calls includes baseline finite differences, all three probes
and scalar trials; support/sensitivity diagnostic passes remain separate. Adopted
outputs combine baseline joint termination and scalar termination; they do not
establish joint stationarity after correction. Inherited status/nfev/njev are the
baseline's; rescue audit and total actual_residual_calls describe additional work.

Run all 54 sphere controls: three layouts, visible/all returns, 0/+.25/-.25 m,
positive/negative seeds and truth starts. Also retain all nine parallel-plane
controls. Keep .5 degree/.20 m/.05 m gates and original convergence/support/rank
checks. Report perturbed and truth-start recovery separately, regressions/gains,
triggers/adoptions, bias errors, budget, pose preservation, baseline reproduction
and plane rejection. Retain all twelve prior failures, including the two prior
regressions and two near-threshold failures. No post-result tuning or exclusions.
Synthetic evidence only; production and real-data claims require later validation.
