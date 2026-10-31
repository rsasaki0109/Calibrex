# Frozen analytic truth-start stopping diagnostic

Investigate the six wrong negative-bias truth-start outputs in the
[analytic-sampler experiment](analytic_sampling_results.md). Include ALL 18
truth-start cases across three layouts, visible/all-return populations and
0,+.25,-.25 m biases; not just failures. Do not rerun fitting or alter solver,
sampler, tolerance, bias initialization or gates. No new recovery claims.

At archived initial and final poses/corrections evaluate the identical saturated
soft-L1 scalar objective. Probe all seven coordinate directions with forward and
central residual differences at h=1e-8,1e-6,1e-4 in original degree/metre units.
Record gradients, singular spectra, forward/central relative Jacobian differences,
raw residual jumps >1 m and support changes for each signed perturbation. These
are controlled step sweeps, not exact replication of SciPy's machine-relative
finite-difference policy or a search for a best-performing step.

Compute bias derivative analytically: corrected residual derivative is +1 for
valid samples, zero for missing penalties; saturation derivative is
(1+(r/.2)^2)^(-3/2) for r>0 and one otherwise, followed by the soft-L1 derivative.
Check fixed bias-only ±.01 m objective changes. Scan the predeclared full bias
bound [-.5,.5] at .01 m spacing while holding pose fixed, retaining every value.
The grid minimum is a diagnostic feasible comparison, never a replacement output
or true-bias initialization. The measurement generator knows injected bias; probes
operate on generated residuals, not a truth-directed update rule.

Interpret exact-bias descent, pose Jacobian discontinuity and stopping status
separately. A better bias-only point proves a missed feasible improvement but
neither proves the global optimum nor isolates every solver failure. If derivatives
are discontinuous, document their origin before changing sampling/optimization.
Preserve max_nfev cases and the six negative-bias controls. No post-outcome tuning.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/diagnose_analytic_stopping.py docs/assets/analytic_stopping_20261029/plan.json \
  docs/assets/analytic_stopping_20261029/evaluation.json
```
