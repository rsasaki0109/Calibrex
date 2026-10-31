# Matched-input MI and metric cost diagnosis

Frozen before execution, 2026-10-30. Follow the failed real metric transfer on
exactly the same 18 starts, 18 restored SGBM arrays and corresponding clouds.
Keep the prior deterministic min(N,4096) file-order selection. Run the unchanged
BorerSixDofSolver defaults once per start (six drives, both perturbations and
one reference start). MI retains its own projection, histogram and support rules;
this is a matched-input comparison of complete solvers, not an isolated comparison
of objective formulas or equal correspondence populations. Default MI limit is
800 evaluations, which is not identical to metric residual-call accounting.

For every archived bilinear/edge/edge-rescue fit, evaluate four fixed states:
reference with bias zero, reference with fitted bias, fitted pose with bias zero,
fitted pose with fitted bias. No metric reoptimization or truth-based bias fitting.
Keep the existing asymmetric residual and soft-L1 cost. Record total/supported
cost, support count, supported raw signed quantiles (0,.1,.5,.9,1), negative
fraction and |raw|>1 m fraction. On the intersection of supported point indices
across four states, also report saturated soft-L1 cost. This intersection is an
offline diagnostic, not a new fitting population or deployment gate.

Verify recomputed fitted costs reproduce archived costs numerically. Report MI
convergence and full-pose recovery (.5 degree/.20 m), all 12 perturbations and
six reference controls separately. Report lower-cost inaccurate metric outputs
at equal fitted bias and on common support. Reference poses are offline evidence,
not a safe online acceptance criterion. Lower cost at wrong poses implicates
objective/measurement mismatch but does not by itself identify stereo, timing,
occlusion or optimizer as the unique cause. Fixed-reference bias is not a known
physical correction in real data. No threshold tuning, fresh data or SOTA claim.
Abort on changed hashes or execution failure; retain partial outputs/logs. Pin
source and input provenance before running. Do not revise historical verdicts.
