# Baseline-preserving scalar bias rescue

Frozen before evaluation, 2026-10-30. This is an oracle analytic control, not a
production solver or SOTA claim. Following the failed block replacement, keep the
unchanged seven-parameter joint solver and all 54 analytic sphere cases: three
layouts, visible/all returns, 0/+.25/-.25 m biases, both perturbations and truth starts.
Archive the analytic baseline and also rerun it inside each candidate fit.

Only accepted joint fits with at least 43 remaining calls are eligible. At the
unchanged fitted pose evaluate the current bias and clipped bias ±.01 m. Trigger
only if a probe lowers the same soft-L1 asymmetric cost by more than
1e-8*max(1,current cost). No injected bias, seed label or truth enters the solver.
Run bounded scalar minimization on [-.5,.5] m, xatol 1e-6, maxiter 40. Adopt only
on scalar success, measured cost improvement by the same threshold and the
unchanged robust identifiability gate. Otherwise return the complete baseline.
Pose and support cannot change. Keep the baseline fit in every result.

The total 800-call limit includes all joint evaluations, finite differences and
three rescue probes and scalar trials. Support/sensitivity diagnostics remain
outside optimizer accounting, as in the baseline. No pose refinement, new sampler,
new objective or relaxed accuracy gates. All original cases remain denominator.
Report baseline numerical convergence separately: an adopted output has baseline
joint termination plus scalar termination, NOT demonstrated seven-dimensional
stationarity after the correction. Existing converged/accepted flags describe this
staged termination; inherited status/nfev/njev describe the baseline stage only.

Compare joint pose+bias recovery using original .5 degree/.20 m/.05 m gates;
truth-start controls separate. Report regressions, gains, triggers, adoption,
call bounds, exact pose preservation and reproducibility of rerun baselines.
Do not convert previously unconverged joint fits to converged via scalar search.
Cost improvement does not establish global accuracy. Preserve negative evidence;
no post-result tuning of thresholds or budget.
