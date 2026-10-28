# Screw-linear clock-domain reliability

The screw-linear fitter now intersects the feasible clock offsets of every fitted
gyro window and accelerometer observation. A coupled update outside this interval
is rejected before evaluating its objective, and damping increases for a retry.
Exhausting the damping limit records `stalled` / `clock_domain_limit`. Timestamps
are never clipped. A zero-width interval cannot support clock estimation; fixed
clock fitting remains available.

All factors use the same feasible derivative direction. Positive clock increments
move trajectory queries left; negative increments move them right. Accelerometer
finite differences use the actual representable signed increment. At exact
internal gyro endpoint knots, an otherwise stationary final fit also checks the
opposite side when feasible. A zero gradient on just one side cannot certify
convergence when the other side has a nonzero gradient. The optional
`fit_diagnostics.clock_domain` records bounds, rejected updates, the primary
derivative increment, and the opposite gradient when checked.

## Retained development comparison

[Evidence](../assets/screw_clock_domain/manifest.json) retains paired results,
source hashes, changed-source snapshots, and the exact runner. Both versions use
the existing joint synthetic recovery defaults, with seeds 20261010, 1, and 2.
These reused development seeds are not independent validation or a speed study.

| Seed | Baseline status / iterations | Current status / iterations | Quality gates | Detected controls |
| --- | --- | --- | --- | --- |
| 20261010 | stalled / 35 | stalled / 31 | both pass | both 32/32 |
| 1 | stalled / 25 | converged / 18 | both pass | both 32/32 |
| 2 | stalled / 34 | stalled / 32 | both pass | both 32/32 |

None of these three runs rejects a clock update. Small parameter and stopping
differences accompany using the representable finite-difference increment;
they do not establish an accuracy or runtime improvement. Dedicated regression
tests exercise rejection before evaluation, damping retries, narrow domains,
large timestamps, consistent derivative sides across factors, an accelerometer
probe constrained by another factor, and a knot with hidden opposite-side descent.

To reproduce, copy `evaluate_clock.py.txt` to a Python file, set
`OPENBLAS_NUM_THREADS=1`, and run it with a fresh output directory. For the
baseline use the manifest's base revision; for the measured version overlay the
three `measured_source` snapshots on that revision. Use `PYTHONPATH` to select
that source tree. Each protocol binds all package Python files by SHA-256.

## Limits

This is a domain guard and conservative stopping audit, not a projected solver
or a constrained-optimality certificate. A valid boundary optimum may still be
reported as stalled. Only fitted observations constrain the interval. The knot
audit uses exact floating-point ties, without snapping nearby timestamps, and
does not provide a general nonsmooth accelerometer stationarity proof. The
existing accelerometer motion stencil is unchanged. B-spline numerical behavior,
marginalization, and a general clock-stationarity audit remain future work.
