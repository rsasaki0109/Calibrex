# Offline direct-disparity pose landscape

Pre-registered 2026-10-30. Diagnose all 171 archived direct-disparity fits without
new calibration fits, changed gates or new data. Reconstruct exact depth arrays,
verify hashes, and probe the original objective. Preserve clean/noisy, both seeds,
all biases, signs, populations and nine plane controls.

For each case, interpolate from reference pose to archived fitted pose at eleven
fractions 0,.1,...,1. Use linear camera-frame translation and shortest-path rotation
SLERP. Evaluate two tracks: fixed injected metric bias, and fixed fitted metric
bias. Reference and injected bias are offline diagnostic information only; no
truth-dependent initialization/gate is proposed. No bias optimization occurs.
Record total cost, support count and cost restricted to the intersection of
supported point indices over the whole track. Reproduce fitted-bias endpoint cost
against the archive. Keep all points and original unsupported penalties.

Report reference-versus-fit preference for each noise realization and clean data,
separating perturbations from reference starts. Also report sampled interior
minima and whether any interior sample falls below both endpoints. A favorable
reference endpoint does not establish a global optimum, local stationarity or
an optimizer's internal failure cause. Eleven samples cannot rule out narrow
barriers or off-path minima. Fixed injected bias under noisy metric offsets is
not a maximum-likelihood nuisance estimate. Plane probes remain diagnostic only.
No revisions to recovery counts, production adoption or SOTA claim. Freeze code,
protocol and archived result hashes before running; preserve partial output on
failure. Diagnostics have their own pass count, not a new optimizer budget.
