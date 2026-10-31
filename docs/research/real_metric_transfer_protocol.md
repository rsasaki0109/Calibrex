# Spent real-data transfer of metric bias fitting

Pre-registered 2026-10-30. Use the six existing KITTI development recordings,
three frames per recording, and archived re-rectified SGBM depth restored to
original left-camera rays. Depth arrays are pinned to the prior joint learned
stereo comparison; no new stereo inference, model choice or fresh data. SGBM is
the fixed frontend that previously recovered 6/12 versus UniMatch 5/12 with MI.
That MI score is historical, not a matched baseline for this experiment.

For each original positive/negative seed and one reference-start control per
drive, run three fixed methods: bilinear seven-parameter metric pose/bias,
image-only edge seven-parameter fit, and edge fit plus scalar rescue. Exactly
18 starts × 3 methods =54 fits. Compare methods within this run. No artificial
bias is injected. Real depth has no bias ground truth; report estimated nuisance
bias without treating it as accuracy or assigning a bias-recovery gate.

Each corresponding cloud is sampled at min(N,4096) integer linspace indices
from 0 to N-1, preserving file order. Sampling uses no reference, projected
visibility, depth values or fitted pose. Freeze these full-file hashes and this
sampling rule before fitting. Use all three corresponding observations without
cross-frame cloud registration. Original intrinsics match restored depths.
Reference extrinsics enter scoring and explicitly labeled reference starts only.

Keep all solver/sampler/loss/bounds/termination/support/identifiability settings
unchanged, including 800 actual optimizer-call cap per method. Additional
support/sensitivity diagnostics remain outside the cap as in previous controls.
The edge-only fit must exactly reproduce the rescue's nested baseline. Scalar
rescue preserves pose and uses measured cost, not truth; its staged termination
does not establish new joint stationarity.

Report rotation <.5 degree and translation <.20 m with convergence, plus stricter
accepted-pose recovery requiring local identifiability. Separate the 12 perturbed
starts from six reference starts, and show every drive, drift, rejection, runtime,
correction and regressions. Keep inaccurate converged fits and all failures.
Support loss can cause rejection. Do not reduce denominator, tune settings or
relax gates after outcomes. Abort on missing/hash-mismatched input or execution
failure, preserve logs and partial output. This is spent-data transfer evidence,
not independent SOTA validation or production adoption.
