# Returned noisy disparity fits leave measured descent along pose paths

Frozen `21203727`, evaluated 2026-10-30. See
[protocol](disparity_landscape_protocol.md),
[plan](../assets/disparity_landscape_20261030/plan.json),
[171 diagnostic records](../assets/disparity_landscape_20261030/evaluation.json) and
[schema](../assets/disparity_landscape_20261030/evaluation.schema.json).

All 171 archived cases were reconstructed with exact depth hashes; all fitted-bias
endpoint costs reproduce archived costs. No calibration was rerun and no recovery
verdict changes. Each case has two eleven-point pose paths: reference to fitted
pose, at fixed injected bias or fixed fitted bias. There are 3762 diagnostic
residual passes, outside prior optimizer budgets.

## Noisy perturbed cases

| Bias held fixed | Noise seed | Reference lower than fit | Fit lower than reference | Interior lower than both endpoints |
| --- | ---: | ---: | ---: | ---: |
| Injected | 730 | 30/36 | 6/36 | 36/36 |
| Injected | 731 | 36/36 | 0/36 | 36/36 |
| Fitted | 730 | 0/36 | 36/36 | 34/36 |
| Fitted | 731 | 0/36 | 36/36 | 35/36 |

Every count is unchanged when restricting cost to the common supported point
indices over the whole path. Thus changing support alone does not explain these
comparisons. With fitted bias fixed, **69/72 returned noisy perturbed fits have
a sampled pose with lower cost than both endpoints**, without changing bias.
This establishes measured descent left unexploited along the tested path in those
cases. The previous run reported all 72 as nonconverged, so this is consistent
with its status; it does not expose falsely converged solutions in that group.

Reference is lower than fit in 66/72 cases at injected bias but in none at fitted
bias. Therefore comparing reference at the wrong nuisance bias can reverse the
ranking. These results do not show that reference is a global minimum or that an
arbitrary optimizer can reach it. Interior minima occur even at injected bias;
noise, interpolation and the robust model still affect the preferred pose.

## Clean and reference-start controls

At fitted bias, all 54 clean sphere fits prefer their returned endpoint over the
reference endpoint; no sampled interior point beats both. At injected bias,
reference is lower in 17/36 perturbed and 7/18 reference-start cases, while sampled
interior points beat both endpoints in all 54. Fixed injected bias is not an
optimized nuisance estimate, so this is diagnostic model preference, not revised
accuracy or identifiability.

For noisy reference-start cases at fitted bias, the returned endpoint beats the
reference in all 36 cases; none has a sampled interior point below both endpoints.
At injected bias, reference is lower in 6/18 cases per seed and an interior point
beats both endpoints in every case. All nine plane controls remain in the report;
this work does not change their previous rejection.

Three focused tests, repository Ruff and strict documentation build pass. Validation
checks report/schema, all frozen source hashes, full case count, generated-depth
identity and endpoint cost reproduction. Rotation interpolation tests verify
endpoints and the half-angle path. No core/default or SOTA claims change.

Next compare a bounded derivative-free coordinate search against the archived
TRF on the same direct-disparity objective and data, with the same 800-call limit.
Start from the original seeds and use only measured costs; reference paths here
must not become initializations or online gates. Keep clean controls, all noise
realizations, bias levels, populations, truth starts and plane controls. Define
step sizes and termination before running rather than raising TRF caps after
failure. This is evidence to test a search change, not proof it will recover pose
or transfer to real recordings. Eleven samples do not rule out narrow barriers,
off-path minima or other measurement-model defects.
