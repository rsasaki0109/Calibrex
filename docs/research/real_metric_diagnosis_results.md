# Real-data failure involves both input sampling and metric cost preference

Frozen at `4a1cf7b8` before evaluation on 2026-10-30. See
[protocol](real_metric_diagnosis_protocol.md),
[plan](../assets/real_metric_diagnosis_20261030/plan.json),
[complete report](../assets/real_metric_diagnosis_20261030/evaluation.json) and
[schema](../assets/real_metric_diagnosis_20261030/evaluation.schema.json).

The metric path remains unsuitable for adoption. The matched-input MI result
also shows that the 4096-point file-order selection cannot be treated as an
innocuous substitute for the historical full-cloud input. Separately, the metric
objective frequently prefers inaccurate poses even on common supported points.
These findings do not identify one exclusive cause or justify another threshold
or optimizer adjustment.

| Method on the same selected input | Perturbed pose recovery | Reference-start controls |
| --- | ---: | ---: |
| Existing MI, new run | 1/12 | 5/6 |
| Bilinear metric, archived paired input | 0/12 | 1/6 |
| Edge metric, archived paired input | 0/12 | 3/6 |
| Edge + rescue, archived paired input | 0/12 | 3/6 |

All 18 MI fits numerically converge. Perturbed median errors are .9374 degrees
and .2307 m; reference-start drift medians are .2500 degrees and .1059 m. The
only perturbed success is 0011-negative. 0005-negative translation .200488 m
and 0011-reference .202427 m remain failures under the .20 m gate.
Historical SGBM MI 6/12 used the full clouds; it is not the score for these selected
inputs. This comparison raises a sampling concern and calls for a full-cloud
control before attributing all real-data failure to the metric method. MI and
metric methods also retain different projection, objective, search and acceptance
rules; matching input files does not isolate a single algorithmic difference.

## Offline cost preference

For each of 54 archived metric fits, compare reference versus fitted pose at the
**same fitted bias**. The common-support column restricts both costs to identical
point indices valid at every diagnostic state. This removes support membership
changes from that comparison without changing any fitting procedure.

| Method (18 fits each) | Fitted cost lower than reference | Of those, inaccurate pose | Lower cost on common support | Of those, inaccurate pose |
| --- | ---: | ---: | ---: | ---: |
| Bilinear | 18 | 17 | 18 | 17 |
| Edge | 17 | 14 | 16 | 13 |
| Edge + rescue | 16 | 13 | 15 | 12 |

Here inaccurate means exceeding rotation or translation accuracy gates, separately
from numerical convergence. Thus cost preference is not explained solely by losing
supported points. It also does not establish global objective minima: fitted bias
is not an independently optimized reference bias, and these are fixed-state probes.
The report additionally includes both poses at zero bias, raw signed residual
quantiles, supported cost and support counts for all four states.

At reference pose with zero bias, only 1414–1737 of 12288 input points are supported
across the three frames (about 11.5–14.1%). Across six unique reference controls,
the median fraction of supported residuals with magnitude >1 m is 22.12% for
bilinear and 22.06% for edge sampling; median negative fractions are 61.22% and
60.81%. Large signed depth disagreement already exists at the reference pose.
Stereo uncertainty, occlusions, alignment and the loss model remain possible
contributors; this diagnostic does not distinguish them or establish a physical
constant bias. No truth-derived masks enter an online solver.

## Every MI outcome

| Case | Rotation (deg) | Translation (m) | Recovered |
| --- | ---: | ---: | --- |
| 2011_09_26_0005-positive | 1.3812 | 0.3204 | False |
| 2011_09_26_0005-reference | 0.2500 | 0.0225 | True |
| 2011_09_26_0005-negative | 0.4411 | 0.2005 | False |
| 2011_09_28_0002-positive | 1.1504 | 0.1941 | False |
| 2011_09_28_0002-reference | 0.2500 | 0.1876 | True |
| 2011_09_28_0002-negative | 0.7969 | 0.1750 | False |
| 2011_09_26_0009-positive | 0.9661 | 0.3955 | False |
| 2011_09_26_0009-reference | 0.0349 | 0.1118 | True |
| 2011_09_26_0009-negative | 0.7384 | 0.3558 | False |
| 2011_09_26_0011-positive | 0.9119 | 0.3288 | False |
| 2011_09_26_0011-reference | 0.1001 | 0.2024 | False |
| 2011_09_26_0011-negative | 0.3072 | 0.1887 | True |
| 2011_09_26_0013-positive | 0.9628 | 0.2519 | False |
| 2011_09_26_0013-reference | 0.2500 | 0.1000 | True |
| 2011_09_26_0013-negative | 1.3657 | 0.2095 | False |
| 2011_09_26_0015-positive | 0.8511 | 0.3659 | False |
| 2011_09_26_0015-reference | 0.2500 | 0.0380 | True |
| 2011_09_26_0015-negative | 1.1788 | 0.1772 | False |

MI used 85–301 evaluations per fit, below 800; total solve time was 8.08 seconds.
This accounting differs from metric residual calls and is not a speed benchmark.

Six focused tests, repository Ruff and strict documentation build pass. Validation
checks all frozen source and input hashes, report schema, 18 MI cases and 54 metric
diagnostics, and reproduction of every archived fitted metric cost (rtol 1e-10,
atol 1e-8). No historical result, core solver or default changes.

Next freeze a full-cloud paired control on these same spent recordings. Reproduce
the historical MI baseline and run the fixed metric candidate without file-order
thinning; retain both seed signs and reference starts. Report supported population
and cost diagnostics alongside accuracy. Do not claim that a synthetic threshold
change fixes the real problem, or spend fresh holdouts before this input-population
confound is resolved. If full-cloud metric recovery still fails while MI recovers,
prioritize a depth-error model that accounts for measurement uncertainty, with an
independent controlled test, rather than strengthening descent on the same cost.
