# Interrupted measurement cohort

The comparison was stopped after its first prior worker completed and its
first current worker started. Review found that the planned history comparison
included the independently verified abbreviated Git source ID. The saved-result
work replay also needed to accept omitted optional known-bad controls.

Both audit issues were fixed with regression tests. This entire incomplete
cohort, including the completed worker and any partial current-worker files,
is preserved and excluded from the final six-run timing statistics. No missing
timing or result is invented. The exact launcher, schemas, inputs, original raw
data, complete source capsules, protocol, attempt and interruption receipt
remain available. Both core source manifests and problem bytes are identical
between this cohort and the final one; only measurement/audit handling changed.
