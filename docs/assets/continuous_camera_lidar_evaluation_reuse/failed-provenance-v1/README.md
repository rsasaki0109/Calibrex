# Initial provenance check failure

The initial prior worker exited with status 1 because its result correctly
stored the core API's abbreviated Git ID, while the original measurement
launcher incorrectly required equality with the full 40-character ID. It saved
no result or API timing. This incomplete run contributes no successful timing
to the comparison. Its whole-process duration, error, exact launcher/schema,
input/generation files and complete source capsules/manifests are retained.

The corrected launcher reads and verifies both actual Git forms before solving,
then checks the result's unchanged abbreviated provenance. Eight regression
cases exercise accepted abbreviations and wrong/missing source identities.
Both variants' complete core source manifests are byte-identical between this
failed cohort and the final six-run cohort. The input is also byte-identical.
The correction changes measurement/provenance handling only.
