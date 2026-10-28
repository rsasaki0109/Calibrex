# Camera–LiDAR: joint three-frame assisted-input experiment

Freeze the same three frames per spent recording, both seed signs, frontend
artifacts, default optimizer settings and strict `<0.5 deg / <0.20 m` event as
the [single-frame matrix](camera_lidar_assisted_matrix.md). The only intended
algorithmic change is optimizing one transform over all three observations.
Two recordings times two seeds gives **four joint cases per method**. Do not
compare that denominator directly with the earlier twelve single-frame cases.

The [plan](../assets/joint_assisted_20261028/plan.json) pins the frame inventory,
perturbations and native source. The existing D2D solver averages frame MI, with
its normal observation-derived histogram ranges now resolved over the three
frames. Koide receives all three PNG/PLY names in its native JSON input and uses
its unchanged multi-frame objective. Clouds are not concatenated into one scan;
each retains its matching camera-time coordinate frame and image.

Before execution, verify every reused PNG/PLY/depth hash against its previous
frontend report and check common camera intrinsics and identical seed/reference
across the three single-frame cases. No data download, frame selection, parameter
search or outcome-based retry is part of this experiment.

The primary diagnostic is joint translation disagreement compared with the
median of the three corresponding single-frame outputs, separately for each
drive/seed/method. Also report rotation, the absolute recovery event, and whether
translation improves over the unoptimized seed. Four correlated cases cannot
establish superiority; a worse or unchanged result remains published. Crashes,
timeouts, insufficient support and invalid output stay failures. Require all
three D2D frames to support the selected solution; do not silently reduce the fit.

Use the previous four-CPU ceiling and 180-second timeout per method/case. There
is no runtime-superiority claim. The unequal stereo input, auxiliary OXTS motion,
modeled point times, only 2/55 development recordings and outstanding learned /
independent-family / fresh-audit requirements are unchanged.

## Result

The plan and both drivers were committed at `752c1f11` before execution. All
four cases completed for both methods; all native fits retained three supporting
frames, and the official Koide logs confirm loading all three named image/cloud
pairs. There were no invalid transforms, timeouts or missing cases. The runtime's
original entrypoint was restored and byte-compared afterward.

| Recording / seed | D2D rotation | D2D translation | D2D recovery | Koide rotation / translation |
| --- | ---: | ---: | --- | --- |
| 0005 positive | 0.8406 deg | 0.2773 m | false | 1.7916 deg / 0.1170 m |
| 0005 negative | 0.2416 deg | 0.1288 m | true | 2.0056 deg / 0.1521 m |
| 0002 positive | 0.1652 deg | 0.1064 m | true | 3.2789 deg / 0.1520 m |
| 0002 negative | 0.1168 deg | 0.1821 m | true | 2.5428 deg / 0.1154 m |

Joint D2D has **3/4** scoped recoveries and Koide **0/4**. This is not a direct
success-rate improvement over the earlier 5/12: there are four combined estimates
instead of twelve separate estimates, on the same correlated data.

The predeclared primary comparison is translation against each matching
single-frame trio's median (negative delta means improvement):

| Recording / seed | D2D translation delta | Koide translation delta |
| --- | ---: | ---: |
| 0005 positive | +0.0074 m | -0.0043 m |
| 0005 negative | -0.0657 m | +0.0061 m |
| 0002 positive | +0.0301 m | +0.0153 m |
| 0002 negative | +0.0103 m | -0.0090 m |

**The translation-stability hypothesis is not supported by this diagnostic.**
D2D improves translation over the corresponding median in only 1/4 cases and
worsens it in 3/4. Only 0002 positive improves translation over the unoptimized
seed. Three absolute recoveries do not erase this negative paired result.
Koide improves over its single-frame median in 2/4 but still fails all four
rotation thresholds. These statements concern this assisted-input setup, not
the methods' general performance or monocular superiority.

The [complete results](../assets/joint_assisted_20261028/matrix.json) and
[paired summary](../assets/joint_assisted_20261028/summary.json) retain every case;
the [summary schema](../assets/joint_assisted_20261028/summary.schema.json) validates
the descriptive comparison. Each case includes source-to-copied-file hashes,
unaltered frontend configuration, seed, complete D2D trace, Koide log/output and
input provenance. The publisher verifies the exact four-case inventory, committed
source identities, input hashes and all-three-frame native support. Arrays remain
in the workspace; no fresh sensor data was downloaded or read.

## Reproduction and next decision

With the previous matrix's local input arrays still available, from the
repository root:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  python tools/run_joint_assisted_experiment.py \
  docs/assets/joint_assisted_20261028/plan.json \
  /path/to/matrix-20261028 /path/to/existing-data-mount/new-joint-directory
```

The output's parent must be the existing container's `/data` bind mount. A new
output directory is required; the previous matrix is never overwritten.
The native driver can separately replay a prepared three-frame bundle via
`tools/run_joint_stereo_d2d.py BUNDLE SEED OUTPUT`.

Retain joint estimation as an experimental candidate, with no native default
promotion or revised success threshold. Freeze this configuration and broaden
spent-drive coverage before deciding whether it is a useful recovery method.
Do not select only negative seeds or tune another two-drive variant. Keep both
absolute recovery and the paired translation-regression diagnostic in the next
report; the latter prevents rotation gains from hiding translation damage.
This experiment does not waive the existing 55-drive, monocular/learned baseline,
independent-family or fresh-audit requirements.
