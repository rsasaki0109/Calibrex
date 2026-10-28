# Camera–LiDAR: fixed joint estimator on additional spent recordings

Freeze the joint three-frame assisted-input setup from
[the previous experiment](camera_lidar_joint_assisted.md), then apply it to two
additional already-spent KITTI raw drives: 2011-09-26 0009 and 0011. These are the
next two eligible same-day IDs after the existing 0005 pilot; they were selected
without inspecting images or calibration outcomes. Both appear in the historical
55-drive development inventory. They are new to this experiment, not fresh or
independent evaluation data.

The [plan](../assets/expanded_joint_20261029/plan.json) and
[schema](../assets/expanded_joint_20261029/plan.schema.json) pin source, archive
identity, frames and the unchanged two perturbation signs:

| Drive | Archive frame count | `floor((N-1)*q/4)`, q=1,2,3 |
| --- | ---: | --- |
| 0009 | 447 | 111, 223, 334 |
| 0011 | 233 | 58, 116, 174 |

Acquire only those left/right images and clouds, complete timestamp files and
OXTS samples from frame `f-2` through `f+2` for each selected frame. Retain original
frame IDs and timestamps. Require the selected downloads to fit within 64 MiB
per drive, validate ZIP CRC and SHA-256, and pin S3 size/ETag before payload reads.
Do not interpolate outside local pose coverage or substitute another frame if
preparation fails. The seed-only input directories are transport for the joint
runner, **not** evidence that single-frame calibration ran on these drives.

The only frontend code change supports sparse acquisition: select a cloud by
its exact original KITTI timestamp rather than its ordinal position in the bag.
The old selector returned byte-identical payloads to the new selector for all
six previously evaluated frames. Tests reject missing and duplicate timestamps.
Motion interpolation, stereo settings, solver defaults, seeds, four-CPU ceilings,
180-second timeouts and strict `<0.5 deg / <0.20 m` recovery criteria are unchanged.
The comparison has four new joint cases per method. Require all three native
frames to support the result, and retain failed cases in the denominator.

Report per-case rotation and translation, recovery, and change from the seed.
Do not tune settings, switch seeds or select only successful drives after the
run. The native arm still has stereo depth while Koide does not; both share
OXTS-assisted motion. This is not evidence of monocular superiority. Even after
successful execution, current assisted-input coverage is only 4 of 55 drives,
not an independent-family or fresh-data audit.

## Preparation correction before calibration

Plan `126ce961` acquired the sparse 0009 inputs but stopped before any optimizer
ran: blank entries in the official timestamp file exposed an ordinal-index
assumption in the frontend. The converter retains original line/frame indices,
whereas list indexing skips blank lines. Frame 111 prepared, but the request for
frame 223 did not match its cloud timestamp and was rejected. The failed plan,
receipt and log remain under `preparation_attempt1/`.

The corrected frontend selects both timestamps and clouds by original identity;
it does not shift timestamps or substitute frames. A regression test with a
missing timestamp line verifies that frame 2 stays frame 2 and missing frame 1
fails explicitly. The revised source hashes are frozen before any calibration.
Previously fetched members can be reused only after verifying their receipt
hashes; acquisition records distinguish cached bytes from new downloads. This
is a preparation correction, not an outcome-dependent optimizer retry.

## Completed execution

The corrected preparation was frozen at `9053ed48` before calibration. Both
recordings prepared successfully, and all four joint cases completed for both
methods. Every native fit retained all three frames; Koide loaded all three
image/cloud pairs and exited 0. Input hashes and unchanged native optimizer
options were checked against the preceding experiment. The original container
entrypoint was restored and byte-compared afterward.

| Recording / seed | D2D rotation / translation | D2D recovery | Koide rotation / translation | Koide recovery |
| --- | --- | --- | --- | --- |
| 0009 positive | 0.8124 deg / 0.2569 m | false | 2.2166 deg / 0.1071 m | false |
| 0009 negative | 0.3340 deg / 0.2915 m | false | 3.1732 deg / 0.1191 m | false |
| 0011 positive | 0.3001 deg / 0.1358 m | true | 1.0709 deg / 0.1162 m | false |
| 0011 negative | 0.1601 deg / 0.2751 m | false | 0.2819 deg / 0.0279 m | true |

Each method recovers **1/4** additional cases. D2D improves rotation in all four,
but worsens translation in all four, relative to the unchanged seed (translation
disagreement approximately 0.1225 m). Its new-case median translation disagreement
is **0.2660 m**. Koide improves translation in all four but still misses rotation
in three. Execution completion is not the limitation in this experiment.

Across the four recordings tested so far (eight joint drive/seed cases), D2D
recovers **4/8**, improving rotation **8/8** and translation **1/8**. Koide recovers
**1/8**, improving rotation **2/8** and translation **6/8**. These combined counts
are descriptive, not an independent statistical test. Unequal stereo information
and only four related KITTI recordings prevent a superiority or SOTA claim.

The [complete new matrix](../assets/expanded_joint_20261029/matrix.json),
[summary](../assets/expanded_joint_20261029/summary.json) and
[summary schema](../assets/expanded_joint_20261029/summary.schema.json) retain every
case. The archived preparation receipt includes original member IDs, range
requests, cached/downloaded flags, CRC-validated member hashes and conversion
identity. Per-frame frontend records and per-case logs/full native traces are
included; images, clouds and depth arrays remain in the workspace.

The two sparse selections required 10,501,305 and 10,496,223 range bytes,
respectively (approximately 20.0 MiB total). The second preparation reused all
0009 payloads after receipt-hash verification and downloaded only 0011. No failed
optimizer attempt was retried. No fresh evaluation sensor payload was consumed.

## Consequence and reproduction

The translation regression persists on additional recordings. Treat this as a
limitation of the current assisted-input D2D recovery pipeline, not a reason to
relax thresholds or choose only favorable seeds. More frames alone have not
established translation stability. Keep this frozen candidate and all failures;
a substantive next direction is testing metric geometric constraints for
translation, with rotation and translation scored separately and jointly.
That alternative has not yet been implemented or shown to help. A check on the
stereo-depth/motion assumptions must accompany it; this experiment does not
identify the statistical objective as the sole cause.

Coverage is still only 4/55 drives with three frames and two seed signs each.
The full development matrix, learned competitors, monocular comparison,
independent-family evidence and fresh preregistration remain outstanding.

Reproduce using a new work directory and the same frozen source:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  python tools/prepare_sparse_joint_inputs.py \
  docs/assets/expanded_joint_20261029/plan.json \
  /path/to/new-prepared-inputs /path/to/calibration-root

PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  python tools/run_joint_assisted_experiment.py \
  docs/assets/expanded_joint_20261029/plan.json \
  /path/to/new-prepared-inputs /path/to/existing-data-mount/new-joint-output
```

The calibration root contains `2011_09_26/` with the original camera and
IMU–LiDAR calibration files. The output parent must match the existing container
`/data` mount. Optional `--reuse-raw PREVIOUS_PREPARATION` reuses only member bytes
that match the prior acquisition receipt; it does not reuse calibration outcomes.
