# KITTI Camera-LiDAR Falsification

This page publishes Calibrex's full-scale KITTI camera-LiDAR falsification benchmark as a
maintained artifact:
[`kitti-camera-lidar-falsification.json`](../assets/kitti-camera-lidar-falsification.json)
(`slac.kitti_lidar_camera_falsification/v0.1`, schema-valid). One frozen protocol scores
two candidates for `T_camera_lidar` on the official KITTI raw drive
`2011_09_26_drive_0005_sync`:

* `dataset_reference`: KITTI's own `calib_velo_to_cam.txt`, which must **pass**;
* `known_bad`: the same transform turned +5 deg about the LiDAR's vertical axis and moved
  +0.25 m along its forward axis (composed on the LiDAR side, so the camera sees a turn
  about its vertical axis and a shift along its optical axis), declared before scoring,
  which must **fail**.

Each trial is a full `calibrex calibrate` run whose evidence bundle (result, evidence,
policy, assessment, observability, protocol, transforms, bundle and verification files)
is bound by SHA-256 in the artifact, as are the 50 camera images and Velodyne scans and
the KITTI calibration files. The data stay outside the repository (KITTI raw, CC BY-NC-SA
3.0).

## Result

{% include-markdown "../assets/kitti-camera-lidar-falsification.md" %}

## What the protocol's pass does and does not show

**The protocol passes, but none of its checks of the images and scans rejects the
known-bad candidate.** The candidate is turned 5 deg and moved 25 cm. On the held-out
frames it scores as well as the reference: an edge alignment of 0.39 against 0.38, and a
depth-edge alignment of 0.72 against 0.74. Every check of the images and scans passes
it, including support, held-out edge alignment and the 1200 perturbation controls.

It fails one check only, *Dataset Reference Consistency*, which compares the candidate
with KITTI's calibration: it is 0.25 m and 5 deg away, beyond 0.1 m and 2 deg. The
decision-boundary gate counts that comparison, so the known-bad fails. Under this
protocol, the falsification therefore rests on knowing the reference, not on the data.
The published test (`tests/integration/test_kitti_falsification_published.py`) pins
this finding, so a protocol change that alters it also has to update this page.

## The data-only decision

The artifact also carries a decision that never looks at the reference: the *Data-only
status* above (`calibrex.evaluation.kitti_edge_falsification`). Each candidate is judged
with the per-axis rule of the bag-level [camera-LiDAR check](camera_lidar_check.md): the
tolerance is the larger of 3 jackknife standard deviations and a floor (0.5 deg, 0.02 m),
and an error beyond twice the tolerance is a `fail`. The decision passes when the
reference passes and the known-bad fails.

* **Rotation** comes from the check's edge alignment, started at the candidate, pooled
  over the frames with a block jackknife and held-out known-bad controls.
* **Translation along the optical axis (`z`)** comes from KITTI's stereo pair, at the
  rotation the edges estimate (`calibrex.evaluation.stereo_lidar_depth`). LiDAR points
  on surfaces that face the camera are block-matched between the rectified images of
  cameras 2 and 3. The disparity residual is fitted as a constant offset of the rig plus
  the candidate's depth error, which fades as `1 / Z^2`; KITTI's rig reads -0.6 to
  -0.8 px, absorbed by the fit. A control moves the candidate 0.1 m along the axis and
  requires the estimate to follow (between half and one and a half of it); on this drive
  it follows by 0.6 to 1.05. This uses the stereo pair's own calibration, never the
  camera-LiDAR one.
* **The order.** Edges see rotation at a translation. When the stereo depth is trusted,
  the edges run again from the candidate with that error along the optical axis
  removed, and judge rotation there. The depth is then measured again at their new
  rotation. Rotation is still judged as the candidate's error, and the trial records
  the shift (`edge_start_depth_shift_m`).
* **`x` and `y` are not judged.**

Edges cannot judge `z` here. In a development run with translation searched, the edge
estimate of `z` ran to its 0.2 m search bound from the reference, and the 5 cm held-out
translation controls were not detected. The reason is the street itself: moving the
candidate along the optical axis keeps a road point on the road and a side-wall point on
its wall, so the shifted cloud still lines up. The same holds for stereo depth when every
point is used: a 0.25 m error then read as about 0.06 m. Points on camera-facing
surfaces (car backs, building fronts; about one in eight matched points) do not have that
freedom; on them, offsets of -0.10, +0.10 and +0.25 m added to the reference read as
-0.09, +0.09 and +0.21 m in a development run.

**The images and scans alone fail the known-bad on both axes it is wrong in, on both
frame selections.** Its 5 deg turn reads as a 4.9 deg pitch error against a 0.5 deg
tolerance (the check names axes in the camera's optical frame, where the vertical axis is
pitch), and its 0.25 m along the optical axis reads as 0.26 m against 0.029 to 0.076 m.

**They pass the reference on both frame selections, so the data-only status is
`pass`.** On the spread frames it passes close to the tolerance: stereo depth puts it
0.050 m along the optical axis against 0.053 m.

The order (depth first, then rotation) was added to let the edges see the known-bad's
pitch on drive 0009 (below). Before it, the edges judged rotation at the candidate's
translation, and on 0005 the reference read `warn` twice:

* on the spread frames, z at 0.0504 m against 0.0496 m;
* on the first 50 frames (5 s of consecutive driving), yaw at 0.50 deg against 0.50 deg.

The data-only status was then `inconclusive`. With the order, the readings barely move
(z 0.050 m, yaw 0.40 deg), but both now fall inside the tolerance. On the first 50
frames the jackknife's yaw standard deviation stays small (blocks a second apart are not
independent), so yaw is judged against the 0.5 deg floor.

### Drive 0009: the same calibration on another drive

Drive `2011_09_26_drive_0009_sync` was recorded on the same day with the same
calibration files (the artifact binds their SHA-256), so its reference and known-bad
are the same transforms. Repeating the decisions there (the second table above,
`--replicate`) separates what belongs to the calibration from what belongs to a drive.

* **The 5 cm reading along the optical axis belongs to drive 0005, not to the
  calibration.** On 0009 stereo depth puts the reference under 1 cm along the optical
  axis on both frame selections. On 0005 it reads 5 cm on both 50 and, in a development
  run, 100 spread frames. The most likely cause is a scene-dependent bias of the stereo
  estimate on 0005, not an error in KITTI's calibration.
* **On the spread frames the decision passes.** Every rotation axis of the reference is
  judged and passes. The known-bad fails on its 0.25 m along the optical axis and on its
  pitch. When the edges started at its translation, its pitch control reached
  t = 1.99, just under 2. Started with the 0.25 m removed, it reaches 2.2.
* **On 0009's first 50 frames (5 s) the known-bad passes.** Roll and yaw are judged and
  pass (they are right). Pitch is not judged, so its 5 deg turn is not corrected, and
  stereo depth at that rotation is not trusted either (jackknife std 0.084 m), so the
  edges are not restarted. Under the check's rule a pass covers only the judged axes,
  so the decision is `fail`: the data-only method passed a candidate that is wrong.

A first replication read 0009 through a pairing bug: KITTI's `timestamps.txt` keeps a
blank line for each missing scan (0009 has none for frames 177-180), and the reader
took each scan's file by its position in the directory, so every pair after the gap
loaded the scan 0.4 s late. With most frames misaligned, the edges judged no rotation on
0009. Files are now taken by frame number, and pairs more than 60 ms apart (the images
whose own scan is missing) are not used.

### Translation across the optical axis: measured, not judged

Edges and camera-facing surfaces leave `x` and `y` unjudged. A joint fit
(`estimate_stereo_alignment`) uses LiDAR points on planar surfaces of any facing.
House fronts along the street see `x`, the road sees `y`. A candidate that puts a point
`d` away from its surface (normal `n`) reads a disparity residual
`base * (n . d) / (Z (n . p))`, linear in a translation error and a small rotation
error, so all six are fitted together with the rig's offset. Each translation axis has a
moved-candidate control. On a rendered street it recovers x, z and rotation; y, which
rests on the near road only, is trusted only when its control follows.

On KITTI it is recorded on every data-only trial as a cross-check (the last table
above) and not judged, because its readings of the same calibration do not agree:

* **x varies more than its uncertainty says.** The reference reads x from -4.6 cm
  (0005, first 50) to +7.3 cm (0009, spread), while each run's jackknife std is 1 to
  3 cm. Judged with the check's rule, that systematic would warn the known-bad on x,
  which it is right in (0009, first 50: 2.7 cm against 2.2 cm).
* **y is untrusted on three of the four runs.** Its control moves the estimate by a
  quarter to a half of the move. On the road, y trades off against a roll of the
  stereo rig.
* **Its z agrees with the judged one.** It reads the known-bad 0.22 to 0.28 m along the
  optical axis.

KITTI's calibration is a reference, not ground truth. These readings do not show it
wrong. The published tests pin them, so a change to either estimator that moves them
also has to update this page.

What was decided after seeing results:

* **The frame selection.** Spread frames are the check's own policy and set the status;
  the first 50 are a sensitivity row. Both were run before one was made primary.
* **The order: depth first, then rotation.** It was added for the known-bad's pitch on
  0009, and it also moved two borderline reference readings on 0005 inside the
  tolerance (above).
* **The occlusion filter of the stereo estimate.** A point with a nearer LiDAR return in
  its matching patch is dropped: it is hidden from the camera, which sits 0.27 m in front
  of the LiDAR, or straddles a depth edge. It was added after a first run in which the
  reference read 0.041 m against 0.052 m (`pass`); with it, the reference reads `warn`.
  The rule and its floors are the check's, unchanged.

The protocol's own status is unchanged: the data-only decision is reported beside it, not
folded into it, so the protocol's finding above stands.

## Reproduce

```bash
python tools/publish_kitti_falsification_benchmark.py \
    --dataset data/public/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync \
    --output-dir outputs/kitti-falsification \
    --replicate data/public/kitti_raw/2011_09_26/2011_09_26_drive_0009_sync
```

This runs both trials (about 40 minutes on 8 cores) and rewrites the two files under
`docs/assets`; `--from-output DIR` republishes a finished run, re-hashing every trial
file first. The opt-in test `test_official_kitti_falsification_benchmark`
(`@pytest.mark.kitti`, `CALIBREX_KITTI_RAW_0005`) runs the same benchmark;
`tests/integration/test_kitti_falsification_published.py` checks the committed artifact
on every CI run: schema-valid, reference pass, known-bad fail, every trial file
digest-bound with a relative path, both data-only rows (the known-bad failing on pitch
and `z`, the reference's `warn` readings), the 0009 replication, and the Markdown above
regenerated byte for byte. Publishing a run made before the data-only decision existed
adds it (about four minutes); each decision records its own git commit.
