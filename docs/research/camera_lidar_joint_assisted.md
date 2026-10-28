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
