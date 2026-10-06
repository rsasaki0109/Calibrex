# What makes a lever arm observable

`calibrex check` and `calibrex estimate` often return the rotation of a pair but
leave its translation (the lever arm) `unobservable`: the exported frame then
carries a `--tf` prior marked `NOT MEASURED`. This page explains why, what the
tools now print about it, and how well that diagnosis predicts what happens when
you record more. Everything here is tool validation on recordings that were
already on disk, not a SOTA claim.

## Why a translation axis is unobservable

Both lever-arm estimators see the lever arm `l` only through the **rotation
between two poses of the same rigid body**:

```text
(R_j - R_i) l  =  R_i (Exp(theta) - I) l,      theta = Log(R_i^T R_j)
```

* **IMU lever arm (imu-lidar).** The accelerometer sits at `l` from the LiDAR, so
  its double-integrated specific force differs from the LiDAR odometry's
  displacement by `(R_k - R_0) l` over a segment (plus gravity, velocity and
  accelerometer bias, which are nuisances and fitted per segment).
* **GNSS lever arm (gnss-lidar).** The antenna is at `l` from the LiDAR, so the
  antenna displacement between two epochs is the LiDAR displacement plus
  `(R_j - R_i) l` (the alignment of each odometry window to ENU is profiled out).

To first order `(Exp(theta) - I) e_a = theta x e_a`, whose length is the part of
`theta` **perpendicular to axis `a`**. Per axis `a` the information on `l_a`
before nuisances is therefore

```text
I_a  =  sum_k |(R_j - R_i) e_a|^2 / sigma_k^2   ~   sum_k |theta_k,perp(a)|^2 / sigma_k^2
```

with a consequence you can read off without running anything:

| Motion in the recording | Axes it makes observable |
| --- | --- |
| pure translation (no rotation) | none |
| rotation about one axis only (a level vehicle turning, a pole spun about its length) | the two axes perpendicular to it; **not the axis it rotates about** |
| rotation about two or three axes (hand-held tilting and turning) | all three, in proportion to the rotation about the other two |

So a **planar ground vehicle** can observe the horizontal components of a GNSS
lever arm from its turns but never the vertical one (it does not roll or pitch
enough); the vertical component has to be measured or given as a prior. A
**hand-held** rig can observe all three, but z needs roll and pitch, which are
typically small next to the walking person's turning.

Most of the rotation information does not survive the nuisance parameters. The
share that does (`information_efficiency` in the artifact) was 1 to 5 % for the
accelerometer lever arm on the validation recordings (a rotation looks like a
change of gravity direction and of velocity over the 2 s segments) and 30 to 75 %
for the GNSS lever arm (only the per-window alignment competes). This is why a
hand-held accelerometer lever arm takes minutes of data while a few seconds of
tilting look like plenty.

## What the tools say

For every translation axis the estimator derives, from the information matrix it
already builds and the noise level it estimates from its own residuals:

* the data-only (analytic) std and the larger of the analytic and jackknife std
  (`scaling_std_m`), which is the one that shrinks with more data;
* the RMS rotation the recording supplied about the sensor x, y and z axes and
  about the two axes perpendicular to this one;
* the information efficiency above;
* the recording needed to reach the bound for the same motion,
  `needed_duration_s = recording_s * (scaling_std / bound)^2` (the std of the
  same motion shrinks as one over the square root of time); and
* a `cause`:

| `cause` | Meaning | What to do |
| --- | --- | --- |
| `observable` | the reported std is within the bound | nothing |
| `insufficient_duration` | the motion excites the axis; `needed_duration_s` (at most one hour) of it would reach the bound | record longer |
| `no_excitation` | repeating this motion would take more than an hour; `missing_rotation_axes` names the rotation to add | change the motion, or measure and use `--tf` |
| `model_limited` | the data alone would reach the bound but the reported std also covers modelling error (the segment-duration sensitivity) | more of the same motion may not help |
| `unsolved` | too few odometry windows to solve | see the pair's skip reason |

It is printed in `next steps` and stored in the artifacts (see below):

```text
next steps:
  - imu-lidar lever arm y: the motion excites it (4 deg RMS rotation about the other axes) but
    too briefly: std 2.96 cm against the 1 cm bound. About 16 min of the same motion would reach
    it (this recording: 2 min). Caution: the segment-duration sensitivity 2.77 cm (modelling
    error) is also over the bound and may not shrink with more of the same motion.
```

For `no_excitation` the text also sizes a reference motion: it assumes a ramp at
30 deg/s over the estimator's pairing interval, the efficiency measured on the
recording and the same noise inflation the recording's jackknife shows, and says
how long of it would supply the missing information. On `stadtgarten_seq2` the
GNSS z axis (std 4 cm, about 5 300 s of the recorded motion needed) reads:

```text
  - gnss-lidar lever arm z: needs rotation about the sensor x or y axis (for a level rig,
    tilting for x or y and turning for z). The recording rotates only 6/4/12 deg RMS about
    x/y/z. Model estimate: about 4 min of it at >= 30 deg/s would reach the 1 cm bound.
```

The model estimate is a model, not an outcome (see the limits below). When one
axis is the only one the recording rotates about, the text says the motion is
planar and that the axis has to be measured and given with `--tf`. When fewer than
12 windows were fitted the extrapolation is marked "indicative only".

## Schema

An optional `excitation` object (`AxisExcitation` in `calibrex.core.excitation`) was
added to

* `slac.imu_lidar_translation` axes and `slac.gnss_lidar_lever_arm` DoFs (x, y, z),
* `slac.calibration_check` `unchecked_axes`, and
* `slac.bag_estimate` axes.

It is optional everywhere, so existing artifacts stay valid
(`tools/check_schema_compat.py` reports the three additions to released schemas as
compatible; `slac.bag_estimate` is not released yet); the other
pairs (vehicle pairs, composed gnss-imu) do not carry it.

## Validation on real data

The predictions were compared with the estimators' own empirical scatter on
recordings already on disk (nothing downloaded). `tools/validate_translation_observability.py`
runs the LiDAR odometry once per recording, then evaluates time prefixes of the
odometry windows (what `--max-duration-s` and `--gnss-max-duration-s` analyse) and
random subsets of the windows (the empirical std is the spread of the subset
estimates, with the finite-population correction `1 / sqrt(1 - m / N)`).

| Recording | Estimator | Windows | Rotation RMS about x / y / z |
| --- | --- | ---: | --- |
| RTK-SLAM `stadtgarten_seq2`, first 600 s (hand-held MID360, gyro deskew) | imu-lidar | 60 | 9 / 4 / 17 deg |
| Koide `indoor_easy_01` (136 s, rigid scans) | imu-lidar | 24 | 4 / 13 / 5 deg |
| RTK-SLAM `stadtgarten_seq2`, 900 s span (RTK-fixed windows, 329 s of data) | gnss-lidar | 38 | 6 / 4 / 12 deg |
| RTK-SLAM `construction_seq1`, 300 s span (156 s of data) | gnss-lidar | 17 | 5 / 3 / 11 deg |

(The imu-lidar runs use the rotation, clock offset and gyro bias of an earlier
`check` run of the same recording as a fixed input for every prefix; the
estimator would refit them per prefix. Window counts are those of the longest
prefix; GNSS data seconds are the RTK-fixed window time, not wall time. The
Koide prefix runs reproduce the CLI's reported stds to within 15 %.)

### Predicted against empirical std

The analytic (information matrix) std is what the estimator can predict from one
run. Against the subset spread it is **optimistic by a factor of 2 to 6.5**,
because odometry residuals are correlated in time, not white:

| Recording | Data (s) | Empirical / predicted, x | y | z |
| --- | --- | ---: | ---: | ---: |
| imu-lidar `stadtgarten_seq2` | 59 / 119 / 237 / 297 | 2.8 / 3.1 / 3.1 / 3.2 | 2.7 / 2.6 / 3.1 / 2.7 | 1.9 / 2.3 / 2.5 / 2.5 |
| imu-lidar Koide `indoor_easy_01` | 13 / 27 / 54 | 3.1 / 2.8 / 3.2 | 2.4 / 2.7 / 2.7 | 3.4 / 3.5 / 3.6 |
| gnss-lidar `stadtgarten_seq2` | 46 / 91 / 174 | 6.0 / 3.9 / 3.7 | 4.0 / 3.1 / 2.5 | 4.3 / 6.5 / 5.0 |
| gnss-lidar `construction_seq1` | 39 / 77 | 4.1 / 4.6 | 6.5 / 4.5 | 3.2 / 3.3 |

The factor is steady within an estimator (about 3 for imu-lidar) and the empirical
std falls with the square root of time: for `stadtgarten_seq2` imu-lidar the fitted
exponent is -0.46 / -0.50 / -0.36 for x / y / z (-0.5 expected). That is why the
duration is extrapolated from the larger of the analytic and the jackknife std,
which already contains the factor, and not from the analytic std alone.

### Does the prediction hold when the recording is extended?

The std at a longer duration, predicted from a short prefix as
`scaling_std * sqrt(T0 / T)`, divided by the empirical subset std and by the std
the estimator reports when run on the longer prefix:

| Recording, prefix T0 | vs empirical subsets (x / y / z) | vs the estimator on longer prefixes (x / y / z) |
| --- | --- | --- |
| imu-lidar seq2, 60 s | 1.46-1.49 / 0.63-0.74 / 0.90-0.98 | 1.10-1.62 / 0.37-0.97 / 0.89-1.36 |
| imu-lidar Koide, 40 s | 2.2-2.5 / 1.7 / 5.6-5.9 | 1.65-2.81 / 1.02-2.23 / 2.63-5.97 |
| gnss-lidar seq2, 100 s | 0.87-0.95 / 0.49-0.64 / 0.28-0.38 | 0.78-1.18 / 0.74-1.17 / 1.18-1.20, then 0.23-0.28 from the 400 s prefix on |
| gnss-lidar construction, 60 s | 0.78 / 1.24 / 0.90 | 0.56-1.28 / 0.66-2.76 / 0.68-0.92 |

Values above 1 mean the prediction was conservative. The prediction is within a
factor of 3 almost everywhere, with these failures:

* **gnss-lidar z on `stadtgarten_seq2`:** the jackknife std of z jumped from 13 to
  55 mm when the 400 s prefix added six windows, so the prediction made at 100 s
  (21 mm, falling as 1/sqrt(T)) was 4 times too optimistic afterwards. The new windows made z
  a poorly determined, heavy-tailed axis (the recording rotates 5 deg RMS about the axes z needs).
* **imu-lidar y on `stadtgarten_seq2`:** the jackknife std stops falling at about
  5 mm, so predictions made at 60 s are 2.7 times too optimistic by 600 s.
* **Koide, 40 s prefix:** a jackknife over 4 windows is useless; z was predicted to
  need 14 000 s (it is about 520 s by the empirical fit). Hence the "indicative
  only" mark below 12 windows.

### Does the predicted duration reach the bound?

`needed_duration_s` predicted from a short prefix, divided by the data seconds
at which the fitted empirical std crosses the same bound (the fit is a power law
over the subset sizes; 10 mm is the estimators' translation bound, 5 mm a
tighter one that the recordings only reach late):

| Recording, prefix T0 | Bound | Predicted / observed (x / y / z) |
| --- | --- | --- |
| imu-lidar seq2, 60 s | 10 mm | 2.50 / 0.49 / 1.31 |
| imu-lidar seq2, 60 s | 5 mm | 2.18 / 0.48 / 0.78 |
| imu-lidar Koide, prefixes of 60 to 136 s | 10 mm | x 0.62-1.8, y 0.35-1.7, z 0.75-3.9 (12 values, 11 within 3x) |
| gnss-lidar seq2, 100 s | 10 mm | 1.65 / 0.33 / **0.09** |
| gnss-lidar seq2, 400 s | 10 mm | 2.70 / 0.25 / 1.47 |
| gnss-lidar construction, 60 s | 10 mm | **n/a** (x fit from two points) / 5.5 / 0.64 |

So the predicted duration is good to about a factor of 3 once 12 or more windows
are fitted, with the misses listed above (z of `stadtgarten_seq2` at 100 s
predicted 416 s; the 400 s prefix, with more data, said 7 000 s and the empirical
fit says 4 800 s). On `stadtgarten_seq2`, which the estimator reports as x
9.9 mm, y 5.0 mm, z 40 mm at 900 s, the diagnosis says: y observable, x observable
only just (predicted from the 100 s prefix: 451 s of data, reached at 329 s),
z `no_excitation` (needs about 5 300 s of the same motion): the antenna's vertical
offset needs roll and pitch that a hand-held pole walk does not supply.

### Limits of this validation

* Subsets of one recording share its low-frequency odometry errors, so the
  empirical std is a lower bound on the total error. This is exactly what the
  segment-duration sensitivity of imu-lidar adds: on Koide it is 2.4 to 3.8 cm and
  does not fall with more windows, so `model_limited` there is the real
  limit and the `Caution` in the hint says so. On `stadtgarten_seq2` it fell from
  20 to about 8 mm and stayed over the 1 cm bound for z (`model_limited` at 600 s).
* Four recordings, two sensors, two LiDAR setups (a Livox MID360 and the Koide
  recording's rigid-scan cloud). Driving sequences with planar motion were not run
  through the new analysis; the planar behaviour is covered by the analytic unit
  tests (yaw-only motion gives no information on the rotated-about axis).
* The recommended reference motion (30 deg/s ramp) is a model, not a measured
  outcome: no recording on disk adds that motion to test it. It is sized from the
  efficiency measured on the recording.
* The std-per-time law assumes the added data repeat the recorded motion. Different
  motion changes the information, which is the point of a `no_excitation` verdict.

Reproduce (one heavy job at a time; the bags are not shipped):

```bash
python tools/validate_translation_observability.py collect imu-lidar \
  --bag ros2/stadtgarten_seq2 --lidar-topic /livox/points --imu-topic /livox/imu \
  --rotation cached_rotation.yaml --max-seconds 600 --work seq2_imu.pkl
python tools/validate_translation_observability.py analyze imu-lidar --work seq2_imu.pkl \
  --durations 40,60,80,100,120,150,180,240,300,360,480,600 --subsets 6,12,24,30 --out seq2_imu.json
python tools/validate_translation_observability.py report --predict-from 60 seq2_imu=seq2_imu.json
```
