# Why radar-imu judges yaw and not roll or pitch (spent data, 2026-11-11)

The radar-imu audits judged only yaw, the rotation about each radar's `z` axis
([radar-imu](../benchmarks/radar_imu.md)). The unjudged roll came out up to 2.9 deg
(4 std) off the authors' rotation. This note looks for the cause, on spent data only:
- ICINS 2021 (9 bags);
- JGN 2022 (4 bags × 3 radars).

The pseudo ground truth of ICINS is used for diagnosis only, never by the estimator. No
method change came out of it.

## The radar's own z velocity is the limit

Each scan's Doppler ego velocity was compared with the radar velocity predicted from the
pseudo ground truth trajectory (VINS with loop closure, about 4 Hz, spline-differentiated)
and the authors' extrinsic. carried_1, carried_2 and carried_4 give:

| Radar axis | error std (m/s) | measured / predicted slope |
| --- | --- | --- |
| x | 0.040–0.042 | 0.98–1.00 |
| y | 0.036–0.040 | 0.99–1.02 |
| z | 0.089–0.106 | 1.03–1.06 |

- The z component is 2.5 times noisier and reads 3–6 % large.
- The z velocity also varies little in these recordings: its std is 0.10–0.13 m/s, about
  the size of its noise (S/N 1.1–1.2), against 0.17–0.24 m/s on x and y.
- A turn about the radar's x or y axis moves the velocity into z. Roll and pitch are
  therefore read almost only through the noisiest, least-excited component.
- Yaw only mixes x and y, which is why it is judged.

## Three remedies tried, none kept

| Remedy | Result |
| --- | --- |
| A fitted scale on the radar's z velocity, as a nuisance | rejected: it came out 0.16–0.73 (against about 1.05 from the ground truth); the noise drags it toward zero (errors-in-variables attenuation), and roll and pitch got worse, with larger stds |
| Filtering the returns by SNR (3 or 6 dB) or by angle (elevation 45 deg, azimuth 60 deg, as x_rio does) | no effect from SNR (every return passes); the angle filter made every component noisier |
| Weighted least squares with a per-radar-axis noise model, estimated from the fit's own residuals (two passes) | no consistent change; the noise model is sound (z 2–4 times x and y), but the roll and pitch errors moved both ways |

## The errors are systematic per recording

The remaining roll and pitch errors are not noise that more weighting would average away:
- **Across recordings:** the same radar's roll changes by 1–3 deg from one recording to
  the next. For example, JGN's right radar gives +0.9, -2.7, -2.5 and -1.1 deg.
- **Within a recording:** the fold spread is 0.1–0.5 deg.

Something that changes with the recording and is not in the model sets them. Candidates:
- the motion-dependent effect of the z scale;
- multipath;
- the IMU's g-sensitivity.

These data cannot tell them apart.

## What follows

- **The method stays as it is.** Its gating already declines to judge roll and pitch:
  their std or their held-out control keeps them out. That is the honest answer for this
  radar.
- **Judging roll and pitch needs better z data.** Either a radar with better elevation
  (for example a cascaded imaging radar), or motion that moves the rig up and down and
  tilts it while it travels.
- **Recordings like that are a candidate for a new pre-registered audit.** One with
  per-recording ground truth would also tell the systematic error's cause apart.
