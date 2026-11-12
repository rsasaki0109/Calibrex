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

## An imaging radar: Snail-Radar (Oculii Eagle G7)

Tested on 2026-11-12, on development data only. Snail-Radar (Huai et al., IJRR; Open
Database License) records an Oculii Eagle G7 4D imaging radar and a Bynav X36D IMU
(100 Hz) on three platforms. Three sequences were spent as development data, chosen before
any of them was read:

| Sequence | Platform | Length | Roll std / error | Pitch std / error | Yaw std / error | Judged |
| --- | --- | --- | --- | --- | --- | --- |
| 20230921/data4 | hand-held | 622 s | 0.63 / -2.0 deg | 0.53 / +0.4 deg | 1.26 / -3.0 deg | none |
| 20231019/data2 | e-bike | 533 s | 1.35 / -1.3 deg | 1.11 / -0.2 deg | 0.86 / +0.1 deg | none |
| 20240113/data5 | SUV | 1509 s | 0.32 / -1.4 deg | 0.28 / +1.4 deg | 0.37 / -1.3 deg | none (stds pass, held-out 1 deg controls not detected) |

The errors are to the authors' Oculii-to-X36D rotation. The authors fitted that rotation
on 20240113/data5 from the radar's ego velocity and the X36D's INS solution. The estimator
is the package's, unchanged.

- **The imaging radar's z velocity is no better.**
  - Measured against the IMU's 1 s velocity changes, the radar's vertical part is
    uncorrelated (r = -0.04 to 0.09).
  - Its RMS difference is 0.26-0.57 m/s, against a vertical excitation of
    0.04-0.14 m/s (S/N 0.1-0.5, against about 1.1 for the TI radars).
  - Against the INS velocity on the SUV, the radar's z follows with a slope of 0.05, a
    residual of 0.27 m/s and a bias of -0.15 m/s.
  - Horizontally the radar is good: r = 0.85-0.96 against the IMU on the e-bike and
    the SUV.
- **Why:** the Eagle's Doppler is quantised in steps of 0.136 m/s, and the returns span
  only about 4.5 deg of elevation in a scan. Its formal z std (0.036 m/s per scan) is
  far from the observed scatter.
- **A z bias proportional to speed looks like a pitch turn.** The two cannot be told
  apart from velocities alone. On the SUV the package's fit and a direct alignment to the
  INS velocity agree on pitch (+1.3 and +1.1 deg from the authors' rotation) but not on
  roll (-1.5 and -2.6 deg) or yaw (-1.2 and -0.1 deg). The clock offset ran to the
  search bound (-100 ms; -157 ms with a 0.5 s bound, std 33 ms), although the dataset is
  synchronised to about 10 ms.
- **The check stays honest here.** No axis would be judged on any of the three, so the
  authors' rotation would not be failed. It also checks nothing.

**Conclusion:** an imaging radar with this elevation behaviour does not make roll and
pitch judgeable. No pre-registered evaluation was run on the unseen Snail-Radar sequences:
with no development sequence judging roll or pitch, its gates would be zero. The other
41 sequences stay unseen.

What was read:
- the three sequences' Eagle point clouds (`eagleg7/pcl`), X36D IMU and `tf_static.launch`,
  taken out of the per-sequence zips over HTTP range requests;
- their rows of the INS reference (`utm50r_T_x36dimu.txt`), used for diagnosis only.

The reference archive (`ref_trajs.zip`) and the time-correction tables were downloaded
whole, but only these three sequences' entries were opened.

One bug was found on the way. The X36D repeats about 1 in 500 IMU stamps, and that
stopped the estimator (attitude interpolation needs increasing times). A repeated stamp is
now kept once.
