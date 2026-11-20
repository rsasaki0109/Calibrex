---
name: calibrex-check
description: Check whether a robot's deployed sensor calibration still agrees with a recording, using calibrex check, from a bag the user points to. Use when the user asks whether their extrinsics / sensor calibration are still right, wants calibrex run on their bag, or has a bag and no idea which calibrex flags to pass.
---

# Checking a calibration with calibrex

You set up and run `calibrex check` for the user and report what it found. calibrex
judges the calibration; you only prepare its inputs and read its output. Your report must
be as honest as calibrex's verdict: never make a run look better than it was.

## The rules (read first)

You may change the **setup** between runs:

- which topics and pairs run (`--pairs`, `--camera`);
- `--frame-map TOPIC=FRAME` and `--topic-kind TOPIC=wheel|ins`;
- how much of the bag is read (`--max-duration-s`, `--gnss-max-duration-s`);
- which existing calibration files are passed with `--tf`;
- `--vehicle-frame`, when the user confirms the platform is a ground vehicle;
- speed and storage only: `--threads`, `--cache-dir`, `--scan-memory-mb`, `--evidence-dir`.

You must not:

- change a tolerance or estimator mode to get another verdict: `--sigma-k`,
  `--rotation-floor-deg`, `--translation-floor-m`, `--focal-scale-floor`,
  `--detection-probe-deg`, `--rigid-scan-rotation-floor-deg`, any `--*-deskew`,
  `--camera-lidar-*` options. If the user asks for one, run it, and label that run
  as using non-default thresholds;
- give your own diagnosis as calibrex's. When you say which sensor you think is off,
  say it is your reading of the pair verdicts, and quote what calibrex's own notes say;
- edit the calibration being judged, or write transforms yourself. A changed calibration
  is a different candidate: report it as such, never as the user's calibration passing;
- map a topic onto a frame unless that frame is that very sensor. A camera topic mapped
  onto `imu` is wrong even when the run then gets further;
- judge a calibration with `calibrex check` on the same recording `calibrex estimate`
  produced it from. That compares the estimate with itself;
- drop a run from the report. Every run goes in the attempts table, failed or not.

When the verdict is bad, the honest outcome is to report it. Do not search for a setup
that turns it into a pass.

## Steps

1. **Look at the bag.** `calibrex check BAG --plan`. It takes seconds and lists the
   topics, their roles, the frame tree and which pairs can run or why not.
2. **Find the calibration (the candidate).**
   - If the plan shows a `/tf_static` source with frames, that is the candidate.
   - Otherwise look in the user's workspace for files they already use, and ask them
     which one is deployed: a URDF/xacro output, a FAST-LIO / Point-LIO / FAST-LIVO
     config (`extrinsic_R`, `extrinsic_T`), a FAST-LIVO camera file, a Kalibr
     `camchain-imucam.yaml`. Pass each with `--tf` (repeatable). A camera intrinsics
     file (FAST-LIVO camera file, Kalibr camchain) goes with `--tf` too; the camera
     pairs need it when the bag has no CameraInfo.
   - If the user cannot be asked, pick the file the plan accepts and that names the
     bag's topics, and say in the report which file you judged and that the user must
     confirm it is the deployed one. Files that hold no transform (an EKF or driver
     config) are not candidates.
   - Never invent a transform. With nothing to pass, offer two things:
     `--write-frames-template frames.yaml` writes a file naming the bag's frames for the
     user to fill in, and `calibrex estimate BAG --output DIR` estimates a calibration
     (to be checked later on a different recording).
   - Re-run `--plan` with the `--tf` files and read the `next steps` block. An
     `UNMAPPED` topic (often one with no header `frame_id`) means its sensor has no
     transform in the candidate; leave it unmapped unless a tree frame is that sensor,
     and list its pairs as not checked. A topic mapped with `(source_topic_hint)` (the
     candidate file names its topic, as a Kalibr camchain or a FAST-LIO config does) is
     mapped already and needs no `--frame-map`.
3. **First run, short.** `calibrex check BAG --tf ... --pairs <runnable pairs>
   --max-duration-s 120 --output run1.json --html run1.html 2> run1.log`. LiDAR and
   camera pairs take minutes: run it in the background, wait for that process to exit,
   and read the progress in `run1.log` (it shows the time left for the pair). A re-run
   with the same bag and the same options reuses the estimates and takes seconds, except
   camera-lidar; a different `--max-duration-s` computes them again.
4. **Read the result** from `run1.json`, not only from the table:
   - `overall_verdict`, and per pair `status`, `coverage`, `axes` (judged, with
     `candidate_error` and `tolerance`), `unchecked_axes` with their reasons;
   - `fail`: on this recording the candidate disagrees with calibrex's estimate on that
     axis by more than twice the tolerance. It is evidence against the calibration, not
     proof: a clock offset or a bad recording can cause it too, and the notes say so
     when calibrex can tell;
   - `warn`: beyond the tolerance, within twice it;
   - `inconclusive`: the recording could not judge that pair. It says nothing about the
     calibration and is never a pass;
   - `pass (partial: ...)`: only the named axes were judged.
5. **Follow up once, within the rules.** If a pair was inconclusive or skipped, the
   `next steps` block says what would help (a longer window, other motion, a missing
   transform). Do what is possible inside the rules; say what needs a new recording.
   A longer `--max-duration-s` helps only if the recording is longer than the first
   window: check its length first (message count over rate in `--plan`, or
   `ros2 bag info`), and do not re-run when it is not.
6. **Report.**

When two runs judged a pair differently (one window passes it, a longer one is
inconclusive), report both. Do not pick the better one; if you lead with one run, say why
(for example, it read the whole recording) and show the other beside it.

## The report

- First line: calibrex's overall verdict, in its own word (`pass`, `warn`, `fail`,
  `inconclusive`), then what it rests on. "Overall `inconclusive`: four pairs pass, one
  could not be judged" is right; "your calibration looks good" is not, because it
  replaces calibrex's verdict with yours.
- The verdict per pair, in the words above, with the judged axes and their numbers.
- What was not checked, and why.
- The attempts table: one row per run with its command (or the flags that changed),
  why you ran it, and its overall verdict.
- What the user can do next: which recording or file would let calibrex judge more.
- Where `run*.json` and the HTML report are.
