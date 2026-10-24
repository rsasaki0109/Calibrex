# Browser demo backlog

This list tracks the seven demo categories discussed for the browser gallery.
These are basic browser-demo categories, not claims of production calibration
accuracy or support for every method. A projection playground alone does not
complete an algorithm comparison. Implementation and publication are tracked
separately.

**Remaining basic demo categories: 0.** All seven additions are implemented and
published. The limited-coverage / outlier / Huber live-fit controls are also
published. The Camera–IMU clock-offset experiment is also published. Its
rotation × time residual heatmap is also published.
The measured high/low camera-rate window comparison is also published (#317).
Linked stereo correspondence close-ups are also published (#318).
The interactive sparse stereo 3D view is also published (#319).
Extending MEI/DS to all three photographed pairs is a separate addition under validation.

| Demo category | Delivered basic demo | Explicit limits |
|---|---|---|
| Real MEI / Double Sphere fisheye | New native fits to measured corners; real capture 000 rectification, matches and depth | KB4 initialization uses all captures; no independent metric depth truth |
| Camera–LiDAR method comparison | Four ACFR native board methods on one split; separate real A2D2 edge/MI evidence | No cross-dataset ranking or independent extrinsic-accuracy claim |
| GNSS alignment | Real QCOM/u-blox/fused tracks and candidate lever-arm geometry | Fusion is not independent truth; sliders do not estimate a lever arm |
| Radar calibration | Real CAN Radar targets, reported relative speed and candidate yaw | Processed targets, not raw Doppler or a native ego-velocity/extrinsic fit |
| Wheels / INS / vehicle frame | Four measured wheel-speed traces and fused camera FRD velocity axes | Camera frame is not asserted equal to vehicle; scale/time/alignment not separately identified |
| Robot hand–eye | Eight native AX=XB methods on measured ETHZ motion blocks, linked closure origins | Closure is consistency, not absolute accuracy; AX=YB is a further extension |
| Live calibration progress | Actual 6-DoF browser point fit, single-step/run/stop/reset and real residual history | Small measured-board example, separate from the full production solver |

The gallery now has eleven demo tabs. Tabs, algorithm families and
this backlog's categories are different counts. A saved recording, manual
projection or synthetic regression test must not be relabeled as live
optimization or independent accuracy evidence.
