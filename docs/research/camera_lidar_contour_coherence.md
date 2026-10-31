# Camera–LiDAR: neighboring points on the same image contour

**Decision: do not integrate this proposal into pose search.** On the saved pitch
failure, a wrong control still beats the dataset reference in five of six folds.
This screen avoids a full optimization batch for a proposal that fails to reject
known incorrect poses. It establishes no new accuracy or SOTA claim.

## Fixed hypothesis

Independently selecting nearby image ridges permits points to switch between
unrelated boundaries. Test whether requiring nearby depth-boundary points to
match the same image contour suppresses those substitutions.

Before scoring, source `40ae85d4` fixes a mutual four-nearest-neighbor graph over
prepared 3D depth-boundary points, with a 0.75 m maximum link length. Label the
existing image ridge pixels with eight-connected components; propagate those
labels to the nearest pixel within six pixels. At each saved pose, multiply each
point's existing nearest-ridge reward by the fraction of its fixed neighbors
matching the same nonzero component. A point with no neighbors keeps its original
reward. Unknown matches count as disagreements for linked points. Preserve all
point weights and the original score denominator.

This is a local consistency test, not semantic correspondence. Image components
can merge different physical boundaries or fragment one boundary. Three-dimensional
neighbors can also cross surfaces. Rounded pixels supply component identity;
the existing reward continues to use bilinear interpolation.

The screen uses every saved reference, baseline and signed nuisance-refitted
control from all six folds and eleven prepared frames of 0002 pitch +3. There
is no new fitting, frame selection or threshold tuning. All original per-frame
scores and pooled fold scores reproduce within absolute 1e-9. Four focused tests
cover graph construction, partial agreement, unknown labels, distance limits,
isolated-point fallback and exact original contribution accounting.

The decision rule was fixed in the source before execution: if a saved control
still beats the reference in at least five folds, do not integrate. Passing this
screen would only justify a full paired development experiment, not adoption.

## Results

The graph covers 6,887 of 7,401 prepared points (93.05%). Its failure is therefore
not explained by most points receiving no constraint.

| Diagnostic | Nearest ridge | With contour consistency |
| --- | ---: | ---: |
| Folds where a saved control beats the reference | 5/6 | 5/6 |
| Folds where a saved control beats the fitted baseline | 4/6 | 4/6 |
| Frames where a saved control beats the reference | 10/11 | 8/11 |

The following scores retain both controls, including failures:

| Fold | Reference | Baseline | Minus-pitch control | Plus-pitch control |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.3110 | 0.2410 | 0.2035 | 0.2302 |
| 1 | 0.2795 | 0.3323 | 0.3066 | 0.2984 |
| 2 | 0.3529 | 0.3562 | 0.3737 | 0.3636 |
| 3 | 0.3283 | 0.3329 | 0.3206 | 0.3439 |
| 4 | 0.3405 | 0.3124 | 0.2977 | 0.3736 |
| 5 | 0.2903 | 0.2957 | 0.2421 | 0.3014 |

The penalty narrows several wrong-pose score advantages but does not reverse the
pooled preference in enough folds. In fold 3 the preferred signed control changes;
the plus-pitch control still has 4.478 deg full rotation disagreement and beats
the reference. The retained scores show that merely sharing a connected contour
is not sufficient to establish correspondence identity.

The next structural hypothesis should check the geometry of neighboring matches
(direction, separation and collapse), not only their component labels. That is
untested here. Do not claim that all structural matching methods have failed,
or that tuning these constants would produce independent validation.

[Complete scores and graph support](../assets/contour_coherence_20261028/report.json)
and the [hash manifest](../assets/contour_coherence_20261028/manifest.json) retain
all poses and outcomes. Run `tools/kitti_camera_lidar_contour_coherence.py` with
the reacquired snapshot, the previous pitch-match diagnostic receipt and a new
`--output` directory. Native estimator defaults and earlier audit verdicts remain
unchanged.
