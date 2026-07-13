# Iterative registration correspondence stability research note

## Primary sources and scope

Besl and McKay's ICP alternates closest-point association with rigid transform
estimation. Segal, Haehnel, and Thrun's Generalized-ICP retains iterative
association while replacing the geometric error model with a probabilistic
plane-to-plane formulation.

Primary references:

- Besl and McKay, *A Method for Registration of 3-D Shapes*, PAMI 1992:
  <https://graphics.stanford.edu/courses/cs164-09-spring/Handouts/paper_icp.pdf>
- Segal, Haehnel, and Thrun, *Generalized-ICP*, RSS 2009:
  <https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf>

No implementation code from either source is copied into `src/calibrex`.

## Diagnostic contract

A fixed-correspondence optimum is not sufficient evidence for an iterative
closest-point frontend: changing the transform can change both which queries
remain inside the gate and which target each retained query selects. Calibrex
therefore represents assignments as typed `(query_id, target_id)` pairs and
reports query retention, query-set Jaccard, exact pair Jaccard, and same-target
fraction separately. It also records the exact dropped, added, and reassigned
query IDs. Empty comparisons remain explicitly undefined rather than being
reported as perfect stability, and duplicate query IDs are rejected.

The next integration rematches optimized public TUM factors against their voxel
plane maps and records these diagnostics alongside fixed-versus-rematched RMSE.
