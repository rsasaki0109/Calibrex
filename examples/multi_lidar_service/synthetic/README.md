# Synthetic-only multi-LiDAR service fixture

This fixture exercises digest binding, replacement edge allowlisting, replay
gates, graph connectedness, and the `READY` decision path. It is synthetic
evidence only: the edge and replay JSON files are not physical measurements,
make no accuracy claim, do not certify a vehicle installation, and must not be
treated as an Autoware deployment artifact.

The files are intentionally self-contained and may be used by CI as a
read-only plan/evaluate/verify smoke. The service records lifecycle or
Autoware references when supplied, but never applies them.
