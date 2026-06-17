# Open3D SLAC Adapter

Calibrex treats Open3D SLAC as an optional backend, not a core dependency.

Run the adapter example:

```bash
calibrex calibrate examples/rgbd_open3d_slac/config.yaml
```

The example dataset manifest declares RGB-D fragments and pose graph metadata.
When `open3d` is not installed, Calibrex reports `open3d_slac_status:
not_executed` in result provenance and emits WARN quality diagnostics. This
keeps `result.yaml`, `report.html`, and evaluation behavior stable across
machines with and without Open3D.
