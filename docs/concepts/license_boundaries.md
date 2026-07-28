# License Boundaries

Calibrex core is Apache-2.0. The core package should remain usable without ROS,
Autoware, GPL tools, proprietary SDKs, or large public datasets.

This document is project policy, not legal advice. Check upstream licenses and
terms before redistributing datasets, binaries, containers, or patched external
tools.

## Core Package

Code under `src/calibrex` may depend on permissive Python packages when they are
needed for the core user experience. Prefer Apache-2.0, MIT, BSD, ISC, or similarly
permissive dependencies.

Do not copy GPL or incompatible source code into `src/calibrex`. Do not make GPL
tools required imports for the default package.

## External Tools

External calibration tools are welcome as adapters or baselines when they improve
comparison quality. They should cross the Calibrex boundary through files,
subprocess calls, or containers, not by making Calibrex internal types depend on
their runtime.

For every external baseline, record provenance when available:

- `tool_name`
- `tool_version`
- `source_commit`
- `license_spdx`
- `execution_mode`
- `container_digest`
- `command`
- `adapter_version`
- input and output schema versions

Subprocess or container execution is a practical engineering boundary, but it is
not a blanket legal guarantee. The external tool should remain independently
usable, and Calibrex core should remain fully functional without it.

## Public Datasets

Large public datasets are not committed to this repository. Calibrex keeps
manifests, configs, source URLs, license notes, hashes, and small metadata
fixtures where allowed.

Dataset-provided calibration is reference evidence, not absolute ground truth,
unless the dataset explicitly documents synthetic truth or independent metrology.

Before adding a public dataset example:

- link to the official source and terms
- keep raw sensor data out of the repository unless redistribution is clearly allowed
- document whether any included artifact is cached, recomputed, or metadata-only
- record source, release, sequence, hashes, and download instructions
- avoid uploading private robot, vehicle, GPS, face, plate, or internal-path data

## Optional Extras

Optional integrations should use extras, adapters, or user-installed tools:

```bash
pip install -e ".[dev,open3d]"
pip install -e ".[dev,opencv]"
```

The OpenCV hand-eye comparison uses `opencv-python-headless` through a runtime
import. OpenCV code is not copied into `src/calibrex`; the adapter records the
installed version, upstream source, Apache-2.0 license, input transform
directions, and whether external code executed.

Default installation should remain small and portable. ROS, Autoware, proprietary
SDKs, GPU models, and GPL tools should not be required for `calibrex doctor`,
schema validation, basic reports, or evidence comparison.
