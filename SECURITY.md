# Security

Please report security issues privately before opening a public issue. Use
GitHub private vulnerability reporting when it is available for this repository;
otherwise contact the repository owner or maintainers directly and avoid
including exploit details in public threads.

## Scope

Security issues include:

- code execution, path traversal, unsafe archive handling, or command injection
- unsafe parsing of calibration, dataset, result, or report artifacts
- accidental exposure of credentials, private dataset paths, or private logs
- supply-chain risks in optional adapters or external tool execution

Calibration quality issues, wrong transforms, weak observability, and failed
evidence checks are safety-relevant but are usually not software security
vulnerabilities. Report those with the calibration failure issue template unless
they also expose a security weakness.

## Robotics Safety

Calibration outputs can affect robot behavior. Treat unreviewed calibration
results as unsafe for deployment until they pass holdout evaluation, protocol
compatibility checks, and operator review.

Dataset-provided calibration is reference evidence, not absolute ground truth
unless the dataset documents independent metrology or synthetic truth.

## Data Handling

Do not attach private rosbag, MCAP, PCD, image, GPS, or vehicle logs to public
issues unless you have verified that redistribution is allowed and sensitive
locations, faces, plates, credentials, and internal paths are removed.

## Supported Versions

slac is currently an alpha project. Security fixes target the default branch
until release branches are created.
