# Support

slac is an alpha project. The most useful support requests include commands,
schemas, evidence artifacts, and the limits of the dataset or reference estimate.

## Where To Ask

- Use a bug report for reproducible CLI, schema, parser, report, or packaging
  failures.
- Use a calibration failure report when an estimate, metric, weak direction,
  overlay, or evidence protocol looks wrong.
- Use a feature request for new dataset adapters, metrics, factors, solvers,
  external tool adapters, or report views.
- Use private security reporting for vulnerabilities or accidental exposure of
  sensitive data.

## What To Include

For calibration or evidence issues, include as much of this as possible:

- command lines used to produce the result
- `result.yaml`
- `comparison.json` when comparing estimates
- `summary.json`, `metrics.json`, `evidence.json`, `observability.json`, and
  `degeneracy.json`
- dataset name, sequence, local manifest, and whether raw data can be shared
- whether the reference is synthetic truth, dataset-provided, factory-provided,
  manually measured, or independently measured
- slac version, Python version, OS, and install command

Do not upload private logs or sensor data unless redistribution is allowed and
sensitive content has been removed.

## Scope

slac can help evaluate whether a calibration estimate is supported by the
declared evidence protocol. It does not certify robot safety or metrology-grade
truth. Public dataset calibration should be treated as reference evidence unless
the dataset documents independent metrology or synthetic truth.
