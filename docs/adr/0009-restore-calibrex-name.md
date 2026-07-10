# ADR 0009: Restore the Calibrex Name

## Status

Accepted.

## Context

ADR 0006 renamed the project from Calibrex to `slac` when simultaneous
localization and calibration became central to the framework. The acronym is
technically descriptive, but it is not a distinctive product name and collides
with the established SLAC National Accelerator Laboratory name.

Since ADR 0006, the repository has accumulated versioned result, evidence,
policy, protocol, timeline, trajectory, and bundle schemas under the `slac.*`
namespace. Reusing those version numbers with different identifiers would
break stored artifacts and violate the project's schema-stability rule.

## Decision

Restore **Calibrex** as the product, Python distribution, import package, and
CLI:

- distribution: `calibrex`
- Python package: `calibrex`
- CLI: `calibrex`
- source root: `src/calibrex`

The existing `slac.*` schema IDs, serialized `slac_version` field,
`slac_native` producer value, and versioned protocol/policy IDs remain as the
legacy wire namespace. They describe an existing artifact contract, not the
current product branding. A future `calibrex.*` wire namespace requires a new
schema family plus explicit dual-read migration; it must not overwrite an
existing `slac.*/v0.1` contract.

SLAC remains a technical term for simultaneous localization and calibration,
including Open3D SLAC and related literature. Those algorithm names are not
renamed.

## Consequences

- Python and CLI consumers must migrate from `slac` to `calibrex`.
- Existing schema-valid artifacts remain readable without rewriting or digest
  invalidation.
- Documentation must distinguish Calibrex branding from SLAC algorithms and
  from the legacy wire namespace.
- The GitHub repository is restored to `rsasaki0109/Calibrex`; local remotes
  and canonical project URLs must use that name.
