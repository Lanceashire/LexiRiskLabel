# Protocol diagnostic publication

This directory publishes reports and dataset-level aggregate evidence for E0/E1
and conditional E2b/E2c/E1b, followed by the six-anchor E2d experiment.

- [Stage report](analysis/阶段实验报告.md)
- [E2d report](e2d/analysis/E2d_数量与构成实验报告.md)
- [E2d validation](e2d/verification.json)
- [Code and reproduction](../../experiments/protocol_next/README.md)
- [Per-stage reporting workflow](../../experiments/protocol_next/REPORTING.md)

Two development seeds per dataset do not establish confirmatory superiority.
E2d does not justify freezing a new BRIDGE candidate, expanding pair search, or
switching datasets solely to seek improvement. E3 is recommended as the next
small CORR mechanism diagnostic; it has not been run in this publication.

Individual-level inputs, IDs, labels, selections and predictions remain local.
Detailed per-branch/per-state logs, intervention/training hashes, and complete
local manifests are also excluded from this publication after automatic approval
review rejected that finer-grained payload. No rejected detailed blob is included
in the published file tree. `PUBLICATION_MANIFEST.json` lists only published code,
reports, dataset-level summaries and validation conclusions with file SHA-256
hashes. Verification summaries describe complete local runs; these aggregate
files alone cannot rerun checks requiring individual predictions.

The repository-root `PUBLICATION_MANIFEST.csv` is the frozen earlier publication
inventory and is not regenerated or republished here. This stage has its own
limited `PUBLICATION_MANIFEST.json`, containing only the files in this directory
and the corresponding diagnostic source code.
