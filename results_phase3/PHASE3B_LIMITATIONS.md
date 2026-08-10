# Phase 3B Limitations

- The external validation is one LendingClub snapshot and does not establish universal generalization.
- The 20-seed paired tests have limited power for very small effect sizes.
- PPDai provenance contains ambiguous outcome-adjacent fields; its primary result is retained but labeled exploratory.
- GPU acceleration changes execution time, not the frozen policy definition; parity checks are required for any future hardware/backend change.
- Legacy partial/CPU-fallback LendingClub runs are excluded from all final tables.