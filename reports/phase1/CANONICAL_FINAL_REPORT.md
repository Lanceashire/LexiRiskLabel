# Phase 1 final closeout report

## Gate result

- `PHASE1_FINAL_STATUS = PASS_WITH_EVALUATION_DISTORTION_FINDING`
- `FORMAL_PHASE1_FROZEN = YES`
- `CLOSEOUT_ADDENDUM_COMPLETE = YES`
- `READY_FOR_PHASE2 = YES` (gate only; Phase 2 was not run)

## Frozen formal experiment

The formal Phase 1 remains the two-dataset, five-treatment, 20-seed, 10-round experiment in `uci_all_rounds.csv` and `hmeq_all_rounds.csv`. The integrity audit and all 20 automated tests pass. Its fixed external test metrics remain authoritative.

The main pattern is dataset-dependent: HMEQ shows a clearer Oracle advantage in recall/BA and stronger calibration/feedback-health separation; UCI shows much smaller downstream classification gaps despite feedback-health differences. Random exploration improves observability but does not guarantee recovery of every downstream metric.

## Original hypothesis decisions

- H1 selective-label feedback distorts visible TDR/FDRatio: supported on both datasets.
- H2 default-class recall/BA may degrade versus Oracle: dataset-dependent; supported on HMEQ, not UCI.
- H3 accuracy/AUC may remain stable while recall/TDR degrade: supported as a mechanism pattern.
- H4 calibration may deteriorate without ranking collapse: mixed and dataset-dependent.
- H5 simple random exploration may recover part of the Oracle gap: supported for feedback health, weak or dataset-dependent downstream recovery.

## Post-hoc Findings

The dual evaluator is an analysis-only post-hoc addendum, not a new H-number. The fixed external evaluator exactly reproduces the frozen formal results; the internal visible-only evaluator produces systematic evaluation distortion in both datasets. See `PHASE1_ADDENDUM_DUAL_EVALUATION.md`.
## Closeout work completed

- Hidden-information summaries reconstructed from the original all-rounds outputs.
- Aggregate recovery ratios corrected while retaining the previous per-seed field as `legacy_per_seed_ratio_mean`.
- Calibration extended statistics added with valid counts, dispersion, bootstrap intervals, and multiplicity correction.
- Dual evaluator addendum completed for Oracle/NoExplore. External parity is exact; internal visible evaluation is diagnostically distorted.
- Figures and source/provenance records packaged.

## Reproducibility

Run `pytest -q D:/ecomic/tests` and inspect `PHASE1_INTEGRITY_AUDIT.md`, `PHASE1_ADDENDUM_DUAL_EVALUATION.md`, and the CSV outputs in this directory. The final archive is `D:/ecomic/risk_budget_exploration_PHASE1_FINAL.zip`.

## Stop condition

This closeout stops at Phase 1. No Phase 2 or additional dataset was started automatically.


