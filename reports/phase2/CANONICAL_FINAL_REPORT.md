# PHASE 2 FINAL CONCLUSION

## Status

`PHASE2_FINAL_STATUS = PASS`

The formal Phase 2 model runs were not changed or rerun during closeout. All corrections below are analysis-only.

## Strongly supported

- Feedback recovery and downstream performance recovery are distinct mechanisms (`H-DECOUPLING = SUPPORTED`).
- Risk-budgeted policies improve feedback observability relative to NoExplore, but this does not imply proportional Recall, BA, or calibration recovery.

## Provisionally supported

- `H-QUANTITY = PROVISIONALLY_SUPPORTED`: CountOnly-MinRisk remains competitive and often occupies a favorable risk-performance/Pareto position.
- `PROVISIONAL_OUTCOME = B/C_MIXED`: simple risk-aware acquisition captures much of the observed value, while quantity-first safe acquisition is highly competitive.

## Not supported

- `H-FAVE = NOT_SUPPORTED`: no stable downstream superiority of FAVE-v2.1 over CountOnly-MinRisk and RiskTilted-Uncertainty.
- `H-SCARCITY = NOT_SUPPORTED`: no consistent stronger complex-policy advantage at the two low budgets.

## Dataset-dependent / weak findings

- `H-POLICY = WEAK / MIXED`: utility distinctions are small and endpoint-dependent. In UCI unit-exposure risk (`EL=p`), CountOnly-MinRisk, LeastRisky, and Uncertainty are mathematically equivalent on the rejected region and the equivalence audit verifies their trajectories and outcomes. Heterogeneous exposure (`EL=p×EAD`) is therefore essential for policy distinctions to emerge.

## Analysis corrections

- Corrected coverage is `sum(n_explored)/sum(n_rejected)` over rounds; the legacy coverage file is explicitly marked `LEGACY_ANALYSIS_DEFINITION / DO_NOT_USE_FOR_PAPER`.
- NoExplore rho repetitions are confirmed identical and labeled `REDUNDANT_COMPUTATION_ONLY`.
- Final core tables and primary comparisons were regenerated from the existing all-rounds results.

## Freeze gate

Coverage correction, UCI equivalence audit, scarcity hypothesis freeze, FAVE-not-tuned declaration, documentation cleanup, and pytest all pass. Phase 3 may use the frozen algorithm declaration, but no external result may change Phase 2 choices.
