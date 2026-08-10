# Phase 1 closeout addendum: dual evaluation

This is an analysis-only addendum. The frozen Phase 1 trajectories and formal treatment results were not changed.

## Design

- Datasets: UCI Default of Credit Card Clients and HMEQ.
- Treatments: Oracle and NoExplore only.
- Seeds: 0--19; rounds: 0--10.
- Evaluator A (`external_fixed`): the frozen fixed external test split, reconstructed with the same split, batch stream, preprocessing, threshold, and fit seeds.
- Evaluator B (`internal_visible`): a diagnostic 80/20 stratified split of the currently visible sample. It is evaluated after the round and never affects the dynamic trajectory.

## Parity and result

`DUAL_EVAL_PARITY.csv` reports 10,560 finite metric comparisons and a maximum absolute difference of 0.0 against the original frozen `uci_all_rounds.csv` and `hmeq_all_rounds.csv`. Therefore Evaluator A is a valid parity-preserving reproduction.

Evaluator B is not interchangeable with Evaluator A. At round 10, the internal-minus-external change in the Oracle--NoExplore contrast was negative for every tested primary ranking/classification metric. The bootstrap intervals exclude zero for both datasets (see `dual_evaluation_statistics.csv`):

- UCI: recall -0.2496, balanced accuracy -0.1061, ROC-AUC -0.0567, PR-AUC -0.1470.
- HMEQ: recall -0.3025, balanced accuracy -0.1460, ROC-AUC -0.0450, PR-AUC -0.2155.

The finding is evaluation distortion, not a correction to the formal experiment. The external fixed-test results remain the authoritative Phase 1 results.

## Files

`dual_evaluation_all_rounds.csv`, `dual_evaluation_gap.csv`, `dual_evaluation_statistics.csv`, `DUAL_EVAL_PARITY.csv`, and the `*_dual_*.png` figures are the complete addendum outputs.
