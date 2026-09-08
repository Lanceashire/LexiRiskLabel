# Supplemental exploration baseline report

This addendum uses 80 fresh final-round trajectories (4 datasets × 10 seeds × 2 risk-bounded randomized methods). Raw rows are immutable after writing.

## Integrity

- Files: 80/80
- Datasets: hmeq, lendingclub, ppdai, uci
- Same split hashes per dataset-seed: True
- Same D0 hashes per dataset-seed: True
- Hidden-label isolation: True
- Outer-test metrics-only use: True
- GPU backend recorded for every row: True

## Interpretation

RiskBounded-Random greedily samples a deterministic pseudo-random order subject to the predicted-risk budget. RiskBounded-Epsilon retains the exact feasible K* count while adding a preregistered 10% random perturbation to the min-risk utility. These are external-style controls, not claims of reproducing any one prior paper's full online algorithm.

The paired comparisons and confidence intervals are in `supplemental_paired_comparisons.csv`; aggregate values are in `supplemental_method_summary.csv`.
