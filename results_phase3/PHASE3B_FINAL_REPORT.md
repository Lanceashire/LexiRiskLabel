# Phase 3B Final Report

## Completion

- External dataset: LendingClub (`data/phase3/lendingclub.csv`), 16,566 rows.
- Formal coverage: 20 seeds × 4 rho values × 8 policies = 640 jobs, with 580 CSV artifacts including 20 NoExplore baselines and 560 exploratory-policy jobs.
- Expanded round output: 7,040 rows.
- All 580 artifacts have 11 rounds, `compute_backend=gpu`, and passed invariant checks.
- Solver parity: pass (120 randomized cases). FAVE numerical parity: pass (100 randomized cases). Dataset parity: pass.

## Main result

LendingClub has heterogeneous exposure (`loan_amnt`; coefficient of variation 0.5796), unlike the unit-exposure PPDai benchmark. Policy differentiation is therefore observable: at rho=0.025 the Recall range is 0.004955, FDRatio range is 0.049789, and Brier range is 0.000359.

The paired 20-seed comparisons do not show statistically reliable superiority after Holm correction. For FAVE versus CountOnly-MinRisk, mean Recall difference is +0.001563, mean FDRatio difference is -0.000252, and mean Brier difference is -0.000093; all Holm-adjusted p-values are ≥0.9498. FAVE versus RiskTilted is similarly non-significant (Holm p ≥0.7574).

Thus the frozen protocol does not support the claim that FAVE consistently wins on this external dataset. It supports a narrower conclusion: exposure heterogeneity creates measurable policy separation, but the tested FAVE advantage is small and not statistically established.

## PPDai status

PPDai remains the frozen primary/development result. Its corrected policy ranges are very small, and the strict provenance audit flags ambiguous outcome-adjacent fields. PPDai findings are therefore reported as exploratory with a leakage-risk limitation; the raw primary artifacts were not modified or rerun.

## Reproducibility

All formal LendingClub jobs used the frozen split, target mapping, risk budget, exact LRBE selection, GPU XGBoost path, and recorded per-job timing profiles. Legacy partial/CPU-fallback artifacts are excluded from final statistics.