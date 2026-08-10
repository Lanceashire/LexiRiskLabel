# Phase 2 coverage correction

The legacy coverage ratio is not used for the paper. The corrected definition is `sum(n_explored) / sum(n_rejected)` over rounds, grouped by dataset, seed, rho, and policy. The corrected table is `summary/coverage_vs_budget_corrected.csv`.

If corrected coverage is not near one, small policy differences cannot simply be attributed to near-complete exploration saturation.
