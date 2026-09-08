# Independent information-gain extension

This additive evidence block does not alter frozen Phase 1--4 code or results.
It asks (1) whether preserving the maximum feasible feedback count remains
important when FAVE is replaced by a bootstrap information-gain score, and
(2) whether IG improves Recall or Brier enough to justify added predicted risk
after that count is preserved.

The preregistered primary block at `rho = 0.1` is complete: 200 dynamic
episodes and 40 same-state audits.  See [RESULTS.md](RESULTS.md) for the
evidence-bounded interpretation and
`results/information_gain_primary_20260907_analysis/` for machine-readable
statistics.  The optional five-budget extension has not been run.

Five policies are rerun together: `CountOnly-MinRisk`, `LRBE-FAVE`,
`QualityFirst-FAVE`, `LRBE-IG`, and `QualityFirst-IG`. Within each
dataset--seed--budget cell they share the initial visible set, outer test set,
ten incoming batches, threshold, exposure definition, canonical XGBoost setup,
preprocessing rules, committee size, and committee seed rule. Each policy then
follows its own feedback history; candidate pools and numerical budgets may
therefore diverge after round one. Paired comparisons condition on shared
exogenous inputs, not identical endogenous round states.

The canonical model and FAVE use the GPU path. The preregistered 20-member
ordinary bootstrap committee uses CPU and affects only acquisition scores. Its
score is empirical-committee mutual information in nats. It is neither an
exact Bayesian posterior quantity nor joint BatchBALD.

PPDai has unit exposure and is always labeled
`reconstructed_source_exploratory`. The actual processed hash is recorded, but
historical processed byte parity is not claimed; these runs must not be pooled
with historical PPDai observations.

## Reproduction in `no_1`

```text
D:\envs_dirs\no_1\python.exe -m pytest -q -p no:cacheprovider tests/test_information_gain.py tests/test_information_gain_analysis.py

D:\envs_dirs\no_1\python.exe experiments/information_gain/run_information_gain.py --datasets hmeq --seeds 1001 --budgets 0.1 --members 20 --state-audit --output results/information_gain_smoke_20260907

D:\envs_dirs\no_1\python.exe experiments/information_gain/run_information_gain.py --budgets 0.1 --members 20 --state-audit --output results/information_gain_primary_20260907

D:\envs_dirs\no_1\python.exe experiments/information_gain/analyze_information_gain.py results/information_gain_primary_20260907 --output results/information_gain_primary_20260907_analysis
```

Output roots must be new. `--resume` completes missing files after an
interruption but never overwrites an existing trajectory. Raw third-party
datasets are not redistributed.
