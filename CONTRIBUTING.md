# Contributing

This repository is a research artifact with frozen primary evidence.  Changes
must preserve provenance and distinguish new experiments from corrections.

## Rules

1. Do not overwrite frozen Phase 0--4 or formal CCFB raw result CSVs.
2. Register any new seed, dataset, model, threshold, budget, or policy as a new
   experiment.
3. Fit preprocessing, calibration, and hyperparameters without outer-test
   access.
4. Preserve negative results and report all registered seeds.
5. Never commit third-party data, credentials, local caches, logs, or archive
   ZIPs.
6. Update provenance and artifact hashes when adding derived results.

## Pull requests

Describe:

- the research question or reproducibility problem;
- files and evidence affected;
- whether the change is confirmatory, exploratory, or post-hoc;
- tests and audits run;
- any claim that must change as a consequence.
