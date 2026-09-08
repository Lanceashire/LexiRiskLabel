# CCFB Extension: LRBE evidence-chain validation

This directory contains the **new**, non-destructive extension requested after
the frozen Phase 1--4 study.  It imports the frozen compatibility runners in
`src/` but does not modify them or their outputs.

## Confirmatory extension design

- **Fresh seeds:** `1001`--`1010`; they do not overlap the frozen `0`--`19`
  schedule.
- **Splits:** the frozen `phase1_runner.fixed_split` procedure is reused
  verbatim for every dataset and seed.  The fixed outer test set is never used
  for preprocessing, fitting, calibration, lambda selection, or acquisition.
- **Datasets:** UCI and HMEQ are development/monetary validation; LendingClub
  is primary external validation; PPDai remains exploratory external evidence.
- **ECL proxy:** unit exposure for UCI, `PD_hat * LOAN` for HMEQ, and
  `PD_hat * exposure` for both external datasets.  LGD is fixed to 1.0, as in
  the frozen protocol.  Results are therefore simulated risk budgets where
  the dataset has no observed EAD/LGD.
- **Exposure validation:** the current extension fails closed unless every
  exposure is numeric, finite, and strictly positive. Missing, malformed,
  infinite, negative, or zero values are never converted into free labels.
- **Sequential comparability:** policies share the split, initial visible set,
  incoming batches, seed, rho, and outer test set. They refit on their own
  feedback histories, so candidate pools and numerical budgets are generally
  policy-specific after the first acquisition.
- **Joint lambda selection:** `Joint-Best-Val-Uncertainty` selects lambda only
  inside the initial visible set, using a deterministic stratified
  train / pseudo-acquisition / validation split.  Its outer test metrics never
  enter selection.  The full validation-grid sensitivity is preserved.

## Outputs

The runner writes immutable run files under `results/ccfb_extension/raw/`.
The analysis command materializes the requested numbered CSVs, bilingual
figure notes, PNG/PDF figures, and `CCFB_extension_report.md` under
`results/ccfb_extension/`.

Run commands are deliberately explicit:

```powershell
D:\envs_dirs\no_1\python.exe experiments\ccfb_extension\run_ccfb_extension.py --mode smoke
D:\envs_dirs\no_1\python.exe experiments\ccfb_extension\run_ccfb_extension.py --mode full
D:\envs_dirs\no_1\python.exe experiments\ccfb_extension\analyze_ccfb_extension.py
D:\envs_dirs\no_1\python.exe experiments\ccfb_extension\audit_exposure_inputs.py --output-dir results\ccfb_extension\exposure_hardening
D:\envs_dirs\no_1\python.exe experiments\ccfb_extension\analyze_exposure_confirmation.py
```

The hardened confirmation uses experiment name
`exposure_hardened_confirmation`, seed 1001, all four datasets, and
`CountOnly-MinRisk` / `LRBE-FAVE`. It is a separate eight-run audit and does
not replace the registered 1,120-run archive.

`--mode full` is computationally expensive.  It performs the preregistered
minimum 10 fresh seeds and never replaces frozen Phase 1--4 results.
