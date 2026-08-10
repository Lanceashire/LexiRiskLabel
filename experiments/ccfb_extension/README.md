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
```

`--mode full` is computationally expensive.  It performs the preregistered
minimum 10 fresh seeds and never replaces frozen Phase 1--4 results.
