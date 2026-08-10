# Reproducibility Guide

## Evidence levels

The repository separates three actions that should not be conflated:

1. **Integrity verification:** run committed tests against frozen outputs.
2. **Analysis reproduction:** regenerate summaries, statistics, figures, and
   reports from committed formal raw result CSVs without model training.
3. **Full experimental reproduction:** obtain third-party datasets and rerun
   dynamic model fitting/acquisition under the frozen protocol.

## Validated environment

The completed CCFB extension was validated with:

| Component | Version |
|---|---:|
| Python | 3.13.13 |
| NumPy | 2.5.1 |
| pandas | 3.0.3 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| XGBoost | 3.2.0 |
| Matplotlib | 3.10.9 |
| pytest | 9.1.1 |
| CuPy CUDA 13.x | 14.1.1 |

The original run used an NVIDIA RTX 5060-class GPU with CUDA 13.1.  XGBoost
and supported FAVE matrix operations use GPU arrays when available.  Exact
mixed-integer selection and some control logic remain CPU-side.

Create an environment with either:

```bash
conda env create -f environment.yml
conda activate lexirisklabel
```

or:

```bash
python -m venv .venv
# activate the environment for your platform
python -m pip install -r requirements.txt
```

If CUDA 13-compatible CuPy is unavailable, install the CuPy build matching
your CUDA runtime or set `PHASE2_USE_GPU=0` for a recorded CPU fallback.

## Integrity verification

No third-party dataset is required for the committed frozen-output tests:

```bash
python -m pytest -q -p no:cacheprovider tests
```

Expected validated result:

```text
39 passed
```

The root-level `results_phase0`--`results_phase4` directories are intentionally
retained because these frozen tests verify their exact historical layout.

## Reproduce CCFB statistics without retraining

```bash
python experiments/ccfb_extension/analyze_ccfb_extension.py
```

Inputs:

```text
results/ccfb_extension/raw/reference/
results/ccfb_extension/raw/kstar_ablation/
results/ccfb_extension/raw/budget_sweep/
results/phase4/statistics/FEEDBACK_PERFORMANCE_SENSITIVITY_FINAL.csv
```

Outputs include:

- `01_kstar_ablation_raw.csv` and `02_kstar_ablation_summary.csv`;
- paired differences, Cohen's dz, paired tests, and bootstrap CIs;
- budget-sweep raw and summary tables;
- predicted/realized risk and calibration summaries;
- fresh and frozen Phase 4 feedback--decision statistics;
- six figures in both PNG and PDF;
- `CCFB_extension_report.md`;
- an extension artifact manifest.

The feedback--decision intervals use 10,000 seed-cluster bootstrap draws.
Repeated seed clusters receive tie-aware average ranks in Spearman bootstrap
samples.

## Full CCFB experiment

First prepare data as documented in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md):

```bash
python scripts/data_preparation/materialize_data_layout.py \
  --uci /path/to/uci_default_credit_card.csv \
  --hmeq /path/to/hmeq.csv \
  --ppdai /path/to/ppdai.csv \
  --lendingclub /path/to/lendingclub.csv
```

Run a smoke test:

```bash
python experiments/ccfb_extension/run_ccfb_extension.py --mode smoke
```

Run the formal minimum design:

```bash
python experiments/ccfb_extension/run_ccfb_extension.py --mode full
```

The formal design contains:

- datasets: UCI, HMEQ, PPDai, LendingClub;
- fresh seeds: 1001--1010;
- 10 dynamic rounds;
- base risk fraction `rho=0.10`;
- budget sweep `rho={0.025, 0.05, 0.10, 0.20, 0.40}`;
- K* ablation, quality-first, normalized joint objective, NoExplore and Oracle
  references.

The completed run generated 1,120 formal run CSVs and 12,320 round-level rows.
Runtime depends strongly on hardware.  On the validated machine the full
extension completed in several hours; CPU-only reproduction may be much
slower.

## Leakage controls

- Outer test indices are fixed by dataset and seed.
- Preprocessing is fit only on currently visible training labels.
- The outer test set is never used for feature selection, fitting, calibration
  fitting, lambda selection, threshold selection, or acquisition.
- `Joint-Best-Val-Uncertainty` chooses lambda only through a deterministic
  train / pseudo-acquisition / validation split inside the initial visible set
  D0.
- All methods in a dataset × seed × round × budget cell share the same
  candidate pool and risk budget.
- Hidden labels are used only after an acquisition decision to simulate
  realized feedback/loss and to evaluate fixed test metrics.

The completed formal audit found zero split-hash inconsistencies, hidden/test
isolation failures, predicted-budget violations, LRBE `count != K*` failures,
or GPU-disabled formal rows.

## Risk semantics

The planning cost is:

```text
R_pred = sum(z_i * PD_hat_i * EAD_i * LGD_i)
```

Realized simulated loss is:

```text
R_real = sum(z_i * y_i * EAD_i * LGD_i)
```

`R_pred <= B` is a hard constraint in the optimizer.  It does **not** imply
`R_real <= B`.  Do not rewrite this distinction when reproducing or reporting
results.

## Determinism and negative results

- Keep seeds, splits, `tau`, model parameters, FAVE parameters, and lambda grid
  fixed unless registering a new experiment.
- Never select hyperparameters from the outer test set.
- Never delete seeds or datasets because a method loses.
- Do not use `--force` to search for favorable reruns.
- Preserve negative results and confidence intervals crossing zero.

## Publication artifact verification

Run:

```bash
python scripts/maintenance/verify_publication.py
```

This checks required documentation/results, forbidden raw-data/log/cache
paths, the formal CCFB run count, files larger than GitHub's regular-object
limit, and the publication manifest when present.
