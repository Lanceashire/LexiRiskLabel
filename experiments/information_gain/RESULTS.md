# Information-Gain Primary Extension: Results

**Evidence block:** `IG_PRIMARY_EXTENSION_20260907`

**Completed:** 2026-09-08
**Scope:** primary experiment at `rho = 0.1`; the optional five-budget extension was not run.

## Protocol and integrity

The experiment reran all five policies in one fresh evidence block:
`CountOnly-MinRisk`, `LRBE-FAVE`, `QualityFirst-FAVE`, `LRBE-IG`, and
`QualityFirst-IG`.  Each dataset used seeds 1001--1010 and ten sequential
rounds.  The information-gain policies used a fixed bootstrap committee of
20 members.  The committee affected only the acquisition score; candidate
classification, predicted risk, and final evaluation continued to use the
single canonical XGBoost model.

The completed block contains 200 dynamic episodes and 40 same-state audit
files.  The analysis verified all of the following:

- all 200 expected dataset--seed--policy cells are present;
- each trajectory contains the initial state plus ten rounds;
- all canonical-model rows used the registered GPU path;
- all predicted-risk selections satisfy their numerical budget;
- all recorded hidden-label isolation checks pass;
- PPDai is retained as reconstructed-source exploratory evidence and is not
  pooled with historical PPDai results;
- inference uses 10,000 seed-level paired bootstrap draws within each
  dataset, with no cross-dataset or cross-budget pooling.

## Primary comparison 1: preserving maximum feasible quantity

The contrast `LRBE-IG - QualityFirst-IG` asks whether the maximum-feasible
cardinality constraint remains consequential when both policies use the same
information-gain score.

| Dataset | Feedback-count difference | 95% paired interval | Recall difference | 95% paired interval | Brier difference | 95% paired interval |
|---|---:|---:|---:|---:|---:|---:|
| HMEQ | +31.3 | [28.3, 34.5] | +0.00535 | [-0.00068, 0.01170] | -0.000302 | [-0.000937, 0.000364] |
| LendingClub | +49.5 | [46.0, 53.10] | -0.00073 | [-0.00524, 0.00425] | -0.000070 | [-0.000251, 0.000103] |
| UCI | +40.1 | [36.8, 43.6] | +0.00166 | [-0.00187, 0.00517] | +0.000156 | [0.000019, 0.000298] |
| PPDai* | +0.2 | [0.0, 0.5] | -0.00056 | [-0.00139, 0.0] | -0.000029 | [-0.000066, 0.0] |

On HMEQ, LendingClub, and UCI, enforcing maximum feasible quantity prevents a
large loss of acquired feedback.  The paired Recall intervals nevertheless
include zero on every dataset.  The Brier interval also includes zero except
on UCI, where LRBE-IG has a small positive difference (higher is worse).  This
evidence supports the claim that the quantity constraint materially changes
feedback acquisition; it does not establish a stable downstream predictive
benefit.

PPDai has a very small candidate/feedback regime under this reconstruction,
so the two IG policies acquire nearly the same number of outcomes.  PPDai is
exploratory and does not determine the main conclusion.

## Primary comparison 2: incremental value of information gain

The contrast `LRBE-IG - CountOnly-MinRisk` asks whether changing composition
with IG is worth its additional predicted risk when maximum feasible quantity
is approximately preserved.

| Dataset | Feedback-count difference | Predicted-risk difference | 95% paired interval | Recall difference | 95% paired interval | Brier difference | 95% paired interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| HMEQ | +0.2 | +28,765.53 | [19,361.23, 39,425.11] | -0.00518 | [-0.01423, 0.00388] | -0.000068 | [-0.000479, 0.000292] |
| LendingClub | -2.9 | +27,025.75 | [19,238.29, 35,010.34] | +0.00045 | [-0.00286, 0.00488] | -0.000029 | [-0.000174, 0.000120] |
| UCI | -1.2 | +1.827 | [1.004, 2.733] | -0.00025 | [-0.00270, 0.00222] | -0.000098 | [-0.000262, 0.000059] |
| PPDai* | +0.4 | +1.024 | [0.497, 1.606] | -0.00052 | [-0.00270, 0.00142] | -0.000064 | [-0.000134, 0.000005] |

LRBE-IG consumes more predicted risk on all four datasets, with paired
intervals excluding zero.  No dataset shows a Recall or Brier difference whose
paired interval excludes zero.  The correct interpretation is therefore that
this bootstrap-committee IG score changes acquisition composition and raises
predicted risk, but the present ten-seed experiment does not demonstrate that
the increment buys a stable downstream Recall or Brier improvement.  An
interval containing zero is treated as inconclusive, not as evidence of
equivalence.

## Same-state mechanism audit

The same-state audit evaluates all five selectors on the CountOnly trajectory
using exactly the same visible set, candidate set, cost vector, and numerical
budget.  It describes instantaneous selection mechanics only; it is not a
counterfactual estimate of downstream performance.

| Dataset | Method | Cells | Mean count sacrifice | Mean composition premium | Mean Jaccard vs. CountOnly |
|---|---|---:|---:|---:|---:|
| HMEQ | LRBE-IG | 100 | 0.00 | 2,887.95 | 0.793 |
| HMEQ | QualityFirst-IG | 100 | 3.46 | 23,505.01 | 0.436 |
| LendingClub | LRBE-IG | 100 | 0.00 | 3,331.71 | 0.882 |
| LendingClub | QualityFirst-IG | 100 | 5.37 | 39,792.55 | 0.602 |
| UCI | LRBE-IG | 100 | 0.00 | 0.275 | 0.660 |
| UCI | QualityFirst-IG | 100 | 4.04 | 2.507 | 0.135 |
| PPDai* | LRBE-IG | 100 | 0.00 | 0.073 | 0.238 |
| PPDai* | QualityFirst-IG | 100 | 0.02 | 0.076 | 0.224 |

LRBE-IG has zero count sacrifice in all 400 audited states by construction and
changes the selected composition within the maximum-count slack.  In the
three non-PPDai datasets, QualityFirst-IG obtains a larger instantaneous
composition premium by giving up feedback count.  The dynamic results above
show that this sacrifice does not translate into a consistently detectable
Recall or Brier improvement.

## Computation

The 200 dynamic episodes record 13,219.5 seconds (3.67 hours) of episode-level
runtime in total.  The mean LRBE-IG episode time is 8.9 seconds on HMEQ, 19.9
seconds on UCI, 498.6 seconds on LendingClub, and 16.6 seconds on PPDai.  The
large LendingClub cost comes from fitting 20 bootstrap committee members in
each of ten rounds.  Same-state audit time is not included in the episode-level
runtime total.

## Evidence-bounded conclusion

The independent IG baseline strengthens the mechanism result without making
IG the proposed winning algorithm.  Maximum-feasible cardinality remains a
consequential design constraint under a score that does not use FAVE's Fisher
or influence-function approximation.  At approximately preserved quantity,
however, IG's additional predicted risk does not yield a stable downstream
advantage.  The supported contribution is therefore the empirical separation
of feedback quantity, feedback composition, and predicted risk under selective
labels, rather than a claim that either FAVE or IG universally improves credit
prediction.

`*` PPDai uses unit exposure and reconstructed-source exploratory evidence.

## Machine-readable outputs

- Dynamic trajectories: `results/information_gain_primary_20260907/trajectories/`
- Same-state audits: `results/information_gain_primary_20260907/state_audit/`
- Integrity summary: `results/information_gain_primary_20260907_analysis/integrity_summary.json`
- Method summaries: `results/information_gain_primary_20260907_analysis/method_summary.csv`
- Paired contrasts: `results/information_gain_primary_20260907_analysis/paired_comparisons.csv`
- Same-state summaries: `results/information_gain_primary_20260907_analysis/state_audit_summary.csv`
