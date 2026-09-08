# Experimental Log

## September 2026 DASFAA Revision — controlling interpretation

This addendum supersedes older wording below where the two conflict. No model was trained for this revision. The exact derived files and provenance hashes are under `results/ccfb_extension/dasfaa_revision/`; the reproducible analysis is `experiments/ccfb_extension/analyze_dasfaa_revision.py`.

* Policies share the split, initial visible set, exogenous incoming batches, seed and budget ratio. After different selections, they do **not** generally share the model, candidate pool, numerical budget or K-star. Comparisons estimate complete policy differences under paired exogenous conditions, not causal quantity/composition effects.
* Similar cumulative counts are descriptive, not a matched-count experiment. No matched-count-plus-matched-risk intervention or parameterized Pareto frontier was run. Intervals containing zero are not equivalence tests.
* The archive name `RiskBounded-Epsilon` denotes a fixed-count randomized-score objective, `-0.9 normalized_cost + 0.1 random_score`; it is not a 10% random-approval probability or standard epsilon-greedy policy.
* PPDai's exact exposure column is identically one, as is UCI's constructed exposure. Monetary heterogeneity is supplied by HMEQ and LendingClub only. LGD is one, so realized loss is defaulted-exposure proxy rather than recovery-adjusted net loss.
* The sequential stream is a random permutation with an initially sampled visible set and immediate simulated outcome revelation. It is not a historical-time experiment and does not contain outcomes for actual rejected LendingClub applicants.
* Primary feedback--prediction analysis uses raw gains from NoExplore. Reference-normalized recovery is supplemental and available only for positive oriented Oracle gaps. For Recall, nonpositive gaps occur in 10 of 40 dataset--seed reference cells (HMEQ 0, LendingClub 2, PPDai 2, UCI 6). Raw-gain results at rho 0.1 are: HMEQ Recall Spearman 0.397 [0.013, 0.660] and Brier 0.062 [-0.175, 0.381]; LendingClub Recall -0.003 [-0.326, 0.423]; PPDai 0.071 [-0.320, 0.481]; UCI 0.077 [-0.341, 0.459]. All four Brier intervals contain zero. Across all five budgets, HMEQ Recall associations are positive with intervals excluding zero; LendingClub is negative at the two smallest budgets and uncertain thereafter. These are pointwise descriptive intervals, not adjusted causal tests.
* The fresh archive has 1,120 run files: 240 objective-comparison, 800 budget-sweep, and 80 reference runs. Some policy--seed--rho settings occur in multiple evidence blocks; they are not independent replications and are not pooled to inflate sample size.
* The hardening regression now fails closed on actual predictive fields (`recall`, `pr_auc`, `roc_auc`, `brier`, `accuracy`, and calibration fields), row keys, one-sided missingness and infinities. Eight checked trajectories match on all required fields. This is a regression check, not added statistical evidence.

## 1. Experimental Setup

* **Research design:** We ran dynamic offline selective-label simulations in which a model was repeatedly fitted on currently visible repayment outcomes, selected additional applicants under a per-round predicted expected-credit-loss budget, revealed only the selected hidden outcomes, and then refitted. The outer test indices were fixed by dataset and seed.
* **Datasets:** We used UCI Default Credit Card (30,000 rows) as a development benchmark, HMEQ (5,960 rows) as monetary validation, PPDai (55,596 rows) as lower-confidence exploratory boundary evidence, and LendingClub (16,566 rows) as the primary external validation dataset.
* **Risk semantics:** UCI and PPDai used unit exposure. HMEQ used predicted default probability multiplied by `LOAN`; LendingClub used a loan-amount proxy. `LGD=1`. The optimization constrained predicted risk, not realized loss.
* **Fresh-seed design:** The CCFB extension used seeds 1001--1010, disjoint from frozen Phase 0--4 seeds 0--19. Each run used 10 dynamic rounds. The formal extension produced 1,120 run CSVs and 12,320 round-level records.
* **Budgets:** The base risk fraction was $\rho=0.10$. The budget sweep used $\rho\in\{0.025,0.05,0.10,0.20,0.40\}$.
* **Methods:** We compared CountOnly-MinRisk, LRBE-FAVE, LRBE-Uncertainty, QualityFirst-FAVE, QualityFirst-Uncertainty, Joint-Best-Val-Uncertainty, NoExplore, and Oracle references where defined. FAVE was frozen before evaluation and was treated as a complex information-value baseline and mechanism probe.
* **Evaluation metrics:** We reported cumulative feedback count, cumulative predicted risk, cumulative realized simulated loss, budget utilization, visible default rate, feedback recovery, Recall, Recall recovery, ROC-AUC, PR-AUC, Brier score, expected calibration error, and feedback--decision correlations.
* **Statistical analysis:** Main comparisons used paired fresh-seed differences, Cohen's $d_z$, paired tests, and 95% paired bootstrap intervals. Feedback--decision analyses used within-budget associations and 10,000-draw seed-cluster bootstrap intervals. Intervals crossing zero and non-estimable cells were retained.
* **Leakage controls:** Preprocessing was fitted only on currently visible training labels. The outer test set was never used for feature selection, fitting, probability calibration, lambda selection, threshold selection, or acquisition. Joint-objective lambda selection used only a deterministic internal train, pseudo-acquisition, and validation split within the initially visible set. Methods share initial conditions and incoming batches; after the first policy-dependent update, their models, candidate pools, numerical budgets, and $K^*$ values can diverge.
* **Validated integrity:** The completed audits found zero split-hash inconsistencies, predicted-budget violations, or LRBE count-versus-$K^*$ failures in the registered formal rows. Compatibility tests and evidence-block-specific perturbation checks are reported with the corresponding experiment rather than summarized as a single timeless test count.

## 2. Raw Numeric Data

### Dataset registry

| Dataset | Evidence role | Rows | Risk or exposure treatment |
|---|---|---:|---|
| UCI | Development benchmark | 30000 | Predicted default probability with unit-exposure proxy |
| HMEQ | Development monetary validation | 5960 | Predicted default probability times `LOAN`, fixed `LGD=1` |
| PPDai | Exploratory lower-confidence boundary evidence | 55596 | Exposure proxy, fixed `LGD=1` |
| LendingClub | Primary external validation | 16566 | `loan_amnt` exposure proxy, fixed `LGD=1` |

### Fresh-seed K-star ablation at rho 0.10

| Dataset | Method | Seeds | Feedback count mean | Predicted risk mean | Realized loss mean | Recall mean | ROC-AUC mean | PR-AUC mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HMEQ | CountOnly-MinRisk | 10 | 156.5 | 660207.8279 | 718760.0 | 0.6219851 | 0.9271624 | 0.8250488 |
| HMEQ | Joint-Best-Val-Uncertainty | 10 | 132.3 | 577599.6950 | 630730.0 | 0.6238329 | 0.9273221 | 0.8247961 |
| HMEQ | LRBE-FAVE | 10 | 157.6 | 692812.8633 | 756360.0 | 0.6213098 | 0.9263534 | 0.8238884 |
| HMEQ | QualityFirst-FAVE | 10 | 90.5 | 679547.6403 | 795480.0 | 0.6038543 | 0.9269988 | 0.8255156 |
| LendingClub | CountOnly-MinRisk | 10 | 535.2 | 2422361.7092 | 2419902.5 | 0.7294646 | 0.9442019 | 0.7977215 |
| LendingClub | Joint-Best-Val-Uncertainty | 10 | 413.9 | 1789515.3380 | 1787937.5 | 0.7287160 | 0.9440977 | 0.7971637 |
| LendingClub | LRBE-FAVE | 10 | 535.1 | 2453910.5167 | 2443700.0 | 0.7294926 | 0.9440580 | 0.7976833 |
| LendingClub | QualityFirst-FAVE | 10 | 325.5 | 2475582.4290 | 2505725.0 | 0.7246766 | 0.9436558 | 0.7934045 |
| PPDai | CountOnly-MinRisk | 10 | 13.3 | 6.7346 | 6.7 | 0.0195384 | 0.7065929 | 0.2821260 |
| PPDai | Joint-Best-Val-Uncertainty | 10 | 13.3 | 6.7346 | 6.7 | 0.0195384 | 0.7065929 | 0.2821260 |
| PPDai | LRBE-FAVE | 10 | 13.3 | 7.4162 | 7.1 | 0.0193202 | 0.7069748 | 0.2820419 |
| PPDai | QualityFirst-FAVE | 10 | 11.3 | 6.3307 | 5.6 | 0.0188156 | 0.7066281 | 0.2816780 |
| UCI | CountOnly-MinRisk | 10 | 273.8 | 145.5805 | 149.4 | 0.3638927 | 0.7807761 | 0.5444334 |
| UCI | Joint-Best-Val-Uncertainty | 10 | 239.3 | 126.3039 | 130.6 | 0.3654017 | 0.7806545 | 0.5445545 |
| UCI | LRBE-FAVE | 10 | 273.5 | 147.7366 | 149.1 | 0.3634936 | 0.7810208 | 0.5454008 |
| UCI | QualityFirst-FAVE | 10 | 217.1 | 143.6781 | 137.0 | 0.3639080 | 0.7810233 | 0.5462080 |

### Selected fresh-seed budget-sweep results

| Dataset | Method | rho | Feedback count mean | Budget utilization mean | Feedback recovery mean | Recall mean | Predicted risk mean | Realized loss mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HMEQ | CountOnly-MinRisk | 0.025 | 57.4 | 0.9024 | 0.0980 | 0.6096419 | 151575.3509 | 174310.0 |
| HMEQ | CountOnly-MinRisk | 0.100 | 156.5 | 0.9622 | 0.2751 | 0.6219851 | 660207.8279 | 718760.0 |
| HMEQ | CountOnly-MinRisk | 0.400 | 374.0 | 0.9666 | 0.6425 | 0.6390706 | 2741170.3593 | 2861510.0 |
| HMEQ | LRBE-FAVE | 0.025 | 58.0 | 0.9772 | 0.0952 | 0.6035335 | 168234.9383 | 192760.0 |
| HMEQ | LRBE-FAVE | 0.100 | 157.6 | 0.9971 | 0.2763 | 0.6213098 | 692812.8633 | 756360.0 |
| HMEQ | LRBE-FAVE | 0.400 | 375.8 | 0.9968 | 0.6525 | 0.6420296 | 2815158.9276 | 3023890.0 |
| LendingClub | CountOnly-MinRisk | 0.025 | 207.6 | 0.9627 | 0.0943 | 0.7235005 | 594776.5144 | 584042.5 |
| LendingClub | CountOnly-MinRisk | 0.100 | 535.2 | 0.9831 | 0.2457 | 0.7294646 | 2422361.7092 | 2419902.5 |
| LendingClub | CountOnly-MinRisk | 0.400 | 1253.1 | 0.9939 | 0.6107 | 0.7365999 | 9778012.3367 | 9752210.0 |
| LendingClub | LRBE-FAVE | 0.025 | 206.3 | 0.9967 | 0.0960 | 0.7264347 | 614978.6536 | 604532.5 |
| LendingClub | LRBE-FAVE | 0.100 | 535.1 | 0.9993 | 0.2471 | 0.7294926 | 2453910.5167 | 2443700.0 |
| LendingClub | LRBE-FAVE | 0.400 | 1253.3 | 0.9999 | 0.6104 | 0.7352126 | 9849744.4256 | 9825557.5 |
| UCI | CountOnly-MinRisk | 0.025 | 67.2 | 0.9297 | 0.0230 | 0.3635373 | 34.1691 | 33.6 |
| UCI | CountOnly-MinRisk | 0.100 | 273.8 | 0.9814 | 0.0963 | 0.3638927 | 145.5805 | 149.4 |
| UCI | CountOnly-MinRisk | 0.400 | 970.8 | 0.9953 | 0.4077 | 0.3623391 | 581.6932 | 585.7 |
| PPDai | CountOnly-MinRisk | 0.025 | 0.1 | 0.0000 | 0.0019 | 0.0189584 | 0.0507 | 0.1 |
| PPDai | CountOnly-MinRisk | 0.100 | 13.3 | 0.6127 | 0.0704 | 0.0195384 | 6.7346 | 6.7 |
| PPDai | CountOnly-MinRisk | 0.400 | 66.7 | 0.9518 | 0.3647 | 0.0209202 | 35.0306 | 36.2 |

### Paired K-star comparisons on HMEQ

| Comparison | Metric | Mean paired difference | Cohen's dz | 95% bootstrap low | 95% bootstrap high | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| LRBE-Uncertainty minus QualityFirst-Uncertainty | Cumulative feedback count | 7.0000 | 1.4703 | 4.6000 | 10.1000 | 0.001953 |
| LRBE-Uncertainty minus QualityFirst-Uncertainty | Recall | -0.002993 | -0.1845 | -0.011929 | 0.007164 | 0.425781 |
| LRBE-FAVE minus QualityFirst-FAVE | Cumulative feedback count | 67.1000 | 8.8712 | 62.6000 | 71.4000 | 0.001953 |
| LRBE-FAVE minus QualityFirst-FAVE | Recall | 0.017456 | 0.7978 | 0.003660 | 0.029471 | 0.037109 |
| LRBE-FAVE minus Joint-Best-Val-Uncertainty | Cumulative feedback count | 25.3000 | 0.9978 | 11.7000 | 41.1000 | 0.001953 |
| LRBE-FAVE minus Joint-Best-Val-Uncertainty | Recall | -0.002523 | -0.1570 | -0.011795 | 0.007141 | 0.687500 |
| LRBE-FAVE minus CountOnly-MinRisk | Cumulative feedback count | 1.1000 | 0.3318 | -0.9000 | 3.0000 | 0.312500 |
| LRBE-FAVE minus CountOnly-MinRisk | Cumulative predicted risk | 32605.0353 | 1.8966 | 23059.7545 | 43126.5839 | 0.001953 |

### Predicted and realized risk reliability

| Quantity | Value |
|---|---:|
| Formal predicted-budget violations | 0 |
| Mean round-level realized budget violation rate across method--dataset--budget groups | 0.4798 |
| Minimum group-level realized violation rate | 0.0100 |
| Maximum group-level realized violation rate | 0.8400 |
| HMEQ CountOnly-MinRisk realized violation rate at rho 0.10 | 0.5500 |
| HMEQ LRBE-FAVE realized violation rate at rho 0.10 | 0.7500 |
| HMEQ QualityFirst-FAVE realized violation rate at rho 0.10 | 0.8100 |

### Frozen within-budget feedback--decision sensitivity

| Dataset | rho | Association | Spearman rho | Cluster-bootstrap low | Cluster-bootstrap high | Seeds |
|---|---:|---|---:|---:|---:|---:|
| HMEQ | 0.025 | Feedback gain versus decision gain | 0.0364 | -0.2409 | 0.3269 | 20 |
| HMEQ | 0.050 | Feedback gain versus decision gain | -0.1374 | -0.3781 | 0.1184 | 20 |
| HMEQ | 0.100 | Feedback gain versus decision gain | -0.1739 | -0.4041 | 0.1028 | 20 |
| HMEQ | 0.200 | Feedback gain versus decision gain | 0.0159 | -0.3242 | 0.3399 | 20 |
| HMEQ | centered | Feedback gain versus decision gain | -0.0668 | -0.3014 | 0.1863 | 20 |
| LendingClub | 0.025 | Feedback gain versus decision gain | -0.1798 | -0.4077 | 0.1114 | 20 |
| LendingClub | 0.050 | Feedback gain versus decision gain | -0.2666 | -0.5578 | 0.0790 | 20 |
| LendingClub | 0.100 | Feedback gain versus decision gain | -0.1408 | -0.4931 | 0.2445 | 20 |
| LendingClub | 0.200 | Feedback gain versus decision gain | -0.0919 | -0.4486 | 0.2602 | 20 |
| LendingClub | centered | Feedback gain versus decision gain | -0.1550 | -0.4291 | 0.1487 | 20 |
| PPDai | centered | Feedback gain versus decision gain | -0.0415 | -0.2950 | 0.2137 | 20 |
| UCI | centered | Feedback gain versus decision gain | -0.0774 | -0.3805 | 0.2527 | 20 |

Rounded manuscript presentation values for the representative budget sweep include HMEQ CountOnly Recall 0.6096, 0.6220, and 0.6391; HMEQ LRBE-FAVE Recall 0.6035, 0.6213, and 0.6420; LendingClub CountOnly Recall 0.7235, 0.7295, and 0.7366; and LendingClub LRBE-FAVE Recall 0.7264, 0.7295, and 0.7352.

### Cross-dataset paired K-star confirmation (fresh seeds 1001--1010)

Source: `results/ccfb_extension/02b_kstar_paired_comparisons.csv`. Values below are rounded only for manuscript presentation; the source CSV retains full precision.

| Dataset | Comparison | Metric | Mean difference | 95% low | 95% high |
|---|---|---|---:|---:|---:|
| HMEQ | LRBE-FAVE minus QualityFirst-FAVE | Feedback count | 67.1 | 62.6 | 71.4 |
| HMEQ | LRBE-FAVE minus QualityFirst-FAVE | Recall | 0.0175 | 0.0037 | 0.0295 |
| LendingClub | LRBE-FAVE minus QualityFirst-FAVE | Feedback count | 209.6 | 202.6 | 216.1 |
| LendingClub | LRBE-FAVE minus QualityFirst-FAVE | Recall | 0.0048 | -0.0037 | 0.0134 |
| PPDai | LRBE-FAVE minus QualityFirst-FAVE | Feedback count | 2.0 | 0.9 | 3.2 |
| PPDai | LRBE-FAVE minus QualityFirst-FAVE | Recall | 0.0005 | -0.0011 | 0.0020 |
| UCI | LRBE-FAVE minus QualityFirst-FAVE | Feedback count | 56.4 | 54.6 | 58.2 |
| UCI | LRBE-FAVE minus QualityFirst-FAVE | Recall | -0.0004 | -0.0049 | 0.0040 |
| HMEQ | LRBE-FAVE minus CountOnly-MinRisk | Feedback count | 1.1 | -0.9 | 3.0 |
| HMEQ | LRBE-FAVE minus CountOnly-MinRisk | Predicted risk | 32605.0 | 23059.8 | 43126.6 |
| HMEQ | LRBE-FAVE minus CountOnly-MinRisk | Recall | -0.0007 | -0.0083 | 0.0089 |
| LendingClub | LRBE-FAVE minus CountOnly-MinRisk | Feedback count | -0.1 | -2.4 | 2.5 |
| LendingClub | LRBE-FAVE minus CountOnly-MinRisk | Predicted risk | 31548.8 | 26756.9 | 35968.2 |
| LendingClub | LRBE-FAVE minus CountOnly-MinRisk | Recall | 0.0000 | -0.0039 | 0.0043 |
| PPDai | LRBE-FAVE minus CountOnly-MinRisk | Feedback count | 0.0 | -1.0 | 1.1 |
| PPDai | LRBE-FAVE minus CountOnly-MinRisk | Predicted risk | 0.6817 | 0.1344 | 1.2699 |
| PPDai | LRBE-FAVE minus CountOnly-MinRisk | Recall | -0.0002 | -0.0028 | 0.0023 |
| UCI | LRBE-FAVE minus CountOnly-MinRisk | Feedback count | -0.3 | -2.0 | 1.4 |
| UCI | LRBE-FAVE minus CountOnly-MinRisk | Predicted risk | 2.1561 | 1.3580 | 2.9849 |
| UCI | LRBE-FAVE minus CountOnly-MinRisk | Recall | -0.0004 | -0.0033 | 0.0029 |

### Fresh feedback--decision association block

Source: `results/ccfb_extension/08_feedback_decision_statistics.csv`; seed-cluster bootstrap intervals use fresh seeds 1001--1010.

| Dataset | Scope | Spearman estimate | 95% low | 95% high |
|---|---|---:|---:|---:|
| HMEQ | within dataset | 0.5863 | 0.5121 | 0.6640 |
| LendingClub | within dataset | 0.1567 | -0.0241 | 0.4177 |
| PPDai | within dataset, lower confidence | 0.2439 | 0.0035 | 0.5211 |
| UCI | within dataset | 0.0098 | -0.1721 | 0.2408 |
| HMEQ | within budget, rho 0.025 | 0.466 | 0.170 | 0.680 |
| HMEQ | within budget, rho 0.05 | 0.426 | 0.063 | 0.701 |
| HMEQ | within budget, rho 0.10 | 0.325 | -0.063 | 0.587 |
| HMEQ | within budget, rho 0.20 | 0.343 | 0.100 | 0.560 |
| HMEQ | within budget, rho 0.40 | 0.557 | 0.362 | 0.708 |

### Frozen Phase 4 matched-cell quantity-first block

Source: `results_phase4/PHASE4_FINAL_ROBUSTNESS_REPORT.md` and `reports/phase4/CANONICAL_FINAL_REPORT.md`.

| Dataset | Policy | Mean labels |
|---|---|---:|
| HMEQ | CountOnly-MinRisk | 130.45 |
| HMEQ | FAVE-v2.1 | 130.56 |
| HMEQ | RiskTilted-Uncertainty | 130.09 |
| LendingClub | CountOnly-MinRisk | 458.96 |
| LendingClub | FAVE-v2.1 | 458.63 |
| LendingClub | RiskTilted-Uncertainty | 459.56 |

CountOnly-MinRisk uses approximately 24,289 fewer predicted-risk units than FAVE-v2.1 on HMEQ and 31,674 fewer on LendingClub in this block, without a stable downstream loss.

## 3. Qualitative Observations

* LRBE reliably implemented the predicted-risk feasibility constraint and the count-equals-K-star invariant in all audited formal rows.
* Fixing K-star materially changed the quantity of acquired feedback relative to quality-first and joint objectives. The largest differences appeared for FAVE-based quality-first selection, which often used the budget on fewer labels.
* LRBE-FAVE and CountOnly-MinRisk acquired nearly identical mean label counts in the main HMEQ and LendingClub fresh-seed cells, but LRBE-FAVE generally used more predicted risk. Their downstream Recall, ROC-AUC, and PR-AUC values were very close and did not support stable superiority by the complex policy.
* FAVE altered feedback composition and sometimes improved feedback-health measures. These changes did not consistently translate into improved downstream discrimination or Recall.
* Feedback--decision associations depended on dataset and budget. HMEQ showed positive fresh-seed associations in several cells, whereas LendingClub was generally weak or uncertain. Frozen within-budget correlations for HMEQ and LendingClub mostly had intervals crossing zero. The supported claim is that feedback recovery does not reliably imply decision recovery, not that the two quantities are always unrelated.
* All formal non-Oracle choices obeyed the predicted ECL budget, yet realized-loss budget violations remained common. Predicted-risk feasibility therefore cannot be described as a deterministic realized-loss guarantee.
* PPDai had very small acquisition counts at low budgets and several low-variation or non-estimable cells. It was retained as lower-confidence exploratory boundary evidence rather than treated as equal-strength external validation.
* The experiments were offline simulations. They did not constitute an online causal policy trial, and some datasets required exposure or LGD proxies.
* Negative results, confidence intervals crossing zero, and method losses were preserved. No seed or dataset was dropped for unfavorable outcomes.

### Iteration History

1. Phase 0 froze the reference implementation and established compatibility, invariance, hidden-label shuffle, and numerical-parity checks.
2. Phase 1 tested the initial dynamic selective-label mechanism under frozen settings.
3. Phase 2 compared policies under shared exogenous inputs and budget ratios. Later trajectory auditing established that policy-specific candidates and numerical budgets diverge after the initial state.
4. Phase 3 added external datasets, with LendingClub designated primary external validation and PPDai retained as lower-confidence evidence.
5. Phase 4 analyzed quantity-first behavior and the relationship between feedback recovery and decision recovery using within-budget and budget-centered statistics.
6. The fresh-seed CCFB extension separated K-star from quality-first and joint objectives, swept five budget levels, audited predicted versus realized risk, and repeated feedback--decision analysis with seeds 1001--1010.

### Open Evidence Boundaries

* Fully observed real EAD and LGD were unavailable in several datasets.
* Ten fresh seeds provide finite uncertainty precision.
* Applicant-level prediction trajectories and overlap identifiers were unavailable for every historical dataset.
* A preregistered safety-margin experiment would be required before claiming stronger realized-loss control.

### Reviewer-requested supplemental exploration baselines (fresh seeds 1001--1010)

To address the reviewer request for an external-style exploration control, we ran
80 additional episodes (4 datasets x 10 fresh seeds x 2 methods) at the registered
`rho=0.1`. The data split, model, threshold, candidate pool, exposure proxy and
outer-test isolation are identical to the CCFB extension. Raw trajectories are in
`results/ccfb_extension/raw/supplemental_exploration/`; the runner is
`experiments/ccfb_extension/run_ccfb_extension.py` and the summary script is
`experiments/ccfb_extension/summarize_supplemental.py`.

`RiskBounded-Random` greedily traverses a deterministic pseudo-random candidate
order while respecting the predicted-risk budget. `RiskBounded-Epsilon` solves the
exact feasible `K*` cardinality problem with a fixed 10% random perturbation of the
minimum-risk utility. They are external-style controls, not claims of reproducing
any single prior online algorithm.

| Dataset | Method | Feedback count | Predicted risk | Recall | Brier | FDRatio |
|---|---|---:|---:|---:|---:|---:|
| HMEQ | CountOnly-MinRisk | 156.5 | 660207.8 | 0.6220 | 0.07714 | 0.69655 |
| HMEQ | LRBE-FAVE | 157.6 | 692812.9 | 0.6213 | 0.07732 | 0.69712 |
| HMEQ | RiskBounded-Epsilon | 157.8 | 681067.7 | 0.6228 | 0.07716 | 0.69448 |
| HMEQ | RiskBounded-Random | 69.5 | 669233.0 | 0.5952 | 0.07760 | 0.63726 |
| LendingClub | CountOnly-MinRisk | 535.2 | 2422361.7 | 0.7295 | 0.07315 | 0.61779 |
| LendingClub | LRBE-FAVE | 535.1 | 2453910.5 | 0.7295 | 0.07331 | 0.61843 |
| LendingClub | RiskBounded-Epsilon | 533.7 | 2447085.4 | 0.7308 | 0.07298 | 0.61815 |
| LendingClub | RiskBounded-Random | 217.6 | 2458968.6 | 0.7266 | 0.07346 | 0.55457 |
| PPDai | CountOnly-MinRisk | 13.3 | 6.7346 | 0.01954 | 0.10497 | 0.98711 |
| PPDai | LRBE-FAVE | 13.3 | 7.4162 | 0.01932 | 0.10495 | 0.98734 |
| PPDai | RiskBounded-Epsilon | 13.6 | 6.9215 | 0.01967 | 0.10497 | 0.98736 |
| PPDai | RiskBounded-Random | 13.0 | 7.3770 | 0.01875 | 0.10498 | 0.98717 |
| UCI | CountOnly-MinRisk | 273.8 | 145.5805 | 0.36389 | 0.13424 | 0.80475 |
| UCI | LRBE-FAVE | 273.5 | 147.7366 | 0.36349 | 0.13406 | 0.80486 |
| UCI | RiskBounded-Epsilon | 271.1 | 144.6566 | 0.36224 | 0.13419 | 0.80544 |
| UCI | RiskBounded-Random | 214.3 | 144.9621 | 0.36347 | 0.13419 | 0.80699 |

The paired comparisons are stored in
`results/ccfb_extension/supplemental_analysis/supplemental_paired_comparisons.csv`.
Against CountOnly-MinRisk, RiskBounded-Random loses 87.0 labels on HMEQ and 317.6
on LendingClub, with lower Recall; the exact-K* RiskBounded-Epsilon control differs
by only +1.3 and -1.5 labels, respectively, and its Recall intervals include zero.
Across all four datasets, the epsilon control therefore adds no stable downstream
benefit at matched quantity, while the greedy random control demonstrates the
quantity loss incurred by unconstrained random ordering under a risk budget.

Integrity audit: 80/80 files; 10 seeds per dataset; split and D0 hashes identical
within each dataset-seed; hidden-label isolation and outer-test-metrics-only flags
true for every row; GPU backend recorded for every row. Realized budget violations
remain stochastic (0.525 across supplemental final rows), so these results do not
turn point predicted risk into a hard realized-loss guarantee.

### Claim-evidence alignment analysis (analysis only, no retraining)

The manuscript revision derives two additional paired summaries exclusively from
committed frozen CSVs. The reproducible script is
`paper/paper_orchestra/workspace/supplemental_experiments/scripts/derive_claim_alignment_tables.py`.
It uses 10,000 bootstrap draws with seed 20260902 and writes
`claim_alignment_brier_pairs.csv` and `claim_alignment_composition_pairs.csv`.

Paired final-round Brier differences (method A minus method B; negative favors A):

| Dataset | FAVE - CountOnly | FAVE - QualityFirst | Epsilon - CountOnly | Random - CountOnly |
|---|---:|---:|---:|---:|
| HMEQ | 0.000181 [-0.000120, 0.000560] | -0.000297 [-0.001072, 0.000487] | 0.000014 [-0.000421, 0.000432] | 0.000459 [-0.000140, 0.001141] |
| LendingClub | 0.000158 [-0.000025, 0.000370] | -0.000121 [-0.000316, 0.000072] | -0.000172 [-0.000418, 0.000058] | 0.000303 [-0.000096, 0.000700] |
| PPDai | -0.000020 [-0.000062, 0.000020] | -0.000032 [-0.000089, 0.000026] | -0.000045 [-0.000123, 0.000031] | 0.000006 [-0.000047, 0.000062] |
| UCI | -0.000178 [-0.000278, -0.000082] | 0.000044 [-0.000072, 0.000157] | -0.000052 [-0.000165, 0.000047] | -0.000052 [-0.000273, 0.000147] |

Seven of eight FAVE intervals and all eight randomized-control intervals cross
zero. UCI shows a FAVE-over-CountOnly Brier advantage; this is retained as a
dataset-specific positive result rather than erased by the cross-dataset summary.

Frozen Phase 4 paired selected-set differences, FAVE-v2.1 minus
CountOnly-MinRisk, averaged within seed across registered budgets with a
seed-cluster bootstrap:

| Dataset | Selected mean PD difference | Acquired default-fraction difference |
|---|---:|---:|
| HMEQ | -0.00503 [-0.00732, -0.00281] | -0.00351 [-0.01136, 0.00421] |
| LendingClub | 0.00175 [0.00095, 0.00259] | 0.00247 [-0.00040, 0.00518] |
| PPDai | 0.04101 [0.03729, 0.04563] | -0.04767 [-0.10088, -0.00136] |
| UCI | 0.01673 [0.01588, 0.01753] | 0.00876 [-0.00933, 0.02705] |

The selected-PD intervals establish a composition mechanism in every dataset,
with dataset-dependent direction. Realized acquired-default composition is less
stable. No parameterized quantity-quality frontier or true matched-count plus
matched-risk intervention was run; the manuscript removes those claims instead
of treating descriptive Pareto flags or unequal-risk comparisons as such evidence.

### Independent information-gain primary extension (completed 2026-09-08)

This additive evidence block is registered as `IG_PRIMARY_EXTENSION_20260907`.
It reruns `CountOnly-MinRisk`, `LRBE-FAVE`, `QualityFirst-FAVE`, `LRBE-IG`,
and `QualityFirst-IG` together at `rho=0.1` for seeds 1001--1010 on HMEQ,
LendingClub, UCI, and PPDai. Each episode has ten rounds. The IG score is
empirical committee mutual information from 20 same-size bootstrap fits of the
current visible set. The committee affects acquisition scores only; the single
canonical XGBoost model determines candidates, predicted costs, and evaluation.
All canonical-model rows record the GPU path. The primary block contains 200
dynamic episodes and 40 same-state audits. It does not include the optional
five-budget extension.

The integrity analysis verifies all 200 expected cells, all ten seeds, the five
registered policies, M=20, 10,000 paired-bootstrap draws, predicted-budget
feasibility on every row, hidden-label isolation on every row, and separate
PPDai evidence status. PPDai uses unit exposure and remains
`reconstructed_source_exploratory`.

Primary paired differences for `LRBE-IG - QualityFirst-IG`:

| Dataset | Feedback count | Predicted risk | Recall | Brier |
|---|---:|---:|---:|---:|
| HMEQ | 31.3 [28.3, 34.5] | 5267.517 [-6697.273, 16208.554] | 0.005348 [-0.000680, 0.011697] | -0.000302 [-0.000937, 0.000364] |
| LendingClub | 49.5 [46.0, 53.1025] | -9906.969 [-21199.703, -1269.271] | -0.000730 [-0.005240, 0.004247] | -0.000070 [-0.000251, 0.000103] |
| UCI | 40.1 [36.8, 43.6] | 1.623984 [0.240813, 2.868848] | 0.001662 [-0.001871, 0.005174] | 0.000156 [0.000019, 0.000298] |
| PPDai | 0.2 [0.0, 0.5] | 0.049872 [0.0, 0.131635] | -0.000557 [-0.001395, 0.0] | -0.000029 [-0.000066, 0.0] |

Primary paired differences for `LRBE-IG - CountOnly-MinRisk`:

| Dataset | Feedback count | Predicted risk | Recall | Brier |
|---|---:|---:|---:|---:|
| HMEQ | 0.2 [-1.6, 1.7] | 28765.532 [19361.232, 39425.111] | -0.005183 [-0.014228, 0.003877] | -0.000068 [-0.000479, 0.000292] |
| LendingClub | -2.9 [-6.0, 0.1] | 27025.748 [19238.290, 35010.339] | 0.000448 [-0.002857, 0.004877] | -0.000029 [-0.000174, 0.000120] |
| UCI | -1.2 [-3.0, 0.5] | 1.827372 [1.004130, 2.732715] | -0.000250 [-0.002696, 0.002223] | -0.000098 [-0.000262, 0.000059] |
| PPDai | 0.4 [-0.6, 1.5] | 1.023611 [0.497458, 1.606020] | -0.000521 [-0.002704, 0.001417] | -0.000064 [-0.000134, 0.000005] |

The same-state audit evaluates all selectors on each CountOnly state using an
identical visible set, candidate set, cost vector, and numerical budget. LRBE-IG
has zero mean count sacrifice in all four datasets. QualityFirst-IG mean count
sacrifice is 3.46 on HMEQ, 5.37 on LendingClub, 4.04 on UCI, and 0.02 on PPDai.
The corresponding mean Jaccard overlap with CountOnly is 0.436, 0.602, 0.135,
and 0.224. These are instantaneous mechanism descriptions, not counterfactual
downstream effects.

The dynamic evidence supports two bounded statements. First, preserving K-star
materially raises acquired feedback relative to QualityFirst-IG on HMEQ,
LendingClub, and UCI, but no Recall interval excludes zero; UCI Brier is slightly
worse for LRBE-IG. Second, LRBE-IG consumes more predicted risk than MinRisk on
all four datasets, while no Recall or Brier interval excludes zero. An interval
containing zero is inconclusive and is not treated as equivalence.

The 200 dynamic episodes record 13219.532 seconds of episode-level runtime.
Mean LRBE-IG episode time is 8.9 seconds on HMEQ, 498.6 on LendingClub, 19.9 on
UCI, and 16.6 on PPDai. Same-state audit time is outside this total.

Machine-readable sources:

- `results/information_gain_primary_20260907/`
- `results/information_gain_primary_20260907_analysis/integrity_summary.json`
- `results/information_gain_primary_20260907_analysis/method_summary.csv`
- `results/information_gain_primary_20260907_analysis/paired_comparisons.csv`
- `results/information_gain_primary_20260907_analysis/state_audit_summary.csv`

### Controlled mechanism-conditions confirmation (completed 2026-09-08)

This synthetic one-step experiment varies exposure heterogeneity (unit exposure
versus normalized lognormal exposure) and the evaluation weight assigned to a
region underrepresented in the initial visible set (0.1 versus 0.5). Twenty
prespecified seeds (2001--2020), four selectors, a risk ratio of 0.1, and an
M=20 bootstrap committee produce 160 training branches and 320 endpoint rows.
All selectors within a seed and exposure condition share the visible set,
candidate set, predicted costs, numerical budget, and branch-training seed.

The seed-controlled confirmation reproduces the original mechanism block
exactly for selected count, predicted risk, IG-score sum, Brier score, and
Recall across all 320 rows. Candidate/test-label perturbation audits pass, and
all canonical fits use the GPU backend. LRBE-IG acquires 2.5 [1.45, 3.65125]
more samples than QualityFirst-IG under unit exposure and 3.2 [1.75, 4.85]
under heterogeneous exposure. Relative to MinRisk, LRBE-IG uses 0.221061
[0.164636, 0.283944] more predicted risk under unit exposure and 0.135967
[0.100955, 0.170973] under heterogeneous exposure. All four primary Brier
intervals for LRBE-IG minus MinRisk contain zero.

The prespecified interactions are inconclusive: the high-minus-low exposure
difference in QualityFirst count sacrifice is 0.7 [0, 1.4]; the change in the
IG-minus-MinRisk Brier contrast when the target-region weight increases from
0.1 to 0.5 is 0.000126052 [-0.000005353, 0.000254801] under unit exposure and
-0.000058231 [-0.000217445, 0.000101574] under heterogeneous exposure; their
difference is -0.000184283 [-0.000412895, 0.000029288]. The average incoming
batch contains about 3,751 ordinary approvals, whereas mean exploratory
$K^*$ is 29.55 under unit exposure and 118.7 under heterogeneous exposure.
This scale contrast is descriptive and does not by itself identify why the
prediction contrasts are small.

Machine-readable sources:

- `results/mechanism_conditions_seed_controlled_confirmation_20260908/`
- `results/mechanism_conditions_seed_controlled_confirmation_20260908_analysis/`
