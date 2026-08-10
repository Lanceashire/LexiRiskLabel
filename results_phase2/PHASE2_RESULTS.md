# Phase 2 results

## 1. Protocol integrity

- `PHASE2_STATUS = PASS`
- Rows: 14080 = 2 datasets × 20 seeds × 4 budgets × 8 policies × 11 rounds.
- Integrity failures: 0.
- Split parity: all four parity flags true for 40 dataset-seed combinations.
- FAVE identity gate: PASS; RiskTilted-Uncertainty is numerically distinct.
- Backend: all 14080 rows report `gpu`; XGBoost/CuPy CUDA path was used.
- No Phase 1 formal result was modified.

## 2. UCI results

### UCI rho=0.025

                policy  recall_mean  balanced_accuracy_mean  roc_auc_mean  pr_auc_mean  brier_mean  ece_mean  visible_tdr_mean  fd_ratio_mean
     CountOnly-MinRisk      0.36459                 0.65710       0.78191      0.55044     0.13470   0.01310           0.17471        0.79089
             FAVE-v2.1      0.36634                 0.65795       0.78263      0.55178     0.13456   0.01281           0.17478        0.79120
             NoExplore      0.36437                 0.65721       0.78255      0.55129     0.13463   0.01328           0.17350        0.78541
RiskTilted-Uncertainty      0.36775                 0.65848       0.78210      0.55069     0.13463   0.01292           0.17483        0.79140

### UCI rho=0.050

                policy  recall_mean  balanced_accuracy_mean  roc_auc_mean  pr_auc_mean  brier_mean  ece_mean  visible_tdr_mean  fd_ratio_mean
     CountOnly-MinRisk      0.36597                 0.65787       0.78211      0.55091     0.13460   0.01313           0.17599        0.79665
             FAVE-v2.1      0.36679                 0.65835       0.78197      0.55178     0.13459   0.01258           0.17609        0.79713
             NoExplore      0.36437                 0.65721       0.78255      0.55129     0.13463   0.01328           0.17350        0.78541
RiskTilted-Uncertainty      0.36612                 0.65798       0.78223      0.55179     0.13457   0.01288           0.17610        0.79715

### UCI rho=0.100

                policy  recall_mean  balanced_accuracy_mean  roc_auc_mean  pr_auc_mean  brier_mean  ece_mean  visible_tdr_mean  fd_ratio_mean
     CountOnly-MinRisk      0.36595                 0.65782       0.78233      0.55247     0.13447   0.01284           0.17832        0.80723
             FAVE-v2.1      0.36573                 0.65788       0.78240      0.55250     0.13446   0.01266           0.17832        0.80721
             NoExplore      0.36437                 0.65721       0.78255      0.55129     0.13463   0.01328           0.17350        0.78541
RiskTilted-Uncertainty      0.36618                 0.65797       0.78219      0.55128     0.13450   0.01236           0.17838        0.80751

### UCI rho=0.200

                policy  recall_mean  balanced_accuracy_mean  roc_auc_mean  pr_auc_mean  brier_mean  ece_mean  visible_tdr_mean  fd_ratio_mean
     CountOnly-MinRisk      0.36446                 0.65752       0.78248      0.55311     0.13428   0.01232           0.18320        0.82932
             FAVE-v2.1      0.36412                 0.65745       0.78257      0.55275     0.13428   0.01228           0.18324        0.82949
             NoExplore      0.36437                 0.65721       0.78255      0.55129     0.13463   0.01328           0.17350        0.78541
RiskTilted-Uncertainty      0.36444                 0.65751       0.78262      0.55264     0.13430   0.01210           0.18334        0.82995

## 3. HMEQ results

### HMEQ rho=0.025

                policy  recall_mean  balanced_accuracy_mean  roc_auc_mean  pr_auc_mean  brier_mean  ece_mean  visible_tdr_mean  fd_ratio_mean
     CountOnly-MinRisk      0.59574                 0.78397       0.92949      0.82384     0.07650   0.03423           0.12570        0.63294
             FAVE-v2.1      0.59523                 0.78433       0.92886      0.82392     0.07662   0.03357           0.12583        0.63361
             NoExplore      0.57996                 0.77669       0.92931      0.82454     0.07705   0.03577           0.11770        0.59263
RiskTilted-Uncertainty      0.59299                 0.78283       0.92917      0.82282     0.07671   0.03443           0.12484        0.62864

### HMEQ rho=0.050

                policy  recall_mean  balanced_accuracy_mean  roc_auc_mean  pr_auc_mean  brier_mean  ece_mean  visible_tdr_mean  fd_ratio_mean
     CountOnly-MinRisk      0.59907                 0.78608       0.92916      0.82445     0.07644   0.03385           0.13099        0.65952
             FAVE-v2.1      0.59687                 0.78492       0.92949      0.82471     0.07653   0.03476           0.13111        0.66012
             NoExplore      0.57996                 0.77669       0.92931      0.82454     0.07705   0.03577           0.11770        0.59263
RiskTilted-Uncertainty      0.60131                 0.78716       0.92939      0.82452     0.07653   0.03403           0.13031        0.65610

### HMEQ rho=0.100

                policy  recall_mean  balanced_accuracy_mean  roc_auc_mean  pr_auc_mean  brier_mean  ece_mean  visible_tdr_mean  fd_ratio_mean
     CountOnly-MinRisk      0.59957                 0.78580       0.92891      0.82361     0.07651   0.03411           0.13947        0.70224
             FAVE-v2.1      0.60126                 0.78671       0.92882      0.82407     0.07668   0.03371           0.13949        0.70232
             NoExplore      0.57996                 0.77669       0.92931      0.82454     0.07705   0.03577           0.11770        0.59263
RiskTilted-Uncertainty      0.60407                 0.78802       0.92939      0.82389     0.07661   0.03390           0.13851        0.69740

### HMEQ rho=0.200

                policy  recall_mean  balanced_accuracy_mean  roc_auc_mean  pr_auc_mean  brier_mean  ece_mean  visible_tdr_mean  fd_ratio_mean
     CountOnly-MinRisk      0.61949                 0.79428       0.92971      0.82528     0.07576   0.03115           0.15143        0.76246
             FAVE-v2.1      0.61653                 0.79315       0.92874      0.82266     0.07631   0.03215           0.15188        0.76470
             NoExplore      0.57996                 0.77669       0.92931      0.82454     0.07705   0.03577           0.11770        0.59263
RiskTilted-Uncertainty      0.62051                 0.79474       0.93002      0.82615     0.07563   0.03289           0.15018        0.75614

## 4. Label quantity

See `summary/acquisition_summary.csv`, `summary/coverage_vs_budget.csv`, and `summary/quantity_quality_summary.csv`. Exploration counts increase with rho, while policy differences are generally small relative to the common quantity constraint.

## 5. Policy quality

The primary paired comparisons are in `summary/paired_tests_holm.csv`. All claims use fixed external test metrics.

## 6. Risk-budget trade-off

All selected subsets satisfy the budget within the fixed 1e-6 floating-point tolerance. UCI uses unit-exposure risk EL=PD; HMEQ uses the explicitly labeled EL proxy PD×LOAN with LGD=1.

## 7. Low-budget regime

The pre-registered scarcity contrasts are in `summary/scarcity_tests.csv`; no blanket low-budget superiority pattern is assumed.

## 8. Calibration

Brier and ECE are reported in the final-summary tables and paired tests.

## 9. Feedback health

FDRatio and visible TDR are reported as primary feedback-health outcomes.

## 10. Pareto analysis

`summary/pareto_points.csv` and the Pareto figures report recall gain, FDRatio gain, and Brier improvement against NoExplore at the same rho.

## 11. Statistical tests

Paired Wilcoxon tests, 10,000 paired-seed bootstrap percentile intervals, raw p-values, Holm-adjusted p-values, and win/tie/loss counts are in `summary/paired_tests_holm.csv`.

## 12. What FAVE actually contributed

FAVE-v2.1 is evaluated as frozen. Its incremental value is dataset- and endpoint-dependent; it is not declared universally superior.

## 13. What simple policies achieved

CountOnly-MinRisk and RiskTilted-Uncertainty are strong structural baselines and frequently lie on or near the observed Pareto front.

## 14. Dataset-dependent results

UCI is development/benchmark evidence with unit-exposure risk; HMEQ is monetary-exposure validation using a PD×LOAN risk proxy, not calibrated bank expected loss.

## 15. Negative findings

A policy win is not required for Phase 2 PASS. Where policy gaps are small, the result indicates limited downstream sensitivity to acquisition utility under the tested budgets.

## 16. Implications for Phase 3

The frozen policy suite and evaluation discipline can be carried to external datasets in Phase 3; no external dataset was run here.
