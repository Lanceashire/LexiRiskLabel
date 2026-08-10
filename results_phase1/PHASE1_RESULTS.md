# Phase 1 - Selective-label Feedback Mechanism Validation

## Status

PHASE1_STATUS = PASS_WITH_MIXED_MECHANISMS

Protocol: PASS; UCI: COMPLETE (20/20 seeds); HMEQ: COMPLETE (20/20 seeds).
Phase 1 is complete. Per protocol, do not enter Phase 2 automatically.

## Treatments

Oracle, NoExplore, RandomExplore-2%, RandomExplore-5%, RandomExplore-10%. FAVE and risk-budget policies were excluded.

## UCI final round (mean over 20 seeds)

                   recall  balanced_accuracy  visible_tdr  fd_ratio   brier     ece  roc_auc  pr_auc
treatment                                                                                           
NoExplore          0.3670             0.6582       0.1737    0.7862  0.1346  0.0128   0.7823  0.5510
Oracle             0.3636             0.6576       0.2209    1.0000  0.1341  0.0115   0.7821  0.5589
RandomExplore-10%  0.3644             0.6572       0.1789    0.8098  0.1346  0.0125   0.7821  0.5527
RandomExplore-2%   0.3632             0.6565       0.1746    0.7906  0.1347  0.0130   0.7821  0.5513
RandomExplore-5%   0.3643             0.6571       0.1762    0.7977  0.1346  0.0138   0.7820  0.5520

## HMEQ final round (mean over 20 seeds)

                   recall  balanced_accuracy  visible_tdr  fd_ratio   brier     ece  roc_auc  pr_auc
treatment                                                                                           
NoExplore          0.5805             0.7776       0.1179    0.5936  0.0772  0.0366   0.9295  0.8231
Oracle             0.6423             0.8034       0.1986    1.0000  0.0753  0.0267   0.9278  0.8249
RandomExplore-10%  0.5868             0.7796       0.1263    0.6360  0.0770  0.0345   0.9284  0.8233
RandomExplore-2%   0.5817             0.7776       0.1196    0.6024  0.0773  0.0361   0.9290  0.8211
RandomExplore-5%   0.5838             0.7782       0.1218    0.6134  0.0773  0.0361   0.9288  0.8222

## Hypothesis decisions

- H1 Selective-label feedback distorts visible TDR/FDRatio: SUPPORTED on both datasets. NoExplore has lower Visible TDR and FDRatio than Oracle; paired Holm-adjusted tests support this.
- H2 Default-class recall/BA may degrade versus Oracle: SUPPORTED for HMEQ (Recall and Balanced Accuracy lower than Oracle), NOT SUPPORTED for UCI (Recall/BA do not show the same degradation).
- H3 Accuracy/AUC may remain stable while recall/TDR degrade: SUPPORTED as a mechanism pattern. ROC-AUC remains close to Oracle while feedback health is distorted; the direction of Recall is dataset-dependent.
- H4 Calibration may deteriorate without ranking collapse: MIXED / DATASET-DEPENDENT. HMEQ shows worse Brier/ECE under NoExplore; UCI shows a smaller and mixed calibration change.
- H5 Simple random exploration may recover part of the Oracle gap: SUPPORTED for feedback health (TDR/FDRatio move toward Oracle), but decision/calibration recovery is weak or inconclusive after Holm correction.

## Interpretation

Selective labels first change which defaults become visible, not necessarily ranking discrimination. HMEQ exhibits a clear Recall/BA and calibration penalty, while UCI mainly shows TDR/FDRatio distortion with stable AUC and no matching Recall collapse. Random current-round exploration recovers part of the feedback-health gap, but its recovery of decision and calibration endpoints is dataset-dependent.

## Post-hoc Findings

- Dual evaluation is post-hoc and is not assigned a new H-number. The external fixed-test evaluator exactly reproduces the frozen all-rounds results; the internal visible-only evaluator produces systematic Oracle--NoExplore distortion in both datasets. See PHASE1_ADDENDUM_DUAL_EVALUATION.md and dual_evaluation_statistics.csv.

## Stop condition

This report is the terminal Phase 1 output. No FAVE comparison, risk-budget optimization, new acquisition policy, PPDai/LendingClub expansion, or Phase 2 execution was performed.
