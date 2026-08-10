# Phase 4 Final Static Robustness Patch

**Mode: analysis-only.** No model, policy, threshold, budget, seed, dataset,
preprocessing step, or Phase 1-3 frozen result was changed.

## Correction to feedback-performance sensitivity

The earlier across-budget correlation is retained only as
BUDGET_CONFOUNDED_DESCRIPTIVE. Primary robustness evidence now uses separate
dataset×rho Spearman estimates and dataset-level budget-centered estimates.
Every 95% interval uses 10,000 seed-cluster bootstrap draws: 20 seeds are
sampled with replacement, retaining all policy and (where applicable) budget
observations for each sampled seed.

Within-budget FeedbackGain versus DecisionGain:

        dataset          rho   analysis_type  spearman_rho  cluster_bootstrap_ci_low  cluster_bootstrap_ci_high
           hmeq        0.025   WITHIN_BUDGET        0.0364                   -0.2409                     0.3269
           hmeq         0.05   WITHIN_BUDGET       -0.1374                   -0.3781                     0.1184
           hmeq          0.1   WITHIN_BUDGET       -0.1739                   -0.4041                     0.1028
           hmeq          0.2   WITHIN_BUDGET        0.0159                   -0.3242                     0.3399
           hmeq ALL_CENTERED BUDGET_CENTERED       -0.0668                   -0.3014                     0.1863
    lendingclub        0.025   WITHIN_BUDGET       -0.1798                   -0.4077                     0.1114
    lendingclub         0.05   WITHIN_BUDGET       -0.2666                   -0.5578                     0.0790
    lendingclub          0.1   WITHIN_BUDGET       -0.1408                   -0.4931                     0.2445
    lendingclub          0.2   WITHIN_BUDGET       -0.0919                   -0.4486                     0.2602
    lendingclub ALL_CENTERED BUDGET_CENTERED       -0.1550                   -0.4291                     0.1487
          ppdai        0.025   WITHIN_BUDGET       -1.0000                   -1.0000                    -1.0000
          ppdai         0.05   WITHIN_BUDGET       -0.1088                   -0.3678                     0.1830
          ppdai          0.1   WITHIN_BUDGET        0.0626                   -0.2852                     0.4223
          ppdai          0.2   WITHIN_BUDGET       -0.0151                   -0.3090                     0.2697
          ppdai ALL_CENTERED BUDGET_CENTERED       -0.0415                   -0.2950                     0.2137
            uci        0.025   WITHIN_BUDGET        0.1671                   -0.1279                     0.4440
            uci         0.05   WITHIN_BUDGET        0.0246                   -0.2987                     0.3407
            uci          0.1   WITHIN_BUDGET       -0.0706                   -0.4063                     0.2906
            uci          0.2   WITHIN_BUDGET       -0.3065                   -0.6411                     0.1576
            uci ALL_CENTERED BUDGET_CENTERED       -0.0774                   -0.3805                     0.2527

Budget-centered results:

        dataset                     metric_pair  spearman_rho  cluster_bootstrap_ci_low  cluster_bootstrap_ci_high
           hmeq    FeedbackGain_vs_DecisionGain       -0.0668                   -0.3014                     0.1863
           hmeq FeedbackGain_vs_CalibrationGain       -0.0931                   -0.3427                     0.1881
    lendingclub    FeedbackGain_vs_DecisionGain       -0.1550                   -0.4291                     0.1487
    lendingclub FeedbackGain_vs_CalibrationGain       -0.0686                   -0.3853                     0.2562
          ppdai    FeedbackGain_vs_DecisionGain       -0.0415                   -0.2950                     0.2137
          ppdai FeedbackGain_vs_CalibrationGain        0.1901                   -0.0260                     0.3956
            uci    FeedbackGain_vs_DecisionGain       -0.0774                   -0.3805                     0.2527
            uci FeedbackGain_vs_CalibrationGain       -0.0812                   -0.2960                     0.1317

The direction and magnitude vary across datasets and budgets. The defensible
interpretation is:

**Recovering more representative feedback does not reliably predict greater
downstream decision recovery under the same risk budget.**

This does not claim that feedback recovery has no effect; it states that
recovery does not reliably translate into proportional downstream gains.

## Quantity-performance correction

Quantity correlations received the identical within-budget, budget-centered,
and seed-cluster treatment:

        dataset                 metric_pair  spearman_rho  cluster_bootstrap_ci_low  cluster_bootstrap_ci_high
           hmeq    Quantity_vs_DecisionGain       -0.0508                   -0.3447                     0.2691
           hmeq Quantity_vs_CalibrationGain       -0.2386                   -0.4995                     0.0942
    lendingclub    Quantity_vs_DecisionGain       -0.2073                   -0.5135                     0.1643
    lendingclub Quantity_vs_CalibrationGain       -0.0145                   -0.3887                     0.3352
          ppdai    Quantity_vs_DecisionGain       -0.1679                   -0.3829                     0.0611
          ppdai Quantity_vs_CalibrationGain        0.1657                   -0.1320                     0.4365
            uci    Quantity_vs_DecisionGain       -0.2602                   -0.5558                     0.1283
            uci Quantity_vs_CalibrationGain       -0.0957                   -0.3689                     0.2084

Quantity-first evidence does not rely on a simple correlation. The primary
evidence is the matched dataset×seed×rho comparison:

        dataset                 policy  labels_mean  label_range_mean  expected_risk_mean  risk_excess_vs_minrisk_mean  recall_gain_mean  brier_improvement_mean
           hmeq      CountOnly-MinRisk     130.4500            3.0750         563273.1278                       0.0000            0.0235                  0.0007
           hmeq              FAVE-v2.1     130.5625            3.0750         587561.7692                   24288.6414            0.0225                  0.0005
           hmeq RiskTilted-Uncertainty     130.0875            3.0750         587509.8690                   24236.7411            0.0248                  0.0007
    lendingclub      CountOnly-MinRisk     458.9625            6.5000        2205325.7412                       0.0000           -0.0003                  0.0005
    lendingclub              FAVE-v2.1     458.6250            6.5000        2236999.6142                   31673.8730           -0.0007                  0.0005
    lendingclub RiskTilted-Uncertainty     459.5625            6.5000        2240312.4215                   34986.6803           -0.0000                  0.0005
          ppdai      CountOnly-MinRisk      11.6125            1.5625              5.9296                       0.0000           -0.0000                  0.0000
          ppdai              FAVE-v2.1      11.8000            1.5625              6.5411                       0.6115            0.0002                  0.0000
          ppdai RiskTilted-Uncertainty      12.0000            1.5625              6.8372                       0.9076           -0.0001                  0.0000
            uci      CountOnly-MinRisk     247.6750            3.6375            134.7116                       0.0000            0.0009                  0.0001
            uci              FAVE-v2.1     246.8125            3.6375            136.4973                       1.7857            0.0014                  0.0002
            uci RiskTilted-Uncertainty     247.6625            3.6375            137.5386                       2.8270            0.0018                  0.0001

Under frozen LRBE, the three focus policies acquire nearly equal label
quantities in matched cells. CountOnly-MinRisk typically uses less expected
risk and does not exhibit a stable downstream loss. No causal statement that
more labels improve recall is made.

## Terminology corrections

- M7 is Mean predicted PD among true defaults — aggregate diagnostic.
- M8 is NET THRESHOLD-STATUS CHANGE.
- Exact applicant-level near-threshold distributions and separate up/down
  crossings remain unavailable.
