# Phase 4 Mechanism Report

All claims below are descriptive or cross-phase syntheses, not causal proofs.

## Quantity-first

Across quality policies, mean within-cell label-count CVs were:
{"hmeq": 0.0157, "lendingclub": 0.0092, "ppdai": 0.0937, "uci": 0.0082}.
This tests the quantity-controlled premise directly. CountOnly-MinRisk is
interpreted through the joint quantity–risk–performance table, not a
leaderboard: it often obtains similar label quantity while reducing expected
risk, with downstream results comparable to more complex policies.

## Feedback composition versus decisions

Within-dataset Spearman associations between FDRatio gain and recall gain were:
{"hmeq": 0.2778, "lendingclub": 0.1567, "ppdai": -0.0296, "uci": -0.0132}.
The full exploratory intervals are:

        dataset performance_metric  spearman_rho  bootstrap95_low  bootstrap95_high
           hmeq        recall_gain        0.2778           0.2017            0.3505
           hmeq  brier_improvement        0.1244           0.0429            0.2071
    lendingclub        recall_gain        0.1567           0.0786            0.2385
    lendingclub  brier_improvement        0.2379           0.1510            0.3244
          ppdai        recall_gain       -0.0296          -0.1155            0.0547
          ppdai  brier_improvement        0.1589           0.0685            0.2437
            uci        recall_gain       -0.0132          -0.0958            0.0746
            uci  brier_improvement        0.4786           0.4109            0.5384

FAVE can recover a more default-heavy observed stream without moving many true
defaults across tau=.5. The frozen aggregates support net threshold-status
changes only; they cannot identify exact up/down crossings. Therefore the
strongest defensible interpretation is prevalence/feedback-composition
recovery without stable decision-boundary superiority.

## Selected-set composition

        dataset                 policy  selected_pd  default_yield  selected_ead
           hmeq      CountOnly-MinRisk       0.7315         0.7808     5444.1661
           hmeq              FAVE-v2.1       0.7265         0.7772     5745.8625
           hmeq RiskTilted-Uncertainty       0.6909         0.7411     6101.1539
    lendingclub      CountOnly-MinRisk       0.6344         0.6184     6724.7891
    lendingclub              FAVE-v2.1       0.6362         0.6209     6811.5707
    lendingclub RiskTilted-Uncertainty       0.6215         0.6082     6999.9604
          ppdai      CountOnly-MinRisk       0.5079         0.5509        1.0000
          ppdai              FAVE-v2.1       0.5489         0.5033        1.0000
          ppdai RiskTilted-Uncertainty       0.5738         0.6082        1.0000
            uci      CountOnly-MinRisk       0.5296         0.5431        1.0000
            uci              FAVE-v2.1       0.5463         0.5518        1.0000
            uci RiskTilted-Uncertainty       0.5485         0.5631        1.0000

These are weighted summaries of frozen round-level selected-set aggregates.
They quantify composition differences but not applicant overlap. Exact overlap
remains unknown.

## Dataset mechanism profile

        dataset  baseline_rejection_rate  hidden_default_fraction  ead_cv  policy_recall_dispersion  feedback_sensitivity_spearman
            uci                   0.1192                   0.3621  0.0000                    0.0017                        -0.0132
           hmeq                   0.1427                   0.5949  0.6023                    0.0085                         0.2778
          ppdai                   0.0046                      NaN  0.0000                    0.0003                        -0.0296
    lendingclub                   0.1973                   0.7279  0.5796                    0.0043                         0.1567

PPDai remains exploratory boundary evidence; LendingClub remains the primary
external evidence. No pooled four-dataset significance test was performed.
