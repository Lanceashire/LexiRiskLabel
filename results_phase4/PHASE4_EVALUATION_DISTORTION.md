# Evaluation Distortion versus Genuine Degradation

Phase 1 dual-evaluation predictions were reorganized without retraining.

    dataset treatment  recall_gap_mean  balanced_accuracy_gap_mean  roc_auc_gap_mean  pr_auc_gap_mean  brier_gap_mean  ece_gap_mean
       hmeq NoExplore           0.1964                      0.0915            0.0285           0.1335          0.0059       -0.0042
       hmeq    Oracle           0.0092                      0.0042            0.0007           0.0059         -0.0000       -0.0120
        uci NoExplore           0.1638                      0.0692            0.0387           0.0960          0.0058       -0.0037
        uci    Oracle          -0.0010                     -0.0007            0.0021           0.0008         -0.0002       -0.0049

An internal-visible score can differ from performance on the fixed external
test set, which is evaluation distortion. Separately, trajectory changes on the
fixed external set indicate genuine model/operating-point degradation. The two
are analytically distinct even when both arise from selective feedback.
