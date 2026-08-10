# Phase 1 final mechanism summary

## Hypothesis numbering

The original preregistered H1--H5 numbering is preserved. The dual evaluator is reported separately under **Post-hoc Findings** and is not treated as H6.
## Final status

`PHASE1_FINAL_STATUS = PASS_WITH_EVALUATION_DISTORTION_FINDING`

The frozen experiment passes its integrity checks. The closeout addendum establishes that a visible-only internal evaluation can materially distort the apparent Oracle--NoExplore comparison; it must not replace the fixed external test evaluation.

## Mechanisms supported by the data

1. Selective-label feedback is present. NoExplore has hidden defaults among rejected applicants, while Oracle observes the full batch. Mean hidden-default fractions are 0.3610 (UCI) and 0.5936 (HMEQ); these are descriptive summaries, not labels used by the learner.
2. Visible TDR and FDRatio are the most stable feedback-health signals. ROC-AUC is comparatively stable and can conceal feedback degradation.
3. Random exploration improves feedback observability, but downstream recovery is dataset-dependent and is not interpreted as a universal monotone performance guarantee.
4. Calibration-slope/intercept diagnostics are now reported with valid-count and uncertainty summaries; they are supplementary and do not alter the frozen primary claims.
5. Internal visible-set evaluation has a systematic evaluation-distortion mechanism. It is useful only as a diagnostic and cannot be used to claim external generalization.

## Scope boundary

No model, threshold, split, batch schedule, treatment definition, seed, round count, or formal result was changed. No Phase 2 experiment was executed.

