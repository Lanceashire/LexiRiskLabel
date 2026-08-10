# Phase 1 mechanism report

The mechanism is selective-label feedback: rejected applicants are not observed unless they are in the Oracle treatment or selected by current-round random exploration.

## Mechanistic conclusions

1. Visible TDR and FDRatio are the most consistent distortion signals across UCI and HMEQ.
2. ROC-AUC is comparatively stable, so aggregate ranking metrics can conceal feedback degradation.
3. HMEQ shows a stronger default-class and calibration penalty than UCI.
4. Random exploration improves observed feedback health monotonically with rate, but it does not guarantee recovery of Recall, BA, or calibration.
5. The effect is dataset-dependent; no forced cross-dataset unification is made.

See PHASE1_INTEGRITY_AUDIT.md, trajectory_summary.csv, threshold_crossing_summary.csv, and reliability_bins.csv for audit and mechanism evidence.
