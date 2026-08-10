# Phase 4 Counterevidence

The following observations constrain or could falsify the explanatory story:

- Exact selected-set overlap is unknown. Similar downstream performance cannot
  be attributed either to high overlap or to interchangeable low-overlap sets.
- Applicant-level boundary distance and exact two-way crossings are unavailable.
  The proposed prevalence-versus-boundary account is supported by aggregate
  patterns, not directly proven at applicant level.
- Exposure heterogeneity alone is insufficient: LendingClub has heterogeneous
  EAD, but sophisticated acquisition still lacks stable downstream superiority.
- Feedback sensitivity is dataset-dependent:

        dataset performance_metric  spearman_rho  bootstrap95_low  bootstrap95_high
           hmeq        recall_gain        0.2778           0.2017            0.3505
           hmeq  brier_improvement        0.1244           0.0429            0.2071
    lendingclub        recall_gain        0.1567           0.0786            0.2385
    lendingclub  brier_improvement        0.2379           0.1510            0.3244
          ppdai        recall_gain       -0.0296          -0.1155            0.0547
          ppdai  brier_improvement        0.1589           0.0685            0.2437
            uci        recall_gain       -0.0132          -0.0958            0.0746
            uci  brier_improvement        0.4786           0.4109            0.5384

  Weak or inconsistent associations imply feedback representativeness may not
  be the downstream bottleneck.
- PPDai's hidden-default fraction is not present in frozen artifacts, and its
  source provenance remains lower-confidence. It cannot confirm superiority or
  fully anchor the selection-severity gradient.
- Cell-level selected PD/EAD boxes are distributions of aggregate means, not
  applicant-level distributions.
- Runtime evidence is implementation/hardware-specific. FAVE is about
  1.64× the CountOnly-MinRisk mean runtime in the frozen
  LendingClub profile, but speed alone does not establish methodological
  invalidity.
- Four datasets provide a descriptive pattern, not a universal law.
