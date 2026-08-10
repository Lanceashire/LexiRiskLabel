# Master experiment plan (evidence-integrated)

1. **Research question:** risk-budgeted acquisition under selective labels.
2. **Formal problem:** obtain feedback subject to an expected-risk budget while decisions reveal labels selectively.
3. **Mechanism:** observed labels are selection-biased; feedback health and decision performance are distinct outcomes.
4. **Risk budget:** frozen Phase 2/3 values are defined by the exact declarations in `configs/phase2/` and `configs/phase3/`; no parameters are reconstructed here.
5. **Frozen policies:** FAVE-v2.1, LRBE, CountOnly-MinRisk, RiskTilted-Uncertainty and stated comparators; see registries.
6. **Datasets:** UCI and HMEQ development; PPDai exploratory boundary; LendingClub primary external validation.
7. **Metrics:** feedback composition/health, labels and expected risk, recall, BA, calibration, paired and bootstrap summaries.
8. **Phase 0:** protocol repair and FAVE reference recovery — complete.
9. **Phase 1:** mechanism validation and evaluation-distortion addendum — complete.
10. **Phase 2:** risk-budgeted policy comparison — complete.
11. **Phase 3:** frozen external validation — complete.
12. **Phase 4:** analysis-only mechanism robustness — complete.
13. **Hypotheses:** decisions are frozen in phase reports; unsupported findings remain retained.
14. **Post-hoc analyses:** severity, heterogeneity, and feedback sensitivity are partially supported explanatory patterns.
15. **Stop rule:** no automatic additional experiment or Phase 5.
16. **Final findings:** feedback recovery is not proportional downstream recovery; FAVE lacks stable downstream superiority; quantity-first safe acquisition is competitive with external support.
17. **Limitations:** PPDai provenance boundary, policy differentiation conditions, and non-causal correlation interpretations; see Phase 3/4 canonical reports.
