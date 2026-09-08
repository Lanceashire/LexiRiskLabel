# Controlled mechanism-conditions results

This report is generated from the complete validated evidence block.

- Seeds: 20
- Retraining branches: 160
- GPU validation: True
- Bootstrap draws: 10000

## Primary paired contrasts

| Exposure | Target region-1 weight | Contrast | Metric | Mean difference | 95% interval |
|---|---:|---|---|---:|---:|
| high | 0.1 | LRBE-IG - QualityFirst-IG | selected_count | 3.2 | [1.75, 4.85] |
| high | 0.1 | LRBE-IG - CountOnly-MinRisk | selected_predicted_risk | 0.135967 | [0.100955, 0.170973] |
| high | 0.1 | LRBE-IG - CountOnly-MinRisk | brier | -0.000101933 | [-0.000415109, 0.000198847] |
| high | 0.5 | LRBE-IG - QualityFirst-IG | selected_count | 3.2 | [1.75, 4.85] |
| high | 0.5 | LRBE-IG - CountOnly-MinRisk | selected_predicted_risk | 0.135967 | [0.100955, 0.170973] |
| high | 0.5 | LRBE-IG - CountOnly-MinRisk | brier | -0.000160164 | [-0.000564917, 0.000202895] |
| low | 0.1 | LRBE-IG - QualityFirst-IG | selected_count | 2.5 | [1.45, 3.65125] |
| low | 0.1 | LRBE-IG - CountOnly-MinRisk | selected_predicted_risk | 0.221061 | [0.164636, 0.283944] |
| low | 0.1 | LRBE-IG - CountOnly-MinRisk | brier | -7.95397e-06 | [-0.000242101, 0.00021023] |
| low | 0.5 | LRBE-IG - QualityFirst-IG | selected_count | 2.5 | [1.45, 3.65125] |
| low | 0.5 | LRBE-IG - CountOnly-MinRisk | selected_predicted_risk | 0.221061 | [0.164636, 0.283944] |
| low | 0.5 | LRBE-IG - CountOnly-MinRisk | brier | 0.000118098 | [-0.000142545, 0.000361566] |

## Preregistered interaction tests

| Test | Mean | 95% interval |
|---|---:|---:|
| H1_high_minus_low_exposure_QF_count_sacrifice | 0.7 | [0, 1.4] |
| H2_weight_0.5_minus_0.1_IG_Brier_contrast_low_exposure | 0.000126052 | [-5.35348e-06, 0.000254801] |
| H2_weight_0.5_minus_0.1_IG_Brier_contrast_high_exposure | -5.82311e-05 | [-0.000217445, 0.000101574] |
| H3_exposure_difference_in_H2 | -0.000184283 | [-0.000412895, 2.92876e-05] |

Intervals containing zero are treated as inconclusive. Same-state one-step results do not identify field causal effects or long-run dynamic policy value.
