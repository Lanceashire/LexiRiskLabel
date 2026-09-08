# Controlled mechanism-conditions experiment: final results

## Completion and integrity

The preregistered primary block is complete. It contains 20 independent seeds
(2001--2020), two exposure conditions, four acquisition policies, and two
fixed target-region weights: 160 acquisition/retraining branches and 320
endpoint rows. The 20-member IG committee was shared within each seed's common
initial state. Every canonical model fit used the registered GPU path.

All common-state, label-isolation, target-weight invariance, predicted-budget
feasibility, and budget-decomposition checks passed. The maximum absolute
budget-identity residual is zero. The complete run took 37.70 recorded seconds
across seeds on the current machine; this is a local runtime observation, not a
hardware-independent benchmark.

## Main findings

The fixed-`K*` constraint changes acquisition quantity under both cost
conditions. `LRBE-IG - QualityFirst-IG` acquires 2.50 more labels under unit
exposure (95% paired bootstrap interval [1.45, 3.65]) and 3.20 more under
heterogeneous exposure ([1.75, 4.85]). Neither corresponding Brier or Recall
contrast has a stable direction. Thus preserving `K*` prevents an immediate
count sacrifice, but the experiment does not show that the extra labels
improve the one-step predictive endpoint.

At the same `K*`, IG changes the selected subset and uses more predicted risk
than MinRisk. `LRBE-IG - CountOnly-MinRisk` predicted-risk differences are
0.221 under unit exposure ([0.165, 0.284]) and 0.136 under heterogeneous
exposure ([0.101, 0.171]). The corresponding IG-score gains are 0.577
([0.441, 0.718]) and 0.200 ([0.154, 0.248]). None of the four primary Brier
intervals excludes zero.

The feasible random control isolates score optimization more directly.
`LRBE-IG` obtains more IG score than `MaxCount-Random` under unit exposure
(difference 0.555, [0.427, 0.689]) and heterogeneous exposure (0.211,
[0.147, 0.279]), while its Brier intervals relative to random contain zero for
both target weights and both exposure conditions. This is direct same-state
evidence that improving the acquisition proxy need not improve the downstream
prediction target in one retraining step.

One secondary Recall contrast is adverse to IG: under unit exposure and target
region-1 weight 0.5, `LRBE-IG - CountOnly-MinRisk` is -0.00218
([-0.00406, -0.00035]). Recall is secondary and the analysis contains several
reported contrasts, so this isolated result is not promoted to a general
claim.

## Preregistered condition tests

| Hypothesis test | Mean | 95% paired bootstrap interval | Decision |
|---|---:|---:|---|
| H1: high-minus-low exposure change in QualityFirst-IG count sacrifice | 0.700 | [0.000, 1.400] | Inconclusive; interval includes zero |
| H2, unit exposure: weight 0.5-minus-0.1 change in IG-minus-MinRisk Brier | 0.000126 | [-0.000005, 0.000255] | Inconclusive |
| H2, heterogeneous exposure | -0.000058 | [-0.000217, 0.000102] | Inconclusive |
| H3: exposure difference in the H2 change | -0.000184 | [-0.000413, 0.000029] | Inconclusive |

The manipulation therefore does not establish that exposure heterogeneity or
target-region weight explains when IG improves prediction. No follow-up
parameter tuning is performed on this evidence block.

## Paper-facing interpretation

This block strengthens a narrow mechanism claim: under an identical candidate
set, predicted-cost vector, and numerical budget, (i) allowing an acquisition
score to choose cardinality sacrifices feasible labels, and (ii) maximizing a
measured information score changes composition and risk use without a detected
improvement in the primary predictive endpoint. It does not establish a
universal quantity-first optimum, equivalence between methods, a causal field
effect, or the conditions under which information-oriented acquisition must
succeed.

Machine-readable results are in
`results/mechanism_conditions_primary_20260908_analysis/`. The generated
`mechanism_conditions.pdf` and `.png` visualize count sacrifice, composition
premium, and the Brier contrasts. Smoke outputs are diagnostic only and are
excluded from the formal evidence block.
