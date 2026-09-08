# Controlled mechanism experiment preregistration

Experiment ID: `MECHANISM_CONDITIONS_20260908`. This additive synthetic
evidence block is fixed before formal endpoints are inspected. It does not
replace or pool with the four-dataset dynamic experiments.

## Questions and directional hypotheses

The experiment asks when information-oriented acquisition is worth its
feedback-count and predicted-risk cost.

- H1: increasing exposure heterogeneity increases the count sacrifice of
  `QualityFirst-IG` relative to the maximum feasible count `K*`.
- H2: increasing the target weight of the initially underrepresented region
  makes the Brier contrast `LRBE-IG - CountOnly-MinRisk` more favorable to IG.
- H3: the H2 change can depend on exposure heterogeneity. The corresponding
  difference-in-differences is reported without assuming its direction.

All four cells and all intervals are reported. An interval containing zero is
inconclusive, not evidence of equivalence. Unsupported hypotheses are retained
as null findings and removed from the paper's positive claims.

## Fixed design

For each of seeds 2001--2020, a transparent synthetic credit task contains an
initial pool of 4,000 applicants, one incoming batch of 4,000, and two disjoint
test strata of 4,000 applicants each. Six standard-normal features and an
observed binary region enter the fixed default mechanism

`logit(P(default)) = -1 + 1.3*x1 - 0.8*x2 + region*(1.5*x3 + 0.7*x4)`.

The 1,000 initial visible observations are the lowest values of a feature-only
selection score that penalizes the second region. Labels do not enter this
selection rule. A single canonical XGBoost model, fitted after preprocessing
on the visible set, partitions the incoming batch at `tau=0.5`. All policies
then share the visible set, accepted applicants, candidate set, predicted PD,
cost vector, and numerical budget within an exposure condition.

Low exposure is one for every candidate. High exposure is a deterministic
log-normal draw with log standard deviation 1.2 and is rescaled so that the
sum of candidate predicted risk equals the low-exposure total. Thus the
numerical budget `B=0.1*sum(PD_hat*exposure)` is identical across exposure
conditions, while the allocation of costs changes. Exposure is excluded from
the prediction features.

The four policies are `CountOnly-MinRisk`, `LRBE-IG`, `QualityFirst-IG`, and
`MaxCount-Random`. IG is the same 20-member ordinary bootstrap-committee mutual
information score used in the independent real-data extension. The committee
uses visible labels only. `MaxCount-Random` uses a deterministic random utility
inside the exact `K*` constraint; it is a reproducible feasible perturbation,
not a uniform draw over all feasible subsets.

After acquisition, each branch is retrained once on the common initial visible
set, common automatically accepted applicants, and its selected rejected
applicants. The same predictions on two held-out region strata are evaluated
under fixed target region weights 0.1 and 0.5. The target weights affect metrics
only and never affect training, scoring, selection, or tuning.

## Outcomes and analysis

Primary predictive outcome: weighted Brier score. Secondary predictive outcome:
weighted Recall at 0.5. Mechanism outcomes include selected count, `K*`, count
sacrifice, predicted risk, realized loss, composition premium, maximum-count
slack, released minimum risk, unused budget, budget-identity residual, selected
region composition, IG sum, and Jaccard overlap with MinRisk.

Primary paired contrasts are `LRBE-IG - QualityFirst-IG` and `LRBE-IG -
CountOnly-MinRisk`. Seed-level paired bootstrap intervals use 10,000 draws.
H1 is the high-minus-low exposure change in QualityFirst-IG count sacrifice.
H2 is the weight-0.5 minus weight-0.1 change in the LRBE-IG-minus-MinRisk Brier
contrast, reported separately by exposure condition. H3 is the high-exposure
minus low-exposure difference between those H2 changes.

The same-state, one-step design identifies immediate selection and retraining
differences under the synthetic task. It is not a field causal estimate, a
dynamic long-horizon policy evaluation, or proof that the manipulated factors
exhaust the conditions governing information value.
