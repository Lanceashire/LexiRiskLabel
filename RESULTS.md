# Evidence-Aligned Results Summary

## What the experiments support

### Lexicographic decomposition

LRBE is empirically distinguishable from a normalized joint
`quality - lambda * risk` objective and from quality-first knapsack selection.
Its defensible contribution is the explicit maximum-safe-count constraint
`sum(z)=K*`, not a claim of universal downstream superiority.

Across the four fresh-seed datasets, LRBE-FAVE minus Joint-Best-Val Recall
differences were small and their 95% paired bootstrap intervals crossed zero.
This is retained as a negative result.

### Budget sensitivity

Five risk levels were evaluated at `rho={0.025, 0.05, 0.10, 0.20, 0.40}`.
The curves show that quantity-first behavior is not merely a single-budget
artifact, but its strength and convergence vary by dataset and budget.

### Predicted versus realized risk

All non-Oracle formal decisions satisfied the predicted risk constraint.  The
mean round-level realized budget violation rate across method--dataset--budget
groups was approximately `0.4798`, with group values from `0.01` to `0.84`.

Therefore the paper may claim predicted ECL feasibility, but must not claim a
deterministic realized-loss guarantee.

### Feedback--decision relationship

Within-budget correlations are heterogeneous:

- HMEQ shows positive association under several budgets;
- LendingClub is mostly weak, negative, or uncertain;
- UCI changes with budget and is generally uncertain;
- PPDai contains lower-confidence and partly non-estimable recovery cells.

The defensible conclusion is that feedback recovery does not **reliably imply**
decision recovery across datasets and budgets.  The evidence does not support
the stronger statement that the two are always unrelated.

### FAVE

FAVE remains a frozen complex information-value baseline and mechanism probe.
It can alter feedback composition, but it does not show stable downstream
superiority.  The contribution is not a state-of-the-art claim for FAVE.

## Evidence hierarchy

1. LendingClub: primary external validation.
2. UCI/HMEQ: formal development and monetary validation.
3. PPDai: exploratory lower-confidence boundary evidence.
4. Phase 4 and CCFB decoupling statistics: mechanism analysis, with post-hoc
   components explicitly labelled.

## Main limitations

- Third-party datasets do not provide fully observed real EAD/LGD in every
  case; several risk budgets are proxy-based simulations.
- Ten fresh seeds give reasonable but finite uncertainty precision.
- Offline selective-label simulation is not an online causal deployment.
- Applicant-level prediction trajectories and overlap identifiers are not
  available for every historical dataset.
- The high realized violation rate motivates a preregistered safety-margin
  appendix before final submission.

For full tables and Reviewer Defense, see
[`results/ccfb_extension/CCFB_extension_report.md`](results/ccfb_extension/CCFB_extension_report.md).
