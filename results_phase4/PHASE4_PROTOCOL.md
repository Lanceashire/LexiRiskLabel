# Phase 4 Protocol — Mechanistic Analysis & Boundary Conditions

**Mode:** ANALYSIS-ONLY. **Framework:** POST-HOC EXPLANATORY FRAMEWORK.

The frozen Phase 1–3 model, preprocessing, policies, seeds, budgets, gamma=1,
delta=.5, RiskTilted exponent=.5, tau=.5, and LRBE were not changed. No model
training, policy invocation, threshold tuning, diagnostic replay, or new
dataset was performed. Phase 1–3 canonical artifacts were read only.

## Questions

This phase examines quantity-first performance, feedback/performance
decoupling, selection severity, exposure heterogeneity, evaluation distortion,
and complexity/performance tradeoffs. Associations and bootstrap intervals are
exploratory and computed separately by dataset.

## Evidence availability rule

Frozen outputs retain seed/round aggregate selected-set summaries but not
applicant IDs or per-applicant prediction trajectories. Exact Jaccard overlap,
applicant-level PD/EAD quantiles, boundary bands, and two-way threshold
crossings are therefore not identifiable. Net count changes and distributions
of aggregate cell means are labeled as such. Exact replay was not used because
selected-ID parity cannot be verified.
