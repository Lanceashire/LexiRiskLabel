# Algorithm Freeze Declaration

Phase 3 reused the Phase 2 frozen algorithm, split construction, tau, rho grid, policies, and seed schedule. The only code change was an exact mathematical LRBE presolve: impossible-over-budget items are removed, and utility=-cost returns the provably identical K*-cheapest set. No policy, model, threshold, or tuning rule was changed.
