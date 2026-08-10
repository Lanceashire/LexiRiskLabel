# Phase 0 protocol

- Frozen parameters: FAVE gamma=1.0, delta=0.5; no tuning.
- Phase 0 policies: NoExplore, CountOnly, CountOnly-MinRisk, LeastRisky, MostRisky, Uncertainty, RiskTilted-Uncertainty, FAVE-v2.1.
- FAVE-v2.1 is recovered from the user-provided frozen reference archive and ported into the repaired current-round-only environment.
- rho: 0.025, 0.05, 0.10, 0.20; seeds: 0 and 1; rounds: 10.
- Batches are np.array_split(remaining, 10) and each applicant appears exactly once.
- Exploration candidates are current-round rejected applicants only; non-explored rejected applicants leave the episode permanently.
- Preprocessing is fit only on current visible labels.
- Risk budget is current-round rho * sum(PD * LOAN).
- Policy selection uses the same two-stage MILP optimizer.
- Oracle labels are used only after selection for evaluator metrics.
- Phase-0 result: 640 combined rows, including 80 FAVE rows; 0 protocol invariant failures.
