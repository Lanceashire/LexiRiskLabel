# PHASE 0 REPORT

## 1. Protocol repairs

- Exploration candidates are current-round rejected applicants only.
- Remaining applicants are partitioned exactly once into ten non-empty batches.
- CountOnly-MinRisk is included with the same lexicographic optimizer.
- Metrics include Brier, 10-bin ECE, Visible TDR, Population Default Rate, and FDRatio.
- Selection never receives true rejected labels.

## 2. Frozen FAVE-v2.1 recovery

The user-provided archive was recovered and audited:

- ZIP SHA-256: E8EB01B7B0019A554D0D49AC64DAF8B5F50C60647A4D7CAF202B783324404C65
- Frozen parameters: gamma=1.0, delta=0.5.
- The score is the archived Fisher/influence EVI:
  candidate weight wi = p_i(1-p_i)p_i^delta;
  future/current-batch matrix G = F^T diag(p_f^gamma)F / |F|;
  value = wi * (r_i^T H^-1 G H^-1 r_i) / (1 + wi * r_i^T H^-1 r_i).
- In Phase 0, the reference set is the current batch and the model/preprocessor are fit on visible applicants only.
- The former p(1-p)sqrt(p) policy remains separately named RiskTilted-Uncertainty.

Detailed evidence is in FAVE_IMPLEMENTATION_AUDIT.md and results_phase0/fave_score_audit.csv.

## 3. Sanity verification

- Seeds: 0, 1.
- rho: 0.025, 0.05, 0.10, 0.20.
- Rounds: 10.
- Policies: 8, including FAVE-v2.1.
- Combined rows: 640; FAVE rows: 80.
- Protocol invariant failures: 0.
- Unit/protocol tests: 13 passed, 0 failed.

These are procedural sanity results, not superiority claims.

## 4. Status

PHASE0_STATUS = PASS

Ready for Phase 1 = YES. The frozen FAVE identity is now recovered, implemented, and verified under the repaired current-round-only protocol. Formal multi-seed HMEQ analysis and the second-dataset validation remain Phase 1 work.
