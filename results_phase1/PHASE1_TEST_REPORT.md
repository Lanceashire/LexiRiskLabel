# Phase 1 test report

- Command (original environment): `<PYTHON> -m pytest -q <PROJECT_ROOT>/tests`
- Result: **20 passed, 0 failed**.
- UCI formal output: 1100 rows (20 seeds x 5 treatments x 11 rounds).
- HMEQ formal output: 1100 rows (20 seeds x 5 treatments x 11 rounds).
- Integrity audit: all checks passed.
- Closeout coverage: RecoveryRatio N/A guard, dual-evaluation artifact presence, and external parity (`max_abs_diff=0`).
- Phase 2: not executed; this is a static Phase 1 closeout.
