from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results_phase1'

def test_closeout_artifacts_and_dual_parity():
    for name in ['hidden_information_summary.csv','recovery_ratios_corrected.csv','paired_tests_calibration_extended.csv','dual_evaluation_all_rounds.csv','dual_evaluation_gap.csv','dual_evaluation_statistics.csv','PHASE1_FINAL_REPORT.md','PHASE1_ADDENDUM_DUAL_EVALUATION.md']:
        assert (RES/name).exists(), name
    p=pd.read_csv(RES/'DUAL_EVAL_PARITY.csv').iloc[0]
    assert bool(p['passed']) and float(p['max_abs_diff']) <= 1e-8
    d=pd.read_csv(RES/'dual_evaluation_all_rounds.csv')
    assert len(d)==1760 and set(d.evaluation_scope)=={'external_fixed','internal_visible'}


