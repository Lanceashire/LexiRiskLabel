from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results_phase3'

def test_static_patch_raw_freeze_and_final_mechanism_parity():
    before=pd.read_csv(R/'PHASE3_STATIC_PATCH_FREEZE_PRE.csv').sort_values('path').reset_index(drop=True)
    after=pd.read_csv(R/'PHASE3_STATIC_PATCH_FREEZE_POST.csv').sort_values('path').reset_index(drop=True)
    assert before.equals(after)
    p=pd.read_csv(R/'FINAL_MECHANISM_CORE_TABLE_PARITY.csv')
    assert bool(p['pass'].all())
    s=pd.read_csv(R/'LENDINGCLUB_FEEDBACK_MECHANISM_SUMMARY_FINAL.csv')
    assert {'visible_tdr_final_mean','visible_tdr_final_std','visible_tdr_final_median','fdratio_final_mean','fdratio_final_std','fdratio_final_median'}.issubset(s.columns)

def test_normalized_cross_dataset_schema_and_parity():
    x=pd.read_csv(R/'CROSS_DATASET_FINAL_TABLE_NORMALIZED.csv')
    assert len(x)==128
    assert set(x.dataset)=={'uci','hmeq','ppdai','lendingclub'}
    assert not x.duplicated(['dataset','rho','policy']).any()
    assert 'Recall_mean' not in x.columns and 'FDRatio_mean' not in x.columns
    p=pd.read_csv(R/'CROSS_DATASET_SCHEMA_PARITY.csv')
    assert len(p)>=60 and bool(p['pass'].all())