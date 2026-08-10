from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def test_phase3_ppdai_complete():
    files=list((ROOT/'results_phase3'/'raw'/'ppdai').glob('*.csv'))
    assert len(files)==580
    x=pd.read_csv(ROOT/'results_phase3'/'summary'/'all_rounds.csv')
    assert len(x)==7040
    assert set(x.compute_backend.dropna().unique())=={'gpu'}

def test_phase3_final_integrity():
    a=pd.read_csv(ROOT/'results_phase3'/'PHASE3_INTEGRITY_AUDIT_FINAL.csv')
    assert len(a)==40
    assert bool(a.drop(columns=['dataset','seed','leakage_risk']).all().all())

def test_lendingclub_is_completed_primary_external_validation():
    s=pd.read_csv(ROOT/'results_phase3'/'CROSS_DATASET_VALIDATION_STATUS.csv').set_index('dataset')
    assert s.loc['lendingclub','external_status']=='COMPLETE'
    assert s.loc['lendingclub','seeds_completed']==20
    assert s.loc['lendingclub','primary_evidence_level']=='CONFIRMATORY_EXTERNAL'