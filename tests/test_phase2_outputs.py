from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results_phase2'

def test_phase2_complete_gpu_and_integrity():
    d=pd.read_csv(R/'summary/all_rounds.csv')
    assert len(d)==14080
    assert set(d.dataset)=={'uci','hmeq'}
    assert set(d.seed)==set(range(20))
    assert set(d.rho)=={0.025,0.05,0.1,0.2}
    assert len(d.policy.unique())==8
    assert set(d.compute_backend)=={'gpu'}
    inv=['risk_ok','selected_subset_current_rejected','no_historical_backlog','hidden_label_isolation','fixed_test_isolation','visible_update_correct','optimizer_ok']
    assert d[inv].all().all()
    p=pd.read_csv(R/'PHASE2_SPLIT_PARITY.csv')
    assert len(p)==40 and p[['same_test_set','same_D0','same_stream_order','same_batches']].all().all()
    assert bool(pd.read_csv(R/'PHASE2_FAVE_IDENTITY.csv').passed.iloc[0])
    assert pd.read_csv(R/'optimizer_invariant.csv').same_k_star.all()
    for name in ['PHASE2_RESULTS.md','PHASE2_STATISTICAL_REPORT.md','PHASE2_PARETO_REPORT.md','PHASE2_POLICY_REPORT.md','PHASE2_PROVISIONAL_CONCLUSION.md','PHASE2_INTEGRITY_AUDIT.csv']:
        assert (R/name).exists(), name
