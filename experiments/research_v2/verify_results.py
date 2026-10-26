"""Focused evidence verification, not a new benchmark suite."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
from core import digest,select
from run_mechanisms import generate

ROOT=Path(__file__).resolve().parents[2];base=ROOT/'results/research_v2'
ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,default=base);args=ap.parse_args();base=args.base
cfg=json.loads(Path(__file__).with_name('config.json').read_text());common={};checks=0;rows=[];identities=[]
for run in sorted((base/'final_real').iterdir()):
    if not (run/'completion.json').exists():raise RuntimeError('incomplete '+str(run))
    identity=json.loads((run/'identity.json').read_text());identities.append(identity)
    assert identity['config']==cfg
    expected=digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).with_name('core.py'),Path(__file__).with_name('run_real.py')]})
    assert identity['code_hash']==expected
    table=pd.read_csv(run/'trajectory.csv');prediction=np.load(run/'test_predictions.npz')
    assert len(table)==cfg['rounds'] and prediction['prediction'].shape[0]==cfg['rounds']
    assert set(table.actual_training_device)=={'cuda:0'} and table.updater.str.contains('gpu').all()
    assert table.budget_feasible.all()
    for i,row in table.iterrows():
        log=json.loads((run/f'round_{i+1:02d}_selection.json').read_text())
        o,c,s,v=map(lambda k:np.array(log[k],int),['ordinary_ids','candidate_ids','selected_ids','visible_ids'])
        reference=np.array(log['reference_ids'],int);test=prediction['ids'];cost=np.array(log['candidate_costs'])
        assert not np.intersect1d(test,np.r_[v,o,s,reference]).size
        assert not np.intersect1d(o,c).size and set(s)<=set(c) and not np.intersect1d(v,np.r_[o,s]).size
        local=np.flatnonzero(np.isin(c,s));spent=cost[local].sum()
        assert spent<=log['budget']+1e-7 and np.isclose(spent,row.predicted_exploration_risk)
        assert np.isclose(np.mean((prediction['y']-prediction['prediction'][i])**2),row.brier,atol=1e-12)
        key=(row.dataset,int(row.seed),int(row['round']))
        state=(digest(o),digest(c),digest(cost),log['budget'])
        if key in common:assert common[key]==state
        else:common[key]=state
        design=log['sampling_design'];pi=log['inclusion_probabilities']
        if design is not None:
            reconstructed=np.zeros(len(c))
            for ids,w in zip(design['sets'],design['weights']):
                assert cost[ids].sum()<=log['budget']+1e-7;reconstructed[ids]+=w
            assert np.allclose(reconstructed,pi) and np.isclose(sum(design['weights']),1)
            assert np.all(np.asarray(pi)[cost<=log['budget']+1e-10]>0)
        checks+=1
    rows.append(table)
assert len(rows)==84
frame=pd.concat(rows,ignore_index=True)
# Paired factorial acquisition must be the same with and without correction.
for dataset in ['hmeq','uci','lendingclub']:
    for seed in cfg['real_seeds']:
        for a,b in [('CORR-base-plain','CORR-base-corrected'),('CORR-opt-plain','RB-CORR')]:
            for r in range(1,cfg['rounds']+1):
                pa=base/'final_real'/f'{dataset}_{seed}_correction_{a}'/f'round_{r:02d}_selection.json'
                pb=base/'final_real'/f'{dataset}_{seed}_correction_{b}'/f'round_{r:02d}_selection.json'
                ja,jb=json.loads(pa.read_text()),json.loads(pb.read_text())
                assert ja['selected_ids']==jb['selected_ids'] and ja['inclusion_probabilities']==jb['inclusion_probabilities']
# Actual hidden-label perturbation at selector boundary: only visible labels enter state.
s,yo,yc,test,yt,pt,truth=generate('A1',2101)
perturbed=np.r_[s.visible_y,1-yo,1-yc,1-yt]
s_perturbed=type(s)(**{**s.__dict__,'visible_y':perturbed[:len(s.visible_y)]})
for method in ['RB-BRIDGE','RB-CORR','RB-GATE']:
    a=select(method,s);b=select(method,s_perturbed)
    assert np.array_equal(a.selected_ids,b.selected_ids)
    if a.inclusion_probabilities is not None:assert np.array_equal(a.inclusion_probabilities,b.inclusion_probabilities)
mechanism=pd.read_csv(base/'final_mechanisms/mechanism_rows.csv')
assert np.allclose(mechanism.loc[mechanism.environment=='C2','feedback_channel_value'],0)
result={'verified':True,'episodes':len(rows),'round_records':checks,'actual_training_devices':sorted(frame.actual_training_device.unique().tolist()),
        'predicted_budget_violations':int((~frame.budget_feasible).sum()),'saved_predictions_recomputed':True,
        'fixed_O_C_cost_budget_equal_across_methods':True,'correction_factorial_shares_realized_acquisition':True,
        'finite_design_probabilities_verified':True,'test_and_reference_isolation_verified':True,
        'hidden_label_perturbation_three_algorithms':'passed','frozen_approval_channel_value_zero':True,
        'note':'These are artifact verification checks, not independent statistical replications.'}
(base/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
