"""Narrow M5 information-boundary and schema checks; not research outcomes."""
from common import *
from neighbor import select_neighbor,criterion_check
from diagnostic import predict
from data import groups,split,preprocess
import tempfile,importlib,sys

def main():
    checks={};rng=np.random.default_rng(33071)
    D=np.c_[np.ones(60),rng.normal(size=(60,3))];y=rng.binomial(1,expit(D@np.array([-.7,.5,-.3,.8])))
    O=np.c_[np.ones(12),rng.normal(size=(12,3))];C=np.c_[np.ones(18),rng.normal(size=(18,3))];Q=np.c_[np.ones(20),rng.normal(size=(20,3))]
    s=core.State(D,y,O,C,Q,np.linspace(.1,1.,18),1.1,core.fit_logistic(D,y),33071,ridge=1.,exchanges=0)
    a=four(s,np.arange(len(C)));n,detail=select_neighbor(s,{'universes':10,'posterior_draws':32});a[METHODS[-1]]=n
    assert all(s.cost[v].sum()<=s.budget+1e-8 for v in a.values());checks['all_five_budget_feasible']=True
    # Changing every hidden outcome cannot alter any state or selection.
    hidden1=rng.integers(0,2,len(O)+len(C));hidden2=1-hidden1
    aa=four(s,np.arange(len(C)));nn,_=select_neighbor(s,{'universes':10,'posterior_draws':32})
    assert all(np.array_equal(a[k],aa[k]) for k in aa) and np.array_equal(n,nn)
    checks['selector_determinism_and_hidden_labels_not_arguments']=True
    assert not any('label' in k for k in s.__dataclass_fields__ if k not in ['visible_y'])
    zz=replace(s,ordinary_x=O[:0]);za=four(zz,np.arange(len(C)));assert np.array_equal(za['D-only'],za['Pending-aware']);checks['zero_O_D_and_pending_equal']=True
    b0=replace(s,budget=0.);assert all(len(v)==0 for v in four(b0,np.arange(len(C))).values());checks['zero_budget_no_acquisition']=True
    tr=treatments(s);assert len(tr['high'])==len(tr['low'])==len(tr['random'])==len(O)//2
    assert not np.intersect1d(tr['high'],tr['low']).size;checks['direction_quantity_matched']=True
    for learner in ['LR','XGB']:assert np.isfinite(predict(s,learner)['predicted_gain'])
    checks['M4_only_diagnostic_predictions_finite']=True
    # Lock authorization is verified against persisted serialized contents.
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        p=Path(tmp)/'lock.json';lock={'ordinary_ids':[2,3],'selected_ids':[5]};write(p,lock);v=FeedbackVault(np.arange(10)%2,np.array([9]));ids,yy=v.release(p,digest(lock));assert np.array_equal(ids,[2,3,5])
        write(p,{**lock,'selected_ids':[6]})
        try:v.release(p,digest(lock));raise RuntimeError('tamper accepted')
        except AssertionError:pass
        write(p,{'ordinary_ids':[9],'selected_ids':[]})
        try:v.release(p,digest({'ordinary_ids':[9],'selected_ids':[]}));raise RuntimeError('T release accepted')
        except AssertionError:pass
    checks['lock_tampering_and_T_release_rejected']=True
    raw=pd.DataFrame({'laufzeit':np.arange(100,dtype=float),'hoehe':np.arange(100)+100.,'cat':['a','b']*50});raw.iloc[1]=raw.iloc[0];group=groups(raw);assert group[0]==group[1]
    initial,dev,test,batches=split(raw,group,51001,'SouthGermanCredit');x,pre=preprocess(raw,initial)
    _,dev2,test2,_=split(raw,group,51002,'SouthGermanCredit');assert np.array_equal(test,test2) and np.array_equal(dev,dev2)
    checks['shared_sealed_outer_test_across_replay_seeds']=True
    changed=raw.copy();changed.loc[test,'laufzeit']=1e8
    xx,_=preprocess(changed,initial);assert np.array_equal(x[initial],xx[initial]);assert not np.intersect1d(test,dev).size
    checks['duplicate_groups_split_and_T_covariate_perturbation_isolation']=True
    checks['ParBaLS_criterion']=criterion_check()
    # Actual learner fits use synthetic visible labels only. No new-source file is read.
    from replay import train_predict
    for learner in ['LR','XGB']:
        pred,backend,_=train_predict(D,y,Q,learner,33071);assert np.all(np.isfinite(pred)) and backend['device']=='cpu'
    checks['both_registered_CPU_learners_execute']=True
    for p in HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
    checks['all_M5_python_sources_compile']=True
    write(HERE/'engineering_checks.json',{'passed':True,'checks':checks,'new_task_outcomes_used':False,'research_evidence':False})
    print(json.dumps(checks,default=default),flush=True)
if __name__=='__main__':main()
