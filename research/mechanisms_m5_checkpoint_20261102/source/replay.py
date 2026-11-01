"""Checkpointed ten-round M5 replay; T outcomes are never loaded by evaluation code here."""
from common import *
from data import load,split,preprocess
from neighbor import select_neighbor
from diagnostic import predict as diagnose
from xgboost import XGBClassifier
import argparse,platform,traceback
CFG=json.loads((HERE/'pre_label_scope.json').read_text())
def verify_code():
    reg=json.loads((HERE/'registration.json').read_text())
    for f,h in reg['source_hashes'].items():assert sha(HERE/f)==h,('source drift',f)
    assert reg['M4_complete'] and reg['new_outcomes_inspected_at_registration']==False
    return sha(HERE/'registration.json')
def train_predict(x,y,reference,learner,seed):
    if learner=='LR':
        theta=core.fit_logistic(x,y,1.);return expit(reference@theta),{'device':'cpu','model':'regularized LR'},theta
    m=XGBClassifier(**CFG['model']['XGB'],random_state=seed,objective='binary:logistic',eval_metric='logloss')
    m.fit(x[:,1:].astype(np.float32),y);conf=json.loads(m.get_booster().save_config())['learner'];assert conf['generic_param']['device']=='cpu'
    assert m.get_booster().num_boosted_rounds()==100
    return m.predict_proba(reference[:,1:].astype(np.float32))[:,1],{'device':'cpu','model':'XGB hist 100 depth3 max_bin256'},None

def lock_and_fit(s,sets,ordinary_ids,candidate_ids,visible,visible_y,x,test,vault,learner,seed,path,extra):
    selected=candidate_ids[sets]
    assert len(s.cost)==len(candidate_ids) and np.all(np.isin(selected,candidate_ids))
    assert not np.intersect1d(ordinary_ids,candidate_ids).size
    assert not np.intersect1d(visible,np.r_[ordinary_ids,candidate_ids]).size
    lock={'visible_ids':visible,'ordinary_ids':ordinary_ids,'candidate_ids':candidate_ids,'selected_ids':selected,
        'cost':s.cost,'budget':s.budget,'state_covariates_hash':digest([s.visible_x,s.ordinary_x,s.candidate_x,s.reference_x]),
        'visible_labels_hash':digest(visible_y),**extra}
    assert s.cost[sets].sum()<=s.budget+1e-7
    write(path,lock);commitment=digest(lock)
    ids,new_y=vault.release(path,commitment);all_ids=np.r_[visible,ids];all_y=np.r_[visible_y,new_y]
    assert len(all_ids)==len(np.unique(all_ids))
    pred,backend,coef=train_predict(x[all_ids],all_y,x[test],learner,seed)
    return all_ids,all_y,pred,backend

def task(name,seed):
    identity=verify_code();dest=HERE/'runs'/f'{name}_{seed}'
    if (dest/'complete.json').exists():
        old=json.loads((dest/'complete.json').read_text());assert old['registration_sha256']==identity;return {'skipped':True,'task':dest.name}
    dest.mkdir(parents=True,exist_ok=True)
    raw,y,exposure,group,receipt=load(name);initial,development,test,batches=split(raw,group,seed,name)
    # A reader object holds raw outcomes; no selector/state has access to it.
    vault=FeedbackVault(y,test);initial_y=vault.initial(initial);del y
    x,pre=preprocess(raw,initial);theta0=core.fit_logistic(x[initial],initial_y,1.)
    price=expit(x@theta0);threshold=float(np.quantile(price[development],.7));exposure=exposure/float(np.mean(exposure[initial]));costall=price*exposure
    qids=np.sort(np.random.default_rng(seed+77).choice(development,min(128,len(development)),replace=False))
    assert not np.intersect1d(qids,test).size
    # This development endpoint only fixes the practical threshold; T outcomes remain sealed.
    dev_y=vault.initial(development);delta=.005*float(np.mean((dev_y-price[development])**2))
    np.savez_compressed(dest/'split.npz',initial=initial,development=development,test=test,group=group,reference=qids)
    write(dest/'identity.json',{'registration_sha256':identity,'source':receipt,'seed':seed,'schema':list(raw),'initial_y_hash':digest(initial_y),'threshold':threshold,'delta':delta,'feature_dimension':x.shape[1],'preprocessing_fit_ids':initial,'Q_ref_ids':qids,'split_hash':digest([initial,development,test,*batches])})
    cost_records=[];common_records=[]
    for learner in CFG['learners']:
        for method in METHODS:
            run=dest/learner/method;run.mkdir(parents=True,exist_ok=True);visible=initial.copy();visible_y=initial_y.copy();predictions=[]
            for r,batch in enumerate(batches,1):
                ordinary=batch[price[batch]<threshold];full_candidate=batch[price[batch]>=threshold]
                g=np.random.default_rng(np.random.SeedSequence([seed,913,r]));candidate=np.sort(g.choice(full_candidate,min(128,len(full_candidate)),replace=False))
                costs=costall[candidate];budget=.1*float(costs.sum());theta=core.fit_logistic(x[visible],visible_y,1.)
                s=core.State(x[visible],visible_y,x[ordinary],x[candidate],x[qids],costs,budget,theta,seed*100+r,ridge=1.,exchanges=0)
                foursets=four(s,candidate)
                if method==METHODS[-1]:chosen,details=select_neighbor(s,CFG['neighbor'])
                else:chosen=foursets[method];details={}
                diag=diagnose(s,learner,foursets)
                if method=='MinRisk' and r in [2,5,8]:
                    for treatment,oi in treatments(s).items():
                        branch=replace(s,ordinary_x=s.ordinary_x[oi]);options=four(branch,candidate);branch_diag=diagnose(branch,learner,options)
                        for action in ['OrdinaryOnly','MinRisk','Pending-aware']:
                            bp=dest/'common_state'/learner/f'r{r}_{treatment}_{action}.json'
                            _,_,pp,be=lock_and_fit(branch,options[action],ordinary[oi],candidate,visible,visible_y,x,test,vault,learner,seed*100+r,bp,{'round':r,'treatment':treatment,'method':action,'common_D_hash':digest(visible),'C_cost_B_hash':digest([candidate,costs,budget])})
                            np.savez_compressed(bp.with_suffix('.npz'),prediction=pp,test_ids=test)
                            common_records.append({'learner':learner,'round':r,'treatment':treatment,'method':action,'ordinary_count':len(oi),'candidate_count':len(candidate),'selected_count':len(options[action]),'predicted_gain':branch_diag['predicted_gain'],'constant_prediction':branch_diag['constant_prediction'],'diagnostic_outside_M4_range':int(np.any(branch_diag['out_of_M4_range']))})
                old_visible=visible.copy()
                visible,visible_y,pp,be=lock_and_fit(s,chosen,ordinary,candidate,visible,visible_y,x,test,vault,learner,seed*100+r,run/f'r{r:02d}_lock.json',{'round':r,'method':method,'learner':learner,'diagnostic':diag,'neighbor_details':details})
                predictions.append(pp)
                cost_records.append({'learner':learner,'method':method,'round':r,'D_count':len(old_visible),'O_count':len(ordinary),'full_C_count':len(full_candidate),'retained_C_count':len(candidate),'selected_count':len(chosen),'budget':budget,'cost':float(costs[chosen].sum()),'predicted_gain':diag['predicted_gain'],'diagnostic_outside_M4_range':int(np.any(diag['out_of_M4_range'])),'actual_backend':be['device']})
            np.savez_compressed(run/'predictions.npz',prediction=np.asarray(predictions),test_ids=test)
    pd.DataFrame(cost_records).to_csv(dest/'trajectory_no_T_outcomes.csv',index=False)
    pd.DataFrame(common_records).to_csv(dest/'common_state_no_T_outcomes.csv',index=False)
    write(dest/'label_release_audit.json',vault.events)
    write(dest/'complete.json',{'registration_sha256':identity,'complete':True,'sealed_T_outcomes_evaluated':False,'trajectories':10,'round_records':100,'common_state_branches':90})
    return {'task':dest.name,'complete':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['LendingClub2017','SouthGermanCredit']);p.add_argument('seed',type=int);a=p.parse_args();assert a.seed in CFG['seeds'];print(task(a.dataset,a.seed),flush=True)
if __name__=='__main__':main()
