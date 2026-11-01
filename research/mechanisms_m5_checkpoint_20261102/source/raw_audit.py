"""Reconstruct every saved M5 fit and all inexpensive acquisition locks."""
from common import *
from data import load,split,preprocess
from replay import train_predict,verify_code,CFG
from neighbor import select_neighbor
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,multiprocessing

def audit(name,seed):
    identity=verify_code();run=HERE/'runs'/f'{name}_{seed}'
    assert json.loads((run/'complete.json').read_text())['registration_sha256']==identity
    raw,y,exposure,group,receipt=load(name);initial,dev,test,batches=split(raw,group,seed,name);x,_=preprocess(raw,initial)
    split_saved=np.load(run/'split.npz');assert np.array_equal(split_saved['test'],test)
    assert not np.intersect1d(group[test],group[np.r_[dev,*batches,initial]]).size
    qids=split_saved['reference'];assert np.all(np.isin(qids,dev)) and not np.intersect1d(qids,test).size
    theta0=core.fit_logistic(x[initial],y[initial],1.);price=expit(x@theta0);threshold=float(np.quantile(price[dev],.7));costall=price*(exposure/exposure[initial].mean())
    locked_identity=json.loads((run/'identity.json').read_text());assert locked_identity['threshold']==threshold
    assert abs(locked_identity['delta']-.005*np.mean((y[dev]-price[dev])**2))<1e-14
    max_error=0.;fits=0;lock_count=0;neighbor_recomputed=0;internal_refits=0
    for learner in CFG['learners']:
        for method in METHODS:
            visible=initial.copy();pred_saved=np.load(run/learner/method/'predictions.npz');assert np.array_equal(pred_saved['test_ids'],test)
            for r,batch in enumerate(batches,1):
                p=run/learner/method/f'r{r:02d}_lock.json';lock=json.loads(p.read_text());ordinary=batch[price[batch]<threshold];full_c=batch[price[batch]>=threshold]
                gen=np.random.default_rng(np.random.SeedSequence([seed,913,r]));candidate=np.sort(gen.choice(full_c,min(128,len(full_c)),replace=False))
                cost=costall[candidate];budget=.1*cost.sum();theta=core.fit_logistic(x[visible],y[visible],1.)
                s=core.State(x[visible],y[visible],x[ordinary],x[candidate],x[qids],cost,budget,theta,seed*100+r,ridge=1.,exchanges=0)
                options=four(s,candidate)
                if method!=METHODS[-1]:expected=candidate[options[method]];assert np.array_equal(lock['selected_ids'],expected)
                else:
                    details=lock['neighbor_details'];internal_refits+=details['internal_LR_refits']
                    selected=np.asarray(lock['selected_ids'],int);assert np.all(np.isin(selected,candidate))
                    assert np.array_equal(np.sort(candidate[[v['candidate'] for v in details['steps']]]),selected)
                    if seed==51001:
                        ids,_=select_neighbor(s,CFG['neighbor']);assert np.array_equal(candidate[ids],selected);neighbor_recomputed+=1
                assert np.array_equal(lock['visible_ids'],visible) and np.array_equal(lock['ordinary_ids'],ordinary) and np.array_equal(lock['candidate_ids'],candidate)
                assert np.allclose(lock['cost'],cost,atol=1e-13,rtol=1e-13) and abs(lock['budget']-budget)<1e-12
                assert lock['state_covariates_hash']==digest([s.visible_x,s.ordinary_x,s.candidate_x,s.reference_x]) and lock['visible_labels_hash']==digest(y[visible])
                if method=='MinRisk' and r in [2,5,8]:
                    for treatment,oi in treatments(s).items():
                        branch=replace(s,ordinary_x=s.ordinary_x[oi]);bo=four(branch,candidate)
                        for action in ['OrdinaryOnly','MinRisk','Pending-aware']:
                            bp=run/'common_state'/learner/f'r{r}_{treatment}_{action}.json';bl=json.loads(bp.read_text())
                            assert bl['common_D_hash']==digest(visible) and bl['C_cost_B_hash']==digest([candidate,cost,budget])
                            assert np.array_equal(bl['ordinary_ids'],ordinary[oi]) and np.array_equal(bl['selected_ids'],candidate[bo[action]])
                            assert bl['state_covariates_hash']==digest([branch.visible_x,branch.ordinary_x,branch.candidate_x,branch.reference_x])
                            ids=np.r_[visible,ordinary[oi],candidate[bo[action]]];assert len(np.unique(ids))==len(ids) and not np.intersect1d(test,ids).size
                            pred,_,_=train_predict(x[ids],y[ids],x[test],learner,seed*100+r)
                            expected=np.load(bp.with_suffix('.npz'))['prediction'];err=float(np.max(np.abs(pred-expected)));max_error=max(max_error,err);assert err<=1e-10
                            fits+=1;lock_count+=1
                selected=np.asarray(lock['selected_ids'],int);local=np.searchsorted(candidate,selected)
                assert cost[local].sum()<=budget+1e-7
                visible=np.r_[visible,ordinary,selected];assert len(visible)==len(np.unique(visible)) and not np.intersect1d(test,visible).size
                pred,_,_=train_predict(x[visible],y[visible],x[test],learner,seed*100+r)
                err=float(np.max(np.abs(pred-pred_saved['prediction'][r-1])));max_error=max(max_error,err);assert err<=1e-10
                fits+=1;lock_count+=1
    events=json.loads((run/'label_release_audit.json').read_text());assert len(events)==192 and sum(e['kind']=='release' for e in events)==190
    assert fits==lock_count==190
    result={'passed':True,'dataset':name,'seed':seed,'fit_predictions_recomputed':fits,'locks_checked':lock_count,'maximum_prediction_error':max_error,
        'neighbor_criteria_recomputed':neighbor_recomputed,'neighbor_internal_refits_in_experiment':internal_refits,
        'no_T_outcomes_needed_for_prediction_audit':True,'registration_sha256':identity}
    write(run/'raw_audit.json',result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();jobs=[];results=[]
    for name in ['LendingClub2017','SouthGermanCredit']:
        for seed in CFG['seeds']:
            p=HERE/'runs'/f'{name}_{seed}'/'raw_audit.json'
            if p.exists():
                r=json.loads(p.read_text());assert r['passed'] and r['registration_sha256']==verify_code();results.append(r)
            else:jobs.append((name,seed))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(audit,*job):job for job in jobs}
        for future in as_completed(futures):
            r=future.result();results.append(r);print('AUDIT',len(results),r['dataset'],r['seed'],flush=True)
    assert len(results)==40
    write(HERE/'raw_audit.json',{'passed':True,'tasks':40,'fit_predictions_recomputed':sum(r['fit_predictions_recomputed'] for r in results),
        'locks_checked':sum(r['locks_checked'] for r in results),'maximum_prediction_error':max(r['maximum_prediction_error'] for r in results),
        'neighbor_criteria_recomputed':sum(r['neighbor_criteria_recomputed'] for r in results),
        'neighbor_internal_refits_in_experiment':sum(r['neighbor_internal_refits_in_experiment'] for r in results),'results':results})
if __name__=='__main__':main()
