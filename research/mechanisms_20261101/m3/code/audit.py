"""Independent evidence reconstruction, from saved inputs/locks/predictions."""
from pathlib import Path
import argparse, concurrent.futures, json, multiprocessing, time
import numpy as np
import pandas as pd
from scipy.special import expit
from study import sha,write,environment,state,core,exact,COSTS,METHODS

def one(args):
    dest,plan=args;dest=Path(dest);w=int(dest.name[1:]);cfg=plan['config'];done=json.loads((dest/'complete.json').read_text())
    assert done['plan_sha256']==plan['_plan_sha256']
    for rel,h in done['files'].items():assert sha(dest/rel)==h,(w,rel)
    with np.load(dest/'inputs.npz') as z:inp={k:z[k] for k in z.files}
    with np.load(dest/'predictions.npz') as z:pred={k:z[k] for k in z.files}
    env=environment(w,cfg)
    for k in ['D','yD','O','C','Q','T','pT','yT','theta','candidate_ids','original_exposure_draw']:
        assert np.array_equal(inp[k],env[k]),(w,k)
    truth=np.array(cfg['truth_theta']);ids=inp['candidate_ids'];B=float(inp['B']);nd=len(inp['D']);no=len(inp['O'])
    for rep in range(plan['label_M']):
        rng=np.random.default_rng(np.random.SeedSequence([w,plan['label_rng_tag'],rep]))
        yo=(rng.random(no)<expit(inp['O']@truth)).astype(np.uint8);yc=(rng.random(len(ids))<expit(inp['C']@truth)).astype(np.uint8)
        assert np.array_equal(yo,inp['yO'][rep]) and np.array_equal(yc,inp['yC'][rep])
    locks=json.loads((dest/'selection_locks.json').read_text());commit=json.loads((dest/'selection_commitment.json').read_text())
    assert commit['sha256']==sha(dest/'selection_locks.json') and commit['all_methods_locked_before_any_future_label_draw_or_fit']
    assert len(locks)==len(COSTS)*len(METHODS)
    fitted={};meta={};ks=[];unmatched=0
    for c in COSTS:
        cost=inp['cost_'+c];ead=inp['effective_EAD_'+c];assert np.array_equal(cost,env['costs'][c])
        assert np.array_equal(ead,cost/.75) and np.allclose(np.sort(ead),np.sort(inp['original_exposure_draw']),rtol=0,atol=1e-15)
        assert np.array_equal(np.sort(cost),np.sort(inp['cost_neutral']))
        k=core.kstar(cost,B);ks.append(k);order=np.lexsort((ids,cost));s=state(env,cost,w,cfg)
        q={x['method']:x for x in locks if x['cost_condition']==c}
        for cond,name in [(False,'D-only'),(True,'Pending-aware')]:
            h,g,u=core.geometry(s,cond);actual=exact.greedy(h,g,u,cost,B,ids)
            assert np.array_equal(q[name]['selected_indices'],actual),(w,c,name)
        pa=np.array(q['Pending-aware']['selected_indices'],int)
        assert len(q['FixedK-Pending']['selected_indices'])==k
        for name,row in q.items():
            meta[c,name]=row
            if not row['estimable']:
                assert name=='CostCountMatched';unmatched+=1;continue
            idx=np.array(row['selected_indices'],int)
            assert len(np.unique(idx))==len(idx) and np.array_equal(ids[idx],row['selected_ids'])
            assert core.digest(ids[idx])==row['select_hash']
            assert abs(cost[idx].sum()-row['cost'])<1e-12 and cost[idx].sum()<=B+1e-8
            assert row['Kstar']==k and row['count_sacrifice']==k-len(idx)
            assert abs(row['composition_premium']-(cost[idx].sum()-np.sort(cost)[:len(idx)].sum()))<1e-12
            assert abs(row['true_expected_selected_loss']-np.dot(expit(inp['C'][idx]@truth),ead[idx]))<1e-12
            if name=='MinRisk':assert np.array_equal(idx,np.sort(order[:k]))
            if name=='MinRisk-k':assert np.array_equal(idx,np.sort(order[:len(pa)]))
            if name=='CostCountMatched':assert len(idx)==len(pa) and abs(cost[idx].sum()-cost[pa].sum())<=.005*B+1e-8
            if row['fit_id'] in fitted:assert np.array_equal(fitted[row['fit_id']],idx)
            else:fitted[row['fit_id']]=idx
    assert len(set(ks))==1
    fits=pd.read_csv(dest/'fits.csv');out=pd.read_csv(dest/'outcomes.csv');maxrisk=0.;cache={}
    for r in fits.itertuples():
        idx=fitted[r.fit_id];x=np.vstack([inp['D'],inp['O'],inp['C'][idx]])
        y=np.r_[inp['yD'],inp['yO'][r.replicate],inp['yC'][r.replicate,idx]]
        assert core.digest(x)==r.train_x_hash and core.digest(y)==r.train_label_hash
        assert core.digest(np.r_[np.arange(nd+no),ids[idx]])==r.train_ids_hash
        assert len(y)==r.train_n
        if r.learner=='LR':p=expit(inp['T']@pred['LR_coefficients'][r.prediction_index])
        else:
            p=pred['XGB_predictions'][r.prediction_index].astype(float)
            assert r.backend.startswith('cuda')
        risk=float(np.mean((p-inp['pT'])**2));brier=float(np.mean((p-inp['yT'])**2))
        maxrisk=max(maxrisk,abs(risk-r.risk),abs(brier-r.brier));cache[r.replicate,r.fit_id,r.learner]=(risk,brier)
    for r in out.itertuples():
        lock=meta[r.cost_condition,r.method]
        if not lock['estimable']:assert pd.isna(r.risk);continue
        risk,brier=cache[r.replicate,int(r.fit_id),r.learner]
        maxrisk=max(maxrisk,abs(r.risk-risk),abs(r.brier-brier))
        idx=np.array(lock['selected_indices'],int);ead=inp['effective_EAD_'+r.cost_condition]
        assert abs(r.realized_selected_loss-np.dot(inp['yC'][r.replicate,idx],ead[idx]))<1e-11
    assert maxrisk<1e-12
    assert len(out)==len(COSTS)*len(METHODS)*2*plan['label_M']
    return {'world_id':w,'fits':len(fits),'outcomes':len(out),'unmatched_states':unmatched,'maximum_risk_error':maxrisk}

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    plan=json.loads(a.plan.read_text());plan['_plan_sha256']=sha(a.plan)
    paths={int(p.name[1:]):p for p in a.run.glob('worker_*/w[0-9]*') if p.is_dir() and p.name[1:].isdigit()}
    assert set(paths)==set(plan['world_ids'])
    start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        results=list(pool.map(one,[(paths[w],plan) for w in plan['world_ids']]))
    write(a.out,{'passed':True,'plan_sha256':sha(a.plan),'worlds':len(results),'fits_reconstructed':sum(r['fits'] for r in results),'outcome_rows':sum(r['outcomes'] for r in results),
                 'unmatched_states':sum(r['unmatched_states'] for r in results),'maximum_risk_error':max(r['maximum_risk_error'] for r in results),'generator_labels_and_legal_training_reconstructed':True,
                 'selection_and_cost_invariants_checked':True,'source_sha256':sha(__file__),'seconds':time.perf_counter()-start,'worlds_summary':results})
    print(json.dumps({'passed':True,'worlds':len(results),'seconds':time.perf_counter()-start}),flush=True)
if __name__=='__main__':main()
