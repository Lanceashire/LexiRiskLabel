"""Registered M3 cost-pairing experiment. Acquisition has no future labels."""
from pathlib import Path
from dataclasses import replace
import argparse, hashlib, json, os, sys, time, platform
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import milp, Bounds, LinearConstraint
from xgboost import XGBClassifier
from deps import core, exact, stage_gate

HERE=Path(__file__).resolve().parent
COSTS=['neutral','information_cheap','information_expensive']
METHODS=['OrdinaryOnly','MinRisk','D-only','Pending-aware','Static-D','Static-Pending',
         'FixedK-Pending','GainBudget-Pending','RandomFeasible','MinRisk-k',
         'MinRisk-spendcap','CostCountMatched']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=clean)+'\n');temp.replace(p)
def rng(w,tag):return np.random.default_rng(np.random.SeedSequence([w,tag]))

def environment(w,cfg):
    """Original generator and RNG tags, restricted to original M3 anchor."""
    nd,nc,no=cfg['D'],cfg['C'],cfg['ordinary_n'];rad=cfg['direction_radius_sd']
    D=np.ones((nd,5));D[:,1]=rng(w,1).uniform(-2,.8,nd);D[:,2]=rng(w,2).normal(0,.12,nd)
    D[:,3]=D[:,2]+rng(w,3).normal(0,.05,nd);D[:,4]=rng(w,4).normal(size=nd)
    C=np.ones((nc,5));C[:,1]=cfg['candidate_a'];C[:,2]=rng(w,5).normal(0,rad,nc);C[:,3:]=rng(w,6).normal(size=(nc,2))
    def target(n,tag):
        x=np.zeros((n,5));x[:,0]=1;x[:,1]=rng(w,tag).uniform(-2,2.2,n);x[:,2]=rng(w,tag+1).normal(0,rad,n);return x
    Q=target(cfg['Q_ref'],10);T=target(cfg['T'],20);truth=np.array(cfg['truth_theta'])
    yD=(rng(w,30).random(nd)<expit(D@truth)).astype(np.uint8)
    a=rng(w,40).uniform(-2,.8,no);r=rng(w,41).normal(0,rad,no)
    O=np.c_[np.ones(no),a,r,np.zeros((no,2))]
    exposure=rng(w,50).uniform(*cfg['EAD_uniform'],nc);prices=.75*exposure
    rank=np.argsort(np.abs(C[:,2]),kind='stable');sorted_cost=np.sort(prices)
    cheap=np.empty(nc);cheap[rank]=sorted_cost[::-1]
    expensive=np.empty(nc);expensive[rank]=sorted_cost
    costs={'neutral':rng(w,51).permutation(prices),'information_cheap':cheap,'information_expensive':expensive}
    assert all(np.array_equal(np.sort(v),sorted_cost) for v in costs.values())
    assert np.all(expit(-1+D[:,1])<=.5) and np.all(expit(-1+O[:,1])<=.5) and np.all(expit(-1+C[:,1])>.5)
    return {'D':D,'yD':yD,'O':O,'C':C,'Q':Q,'T':T,'pT':expit(T@truth),
            'yT':(rng(w,32).random(len(T))<expit(T@truth)).astype(np.uint8),
            'original_exposure_draw':exposure,'costs':costs,'B':float(cfg['rho']*prices.sum()),
            'theta':core.fit_logistic(D,yD,cfg['lr_ridge']),
            'candidate_ids':np.arange(nd+no,nd+no+nc)}

def state(env,cost,w,cfg):
    return core.State(env['D'],env['yD'],env['O'],env['C'],env['Q'],cost,env['B'],env['theta'],w,ridge=cfg['lr_ridge'],exchanges=0)

def variant(h,g,u,c,b,ids,*,static=False,fixed_count=False,raw_gain=False):
    order=np.argsort(ids,kind='stable');inv=np.linalg.inv(h);initial=core.marginal(inv,g,u)
    chosen=[];free=np.ones(len(c),bool);left=b;k=core.kstar(c,b)
    for _ in c:
        valid=free&(c<=left+1e-12)
        if fixed_count:
            slots=k-len(chosen)
            if slots<=0:break
            for i in np.flatnonzero(valid):
                rest=c[free&(np.arange(len(c))!=i)]
                if slots>1 and (len(rest)<slots-1 or np.partition(rest,slots-2)[:slots-1].sum()>left-c[i]+1e-12):valid[i]=False
        valid_ids=order[valid[order]]
        if not len(valid_ids):break
        gain=initial if static else core.marginal(inv,g,u)
        scores=gain[valid_ids] if raw_gain else gain[valid_ids]/np.maximum(c[valid_ids],1e-30)
        i=int(valid_ids[np.argmax(scores)])
        if gain[i]<=1e-15 and not fixed_count:break
        chosen.append(i);free[i]=False;left-=c[i]
        if not static:inv=core.update(inv,u[i])
    if fixed_count:assert len(chosen)==k
    return np.array(sorted(chosen,key=lambda i:ids[i]),int)

def matched(cost,ids,k,target,budget,world,opts):
    if not k:return np.empty(0,int),{'status':0,'valid_incumbent':True}
    order=np.argsort(ids,kind='stable');canonical_cost=cost[order]
    weights=rng(world,93001).uniform(size=len(cost));tol=.005*budget
    con=LinearConstraint(np.vstack([np.ones(len(cost)),canonical_cost/budget]),[k,max(0,target-tol)/budget],[k,min(budget,target+tol)/budget])
    ans=milp(weights,integrality=np.ones(len(cost)),bounds=Bounds(0,1),constraints=con,options=opts)
    idx=order[np.flatnonzero(ans.x>.5)] if ans.x is not None else np.empty(0,int)
    valid=len(idx)==k and abs(cost[idx].sum()-target)<=tol+1e-8 and cost[idx].sum()<=budget+1e-8
    return (np.sort(idx) if valid else None),{'status':int(ans.status),'message':ans.message,'valid_incumbent':bool(valid)}

def acquire(s,ids,match_opts):
    """Accepts only legal visible State and stable IDs; no generator truth or T."""
    order=np.lexsort((ids,s.cost));k=core.kstar(s.cost,s.budget)
    out={'OrdinaryOnly':np.empty(0,int),'MinRisk':np.sort(order[:k])}
    for conditional,name in [(False,'D-only'),(True,'Pending-aware')]:
        h,g,u=core.geometry(s,conditional)
        out[name]=exact.greedy(h,g,u,s.cost,s.budget,ids)
        out['Static-Pending' if conditional else 'Static-D']=variant(h,g,u,s.cost,s.budget,ids,static=True)
    h,g,u=core.geometry(s,True)
    out['FixedK-Pending']=variant(h,g,u,s.cost,s.budget,ids,fixed_count=True)
    out['GainBudget-Pending']=variant(h,g,u,s.cost,s.budget,ids,raw_gain=True)
    permutation=rng(s.seed,93002).permutation(len(ids));permutation=np.argsort(ids,kind='stable')[permutation]
    chosen=[];left=s.budget
    for i in permutation:
        if s.cost[i]<=left+1e-12:chosen.append(int(i));left-=s.cost[i]
    out['RandomFeasible']=np.sort(np.asarray(chosen,dtype=int))
    pa=out['Pending-aware'];spent=float(s.cost[pa].sum())
    out['MinRisk-k']=np.sort(order[:len(pa)])
    cap=np.searchsorted(np.cumsum(s.cost[order]),spent+1e-10,side='right')
    out['MinRisk-spendcap']=np.sort(order[:cap])
    out['CostCountMatched'],status=matched(s.cost,ids,len(pa),spent,s.budget,s.seed,match_opts)
    for m in METHODS:
        idx=out[m]
        if idx is not None:
            assert len(np.unique(idx))==len(idx) and float(s.cost[idx].sum())<=s.budget+1e-8
    return out,status

def source_check(plan):
    for rel,h in plan['source_hashes'].items():
        assert sha(HERE/rel)==h,('frozen source drift',rel)
    policy=json.loads((HERE/'stage_governance_v1.json').read_text())
    stage_gate.require_stage('M3',policy)
    assert policy['M3_training_authorized'] and plan['registration_frozen']
    assert plan['role'] in ['engineering','variance_pilot','confirmation']
    if plan['role']=='confirmation':assert plan['confirmation_n']==len(plan['world_ids'])

def run_world(world,plan,out):
    source_check(plan);dest=Path(out)/f'w{world}';done=dest/'complete.json'
    if done.exists():
        result=json.loads(done.read_text());assert result['plan_sha256']==plan['_plan_sha256']
        return result
    if dest.exists():dest.rename(dest.with_name(dest.name+'_interrupted_'+str(time.time_ns())))
    dest.mkdir(parents=True);started=time.perf_counter();cfg=plan['config'];env=environment(world,cfg)
    all_actions={};locks=[];meta=[];status=[];data={k:v for k,v in env.items() if k not in ['costs','B']};data['B']=env['B']
    ids=env['candidate_ids'];no=len(env['O']);nd=len(env['D']);truth=np.array(cfg['truth_theta'])
    unique={}
    for cname in COSTS:
        s=state(env,env['costs'][cname],world,cfg);acts,match_status=acquire(s,ids,plan['matching_solver']);all_actions[cname]=acts
        data['cost_'+cname]=s.cost;data['effective_EAD_'+cname]=s.cost/.75
        h,g,u=core.geometry(s,True)
        for method in METHODS:
            idx=acts[method];fit_id=None
            if idx is not None:
                idx=np.sort(idx);key=tuple(idx.tolist())
                if key not in unique:unique[key]=len(unique)
                fit_id=unique[key];spent=float(s.cost[idx].sum());count=len(idx)
                base=float(np.sort(s.cost)[:count].sum());maxk=core.kstar(s.cost,s.budget);sortedc=np.sort(s.cost);mk=float(sortedc[:maxk].sum())
                maxswap=0
                for mm in range(1,min(maxk,len(s.cost)-maxk)+1):
                    if sortedc[maxk:maxk+mm].sum()-sortedc[maxk-mm:maxk].sum()<=s.budget-mk+1e-12:maxswap=mm
                d={'count':count,'cost':spent,'budget':s.budget,'Kstar':maxk,'count_sacrifice':maxk-count,'composition_premium':spent-base,'remaining_budget':s.budget-spent,'m_star':maxswap,
                   'proxy_pending':core.set_value(h,g,u,idx),'true_expected_selected_loss':float(np.dot(expit(env['C'][idx]@truth),s.cost[idx]/.75))}
            else:d={k:None for k in ['count','cost','budget','Kstar','count_sacrifice','composition_premium','remaining_budget','m_star','proxy_pending','true_expected_selected_loss']}
            lock={'world_id':world,'cost_condition':cname,'method':method,'estimable':idx is not None,'selected_indices':idx,'selected_ids':ids[idx] if idx is not None else None,'fit_id':fit_id,'select_hash':core.digest(ids[idx]) if idx is not None else None,**d}
            locks.append(lock)
        status.append({'cost_condition':cname,**match_status})
    write(dest/'selection_locks.json',locks);lock_sha=sha(dest/'selection_locks.json')
    write(dest/'selection_commitment.json',{'all_methods_locked_before_any_future_label_draw_or_fit':True,'sha256':lock_sha,'methods':METHODS,'cost_conditions':COSTS,'matching_status':status})
    ysO=[];ysC=[];rows=[];coefficients=[];predictions=[];fit_records=[];configs={}
    for rep in range(plan['label_M']):
        label_rng=np.random.default_rng(np.random.SeedSequence([world,plan['label_rng_tag'],rep]))
        yo=(label_rng.random(no)<expit(env['O']@truth)).astype(np.uint8)
        yc=(label_rng.random(len(ids))<expit(env['C']@truth)).astype(np.uint8);ysO.append(yo);ysC.append(yc)
        cache={}
        for key,fit_id in unique.items():
            idx=np.array(key,int);x=np.vstack([env['D'],env['O'],env['C'][idx]])
            y=np.r_[env['yD'],yo,yc[idx]];train_ids=np.r_[np.arange(nd+no),ids[idx]]
            for learner in ['LR','XGB']:
                tick=time.perf_counter()
                if learner=='LR':
                    coef=core.fit_logistic(x,y,cfg['lr_ridge']);pred=expit(env['T']@coef);coefficients.append(coef)
                    pred_index=len(coefficients)-1;backend='cpu';config_hash=None
                else:
                    model=XGBClassifier(**cfg['xgb'],random_state=world+7000,objective='binary:logistic',eval_metric='logloss')
                    model.fit(x[:,1:].astype(np.float32),y)
                    actual=json.loads(model.get_booster().save_config())['learner'];backend=actual['generic_param']['device']
                    assert backend.startswith('cuda'),('No CPU fallback',backend)
                    def find_key(obj,key):
                        if not isinstance(obj,dict):return []
                        return ([obj[key]] if key in obj else [])+[v for child in obj.values() for v in find_key(child,key)]
                    assert set(find_key(actual,'max_bin'))=={'256'}
                    assert set(find_key(actual,'max_depth'))=={'3'}
                    assert set(find_key(actual,'num_trees'))=={'100'}
                    config_text=json.dumps(actual,sort_keys=True);config_hash=hashlib.sha256(config_text.encode()).hexdigest()
                    if 'XGB' not in configs:configs['XGB']=actual
                    pred=model.predict_proba(env['T'][:,1:].astype(np.float32))[:,1];predictions.append(pred)
                    pred_index=len(predictions)-1
                risk=float(np.mean((pred.astype(float)-env['pT'])**2));brier=float(np.mean((pred.astype(float)-env['yT'])**2))
                record={'world_id':world,'replicate':rep,'fit_id':fit_id,'learner':learner,'risk':risk,'brier':brier,'prediction_index':pred_index,'backend':backend,
                        'train_n':len(y),'train_x_hash':core.digest(x),'train_label_hash':core.digest(y),'train_ids_hash':core.digest(train_ids),
                        'learning_seed':world+7000,'actual_config_hash':config_hash,'fit_seconds':time.perf_counter()-tick}
                fit_records.append(record);cache[fit_id,learner]=record
        for lock in locks:
            for learner in ['LR','XGB']:
                idx=lock['selected_indices'];record=cache[lock['fit_id'],learner] if idx is not None else {k:None for k in ['risk','brier','prediction_index','backend','train_n','train_x_hash','train_label_hash','train_ids_hash','learning_seed','actual_config_hash','fit_seconds']}
                cname=lock['cost_condition'];cost=env['costs'][cname]
                rows.append({k:v for k,v in lock.items() if k not in ['selected_indices','selected_ids']}|record|{'world_id':world,'replicate':rep,'learner':learner,
                             'realized_selected_loss':float(np.dot(yc[idx],cost[idx]/.75)) if idx is not None else None})
    data['yO']=np.array(ysO);data['yC']=np.array(ysC)
    np.savez_compressed(dest/'inputs.npz',**data)
    np.savez_compressed(dest/'predictions.npz',LR_coefficients=np.array(coefficients),XGB_predictions=np.array(predictions,dtype=np.float32))
    pd.DataFrame(rows).to_csv(dest/'outcomes.csv',index=False);pd.DataFrame(fit_records).to_csv(dest/'fits.csv',index=False)
    write(dest/'actual_config_examples.json',configs)
    assert sha(dest/'selection_locks.json')==lock_sha
    source_check(plan)
    result={'completed':True,'world_id':world,'role':plan['role'],'plan_sha256':plan['_plan_sha256'],'seconds':time.perf_counter()-started,
            'label_replicates':plan['label_M'],'outcome_rows':len(rows),'unique_fits_per_learner':len(unique)*plan['label_M'],
            'unmatched_cost_conditions':[x['cost_condition'] for x in status if not x['valid_incumbent']],
            'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}}
    write(done,result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());plan['_plan_sha256']=sha(args.plan);source_check(plan)
    args.out.mkdir(parents=True,exist_ok=True);worlds=plan['world_ids'][args.worker::args.workers]
    from importlib.metadata import version
    registration={'plan_sha256':sha(args.plan),'source_hashes':plan['source_hashes'],'worker':args.worker,'world_ids':worlds,'environment':{'python':platform.python_version(),**{k:version(k) for k in ['numpy','scipy','scikit-learn','xgboost']}}}
    rp=args.out/'registration.json'
    if rp.exists():assert json.loads(rp.read_text())==registration
    else:write(rp,registration)
    reports=[]
    for world in worlds:
        r=run_world(world,plan,args.out);reports.append(r)
        write(args.out/'progress.json',{'completed_worlds':len(reports),'assigned_worlds':len(worlds),'last_world':world})
        print(json.dumps(r),flush=True)
    write(args.out/'completion.json',{'completed':True,'worlds':len(reports),'fits_per_learner':sum(x['unique_fits_per_learner'] for x in reports),'plan_sha256':sha(args.plan),'sources_unchanged':True})

if __name__=='__main__':main()
