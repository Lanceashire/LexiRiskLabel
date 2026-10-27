"""Predeclared six-anchor E2d; exact counts and local risk-matched swaps.

Only frozen D labels, cached D-only IG, features and frozen prices enter selection.
All actions in a state are locked before labels or evaluation are accessed.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from runtime import *
from run_e2b import load_e1_cache


def exact_ig(s, scores, k, limit):
    if not k:return np.empty(0, int), {'solver': 'empty'}
    scale=max(float(np.max(np.abs(scores))), 1e-15)
    a=np.vstack([s.cost,np.ones(len(s.cost))])
    result=milp(-scores/scale, integrality=np.ones(len(s.cost)),
                bounds=Bounds(0,1), constraints=LinearConstraint(a,[-np.inf,k],[s.budget,k]),
                options={'time_limit':limit,'mip_rel_gap':0.0})
    if not result.success or result.status!=0:raise RuntimeError('exact IG optimization failed: '+result.message)
    selected=np.flatnonzero(result.x>.5)
    return selected, {'solver':'scipy_milp','solver_status':int(result.status),'mip_gap':float(result.mip_gap),
                      'ig_sum':float(scores[selected].sum())}


def exact_vopt(s,k,h,g,u):
    c=frozen.inverse(h);free=np.ones(len(s.cost),bool);chosen=[];remain=s.budget
    while len(chosen)<k:
        slots=k-len(chosen);valid=free & (s.cost<=remain+1e-10)
        for i in np.flatnonzero(valid):
            others=np.sort(s.cost[free & (np.arange(len(free))!=i)])
            if len(others)<slots-1 or others[:slots-1].sum()>remain-s.cost[i]+1e-10:valid[i]=False
        if not valid.any():raise RuntimeError('fixed-count completion infeasible')
        gains=frozen.marginal(c,g,u)
        i=int(np.argmax(np.where(valid,gains/np.maximum(s.cost,1e-15),-np.inf)))
        chosen.append(i);free[i]=False;remain-=s.cost[i];c=frozen.update(c,u[i])
    return np.asarray(chosen,int),{'proxy_value':frozen.set_value(h,g,u,np.asarray(chosen,int))}


def matched_swap(s, selected, tolerance, rng):
    outside=np.setdiff1d(np.arange(len(s.cost)),selected)
    cost=float(s.cost[selected].sum());pairs=[]
    for remove in selected:
        for add in outside:
            change=float(s.cost[add]-s.cost[remove])
            if abs(change)<=tolerance+1e-12 and cost+change<=s.budget+1e-10:pairs.append((int(remove),int(add)))
    if not pairs:return None,0
    remove,add=pairs[int(rng.integers(len(pairs)))];ids=np.r_[selected[selected!=remove],add]
    return ids,len(pairs)


def verify_outputs(out, private, contexts):
    selections=[json.loads(x) for x in (out/'selection_summary.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (out/'branch_outcomes.jsonl').read_text().splitlines()]
    bystate={c.state_id:c for c in contexts};groups={}
    for row in rows:
        key=(row['state_id'],row['k'],row['method']);groups.setdefault(key,[]).append(row)
        pred=np.load(private/'predictions'/f"{row['fit_key']}.npz")
        assert abs(float(np.mean((pred['prediction']-pred['y'])**2))-row['brier'])<1e-12
        assert row['actual_backend'].startswith('cuda') if row['learner']=='XGB' else row['actual_backend']=='cpu'
    for group in groups.values():
        assert len(group)==2 and len({r['selected_hash'] for r in group})==1 and len({r['train_hash'] for r in group})==1
    for row in selections:
        c=bystate[row['state_id']];record=json.loads((private/'e2d_sets'/f"{row['action_id']}.json").read_text())
        ids=np.asarray(record['local_ids'],int)
        assert len(ids)==row['k']==len(np.unique(ids))
        assert row['cost']<=row['budget']+1e-7
        assert abs(float(c.acquisition.cost[ids].sum())-row['cost'])<1e-10
        assert digest(np.sort(c.feedback.c_ids[ids]))==row['selected_hash']
        assert (private/'selection_locks'/f"{record['lock_key']}.json").exists()
        if row['method'].startswith('RiskMatched-'):
            target=next(x for x in selections if x['state_id']==row['state_id'] and x['k']==row['k'] and x['method']==row['matched_to'])
            assert abs(row['cost']-target['cost'])<=row['risk_tolerance']+1e-10
            target_record=json.loads((private/'e2d_sets'/f"{target['action_id']}.json").read_text())
            assert len(set(ids)^set(target_record['local_ids']))==2
    result={'passed':True,'checks':['exact unique k','common budget','same actions and training rows across learners',
            'saved-prediction Brier recomputation','actual CUDA backend','matched-risk tolerance and exactly one swap','selection locks present'],
            'selection_slots':len(selections),'learner_slots':len(rows)}
    write_json(out/'verification.json',result)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args()
    base=args.base;out=base/'e2d';out.mkdir(exist_ok=False);private=base/'private'
    identity=json.loads((base/'source_identity.json').read_text());config=identity['configuration']
    verify_identity(identity,run_identity(config))
    stage=json.loads((HERE/'e2d_config.json').read_text())
    write_json(out/'activation.json',{'configuration':stage,'source_hash':sha(__file__),
               'config_hash':sha(HERE/'e2d_config.json'),'parent_identity':identity['identity_hash'],
               'scope':'all six round-5 anchors, fixed before any E2d endpoints',
               'interpretation':'two development seeds; risk-matched swap is a local composition diagnostic, not a global random-policy estimate'})
    cache=FitCache(config,identity['identity_hash'],private);load_e1_cache(cache,base)
    selections=[];branches=[];matching=[];states=[];contexts=[]
    for dataset in config['datasets']:
        for seed in config['seeds']:
            c=build_context(dataset,seed,stage['anchor_round'],config,private);contexts.append(c);s=c.acquisition
            old=json.loads((private/'states'/f'{c.state_id}.json').read_text())
            assert np.array_equal(old['C_ids'],c.feedback.c_ids)
            assert np.array_equal(old['D_ids'],c.feedback.d_ids)
            assert np.array_equal(old['candidate_costs'],s.cost)
            scores=np.asarray(old['IG_scores']);ks=sorted({int(np.floor(f*frozen.kstar(s.cost,s.budget))) for f in stage['count_fractions']})
            h,g,u=frozen.geometry(s);order=np.argsort(s.cost,kind='stable');actions=[]
            states.append({**c.manifest,'Kstar':frozen.kstar(s.cost,s.budget),'unique_k':ks,
                           'fraction_to_k':{str(f):int(np.floor(f*frozen.kstar(s.cost,s.budget))) for f in stage['count_fractions']}})
            for k in ks:
                for method in stage['methods']:
                    started=time.perf_counter()
                    if method=='MinRisk-k':ids,diag=order[:k],{}
                    elif method=='IG-k':ids,diag=exact_ig(s,scores,k,stage['milp_time_limit_seconds'])
                    else:ids,diag=exact_vopt(s,k,h,g,u)
                    assert len(ids)==k and s.cost[ids].sum()<=s.budget+1e-7
                    diag['selection_seconds']=time.perf_counter()-started
                    actions.append({'method':method,'k':k,'ids':ids,'diag':diag})
            if stage['include_zero_baseline'] and 0 not in ks:
                actions.append({'method':'OrdinaryOnly','k':0,'ids':np.empty(0,int),'diag':{}})
            for a in list(actions):
                if a['method'] not in stage['matched_control_targets']:continue
                tolerance=stage['risk_match_absolute_tolerance_budget_fraction']*s.budget
                rngseed=stage['selector_seed_offset']+s.seed*100+10*a['k']+stage['methods'].index(a['method'])
                ids,n=matched_swap(s,a['ids'],tolerance,np.random.default_rng(rngseed))
                m={'state_id':c.state_id,'dataset':dataset,'seed':seed,'k':a['k'],'target_method':a['method'],
                   'risk_tolerance':tolerance,'eligible_one_swap_sets':n,'selector_seed':rngseed,
                   'status':'matched' if ids is not None else 'not_estimable','tolerance_widened':False}
                matching.append(m)
                if ids is not None:actions.append({'method':'RiskMatched-'+a['method'],'k':a['k'],'ids':ids,
                    'diag':{'matched_to':a['method'],'risk_tolerance':tolerance,'eligible_one_swap_sets':n,'selector_seed':rngseed}})
            # Lock every action before resource outcomes or labels can be inspected.
            for a in actions:
                a['lock']=c.feedback.lock(a['ids'],a['method'],s.seed)
                a['action_id']=digest({'state':c.state_id,'k':a['k'],'method':a['method']})
                write_json(private/'e2d_sets'/f"{a['action_id']}.json",{'local_ids':a['ids'],
                    'global_ids':c.feedback.c_ids[a['ids']],'lock_key':a['lock'],'method':a['method'],'k':a['k']})
            for a in actions:
                ids=a['ids'];k=a['k'];cost=float(s.cost[ids].sum())
                row={'state_id':c.state_id,'dataset':dataset,'seed':seed,'round':c.round,'method':a['method'],'k':k,
                     'Kstar':frozen.kstar(s.cost,s.budget),'action_id':a['action_id'],
                     'selected_hash':digest(np.sort(c.feedback.c_ids[ids])),'count':len(ids),'cost':cost,'budget':s.budget,
                     'composition_premium':cost-float(s.cost[order[:k]].sum()),'proxy_value':frozen.set_value(h,g,u,ids),**a['diag']}
                selections.append(row)
                for learner in config['learners']:
                    fitted=cache.fit(c,ids,learner,a['lock'])
                    branches.append({**row,'learner':learner,**fitted['metrics'],
                        **{key:fitted[key] for key in ['fit_key','train_hash','train_n','actual_backend','cache_hit','charged_fit_seconds']},
                        'original_fit_time':fitted['fit_seconds'],'updater':fitted['updater']})
            print('E2d',c.state_id,'Kstar',frozen.kstar(s.cost,s.budget),'k',ks,'actions',len(actions),'new fits',cache.fits,'reused',cache.hits,flush=True)
    write_jsonl(out/'state_manifest.jsonl',states);write_jsonl(out/'selection_summary.jsonl',selections)
    write_jsonl(out/'branch_outcomes.jsonl',branches);write_jsonl(out/'matching_status.jsonl',matching)
    verify_outputs(out,private,contexts)
    completion={'completed':True,'states':len(states),'independent_trajectory_clusters':6,'independent_seed_values':2,
                'unique_state_k_cells':sum(len(x['unique_k']) for x in states),'selection_slots':len(selections),
                'nominal_fit_slots':len(branches),'actual_new_fits':cache.fits,'reused_fits':cache.hits,
                'unique_training_interventions':len({(r['state_id'],r['selected_hash']) for r in selections}),
                'matched_controls':sum(x['status']=='matched' for x in matching),
                'not_estimable_controls':sum(x['status']=='not_estimable' for x in matching)}
    write_json(out/'completion.json',completion);print(json.dumps(completion),flush=True)


if __name__=='__main__':main()
