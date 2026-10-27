"""Conditional E2b: all-pair complementarity and pair-search x exchange factorial.

All four cells share the same conditional geometry, full candidate pool, best
singleton safeguard, and (when enabled) ONE 1->1 exchange with 8 removal probes.
No result-driven expansion of partners/exchange passes is performed.
"""
from __future__ import annotations
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from runtime import *


def signed_pair_gain(c,g,u,pairs):
    a=u@c;b=a@g;i,j=pairs.T
    aa=1+np.einsum('ij,ij->i',a[i],u[i]);dd=1+np.einsum('ij,ij->i',a[j],u[j]);ab=np.einsum('ij,ij->i',a[i],u[j])
    ni=np.einsum('ij,ij->i',b[i],a[i]);nj=np.einsum('ij,ij->i',b[j],a[j]);nij=np.einsum('ij,ij->i',b[i],a[j])
    return (dd*ni+aa*nj-2*ab*nij)/(aa*dd-ab*ab)


def four_cell(s,paired,exchange,config):
    started=time.perf_counter();h,g,u=frozen.geometry(s);c=frozen.inverse(h)
    pairs=np.array(list(itertools.combinations(range(len(u)),2)),int).reshape(-1,2)
    available=np.ones(len(u),bool);chosen=[];remain=s.budget;steps=[];evaluations=0;gamma_logs=[];cap=config['e2b']['max_proxy_evaluations'];capped=False
    while available.any():
        if time.perf_counter()-started>config['e2b']['max_seconds'] or evaluations>=cap:capped=True;break
        single=frozen.marginal(c,g,u);evaluations+=len(u);valid=available&(s.cost<=remain+1e-10)
        feasible_pairs=pairs[available[pairs].all(axis=1)&(s.cost[pairs].sum(axis=1)<=remain+1e-10)]
        pg=signed_pair_gain(c,g,u,feasible_pairs);gamma=pg-single[feasible_pairs].sum(axis=1)
        tolerance=config['gamma_absolute_tolerance']+config['gamma_relative_tolerance']*np.maximum(np.abs(pg),np.abs(single[feasible_pairs].sum(axis=1)))
        gamma_logs.append({'step':len(steps),'pairs':feasible_pairs,'gamma':gamma,'tolerance':tolerance})
        # Audit all pairs for every cell; only paired cells use the pair ranking.
        evaluations+=len(feasible_pairs)
        best=(-np.inf,[],0.,0.)
        if valid.any():
            score=np.where(valid,single/np.maximum(s.cost,1e-15),-np.inf);i=int(np.argmax(score));best=(score[i],[i],single[i],0.)
        if paired and len(pg):
            ratio=pg/np.maximum(s.cost[feasible_pairs].sum(axis=1),1e-15);j=int(np.argmax(ratio))
            if ratio[j]>best[0]:best=(ratio[j],feasible_pairs[j].tolist(),pg[j],gamma[j])
        if not best[1] or best[2]<=1e-18:break
        steps.append({'ids':best[1],'gain':float(best[2]),'gamma':float(best[3])})
        for i in best[1]:chosen.append(i);available[i]=False;remain-=s.cost[i];c=frozen.update(c,u[i])
    feasible=np.flatnonzero(s.cost<=s.budget+1e-10);initial=frozen.marginal(frozen.inverse(h),g,u)
    if len(feasible):
        best=int(feasible[np.argmax(initial[feasible])])
        if initial[best]>frozen.set_value(h,g,u,np.asarray(chosen,int))+1e-14:chosen=[best]
    exchange_count=0
    if exchange:
        for _ in range(config['e2b']['exchange_passes']):
            current=np.asarray(chosen,int);old=frozen.set_value(h,g,u,current);best_set=current;best_value=old;removals=[]
            for j in current:
                rest=current[current!=j];removals.append((old-frozen.set_value(h,g,u,rest),int(j)))
            for _,j in sorted(removals)[:config['e2b']['exchange_remove_candidates']]:
                rest=current[current!=j];cc=frozen.inverse(h+u[rest].T@u[rest]);base=frozen.set_value(h,g,u,rest)
                eligible=np.ones(len(u),bool);eligible[rest]=False;eligible&=(s.cost<=s.budget-s.cost[rest].sum()+1e-10)
                gains=frozen.marginal(cc,g,u);evaluations+=len(u)
                if eligible.any():
                    gains=np.where(eligible,gains,-np.inf);i=int(np.argmax(gains))
                    if base+gains[i]>best_value+1e-14:best_value=base+gains[i];best_set=np.r_[rest,i]
            if best_value>old+1e-14:chosen=best_set.tolist();exchange_count+=1
    result=frozen.finish(s,np.asarray(chosen,int),{'proxy_value':frozen.set_value(h,g,u,np.asarray(chosen,int)),'paired_construction':paired,
                         'exchange_enabled':exchange,'accepted_exchanges':exchange_count,'proxy_evaluations':evaluations,
                         'computation_capped':capped,'selection_seconds':time.perf_counter()-started,'steps':steps})
    return result,gamma_logs


def load_e1_cache(cache,base):
    records=[json.loads(x) for x in (base/'branch_outcomes.jsonl').read_text().splitlines()]
    for row in records:
        key=row['fit_key']
        if key in cache.entries:continue
        data=np.load(base/'private/predictions'/f'{key}.npz')
        metrics={k:row[k] for k in ['brier','target_brier','rest_brier','target_share','test_n']}
        cache.entries[key]={'prediction':data['prediction'],'metrics':metrics,'actual_backend':row['actual_backend'],'updater':row['updater'],
                            'fit_seconds':row['original_fit_time'],'cache_hit':True,'charged_fit_seconds':0.,'fit_key':key,
                            'train_n':row['train_n'],'train_hash':row['train_hash']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base;out=base/'e2b';out.mkdir(exist_ok=False)
    identity=json.loads((base/'source_identity.json').read_text());config=identity['configuration'];private=base/'private'
    write_json(out/'activation.json',{'basis':'E1 learner-specific effects differ and BRIDGE has zero positive pair-selection steps in all 18 states; E2b four-cell identification is required.',
                                    'scope':'all registered 18 anchors; no seed/state selection by improvement','source_hash':sha(__file__),'parent_identity':identity['identity_hash'],
                                    'exchange_neighborhood':'common 1-to-1 only, maximum one accepted pass, same eight exact removal-loss probes for all exchange cells',
                                    'decision_precommitted':'if visited all-pair states contain no positive Gamma above registered tolerance, do not increase partners or exchange passes'})
    cache=FitCache(config,identity['identity_hash'],private);load_e1_cache(cache,base)
    e1=pd.read_json(base/'branch_outcomes.jsonl',lines=True);selections=[];branches=[];gamma_summaries=[]
    for dataset in config['datasets']:
        for seed in config['seeds']:
            for r in config['anchor_rounds']:
                context=build_context(dataset,seed,r,config,private);s=context.acquisition
                for paired,exchange in itertools.product([False,True],[False,True]):
                    selection,gammas=four_cell(s,paired,exchange,config);ids=selection.selected_ids
                    method=('pair' if paired else 'single')+('_exchange' if exchange else '_noexchange')
                    key=context.feedback.lock(ids,method,s.seed);selected_hash=digest(np.sort(context.feedback.c_ids[ids]))
                    diag={k:v for k,v in selection.diagnostics.items() if k!='steps'}
                    row={'state_id':context.state_id,'dataset':dataset,'seed':seed,'round':r,'method':method,'selected_hash':selected_hash,
                         'count':len(ids),'cost':selection.predicted_cost,'budget':s.budget,'representation_hash':context.manifest['representation_hash'],**diag};selections.append(row)
                    if gammas:
                        gamma=np.concatenate([v['gamma'] for v in gammas]);tolerance=np.concatenate([v['tolerance'] for v in gammas]);pp=np.concatenate([v['pairs'] for v in gammas]);step=np.concatenate([np.full(len(v['gamma']),v['step']) for v in gammas])
                    else:gamma=np.empty(0);tolerance=np.empty(0);pp=np.empty((0,2),int);step=np.empty(0,int)
                    path=private/'e2b_gamma'/f'{context.state_id}_{method}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,gamma=gamma,tolerance=tolerance,pairs=pp,step=step)
                    gamma_summaries.append({'state_id':context.state_id,'dataset':dataset,'seed':seed,'round':r,'method':method,
                                            'pair_evaluations':len(gamma),'positive_above_tolerance':int((gamma>tolerance).sum()),
                                            'negative_below_tolerance':int((gamma<-tolerance).sum()),'gamma_max':float(gamma.max()) if len(gamma) else None,
                                            'gamma_min':float(gamma.min()) if len(gamma) else None,'raw_gamma_file_hash':sha(path),
                                            'selected_pair_steps':sum(len(t['ids'])==2 for t in selection.diagnostics['steps'])})
                    for learner in config['learners']:
                        fitted=cache.fit(context,ids,learner,key)
                        ref=e1[(e1.state_id==context.state_id)&(e1.learner==learner)]
                        ordinary=float(ref[ref.selector=='OrdinaryOnly'].brier.iloc[0]);mr=float(ref[ref.selector=='MinRisk'].brier.iloc[0])
                        branches.append({'state_id':context.state_id,'dataset':dataset,'seed':seed,'round':r,'method':method,'learner':learner,
                                         'selected_hash':selected_hash,'proxy_value':diag['proxy_value'],'brier':fitted['metrics']['brier'],
                                         'gain_vs_ordinary':ordinary-fitted['metrics']['brier'],'gain_vs_minrisk':mr-fitted['metrics']['brier'],
                                         'actual_backend':fitted['actual_backend'],'cache_hit':fitted['cache_hit'],'fit_key':fitted['fit_key']})
                print('E2b',context.state_id,'new fits',cache.fits,'reused fits',cache.hits,flush=True)
    write_jsonl(out/'selection_summary.jsonl',selections);write_jsonl(out/'branch_outcomes.jsonl',branches);write_jsonl(out/'gamma_summary.jsonl',gamma_summaries)
    count=sum(x['positive_above_tolerance'] for x in gamma_summaries)
    completion={'completed':True,'states':18,'four_cell_method_slots':len(selections),'nominal_fit_slots':len(branches),
                'actual_new_fits':cache.fits,'reused_fits':cache.hits,'positive_gamma_evaluations':count,
                'capped_cells':sum(x['computation_capped'] for x in selections),
                'pair_route_decision':'do not expand partner count/exchange passes' if count==0 else 'positive surrogate complementarity exists; actual benefit must be checked separately'}
    write_json(out/'completion.json',completion);print(json.dumps(completion),flush=True)


if __name__=='__main__':main()
