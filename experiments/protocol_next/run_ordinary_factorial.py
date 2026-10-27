"""Small 2x2 component diagnosis with one common greedy solver; no new method search."""
from __future__ import annotations
import argparse
from dataclasses import replace
from runtime import *
import run_mechanisms as previous


def component_select(s, conditional, batch):
    h,g,u=frozen.geometry(s,conditional);c=frozen.inverse(h)
    initial=frozen.marginal(c,g,u);free=np.ones(len(u),bool);ids=[];remaining=s.budget;steps=[]
    while free.any():
        gains=frozen.marginal(c,g,u) if batch else initial
        feasible=free&(s.cost<=remaining+1e-10)
        if not feasible.any():break
        scores=np.where(feasible,gains/np.maximum(s.cost,1e-15),-np.inf)
        j=int(np.argmax(scores))
        if gains[j]<=1e-18:break
        ids.append(j);free[j]=False;remaining-=s.cost[j]
        steps.append({'id':j,'used_gain':float(gains[j]),'remaining_budget':float(remaining)})
        if batch:c=frozen.update(c,u[j])
    return np.sort(np.asarray(ids,int)),steps


def family(s):
    out=[{'method':'OrdinaryOnly','ids':np.empty(0,int),'conditional':None,'batch':None,'steps':[]}]
    costorder=np.argsort(s.cost,kind='stable');k=frozen.kstar(s.cost,s.budget)
    out.append({'method':'MinRisk','ids':np.sort(costorder[:k]),'conditional':None,'batch':None,'steps':[]})
    for conditional in [False,True]:
        for batch in [False,True]:
            ids,steps=component_select(s,conditional,batch)
            out.append({'method':f'C{int(conditional)}B{int(batch)}','ids':ids,'conditional':conditional,'batch':batch,'steps':steps})
    return out


def common_identity(s,test,yt,pt,yc):
    return {name:digest(x) for name,x in {'D':s.visible_x,'yD':s.visible_y,'C':s.candidate_x,'yC':yc,
        'Q':s.reference_x,'theta':s.theta,'cost':s.cost,'test':test,'yTest':yt,'pTest':pt}.items()}|{'budget':s.budget}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args()
    base=args.base;out=base/'ordinary_factorial_small';out.mkdir(exist_ok=False)
    private=base/'private/ordinary_factorial_small';private.mkdir(parents=True,exist_ok=True)
    stage=json.loads((HERE/'ordinary_factorial_config.json').read_text())
    parent=json.loads((base/'source_identity.json').read_text());verify_identity(parent,run_identity(parent['configuration']))
    files=[HERE/'ordinary_factorial_config.json',HERE/'run_ordinary_factorial.py',HERE/'runtime.py',
           ROOT/'experiments/research_v2/core.py',ROOT/'experiments/research_v2/config.json',ROOT/'experiments/research_v2/run_mechanisms.py']
    sources={str(p.relative_to(ROOT)):sha(p) for p in files}
    write_json(out/'activation.json',{'configuration':stage,'source':sources,'parent_identity':parent['identity_hash'],
        'phase':'configuration and all actions frozen before any outcome fitting; old development seeds'})
    start=time.perf_counter();pending=[];locks=[];state_rows=[];max_zero_error=0.
    for seed in stage['seeds']:
        a=previous.generate('A1',seed);b=previous.generate('A2',seed)
        sa,yoa,yca,ta,yta,pta,_=a;sb,yob,ycb,tb,ytb,ptb,_=b
        assert common_identity(sa,ta,yta,pta,yca)==common_identity(sb,tb,ytb,ptb,ycb)
        assert np.array_equal(yoa,yob) and np.array_equal(previous.generate('A1',seed)[0].ordinary_x,sa.ordinary_x)
        # The generator's ordinary target coordinate is zero, hence both coverage cases have the same true probabilities.
        assert np.array_equal(a[-1](sa.ordinary_x),b[-1](sb.ordinary_x))
        empty=replace(sa,ordinary_x=sa.ordinary_x[:0])
        cases=[('empty_O',empty,yoa[:0]),('A1_irrelevant_direction',sa,yoa),('A2_nuisance_direction',sb,yob)]
        for case,s,yo in cases:
            actions=family(s);h,g,u=frozen.geometry(s,True)
            if case=='empty_O':
                h0,g0,u0=frozen.geometry(s,False)
                max_zero_error=max(max_zero_error,float(np.max(np.abs(h-h0))),float(np.max(np.abs(g-g0))),float(np.max(np.abs(u-u0))))
                for batch in [False,True]:
                    assert np.array_equal(actions[2+int(batch)]['ids'],actions[4+int(batch)]['ids'])
            for p in actions:
                ids=p['ids'];assert len(ids)==len(np.unique(ids)) and s.cost[ids].sum()<=s.budget+1e-7
                own_h,own_g,own_u=frozen.geometry(s,bool(p['conditional']))
                p.update({'seed':seed,'coverage':case,'selected_hash':digest(ids),'count':len(ids),
                    'cost':float(s.cost[ids].sum()),'budget':s.budget,'Kstar':frozen.kstar(s.cost,s.budget),
                    'common_conditional_proxy':frozen.set_value(h,g,u,ids),
                    'own_proxy':frozen.set_value(own_h,own_g,own_u,ids)})
                locks.append(p)
            state_rows.append({'seed':seed,'coverage':case,'common_non_O_identity':common_identity(s,ta,yta,pta,yca),
                'ordinary_x_hash':digest(s.ordinary_x),'ordinary_n':len(s.ordinary_x),
                'family_size':len(actions),'unique_selected_sets':len({p['selected_hash'] for p in actions})})
            pending.append((seed,case,s,yo,yca,ta,yta,pta,actions))
    # All nine-state action families are physically persisted before evaluation/training starts.
    write_jsonl(private/'selection_locks.jsonl',locks);lock_hash=sha(private/'selection_locks.jsonl')
    write_json(out/'selection_lock_commitment.json',{'sha256':lock_hash,'actions':len(locks),'all_locked_before_training':True})
    write_jsonl(out/'state_manifest.jsonl',state_rows)
    # Exact null-direction geometry control: zero target covariance along the added direction, block-diagonal precision.
    hh=np.eye(2);gg=np.diag([1.,0.]);uu=np.array([[1.,0.],[0.,5.]])
    before=frozen.marginal(frozen.inverse(hh),gg,uu)
    after=frozen.marginal(frozen.inverse(np.diag([1.,100.])),gg,uu)
    assert np.array_equal(before,after) and before[1]==0
    cache={};records=[];fit_logs=[];hits=0;gpu_devices=set();max_recomputed_error=0.
    for seed,case,s,yo,yc,test,yt,pt,actions in pending:
        nd,no,nc=len(s.visible_y),len(yo),len(yc)
        np.savez_compressed(private/f'{seed}_{case}_environment.npz',D=s.visible_x,yD=s.visible_y,O=s.ordinary_x,yO=yo,
            C=s.candidate_x,yC=yc,Q=s.reference_x,cost=s.cost,budget=s.budget,test=test,yTest=yt,pTest=pt)
        for learner in ['LR','XGB']:
            for p in actions:
                ids=p['ids'];x=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]]);y=np.r_[s.visible_y,yo,yc[ids]]
                key=digest({'learner':learner,'x':digest(x),'y':digest(y),'fit_seed':seed+7000,
                    'parameters':stage['xgb'] if learner=='XGB' else stage['lr_ridge']})
                reused=key in cache;tick=time.perf_counter()
                if reused:model,backend=cache[key];hits+=1
                elif learner=='LR':model=frozen.fit_logistic(x,y,stage['lr_ridge']);backend='cpu';cache[key]=(model,backend)
                else:
                    model=XGBClassifier(**stage['xgb'],random_state=seed+7000,objective='binary:logistic',eval_metric='logloss')
                    model.fit(x[:,1:].astype(np.float32),y)
                    backend=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
                    if not backend.startswith('cuda'):raise RuntimeError('actual CUDA required; no silent CPU fallback')
                    gpu_devices.add(backend);cache[key]=(model,backend)
                prediction=expit(test@model) if learner=='LR' else model.predict_proba(test[:,1:].astype(np.float32))[:,1]
                excess=float(np.mean((prediction-pt)**2));brier=float(np.mean((prediction-yt)**2))
                evalkey=digest({'fit_key':key,'test':digest(test),'pTest':digest(pt),'yTest':digest(yt)})
                path=private/'predictions'/f'{evalkey}.npz';path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,prediction=prediction,pTest=pt,yTest=yt,selected_ids=ids)
                saved=np.load(path)
                max_recomputed_error=max(max_recomputed_error,abs(excess-float(np.mean((saved['prediction']-saved['pTest'])**2))),
                    abs(brier-float(np.mean((saved['prediction']-saved['yTest'])**2))))
                row={k:v for k,v in p.items() if k not in ['ids','steps']}
                row.update({'learner':learner,'fit_key':key,'evaluation_key':evalkey,'cache_hit':reused,'actual_backend':backend,
                    'train_n':len(y),'excess_brier':excess,'brier':brier,'realized_selected_loss':float(yc[ids].sum()),
                    'ordinary_predicted_risk':float(expit(s.ordinary_x@s.theta).sum()),'ordinary_realized_loss':float(yo.sum())})
                records.append(row);fit_logs.append({'seed':seed,'coverage':case,'method':p['method'],'learner':learner,
                    'fit_key':key,'cache_hit':reused,'backend':backend,'seconds':time.perf_counter()-tick})
        print('factorial complete',seed,case,'unique_fits',len(cache),'cache_hits',hits,flush=True)
    frame=pd.DataFrame(records);frame.to_csv(out/'branch_outcomes.csv',index=False);write_jsonl(private/'fit_log.jsonl',fit_logs)
    contrasts=[]
    for (seed,case,learner),q in frame.groupby(['seed','coverage','learner']):
        m=q.set_index('method');ex=m.excess_brier;sample=m.brier
        pairs={'conditioning_static':('C0B0','C1B0'),'conditioning_batch':('C0B1','C1B1'),
               'batch_unconditional':('C0B0','C0B1'),'batch_conditional':('C1B0','C1B1'),
               'combined_vs_MinRisk':('MinRisk','C1B1'),'combined_vs_OrdinaryOnly':('OrdinaryOnly','C1B1')}
        for contrast,(left,right) in pairs.items():
            contrasts.append({'seed':seed,'coverage':case,'learner':learner,'contrast':contrast,
                'excess_gain':float(ex[left]-ex[right]),'sampled_brier_gain':float(sample[left]-sample[right]),
                'changed_selected_set':m.loc[left,'selected_hash']!=m.loc[right,'selected_hash'],
                'right_minus_left_count':int(m.loc[right,'count']-m.loc[left,'count']),
                'right_minus_left_cost':float(m.loc[right,'cost']-m.loc[left,'cost'])})
        contrasts.append({'seed':seed,'coverage':case,'learner':learner,'contrast':'interaction',
            'excess_gain':float((ex['C1B0']-ex['C1B1'])-(ex['C0B0']-ex['C0B1'])),
            'sampled_brier_gain':float((sample['C1B0']-sample['C1B1'])-(sample['C0B0']-sample['C0B1'])),
            'changed_selected_set':None,'right_minus_left_count':None,'right_minus_left_cost':None})
        if case=='empty_O':
            assert ex['C0B0']==ex['C1B0'] and ex['C0B1']==ex['C1B1']
            assert m.loc['C0B0','fit_key']==m.loc['C1B0','fit_key'] and m.loc['C0B1','fit_key']==m.loc['C1B1','fit_key']
    cc=pd.DataFrame(contrasts);cc.to_csv(out/'paired_component_contrasts.csv',index=False)
    summary=cc.groupby(['coverage','learner','contrast'],as_index=False).agg(mean_excess_gain=('excess_gain','mean'),
        positive_seeds=('excess_gain',lambda x:int((x>1e-12).sum())),negative_seeds=('excess_gain',lambda x:int((x< -1e-12).sum())),
        zero_seeds=('excess_gain',lambda x:int((x.abs()<=1e-12).sum())),mean_sampled_brier_gain=('sampled_brier_gain','mean'),
        mean_count_difference=('right_minus_left_count','mean'),mean_cost_difference=('right_minus_left_cost','mean'))
    summary.to_csv(out/'component_summary.csv',index=False)
    directional=[]
    for (seed,learner,contrast),q in cc[cc.coverage!='empty_O'].groupby(['seed','learner','contrast']):
        v=q.set_index('coverage')
        directional.append({'seed':seed,'learner':learner,'contrast':contrast,
            'A2_minus_A1_excess_gain':float(v.loc['A2_nuisance_direction','excess_gain']-v.loc['A1_irrelevant_direction','excess_gain'])})
    pd.DataFrame(directional).to_csv(out/'direction_difference.csv',index=False)
    assert lock_hash==sha(private/'selection_locks.jsonl') and max_zero_error==0 and max_recomputed_error<1e-12
    assert len(records)==stage['max_fit_slots'] and len(locks)==stage['max_selector_slots']
    actual_gpu=sum(1 for model,backend in cache.values() if backend.startswith('cuda'))
    completion={'completed':True,'selector_slots':len(locks),'fit_slots':len(records),'actual_unique_LR_fits':len(cache)-actual_gpu,
        'actual_unique_GPU_fits':actual_gpu,'cache_hits':hits,'independent_seed_clusters':3,'actual_GPU_devices':sorted(gpu_devices),
        'elapsed_seconds':time.perf_counter()-start,'new_method_candidates':0,'confirmation':False}
    verification={'passed':True,'parent_identity_verified':True,'source_hashes_match':all(sha(ROOT/k)==v for k,v in sources.items()),
        'all_actions_locked_before_training':True,'paired_non_O_state_identical':True,'same_nonempty_O_counts_labels':True,
        'same_greedy_solver_across_four_cells':True,'empty_O_geometry_error':max_zero_error,'empty_O_predictions_equal':True,
        'orthogonal_direction_null_control_passed':True,'saved_prediction_metric_max_error':max_recomputed_error,
        'actual_GPU_backend_verified':True,'approval_constrained_generator':False,'full_DOCX_confirmation':False}
    write_json(out/'completion.json',completion);write_json(out/'verification.json',verification)
    snap=out/'source_snapshot';snap.mkdir()
    for p in files:(snap/p.name).write_bytes(p.read_bytes())
    print(json.dumps(completion),flush=True)
    print(summary[summary.contrast.isin(['conditioning_static','conditioning_batch','interaction'])].to_string(index=False),flush=True)


if __name__=='__main__':main()
