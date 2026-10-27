"""ParBaLS EPIG, explicitly local Laplace/cost/pending adaptation, six fixed real states."""
from runtime import *
from run_e1b import validation_blocks
import argparse

METHOD='ParBaLS-EPIG-Laplace-cost-pending'


def select_neighbor(s,cfg):
    started=time.perf_counter();rng=np.random.default_rng(s.seed+470)
    hd,_,_=frozen.geometry(s,False);draws=rng.multivariate_normal(s.theta,frozen.inverse(hd),size=cfg['posterior_draws'])
    world=draws[rng.choice(len(draws),cfg['universes'],replace=False)]
    # This hard pseudo-label convention follows current upstream source; not a Bernoulli label sampler claim.
    yo=(world@s.ordinary_x.T>=0).astype(int);yc=(world@s.candidate_x.T>=0).astype(int)
    chosen=[];free=np.ones(len(s.cost),bool);steps=[];refits=0
    while True:
        candidates=np.flatnonzero(free&(s.cost<=s.budget-s.cost[chosen].sum()+1e-10))
        if not len(candidates):break
        score=np.zeros(len(candidates))
        for w in range(cfg['universes']):
            x=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[chosen]])
            y=np.r_[s.visible_y,yo[w],yc[w,chosen]]
            theta=frozen.fit_logistic(x,y,s.ridge);v=frozen.info(x,theta)
            post=rng.multivariate_normal(theta,frozen.inverse(s.ridge*np.eye(x.shape[1])+v.T@v),size=cfg['posterior_draws'])
            score+=frozen.epig_scores(expit(post@s.candidate_x[candidates].T),expit(post@s.reference_x.T))/cfg['universes']
            refits+=1
        ratios=score/np.maximum(s.cost[candidates],1e-15);j=int(np.argmax(ratios));i=int(candidates[j])
        chosen.append(i);free[i]=False;steps.append({'candidate':i,'full_EPIG':float(score[j]),'risk_cost':float(s.cost[i]),'ratio':float(ratios[j])})
    ids=np.sort(np.asarray(chosen,int));assert s.cost[ids].sum()<=s.budget+1e-7
    return ids,{'internal_LR_refits':refits,'selection_seconds':time.perf_counter()-started,'steps':steps,
        'universe_parameters':world,'fantasy_ordinary_labels':yo,'fantasy_candidate_labels':yc}


def criterion_check():
    rng=np.random.default_rng(64179);pc=rng.uniform(.02,.98,(32,7));pq=rng.uniform(.02,.98,(32,9))
    score=frozen.epig_scores(pc,pq);direct=[];upstream=[]
    for i in range(pc.shape[1]):
        val=[];raw=0.
        for j in range(pq.shape[1]):
            a,b=pc[:,i].mean(),pq[:,j].mean();both=np.mean(pc[:,i]*pq[:,j]);joint=np.array([[1-a-b+both,b-both],[a-both,both]])
            expected=np.outer([1-a,a],[1-b,b]);val.append(float(np.sum(joint*np.log(joint/expected))))
            raw+=float(frozen.entropy(np.asarray(a))+np.sum(joint*np.log(joint)))
        direct.append(np.mean(val));upstream.append(raw)
    error=float(np.max(np.abs(score-direct)))
    # In equal costs, restoring the target entropy constant preserves original source ranking.
    offset=score-np.asarray(upstream)/len(pq.T);constant_error=float(np.ptp(offset))
    assert error<1e-12 and constant_error<1e-12 and np.argmax(score)==np.argmax(upstream)
    return {'direct_mutual_information_max_error':error,'equal_cost_constant_range':constant_error,
            'equal_cost_upstream_argmax_matches':True,'heterogeneous_cost_uses_full_nonnegative_MI':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base
    out=base/'nearest_neighbor_small';out.mkdir(exist_ok=False);private=base/'private/nearest_neighbor_small';private.mkdir(parents=True,exist_ok=True)
    parent=json.loads((base/'source_identity.json').read_text());config=parent['configuration'];verify_identity(parent,run_identity(config))
    stage=json.loads((HERE/'nearest_neighbor_config.json').read_text());source={str(p.relative_to(ROOT)):sha(p) for p in
        [HERE/'run_nearest_neighbor.py',HERE/'nearest_neighbor_config.json',HERE/'runtime.py',HERE/'run_e1b.py',ROOT/'experiments/research_v2/core.py']}
    write_json(out/'activation.json',{'config':stage,'source':source,'parent_identity':parent['identity_hash'],'criterion_check':criterion_check()})
    pending=[];locks=[];start=time.perf_counter();internal=0
    for dataset in config['datasets']:
        for seed in config['seeds']:
            c=build_context(dataset,seed,stage['round'],config,private);s=c.acquisition
            ids,diag=select_neighbor(s,stage);token=c.feedback.lock(ids,METHOD,s.seed);key=digest(np.sort(c.feedback.c_ids[ids]));internal+=diag['internal_LR_refits']
            np.savez_compressed(private/f'{c.state_id}_fantasies.npz',theta=diag['universe_parameters'],ordinary=diag['fantasy_ordinary_labels'],candidate=diag['fantasy_candidate_labels'])
            lock={'state_id':c.state_id,'dataset':dataset,'seed':seed,'round':c.round,'method':METHOD,'selected_hash':key,
                'count':len(ids),'cost':float(s.cost[ids].sum()),'budget':s.budget,'candidate_count':len(s.cost),'reference_count':len(s.reference_x),
                'selection_seconds':diag['selection_seconds'],'internal_LR_refits':diag['internal_LR_refits'],'all_O_C_labels_unknown':True}
            locks.append(lock);write_json(private/f'{c.state_id}_selection.json',{'ids':ids,'steps':diag['steps'],'commitment':lock})
            pending.append((c,ids,token,lock))
            print('nearest action locked',c.state_id,'count',len(ids),'internal LR',diag['internal_LR_refits'],flush=True)
    write_jsonl(out/'locked_actions.jsonl',locks);locks_hash=sha(out/'locked_actions.jsonl')
    # Evaluation files are first loaded only after every state has a locked new action.
    oldV=pd.read_json(base/'e1b/all_set_outcomes.jsonl',lines=True);oldT=pd.read_json(base/'branch_outcomes.jsonl',lines=True)
    outcomes=[];newfits=0;gpu=0;reused=0;error=0.
    for c,selected,token,lock in pending:
        for learner in config['learners']:
            vr=oldV[(oldV.state_id==c.state_id)&(oldV.learner==learner)]
            tr=oldT[(oldT.state_id==c.state_id)&(oldT.learner==learner)]
            for method in stage['comparison_methods'][:-1]:
                v=vr[vr.sources.map(lambda x:method in x)];t=tr[tr.selector==method]
                assert len(v)==1 and len(t)==1
                v=v.iloc[0];t=t.iloc[0]
                assert v.selected_hash==t.selected_hash
                outcomes.append({'state_id':c.state_id,'dataset':c.dataset,'seed':c.seed,'learner':learner,'method':method,
                    'selected_hash':v.selected_hash,'count':int(v['count']),'cost':float(v.cost),'budget':float(v.budget),
                    'V_eval_brier':float(v.V_eval_brier),'T_brier':float(t.brier),'actual_backend':v.actual_backend,'reused':True})
                reused+=1
            match=vr[vr.selected_hash==lock['selected_hash']]
            if len(match):
                v=match.iloc[0];t=tr[tr.selected_hash==lock['selected_hash']];assert len(t)>0
                vb=float(v.V_eval_brier);tb=float(t.brier.iloc[0]);backend=v.actual_backend;reused+=1;isreuse=True
            else:
                train_ids,x,y=c.feedback.reveal(selected,token)
                _,eval_ids,_,evx,_,evy=validation_blocks(c,config)
                if learner=='LR':model=frozen.fit_logistic(x,y,config['lr_ridge']);backend='cpu';vp=expit(evx@model)
                else:
                    model=XGBClassifier(**config['xgb'],random_state=c.seed*1000+c.round,objective='binary:logistic',eval_metric='logloss')
                    model.fit(x[:,1:].astype(np.float32),y);backend=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
                    assert backend.startswith('cuda');gpu+=1;vp=model.predict_proba(evx[:,1:].astype(np.float32))[:,1]
                tp=c.evaluation.predict(model,learner);vb=float(np.mean((vp-evy)**2));tb=c.evaluation.evaluate(tp)['brier'];newfits+=1;isreuse=False
                path=private/f'{c.state_id}_{learner}_predictions.npz'
                np.savez_compressed(path,train_ids=train_ids,train_y=y,selected_global_ids=np.sort(c.feedback.c_ids[selected]),eval_ids=eval_ids,
                    eval_prediction=vp,eval_y=evy,T_prediction=tp,T_y=c.evaluation._y)
                saved=np.load(path);error=max(error,abs(vb-float(np.mean((saved['eval_prediction']-saved['eval_y'])**2))),
                    abs(tb-float(np.mean((saved['T_prediction']-saved['T_y'])**2))))
            outcomes.append({**lock,'learner':learner,'V_eval_brier':vb,'T_brier':tb,'actual_backend':backend,'reused':isreuse})
        print('nearest evaluated',c.state_id,'new actual fits',newfits,flush=True)
    frame=pd.DataFrame(outcomes);frame.to_csv(out/'branch_outcomes.csv',index=False);contrasts=[]
    for (dataset,seed,learner),g in frame.groupby(['dataset','seed','learner']):
        q=g.set_index('method')
        for name,left,right in [('neighbor_vs_MinRisk','MinRisk',METHOD),('neighbor_vs_OrdinaryOnly','OrdinaryOnly',METHOD),
            ('Vopt_vs_neighbor',METHOD,'Vopt'),('BRIDGE_vs_neighbor',METHOD,'RB-BRIDGE')]:
            contrasts.append({'dataset':dataset,'seed':seed,'learner':learner,'contrast':name,
                'V_eval_gain':q.loc[left,'V_eval_brier']-q.loc[right,'V_eval_brier'],'T_gain':q.loc[left,'T_brier']-q.loc[right,'T_brier'],
                'count_difference':int(q.loc[right,'count']-q.loc[left,'count']),'cost_difference':q.loc[right,'cost']-q.loc[left,'cost'],
                'same_action':q.loc[right,'selected_hash']==q.loc[left,'selected_hash']})
    p=pd.DataFrame(contrasts);p.to_csv(out/'paired_contrasts.csv',index=False)
    summary=p.groupby(['dataset','learner','contrast'],as_index=False).agg(V_eval_gain=('V_eval_gain','mean'),T_gain=('T_gain','mean'),
        positive_seeds=('V_eval_gain',lambda x:int((x>1e-12).sum())),negative_seeds=('V_eval_gain',lambda x:int((x< -1e-12).sum())),
        mean_count_difference=('count_difference','mean'),mean_cost_difference=('cost_difference','mean'))
    summary.to_csv(out/'descriptive_summary.csv',index=False)
    assert newfits<=stage['new_actual_fit_slots_upper_bound'] and sha(out/'locked_actions.jsonl')==locks_hash and error<1e-12
    write_json(out/'completion.json',{'completed':True,'states':6,'new_unique_LR_fits':newfits-gpu,'new_unique_GPU_fits':gpu,
        'reused_actual_reference_slots':reused,'internal_hypothetical_LR_refits':internal,'seconds':time.perf_counter()-start,
        'independent_data_seed_values_per_task':2,'confirmation':False})
    write_json(out/'verification.json',{'passed':True,'parent_identity':True,'all_six_actions_locked_before_evaluation':True,
        'candidate_and_reference_not_subsampled':True,'same_pending_information_access':True,'same_frozen_training_and_preprocessing':True,
        'saved_prediction_error':error,'actual_GPU_backend_checked':True,'V_eval_development_reused':True,
        'upstream_exact_MCMC_reproduction':False,'criterion':criterion_check(),'sources_unchanged':all(sha(ROOT/k)==h for k,h in source.items())})
    snap=out/'source_snapshot';snap.mkdir()
    for p in source:(snap/Path(p).name).write_bytes((ROOT/p).read_bytes())
    print(json.dumps(json.loads((out/'completion.json').read_text())),flush=True)
    print(summary[summary.learner=='XGB'].to_string(index=False),flush=True)


if __name__=='__main__':main()
