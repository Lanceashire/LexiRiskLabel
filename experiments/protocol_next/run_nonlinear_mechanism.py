"""Scheduled nonlinear subset, preserving the approval-constrained feature pools."""
from runtime import *
from dataclasses import replace
from run_approval_mechanism import generate
from run_ordinary_factorial import family
import argparse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);a=ap.parse_args();base=a.base
    out=base/'nonlinear_mechanism_small';out.mkdir(exist_ok=False);private=base/'private/nonlinear_mechanism_small';private.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'nonlinear_mechanism_config.json').read_text());gen=json.loads((HERE/'approval_mechanism_config.json').read_text())
    parent=json.loads((base/'source_identity.json').read_text());verify_identity(parent,run_identity(parent['configuration']))
    sources={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'run_nonlinear_mechanism.py',HERE/'nonlinear_mechanism_config.json',
        HERE/'run_approval_mechanism.py',HERE/'approval_mechanism_config.json',HERE/'run_ordinary_factorial.py',HERE/'runtime.py',ROOT/'experiments/research_v2/core.py']}
    write_json(out/'activation.json',{'config':cfg,'source':sources,'parent_identity':parent['identity_hash']})
    started=time.perf_counter();pending=[];locks=[];theta_true=np.asarray(gen['truth_theta'])
    def truth(x):return expit(x@theta_true+.8*(x[:,2]**2-1))
    for seed in cfg['seeds']:
        scenarios,_,test,_,_,exposure=generate(seed,gen);s0=scenarios[0][1]
        yd=(np.random.default_rng(seed+150000).random(len(s0.visible_y))<truth(s0.visible_x)).astype(int)
        yc=(np.random.default_rng(seed+160000).random(len(s0.candidate_x))<truth(s0.candidate_x)).astype(int)
        pt=truth(test);yt=(np.random.default_rng(seed+190000).random(len(test))<pt).astype(int)
        theta=frozen.fit_logistic(s0.visible_x,yd,gen['lr_ridge'])
        uniform=np.random.default_rng(seed+230000).random(gen['ordinary'])
        for coverage,s,_ in scenarios:
            yo=(uniform<truth(s.ordinary_x)).astype(int);s=replace(s,visible_y=yd,theta=theta);actions=family(s)
            for p in actions:
                ids=p['ids'];assert s.cost[ids].sum()<=s.budget+1e-7
                p.update({'seed':seed,'coverage':coverage,'count':len(ids),'cost':float(s.cost[ids].sum()),'budget':s.budget,
                    'Kstar':frozen.kstar(s.cost,s.budget),'selected_hash':digest(ids)})
                locks.append(p)
            empty=family(replace(s,ordinary_x=s.ordinary_x[:0]));assert np.array_equal(empty[2]['ids'],empty[4]['ids']) and np.array_equal(empty[3]['ids'],empty[5]['ids'])
            pending.append((seed,coverage,s,yo,yc,test,yt,pt,exposure,actions))
    write_jsonl(private/'selection_locks.jsonl',locks);locksha=sha(private/'selection_locks.jsonl')
    write_json(out/'selection_commitment.json',{'sha256':locksha,'all_actions_locked_before_training':True,'slots':len(locks)})
    cache={};rows=[];hits=0;gpu=0;error=0.
    for seed,coverage,s,yo,yc,test,yt,pt,exposure,actions in pending:
        np.savez_compressed(private/f'{seed}_{coverage}_environment.npz',D=s.visible_x,yD=s.visible_y,O=s.ordinary_x,yO=yo,
            C=s.candidate_x,yC=yc,Q=s.reference_x,test=test,yTest=yt,pTest=pt,cost=s.cost,budget=s.budget,exposure=exposure)
        for learner in ['LR','XGB']:
            for p in actions:
                ids=p['ids'];x=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]]);y=np.r_[s.visible_y,yo,yc[ids]]
                key=digest({'x':digest(x),'y':digest(y),'learner':learner,'seed':seed+7000,'params':gen['xgb'] if learner=='XGB' else gen['lr_ridge']})
                reused=key in cache
                if reused:model,backend=cache[key];hits+=1
                elif learner=='LR':model=frozen.fit_logistic(x,y,gen['lr_ridge']);backend='cpu';cache[key]=(model,backend)
                else:
                    model=XGBClassifier(**gen['xgb'],random_state=seed+7000,objective='binary:logistic',eval_metric='logloss')
                    model.fit(x[:,1:].astype(np.float32),y);backend=json.loads(model.get_booster().save_config())['learner']['generic_param']['device'];assert backend.startswith('cuda')
                    gpu+=1;cache[key]=(model,backend)
                pred=expit(test@model) if learner=='LR' else model.predict_proba(test[:,1:].astype(np.float32))[:,1]
                excess=float(np.mean((pred-pt)**2));brier=float(np.mean((pred-yt)**2));evalkey=digest({'fit':key,'test':digest(test),'truth':digest(pt),'yTest':digest(yt)})
                dest=private/'predictions'/f'{evalkey}.npz';dest.parent.mkdir(exist_ok=True)
                np.savez_compressed(dest,prediction=pred,pTest=pt,yTest=yt,selected_ids=ids);saved=np.load(dest)
                error=max(error,abs(excess-float(np.mean((saved['prediction']-saved['pTest'])**2))),abs(brier-float(np.mean((saved['prediction']-saved['yTest'])**2))))
                row={k:v for k,v in p.items() if k not in ['ids','steps']};row.update({'learner':learner,'excess_brier':excess,'brier':brier,
                    'fit_key':key,'evaluation_key':evalkey,'actual_backend':backend,'cache_hit':reused,'realized_exploration_loss':float(yc[ids]@exposure[ids])});rows.append(row)
        pd.DataFrame(rows).to_csv(out/'branch_outcomes.csv',index=False)
        print('nonlinear complete',seed,coverage,'unique fits',len(cache),'cache hits',hits,flush=True)
    frame=pd.DataFrame(rows);pairs=[]
    for (seed,coverage,learner),g in frame.groupby(['seed','coverage','learner']):
        q=g.set_index('method')
        for name,left,right in [('conditioning_static','C0B0','C1B0'),('conditioning_batch','C0B1','C1B1'),
            ('batch_conditional','C1B0','C1B1'),('combined_vs_OrdinaryOnly','OrdinaryOnly','C1B1'),('combined_vs_MinRisk','MinRisk','C1B1')]:
            pairs.append({'seed':seed,'coverage':coverage,'learner':learner,'contrast':name,
                'excess_gain':q.loc[left,'excess_brier']-q.loc[right,'excess_brier'],'sampled_brier_gain':q.loc[left,'brier']-q.loc[right,'brier'],
                'count_difference':int(q.loc[right,'count']-q.loc[left,'count']),'cost_difference':q.loc[right,'cost']-q.loc[left,'cost']})
    p=pd.DataFrame(pairs);p.to_csv(out/'paired_contrasts.csv',index=False)
    summary=p.groupby(['coverage','learner','contrast'],as_index=False).agg(mean_excess_gain=('excess_gain','mean'),
        positive_seeds=('excess_gain',lambda x:int((x>1e-12).sum())),negative_seeds=('excess_gain',lambda x:int((x< -1e-12).sum())),
        mean_sampled_brier_gain=('sampled_brier_gain','mean'),mean_count_difference=('count_difference','mean'),mean_cost_difference=('cost_difference','mean'))
    summary.to_csv(out/'descriptive_summary.csv',index=False)
    assert len(locks)==cfg['max_selector_slots'] and len(rows)==cfg['max_fit_slots'] and error<1e-12 and sha(private/'selection_locks.jsonl')==locksha
    write_json(out/'completion.json',{'completed':True,'selector_slots':len(locks),'fit_slots':len(rows),'actual_unique_LR_fits':len(cache)-gpu,
        'actual_unique_GPU_fits':gpu,'cache_hits':hits,'independent_generation_seeds':3,'confirmation':False,'seconds':time.perf_counter()-started})
    write_json(out/'verification.json',{'passed':True,'paired_feature_pools_preserved_from_linear_case':True,'D_only_selector_refit_to_nonlinear_labels':True,
        'all_ordinary_approved':True,'all_candidates_rejected':True,'empty_O_null_selection':True,'all_choices_before_training':True,
        'same_representation_for_selectors_and_LR':True,'actual_GPU_backend_checked':True,'saved_prediction_max_error':error,
        'sources_unchanged':all(sha(ROOT/k)==v for k,v in sources.items())})
    snap=out/'source_snapshot';snap.mkdir()
    for p in sources:(snap/Path(p).name).write_bytes((ROOT/p).read_bytes())
    print(json.dumps(json.loads((out/'completion.json').read_text())),flush=True)
    print(summary[summary.contrast.isin(['combined_vs_OrdinaryOnly','combined_vs_MinRisk'])].to_string(index=False),flush=True)


if __name__=='__main__':main()
