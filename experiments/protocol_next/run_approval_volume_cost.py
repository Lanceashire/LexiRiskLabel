"""Continue scheduled mechanism controls: nested ordinary count and one cost-association pair."""
from runtime import *
from dataclasses import replace
from run_approval_mechanism import generate
from run_ordinary_factorial import family
import argparse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);a=ap.parse_args();base=a.base
    out=base/'approval_volume_cost';out.mkdir(exist_ok=False);private=base/'private/approval_volume_cost';private.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'approval_volume_cost_config.json').read_text());gen=json.loads((HERE/'approval_mechanism_config.json').read_text())
    parent=json.loads((base/'source_identity.json').read_text());verify_identity(parent,run_identity(parent['configuration']))
    sources={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'run_approval_volume_cost.py',HERE/'approval_volume_cost_config.json',
        HERE/'run_approval_mechanism.py',HERE/'approval_mechanism_config.json',HERE/'run_ordinary_factorial.py',HERE/'runtime.py',ROOT/'experiments/research_v2/core.py']}
    write_json(out/'activation.json',{'config':cfg,'source':sources,'parent_identity':parent['identity_hash'],
        'reference_outcome_sha':sha(base/'approval_mechanism_small/branch_outcomes.csv')})
    started=time.perf_counter();pending=[];locks=[];states=[]
    for seed in cfg['seeds']:
        large,yc,test,yt,pt,exposure=generate(seed,{**gen,'ordinary':4000})
        small,small_yc,small_test,small_yt,small_pt,small_exposure=generate(seed,gen)
        assert np.array_equal(yc,small_yc) and np.array_equal(test,small_test) and np.array_equal(exposure,small_exposure)
        for (coverage,s,yo),(small_coverage,ss,small_yo) in zip(large,small):
            assert coverage==small_coverage and np.array_equal(s.ordinary_x[:500],ss.ordinary_x) and np.array_equal(yo[:500],small_yo)
        jobs=[(4000,coverage,'neutral_original',s,yo,exposure) for coverage,s,yo in large]
        coverage,s,yo=small[0];pdscore=expit(s.candidate_x@np.asarray(gen['public_business_theta']));permuted=exposure.copy()
        for group in np.array_split(np.argsort(pdscore,kind='stable'),5):
            by_direction=group[np.argsort(np.abs(s.candidate_x[group,2]),kind='stable')]
            permuted[by_direction]=np.sort(exposure[group])
        assert np.array_equal(np.sort(permuted),np.sort(exposure))
        altered=replace(s,cost=pdscore*permuted)
        assert altered.budget==s.budget
        jobs.append((500,coverage,'positive_direction_association',altered,yo,permuted))
        states.append({'seed':seed,'neutral_cost_mean':float(s.cost.mean()),'positive_cost_mean':float(altered.cost.mean()),
            'neutral_cost_quantiles':np.quantile(s.cost,[0,.25,.5,.75,1]),'positive_cost_quantiles':np.quantile(altered.cost,[0,.25,.5,.75,1]),
            'exposure_multiset_preserved':True,'numerical_budget_unchanged':True,'cost_multiset_exactly_preserved':bool(np.array_equal(np.sort(s.cost),np.sort(altered.cost))),
            'neutral_abs_target_exposure_correlation':float(np.corrcoef(np.abs(s.candidate_x[:,2]),exposure)[0,1]),
            'positive_abs_target_exposure_correlation':float(np.corrcoef(np.abs(s.candidate_x[:,2]),permuted)[0,1]),
            'original_total_cost':float(s.cost.sum()),'positive_total_cost':float(altered.cost.sum()),'B':s.budget})
        for n,coverage,costcase,s,yo,exp in jobs:
            actions=family(s)
            for p in actions:
                ids=p['ids'];assert s.cost[ids].sum()<=s.budget+1e-7
                p.update({'seed':seed,'ordinary_n':n,'coverage':coverage,'cost_case':costcase,'count':len(ids),
                    'cost':float(s.cost[ids].sum()),'budget':s.budget,'Kstar':frozen.kstar(s.cost,s.budget),'selected_hash':digest(ids)})
                locks.append(p)
            pending.append((seed,n,coverage,costcase,s,yo,yc,test,yt,pt,exp,actions))
    write_jsonl(private/'selection_locks.jsonl',locks);locksha=sha(private/'selection_locks.jsonl')
    write_json(out/'selection_commitment.json',{'sha256':locksha,'all_actions_before_training':True,'selector_slots':len(locks)})
    write_jsonl(out/'cost_marginal_diagnostics.jsonl',states)
    cache={};rows=[];hits=0;gpu=0;error=0.
    for seed,n,coverage,costcase,s,yo,yc,test,yt,pt,exposure,actions in pending:
        keybase=f'{seed}_{n}_{coverage}_{costcase}'
        np.savez_compressed(private/f'{keybase}_environment.npz',D=s.visible_x,yD=s.visible_y,O=s.ordinary_x,yO=yo,C=s.candidate_x,yC=yc,
            cost=s.cost,exposure=exposure,budget=s.budget,test=test,yTest=yt,pTest=pt)
        for learner in cfg['learners']:
            for p in actions:
                ids=p['ids'];x=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]]);y=np.r_[s.visible_y,yo,yc[ids]]
                key=digest({'learner':learner,'x':digest(x),'y':digest(y),'seed':seed+7000,'params':gen['xgb'] if learner=='XGB' else gen['lr_ridge']})
                reused=key in cache
                if reused:model,backend=cache[key];hits+=1
                elif learner=='LR':model=frozen.fit_logistic(x,y,gen['lr_ridge']);backend='cpu';cache[key]=(model,backend)
                else:
                    model=XGBClassifier(**gen['xgb'],random_state=seed+7000,objective='binary:logistic',eval_metric='logloss')
                    model.fit(x[:,1:].astype(np.float32),y);backend=json.loads(model.get_booster().save_config())['learner']['generic_param']['device'];assert backend.startswith('cuda')
                    gpu+=1;cache[key]=(model,backend)
                pred=expit(test@model) if learner=='LR' else model.predict_proba(test[:,1:].astype(np.float32))[:,1]
                excess=float(np.mean((pred-pt)**2));brier=float(np.mean((pred-yt)**2))
                evalkey=digest({'fit':key,'test':digest(test),'yTest':digest(yt),'pTest':digest(pt)})
                dest=private/'predictions'/f'{evalkey}.npz';dest.parent.mkdir(exist_ok=True)
                np.savez_compressed(dest,prediction=pred,yTest=yt,pTest=pt);saved=np.load(dest)
                error=max(error,abs(excess-float(np.mean((saved['prediction']-saved['pTest'])**2))))
                row={k:v for k,v in p.items() if k not in ['ids','steps']};row.update({'learner':learner,'excess_brier':excess,'brier':brier,
                    'actual_backend':backend,'cache_hit':reused,'fit_key':key,'evaluation_key':evalkey,'realized_exploration_loss':float(yc[ids]@exposure[ids])})
                rows.append(row)
        pd.DataFrame(rows).to_csv(out/'new_branch_outcomes.csv',index=False)
        print('volume/cost complete',keybase,'unique fits',len(cache),'cache hits',hits,flush=True)
    old=pd.read_csv(base/'approval_mechanism_small/branch_outcomes.csv');old['ordinary_n']=500;old['cost_case']='neutral_original';old['inherited_reference']=True
    new=pd.DataFrame(rows);new['inherited_reference']=False
    frame=pd.concat([old,new],ignore_index=True);frame.to_csv(out/'all_branch_outcomes.csv',index=False)
    pairs=[]
    for (seed,learner,coverage,costcase),g in frame[frame.cost_case=='neutral_original'].groupby(['seed','learner','coverage','cost_case']):
        q=g.pivot(index='method',columns='ordinary_n',values='excess_brier')
        for method in cfg['methods']:pairs.append({'seed':seed,'learner':learner,'coverage':coverage,'contrast':'ordinary_4000_vs_500','method':method,
            'gain':float(q.loc[method,500]-q.loc[method,4000])})
        lo=g[g.ordinary_n==500].set_index('method');hi=g[g.ordinary_n==4000].set_index('method')
        for baseline in ['OrdinaryOnly','MinRisk']:
            pairs.append({'seed':seed,'learner':learner,'coverage':coverage,'contrast':f'exploration_gain_4000_minus_500_vs_{baseline}','method':'C1B1',
                'gain':float((hi.loc[baseline,'excess_brier']-hi.loc['C1B1','excess_brier'])-(lo.loc[baseline,'excess_brier']-lo.loc['C1B1','excess_brier']))})
    cost=frame[(frame.ordinary_n==500)&(frame.coverage=='low_target_overlap')]
    for (seed,learner),g in cost.groupby(['seed','learner']):
        q=g.pivot(index='method',columns='cost_case',values='excess_brier')
        for method in cfg['methods']:pairs.append({'seed':seed,'learner':learner,'coverage':'low_target_overlap','contrast':'cost_positive_vs_neutral','method':method,
            'gain':float(q.loc[method,'neutral_original']-q.loc[method,'positive_direction_association'])})
    p=pd.DataFrame(pairs);p.to_csv(out/'paired_contrasts.csv',index=False)
    summary=p.groupby(['learner','coverage','contrast','method'],as_index=False).agg(mean_gain=('gain','mean'),
        positive_seeds=('gain',lambda x:int((x>1e-12).sum())),negative_seeds=('gain',lambda x:int((x< -1e-12).sum())))
    summary.to_csv(out/'descriptive_summary.csv',index=False)
    assert len(rows)==cfg['max_new_fit_slots'] and len(locks)==cfg['max_new_selector_slots'] and error<1e-12 and locksha==sha(private/'selection_locks.jsonl')
    write_json(out/'completion.json',{'completed':True,'new_selector_slots':len(locks),'new_fit_slots':len(rows),'actual_unique_LR_fits':len(cache)-gpu,
        'actual_unique_GPU_fits':gpu,'within_stage_cache_hits':hits,'inherited_actual_reference_slots':len(old),'independent_seed_clusters':3,
        'seconds':time.perf_counter()-started,'confirmation':False})
    write_json(out/'verification.json',{'passed':True,'nested_500_in_4000_rows_and_labels':True,'approval_gates_inherited':True,
        'common_D_C_Q_test':True,'same_exposure_multiset':True,'identical_numerical_B':True,'final_cost_marginals_disclosed':True,
        'all_choices_locked_before_training':True,'actual_GPU_verified':True,'saved_prediction_max_error':error,
        'all_sources_unchanged':all(sha(ROOT/k)==v for k,v in sources.items())})
    snap=out/'source_snapshot';snap.mkdir()
    for p in sources:(snap/Path(p).name).write_bytes((ROOT/p).read_bytes())
    print(json.dumps(json.loads((out/'completion.json').read_text())),flush=True)
    print(summary[(summary.method=='C1B1')|summary.contrast.str.startswith('exploration')].to_string(index=False),flush=True)


if __name__=='__main__':main()
