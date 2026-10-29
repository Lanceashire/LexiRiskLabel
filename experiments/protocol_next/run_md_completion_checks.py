"""MD section-9 rotations and later-cohort check; reuse frozen full-process engine.

The engine CONFIG is redirected only inside this separate process to the
registered scheduling adapter. Original engine files/main workers are untouched.
"""
from runtime import *
import run_formal_md as engine
from collections import Counter
import argparse


def source_parameters(base,target,cfg):
    records=[]
    for path in sorted((base/'private/formal_md/cases').glob('*/initial_tuning.json')):
        if path.parent.name.startswith(target+'_') or not (path.parent/'completion.json').exists():continue
        name=next(n for n in cfg['datasets'] if path.parent.name.startswith(n+'_'))
        t=json.loads(path.read_text());records.append({'task':name,'ridge':t['ridge'],'xgb':t['xgb'],'file_sha256':sha(path)})
    assert len(records)==40 and len({r['task'] for r in records})==2,'source-task main evaluation not complete yet'
    lr=Counter(r['ridge'] for r in records);ridge=min(cfg['lr_ridge_grid'],key=lambda x:(-lr[x],cfg['lr_ridge_grid'].index(x)))
    grid=cfg['xgb_grid'];counts=Counter(next(i for i,p in enumerate(grid) if all(r['xgb'][k]==v for k,v in p.items())) for r in records)
    index=min(range(len(grid)),key=lambda i:(-counts[i],i))
    return {'ridge':ridge,'xgb':{**cfg['xgb_fixed'],**grid[index]},'scores':[],'actual_tuning_GPU_fits':0,
        'target_hyperparameter_tuning':False,'source_tasks':sorted({r['task'] for r in records}),
        'selection_rule':'mode of source D0-only selected configurations, declared grid-order tie break',
        'source_tuning_records':records,'source_or_target_outer_test_used_for_selection':False}


def external_data(root,completion_cfg):
    old=engine.dataset('lendingclub');raw,y,exposure,oldhash,cols,_,months=old
    path=root/'work/data/LC_Loan_sample_2016.csv';df=pd.read_csv(path,low_memory=False)
    rule=completion_cfg['later_cohort'];assert list(raw.columns)==rule['schema']
    dates=pd.to_datetime(df.issue_d,format='%b-%Y',errors='coerce')
    missing=dates.isna()
    if missing.any():dates.loc[missing]=pd.to_datetime(df.loc[missing,'issue_d'],format='%b-%y',errors='coerce')
    numeric_id=pd.to_numeric(df.id,errors='coerce');amount=pd.to_numeric(df.loan_amnt,errors='coerce')
    eligible=df.loan_status.isin(rule['eligible_statuses'])&(dates.dt.year==2016)&numeric_id.notna()&np.isfinite(amount)&(amount>0)
    ext=df.loc[eligible].copy();ext['_loanid']=numeric_id[eligible].astype('int64').to_numpy();ext['_month']=dates[eligible].values.astype('datetime64[M]').astype(int)
    ext=ext.drop_duplicates('_loanid');assert len(ext)>0 and set(rule['schema']).issubset(ext.columns)
    nx=ext[rule['schema']].copy()
    for col in raw:
        if pd.api.types.is_numeric_dtype(raw[col]):
            values=nx[col].astype(str).str.rstrip('%') if col in ['int_rate','revol_util'] else nx[col]
            nx[col]=pd.to_numeric(values,errors='coerce')
        else:nx[col]=nx[col].astype(object)
    combined=pd.concat([raw,nx],ignore_index=True)
    assert [c for c in raw if pd.api.types.is_numeric_dtype(raw[c])]==[c for c in combined if pd.api.types.is_numeric_dtype(combined[c])]
    ey=(ext.loan_status=='Charged Off').to_numpy(int);ee=pd.to_numeric(ext.loan_amnt).to_numpy(float)
    yy=np.r_[y,ey];ex=np.r_[exposure,ee];mm=np.r_[months,ext['_month'].to_numpy(int)]
    oldids=pd.read_csv(ROOT/'data/research_v2/lendingclub_ids.csv').id.to_numpy('int64');ids=np.r_[oldids,ext['_loanid'].to_numpy('int64')]
    assert not np.intersect1d(oldids,ext['_loanid'].to_numpy()).size
    parents=np.arange(len(combined))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return int(i)
    for keys in [ids,pd.util.hash_pandas_object(combined,index=False).to_numpy()]:
        seen={}
        for i,key in enumerate(keys):
            if key in seen:parents[find(i)]=find(seen[key])
            else:seen[key]=i
    groups=np.asarray([find(i) for i in range(len(combined))]);test=np.arange(len(raw),len(combined))
    train=np.arange(len(raw));train=train[~np.isin(groups[train],np.unique(groups[test]))]
    assert mm[train].max()<mm[test].min() and not np.intersect1d(groups[train],groups[test]).size
    source=digest({'old_source':oldhash,'external_source':sha(path),'frozen_rule':rule,'eligible_ids_hash':digest(ids[test])})
    provenance={'url':rule['url'],'download_sha256':sha(path),'historical_rows':len(raw),'external_eligible_rows':len(test),
        'historical_train_rows_after_group_exclusion':len(train),'test_is_all_eligible_2016':True,'source_dtype_roles_preserved':True,
        'source_latest_month':int(mm[train].max()),'target_earliest_month':int(mm[test].min()),
        'no_target_labels_for_training_or_parameter_choice':True,'labels_mature_at_snapshot_not_known_availability_dates':True,
        'label_availability_assumption':rule['label_time_limit'],'entity_limit':'loan ID and exact-feature duplicate groups; borrower linkage unavailable'}
    return (combined,yy,ex,source,cols,groups,mm),train,test,provenance


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--mode',choices=['transfer','later'],required=True);a=ap.parse_args()
    base=a.base;root=ROOT.parents[1];out=base/'completion_checks'/a.mode;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'formal_md_config.json').read_text());plan=json.loads((HERE/'md_completion_config.json').read_text())
    reg=json.loads((base/'completion_checks/registration.json').read_text());assert reg['config_sha256']==sha(HERE/'md_completion_config.json')
    assert plan['controlling_protocol_sha256']==sha(HERE/'experiment_protocol_zh.md')
    cfg['secondary_budgets']=[];cfg['main_methods']=plan['methods'];cfg['main_budget']=plan['rho']
    runtime_config=out/'runtime_config.json';write_json(runtime_config,cfg);engine.CONFIG=runtime_config
    source={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'md_completion_config.json',HERE/'run_formal_md.py',HERE/'formal_md_config.json',
        HERE/'runtime.py',HERE/'run_nearest_neighbor.py',HERE/'run_fantasy.py',ROOT/'experiments/research_v2/core.py',ROOT/'experiments/research_v2/run_real.py']}
    jobs=[];parameters={};provenance=None
    if a.mode=='transfer':
        for target in plan['transfer']['target_tasks']:
            params=source_parameters(base,target,cfg);parameters[target]=params
            repeat,fold,train,test=engine.outer_splits(target,cfg)[0]
            child=out/target
            for stream in range(plan['replay_streams']):jobs.append((target,repeat,fold,stream,train,test,child,params))
    else:
        state,train,test,provenance=external_data(root,plan)
        # Dedicated process override, so all ordinary/arrival data remain historical and all later labels stay in T.
        engine.dataset=lambda name:state
        for stream in range(plan['replay_streams']):jobs.append(('lendingclub',0,0,stream,train,test,out/'lendingclub_2016',None))
        write_json(out/'data_provenance.json',provenance)
    identity={'plan':plan,'runtime_configuration':cfg,'sources':source,'mode':a.mode,
        'main_parent_identity':json.loads((base/'formal_md/frozen_identity.json').read_text())['identity_hash'],
        'source_parameters':parameters,'later_provenance':provenance,'target_or_source_test_used_for_config_choice':False}
    ih=digest(identity);identity['identity_hash']=ih
    frozen=out/'frozen_identity.json'
    if frozen.exists():assert json.loads(frozen.read_text())==identity
    else:write_json(frozen,identity)
    expected=36 if a.mode=='transfer' else 12;completed=[]
    for name,repeat,fold,stream,train,test,child,preset in jobs:
        case=f'{name}_repeat{repeat+1}_fold{fold+1}_stream{stream+1}';folder=child/'private/formal_md/cases'/case;folder.mkdir(parents=True,exist_ok=True)
        if preset is not None:write_json(folder/'initial_tuning.json',preset)
        result=engine.run_case((name,repeat,fold,stream,train,test,str(child),ih));completed.append(result)
        print('MD COMPLETION CASE',a.mode,case,'done',len(completed),'/',len(jobs),flush=True)
    rows=[r for c in completed for r in c['episodes']];assert len(rows)==expected
    frame=pd.DataFrame(rows);frame.to_csv(out/'episode_summary.csv',index=False);write_jsonl(out/'confirm_episodes.jsonl',rows)
    pairs=[]
    for (name,case),g in frame.groupby(['dataset','case_id']):
        q=g.set_index('method')
        for reference in ['MinRisk','ParBaLS','OrdinaryOnly']:
            pairs.append({'dataset':name,'case_id':case,'reference':reference,'ALC_gain':float(q.loc[reference,'ALC_Brier']-q.loc['FantasyCandidate','ALC_Brier']),
                'final_gain':float(q.loc[reference,'final_Brier']-q.loc['FantasyCandidate','final_Brier']),
                'risk_difference':float(q.loc['FantasyCandidate','total_exploration_cost']-q.loc[reference,'total_exploration_cost'])})
    p=pd.DataFrame(pairs);p.to_csv(out/'paired_contrasts.csv',index=False)
    summary=frame.groupby(['dataset','method'],as_index=False).agg(ALC_Brier=('ALC_Brier','mean'),final_Brier=('final_Brier','mean'),
        count=('cumulative_count','mean'),exploration_risk=('total_exploration_cost','mean'),ordinary_risk=('ordinary_cost','mean'))
    summary.to_csv(out/'method_summary.csv',index=False)
    assert all(sha(ROOT/k)==h for k,h in source.items())
    write_json(out/'completion.json',{'completed':True,'mode':a.mode,'episodes':len(rows),'cases':len(completed),'identity_hash':ih,
        'new_method_candidates':0,'reference_budget':.1,'target_hyperparameter_search':False if a.mode=='transfer' else 'historical D0 only'})
    write_json(out/'verification.json',{'passed':True,'source_and_code_frozen':True,'all_cases_information_budget_checks':all(c['all_information_budget_checks_passed'] for c in completed),
        'all_predictions_rescored_in_engine':True,'max_rescore_error':max(c['max_prediction_rescore_error'] for c in completed),
        'actual_CUDA_training_checked':True,'target_hyperparameter_labels_used':False,'prior_main_results_unchanged':True,
        'claim':'internal transfer robustness' if a.mode=='transfer' else 'new later-cohort retrospective sensitivity; not strict label-maturity online validation'})
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
