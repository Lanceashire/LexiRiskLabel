"""MD sections 9/10: entire acquisition process inside each outer split.

One immutable configuration; --limit-cases is scheduling only, never a changed
experiment. Test features/labels are not arguments to any acquisition routine.
"""
from runtime import *
from run_nearest_neighbor import select_neighbor
from run_fantasy import propose
from sklearn.model_selection import StratifiedGroupKFold
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import argparse,traceback

CONFIG=HERE/'formal_md_config.json'


def gpu_fit(x,y,params,seed):
    if len(np.unique(y))<2:raise RuntimeError('registered fit has only one observed class')
    m=XGBClassifier(**params,random_state=int(seed));m.fit(np.asarray(x,np.float32),y)
    actual=json.loads(m.get_booster().save_config())['learner']['generic_param']['device']
    if not actual.startswith('cuda'):raise RuntimeError('required actual CUDA backend: '+actual)
    return m


@lru_cache(maxsize=3)
def dataset(name):
    raw,y,exposure,source,columns=legacy.load(name);n=len(raw)
    full=pd.read_csv(ROOT/'data/research_v2'/f'{name}.csv',low_memory=False)
    parent=np.arange(n);ids=full['ID'].astype(str).to_numpy() if name=='uci' else None;months=None
    if name=='lendingclub':
        side=pd.read_csv(ROOT/'data/research_v2/lendingclub_ids.csv');assert len(side)==n
        ids=side.id.astype(str).to_numpy();months=pd.to_datetime(side.issue_d,format='%b-%Y').values.astype('datetime64[M]').astype(int)
        assert np.array_equal(pd.to_datetime(side.issue_d,format='%b-%Y').dt.year.to_numpy(),full.year.to_numpy())
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    # Entity identifiers when supplied; exact model-feature duplicates also remain together.
    for values in [ids,pd.util.hash_pandas_object(raw,index=False).to_numpy()]:
        if values is None:continue
        seen={}
        for i,k in enumerate(values):
            if k in seen:parent[find(i)]=find(seen[k])
            else:seen[k]=i
    groups=np.array([find(i) for i in range(n)])
    return raw,y,exposure,source,columns,groups,months


def outer_splits(name,cfg):
    raw,y,_,_,_,groups,months=dataset(name);result=[]
    for repeat in range(cfg['outer_repeats']):
        if months is None:
            splitter=StratifiedGroupKFold(cfg['outer_folds'],shuffle=True,random_state=91000+repeat)
            splits=list(splitter.split(np.zeros(len(y)),y,groups))
        else:
            unique=np.unique(months);start=4+repeat
            windows=np.array_split(unique[start:],cfg['outer_folds']);splits=[]
            for window in windows:
                test=np.flatnonzero(np.isin(months,window));train=np.flatnonzero(months<window.min())
                train=train[~np.isin(groups[train],np.unique(groups[test]))]
                assert months[train].max()<months[test].min()
                splits.append((train,test))
        for fold,(train,test) in enumerate(splits):
            assert len(train) and len(test) and not np.intersect1d(groups[train],groups[test]).size
            result.append((repeat,fold,train,test))
    return result


def case_data(name,repeat,fold,stream,train,test,cfg):
    raw,y,exposure,source,columns,groups,months=dataset(name)
    ds=cfg['datasets'].index(name);seed=20261027+ds*100000+repeat*10000+fold*100+stream*10
    rng=np.random.default_rng(seed)
    if months is None:order=rng.permutation(train)
    else:order=train[np.lexsort((rng.random(len(train)),months[train]))]
    pool=order[:int(cfg['initial_pool_fraction']*len(order))]
    rids=rng.choice(pool,min(cfg['reference_count'],len(pool)//3),replace=False);rg=np.unique(groups[rids])
    allowed=pool[~np.isin(groups[pool],rg)]
    if name=='hmeq':score=raw.DELINQ.fillna(0).to_numpy()+.04*raw.DEBTINC.fillna(30).to_numpy()
    elif name=='uci':score=raw.PAY_0.to_numpy()+.2*raw.PAY_2.to_numpy()
    else:score=.04*raw.dti.fillna(20).to_numpy()-.01*raw.fico_range_low.fillna(680).to_numpy()
    d0=allowed[np.argsort(score[allowed],kind='stable')[:int(cfg['initial_visible_fraction']*len(allowed))]]
    arrivals=order[~np.isin(order,d0)&~np.isin(groups[order],rg)]
    batches=np.array_split(arrivals,cfg['rounds'])
    assert not np.intersect1d(np.r_[d0,arrivals,rids],test).size
    assert not np.intersect1d(groups[rids],groups[np.r_[d0,arrivals]]).size
    assert len(np.unique(np.r_[d0,arrivals]))==len(d0)+len(arrivals)
    manifest={'dataset':name,'outer_repeat':repeat,'outer_fold':fold,'replay_stream':stream,'seed':seed,
        'source_hash':source,'train_hash':digest(train),'test_hash':digest(test),'D0_hash':digest(d0),'reference_hash':digest(rids),
        'arrival_hash':digest(arrivals),'D0_n':len(d0),'test_n':len(test),'reference_n':len(rids),
        'group_train_test_disjoint':True,'reference_labels_excluded':True,
        'entity_group_limit':'available loan/client identifiers and exact-feature duplicate groups; no claim of verified absent borrower linkage',
        'temporal':months is not None,'time0_initial_feature_pool_max_month':None if months is None else int(months[pool].max()),
        'max_train_month':None if months is None else int(months[train].max()),'min_test_month':None if months is None else int(months[test].min()),
        'label_availability':'immediate replay revelation; final-outcome dates unavailable, retrospective temporal evaluation only'}
    return raw,y,exposure,groups,seed,d0,rids,batches,manifest


def tune_initial(raw,y,d0,groups,seed,cfg,folder):
    path=folder/'initial_tuning.json'
    if path.exists():return json.loads(path.read_text())
    splitter=StratifiedGroupKFold(cfg['inner_folds'],shuffle=True,random_state=seed+31)
    splits=list(splitter.split(np.zeros(len(d0)),y[d0],groups[d0]));scores=[];gpu_count=0
    for j,(a,b) in enumerate(splits):
        train,ev=d0[a],d0[b];pre=legacy.preprocessor(raw);pre.fit(raw.iloc[train])
        tx=np.asarray(pre.transform(raw.iloc[train]),float);vx=np.asarray(pre.transform(raw.iloc[ev]),float)
        for ridge in cfg['lr_ridge_grid']:
            m=frozen.fit_logistic(np.c_[np.ones(len(tx)),tx],y[train],ridge)
            p=expit(np.c_[np.ones(len(vx)),vx]@m)
            scores.append({'learner':'LR','parameter':ridge,'inner_fold':j,'brier':float(np.mean((p-y[ev])**2))})
        for k,param in enumerate(cfg['xgb_grid']):
            m=gpu_fit(tx,y[train],{**cfg['xgb_fixed'],**param},seed+j)
            p=m.predict_proba(vx.astype(np.float32))[:,1];gpu_count+=1
            scores.append({'learner':'XGB','parameter':k,'inner_fold':j,'brier':float(np.mean((p-y[ev])**2))})
    frame=pd.DataFrame(scores);lr=frame[frame.learner=='LR'].groupby('parameter').brier.mean();xx=frame[frame.learner=='XGB'].groupby('parameter').brier.mean()
    ridge=float(min(lr.index,key=lambda k:(lr[k],list(cfg['lr_ridge_grid']).index(k))))
    choice=int(min(xx.index,key=lambda k:(xx[k],k)))
    result={'ridge':ridge,'xgb':{**cfg['xgb_fixed'],**cfg['xgb_grid'][choice]},'scores':scores,'actual_tuning_GPU_fits':gpu_count,
        'training_labels':'D0 only','outer_T_used':False,'preprocessor_refitted_each_inner_fold':True,'acquisition_search':False}
    write_json(path,result);return result


def fantasy_select(s,global_d,global_o,global_c,params,fit_seed,cfg):
    family=propose(s);h,_,_=frozen.geometry(s,False)
    theta=np.random.default_rng(s.seed+132000).multivariate_normal(s.theta,frozen.inverse(h),cfg['candidate_selection_worlds'])
    uniforms=np.random.default_rng(s.seed+152000).random((len(theta),len(s.ordinary_x)+len(s.candidate_x)))
    scores=np.zeros((len(theta),len(family)));fits=0
    for w,t in enumerate(theta):
        p=expit(np.vstack([s.ordinary_x,s.candidate_x])@t);fy=(uniforms[w]<p).astype(int);target=expit(s.reference_x@t)
        for j,proposal in enumerate(family):
            ids=proposal['ids'];x=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]])
            yy=np.r_[s.visible_y,fy[:len(s.ordinary_x)],fy[len(s.ordinary_x)+ids]]
            gid=np.r_[global_d,global_o,global_c[ids]];order=np.argsort(gid,kind='stable')
            m=gpu_fit(x[order,1:],yy[order],params,fit_seed);fits+=1
            q=m.predict_proba(s.reference_x[:,1:].astype(np.float32))[:,1];scores[w,j]=np.mean((q-target)**2)
    chosen=int(np.argmin(scores.mean(axis=0)))
    return family[chosen]['ids'],{'sources':family[chosen]['sources'],'family':[{'ids':p['ids'],'sources':p['sources']} for p in family],
        'selection_scores':scores.mean(axis=0),'actual_fantasy_GPU_fits':fits}


def run_case(args):
    name,repeat,fold,stream,train,test,base,identity=args;base=Path(base);cfg=json.loads(CONFIG.read_text())
    cid=f'{name}_repeat{repeat+1}_fold{fold+1}_stream{stream+1}';folder=base/'private/formal_md/cases'/cid;folder.mkdir(parents=True,exist_ok=True)
    done=folder/'completion.json'
    if done.exists():
        result=json.loads(done.read_text());assert result['identity_hash']==identity;return result
    raw,y,exposure,groups,seed,d0,rids,batches,manifest=case_data(name,repeat,fold,stream,train,test,cfg)
    write_json(folder/'state_manifest.json',manifest);np.savez_compressed(folder/'split.npz',D0=d0,reference=rids,outer_train=train,outer_test=test,**{f'batch_{i+1}':b for i,b in enumerate(batches)})
    tuning=tune_initial(raw,y,d0,groups,seed,cfg,folder);params=tuning['xgb'];ridge=tuning['ridge']
    pre=legacy.preprocessor(raw);pre.fit(raw.iloc[d0]);z=np.asarray(pre.transform(raw),float);x=np.c_[np.ones(len(z)),z]
    teacher=gpu_fit(z[d0],y[d0],params,seed);price=teacher.predict_proba(z.astype(np.float32))[:,1]
    target=price[test]>=.5;ordinary=[b[price[b]<.5] for b in batches];candidates=[b[price[b]>=.5] for b in batches]
    combinations=[(cfg['main_budget'],m) for m in cfg['main_methods']]+[(rho,m) for rho in cfg['secondary_budgets'] for m in cfg['secondary_methods']]
    actual_cache={};episodes=[];new_actual=0;new_fantasy=0;new_committee=0;internal_lr=0;all_budget_ok=True;max_rescore_error=0.
    for rho,method in combinations:
        eid=f'{cid}_rho{rho:g}_{method}';ep=folder/eid;ep.mkdir(exist_ok=True);epdone=ep/'completion.json'
        if epdone.exists():
            record=json.loads(epdone.read_text());assert record['identity_hash']==identity;episodes.append(record['summary']);continue
        visible=np.sort(d0.copy());rows=[];predictions=[];cum_cost=cum_loss=cum_ocost=cum_oloss=0.;total_count=0
        for rid,(oid,cid_pool) in enumerate(zip(ordinary,candidates),1):
            cost=price[cid_pool]*exposure[cid_pool];budget=float(rho*cost.sum());theta=frozen.fit_logistic(x[visible],y[visible],ridge)
            s=frozen.State(x[visible],y[visible],x[oid],x[cid_pool],x[rids],cost,budget,theta,seed+rid*1000,ridge=ridge)
            tick=time.perf_counter();diag={};fit_seed=seed+rid
            if method=='ParBaLS':
                selected,di=select_neighbor(s,{'universes':cfg['neighbor_universes'],'posterior_draws':cfg['neighbor_posterior_draws']})
                internal_lr+=di['internal_LR_refits'];diag={'internal_LR_refits':di['internal_LR_refits']}
            elif method=='FantasyCandidate':
                selected,diag=fantasy_select(s,visible,oid,cid_pool,params,fit_seed,cfg);new_fantasy+=diag['actual_fantasy_GPU_fits']
            elif method=='LRBE-IG':
                rng=np.random.default_rng(s.seed);pp=[]
                if len(cid_pool):
                    for member in range(cfg['IG_bootstrap_members']):
                        sampled=rng.integers(0,len(visible),len(visible));yy=y[visible[sampled]]
                        if len(np.unique(yy))<2:pp.append(np.full(len(cid_pool),yy.mean()));continue
                        m=gpu_fit(z[visible[sampled]],yy,params,s.seed+member);new_committee+=1;pp.append(m.predict_proba(z[cid_pool].astype(np.float32))[:,1])
                    pp=np.asarray(pp);utility=np.maximum(frozen.entropy(pp.mean(axis=0))-frozen.entropy(pp).mean(axis=0),0)
                else:utility=np.empty(0)
                selected=frozen.select(method,s,utility).selected_ids
            else:selected=frozen.select(method,s).selected_ids
            selected=np.sort(np.asarray(selected,int));spent=float(cost[selected].sum());assert spent<=budget+1e-7 and len(np.unique(selected))==len(selected)
            global_selected=cid_pool[selected];assert not np.intersect1d(visible,np.r_[oid,global_selected]).size
            # Persist action before revealing current ordinary/candidate labels to the real training procedure.
            write_json(ep/f'round_{rid:02d}_selection.json',{'round':rid,'D_ids':visible,'O_ids':oid,'C_ids':cid_pool,
                'selected_ids':global_selected,'reference_ids':rids,'costs':cost,'B':budget,'selected_hash':digest(global_selected),
                'only_D_labels_in_selector':True,'selection_diagnostics':diag,'selection_seconds':time.perf_counter()-tick})
            visible=np.sort(np.r_[visible,oid,global_selected]);assert len(np.unique(visible))==len(visible) and not np.intersect1d(visible,test).size and not np.intersect1d(visible,rids).size
            key=digest({'training_ids':visible,'labels':digest(y[visible]),'params':params,'seed':fit_seed,'case':cid,'identity':identity})
            reused=key in actual_cache
            if reused:m=actual_cache[key]
            else:m=gpu_fit(z[visible],y[visible],params,fit_seed);actual_cache[key]=m;new_actual+=1
            p=m.predict_proba(z[test].astype(np.float32))[:,1];predictions.append(p);loss=(p-y[test])**2
            cum_cost+=spent;cum_loss+=float(exposure[global_selected]@y[global_selected]);cum_ocost+=float(exposure[oid]@price[oid]);cum_oloss+=float(exposure[oid]@y[oid]);total_count+=len(selected)
            rows.append({'case_id':cid,'dataset':name,'outer_repeat':repeat,'outer_fold':fold,'replay_stream':stream,'method':method,'rho':rho,'round':rid,
                'Brier':float(loss.mean()),'target_Brier':float(loss[target].mean()) if target.any() else None,
                'rest_Brier':float(loss[~target].mean()) if (~target).any() else None,'target_share':float(target.mean()),
                'count':len(selected),'Kstar':frozen.kstar(cost,budget),'risk':spent,'B':budget,'budget_feasible':True,
                'ordinary_n':len(oid),'candidate_n':len(cid_pool),'D_n':len(visible),'cumulative_count':total_count,
                'cumulative_exploration_cost':cum_cost,'cumulative_exploration_loss':cum_loss,'ordinary_cost':cum_ocost,'ordinary_loss':cum_oloss,
                'actual_backend':'cuda:0','cached_real_model':reused,'selected_hash':digest(global_selected),'representation_hash':digest(pre.get_feature_names_out().tolist())})
            pd.DataFrame(rows).to_csv(ep/'trajectory.csv',index=False)
        preds=np.asarray(predictions);np.savez_compressed(ep/'test_predictions.npz',ids=test,groups=groups[test],y=y[test],target=target,prediction=preds)
        saved=np.load(ep/'test_predictions.npz');rescored=((saved['prediction']-saved['y'])**2).mean(axis=1)
        error=float(np.max(np.abs(rescored-np.asarray([r['Brier'] for r in rows]))));assert error<1e-12;max_rescore_error=max(max_rescore_error,error)
        result={'episode_id':eid,'case_id':cid,'dataset':name,'outer_repeat':repeat,'outer_fold':fold,'replay_stream':stream,'method':method,'rho':rho,
            'ALC_Brier':float(rescored.mean()),'final_Brier':float(rescored[-1]),'cumulative_count':total_count,
            'total_exploration_cost':cum_cost,'total_realized_loss':cum_loss,'ordinary_cost':cum_ocost,'ordinary_loss':cum_oloss,
            'target_ALC_Brier':float(np.mean([r['target_Brier'] for r in rows])) if target.any() else None,
            'all_round_metrics_hash':sha(ep/'trajectory.csv'),'prediction_hash':sha(ep/'test_predictions.npz'),
            'actual_backend':'cuda:0','validation_role':'internal full-process evaluation; not first unseen external cohort','test_n':len(test)}
        write_json(epdone,{'completed':True,'identity_hash':identity,'summary':result,'all_rounds':10,'rescore_error':error});episodes.append(result)
        print('FORMAL EPISODE',eid,'ALC',result['ALC_Brier'],'actual',new_actual,'fantasy',new_fantasy,flush=True)
    summary={'case_id':cid,'completed':True,'identity_hash':identity,'episodes':episodes,'actual_new_real_GPU_fits':new_actual,
        'actual_new_fantasy_GPU_fits':new_fantasy,'actual_new_committee_GPU_fits':new_committee,'initial_tuning_GPU_fits':tuning['actual_tuning_GPU_fits'],
        'initial_teacher_GPU_fits':1,'internal_neighbor_LR_refits':internal_lr,'max_prediction_rescore_error':max_rescore_error,
        'selected_ridge':ridge,'selected_xgb':params,'all_information_budget_checks_passed':True}
    write_json(done,summary);return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--limit-cases',type=int);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args();base=args.base
    out=base/'formal_md';out.mkdir(exist_ok=True);cfg=json.loads(CONFIG.read_text())
    assert sha(HERE/'experiment_protocol_zh.md')==cfg['controlling_protocol_sha256']
    paths=[Path(__file__),CONFIG,HERE/'runtime.py',HERE/'run_fantasy.py',HERE/'run_nearest_neighbor.py',ROOT/'experiments/research_v2/core.py',ROOT/'experiments/research_v2/run_real.py']
    old=pd.read_csv(ROOT/'results/research_v2/analysis/method_summary.csv');delta={}
    for name in cfg['datasets']:
        q=old[(old.dataset==name)&(old.protocol=='fixed')&(old.method=='OrdinaryOnly')];assert len(q)==1;delta[name]=.005*float(q.mean_alc_brier.iloc[0])
    identity={'config':cfg,'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'data':{name:sha(ROOT/'data/research_v2'/f'{name}.csv') for name in cfg['datasets']},
        'LC_time_ID_source':sha(ROOT/'data/research_v2/lendingclub_ids.csv'),'delta':delta,
        'delta_source_hash':sha(ROOT/'results/research_v2/analysis/method_summary.csv'),
        'versions':{'numpy':np.__version__,'xgboost':xgboost.__version__,'sklearn':sklearn.__version__}}
    identity['identity_hash']=digest(identity)
    p=out/'frozen_identity.json'
    if p.exists():assert json.loads(p.read_text())==identity,'frozen config/source/data changed; do not silently resume'
    else:write_json(p,identity)
    jobs=[];manifests=[]
    for name in cfg['datasets']:
        for repeat,fold,train,test in outer_splits(name,cfg):
            for stream in range(cfg['replay_streams']):
                *_,manifest=case_data(name,repeat,fold,stream,train,test,cfg);manifests.append(manifest)
                jobs.append((name,repeat,fold,stream,train,test,str(base),identity['identity_hash']))
    assert len(jobs)==60
    write_jsonl(out/'registered_case_manifest.jsonl',manifests)
    write_json(out/'registration.json',{'registered_cases':60,'expected_main_episodes':360,'expected_secondary_episodes':360,'expected_total_episodes':720,
        'all_splits_registered_before_new_T_outcomes':True,'inner_model_tuning':'D0 labels only, preprocessing refit inside three inner folds',
        'acquisition_parameters':'one frozen configuration per method, no acquisition parameter search',
        'failed_candidate_status':cfg['candidate_status'],'LC_monthly_forward_splits':True,'fresh_external_blind_claim':False,'delta':delta})
    print('FROZEN',identity['identity_hash'],'registered cases',len(jobs),'LC monthly forward; delta',delta,flush=True)
    if args.prepare_only:return
    selected=jobs if args.limit_cases is None else jobs[:args.limit_cases]
    completed=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,job):job for job in selected}
        for future in as_completed(futures):
            try:result=future.result()
            except Exception:
                write_json(out/'last_failure.json',{'job':list(futures[future][:4]),'traceback':traceback.format_exc(),'identity_hash':identity['identity_hash']});raise
            completed.append(result);episodes=[r for c in completed for r in c['episodes']]
            write_jsonl(out/'confirm_episodes.jsonl',episodes)
            write_json(out/'progress.json',{'completed_cases_in_this_schedule':len(completed),'scheduled_cases':len(selected),'registered_cases':60,
                'completed_episodes_in_this_schedule':len(episodes),'registered_episodes':720,'identity_hash':identity['identity_hash']})
            print('FORMAL CASE COMPLETE',result['case_id'],'cases',len(completed),'/',len(selected),flush=True)
    all_completed=[]
    for p in (base/'private/formal_md/cases').glob('*/completion.json'):
        value=json.loads(p.read_text());assert value['identity_hash']==identity['identity_hash'];all_completed.append(value)
    episodes=[r for c in all_completed for r in c['episodes']];write_jsonl(out/'confirm_episodes.jsonl',episodes)
    write_json(out/'completion.json',{'complete':len(episodes)==720,'completed_cases':len(all_completed),'completed_episodes':len(episodes),
        'expected_episodes':720,'main_episodes':sum(r['rho']==.1 for r in episodes),'secondary_episodes':sum(r['rho']!=.1 for r in episodes),
        'identity_hash':identity['identity_hash'],'all_cases_use_fresh_full_replay':True})


if __name__=='__main__':main()
