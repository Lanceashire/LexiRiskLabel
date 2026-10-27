"""Complete the MD section-6 10-seed grid, retaining the unchanged pilot evidence."""
from runtime import *
from run_e2a_small import exhaustive,greedy,table
from scipy.stats import norm
import argparse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);a=ap.parse_args();base=a.base
    out=base/'e2a_full';out.mkdir(exist_ok=False);private=base/'private/e2a_full';private.mkdir(parents=True,exist_ok=True)
    config=json.loads((HERE/'e2a_small_config.json').read_text())
    config.update({'stage':'E2a_MD_full_10_seed_grid','status':'registered_10_seed_development_positive_control',
        'seeds':list(range(2101,2111)),'endpoint_slots':300,'independent_seed_clusters':10,'full_10_seed_grid_run':True})
    identity=json.loads((base/'source_identity.json').read_text());verify_identity(identity,run_identity(identity['configuration']))
    source={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'run_e2a_small.py',HERE/'e2a_small_config.json',HERE/'runtime.py',ROOT/'experiments/research_v2/core.py']}
    write_json(out/'activation.json',{'configuration':config,'source':source,'controlling_protocol':sha(HERE/'experiment_protocol_zh.md'),
        'pilot_outcomes_sha256':sha(base/'e2a_small/branch_outcomes.jsonl'),'pilot_controls_sha256':sha(base/'e2a_small/mechanism_controls.jsonl'),
        'extension_rule':'retain 2101-2103; execute contiguous 2104-2110 without outcome-based selection',
        'confirmation':False})
    old=[json.loads(l) for l in (base/'e2a_small/branch_outcomes.jsonl').read_text().splitlines()]
    controls=[json.loads(l) for l in (base/'e2a_small/mechanism_controls.jsonl').read_text().splitlines()]
    rows=[{**r,'prediction_source':'e2a_small','reused_pilot_endpoint':True} for r in old];shared={};cache={};hits=0;inverse_error=0.;metric_error=0.
    for seed in range(2104,2111):
        rng=np.random.default_rng(seed+120000);truth=rng.normal(size=2)
        extra=np.c_[rng.uniform(*config['other_target_coordinate_uniform'],9),rng.uniform(*config['other_nuisance_magnitude_uniform'],9)*rng.choice([-1,1],9)]
        semantic=np.vstack([config['core_pair'],config['bait'],extra]);perm=rng.permutation(12);u=semantic[perm]
        cc=np.asarray(config['cost_multiset'],float);expensive=cc.copy()
        for i,j in config['expensive_case_swaps_semantic_indices']:expensive[[i,j]]=expensive[[j,i]]
        assert np.array_equal(np.sort(cc),np.sort(expensive))
        shared[seed]={'truth':truth,'u':u,'perm':perm,'costs':{'core_pair_cheap':cc[perm],'core_pair_expensive':expensive[perm]}}
        for b in config['ordinary_b']:
            h=np.diag([1.,1.+b]);g=np.diag([1.,0.]);core=np.asarray(config['core_pair']);c=frozen.inverse(h)
            gamma=frozen.group_gain(c,g,core)-frozen.marginal(c,g,core).sum();analytic=2/3-2/(2+config['L']**2/(1+b))
            assert abs(gamma-analytic)<1e-12
            for price,cost in shared[seed]['costs'].items():
                opt,value,nfeasible=exhaustive(h,g,u,cost,config['budget'])
                proposals={'MinRisk':np.argsort(cost,kind='stable')[:frozen.kstar(cost,config['budget'])]}
                for method,paired in [('Vopt-single-noexchange',False),('BRIDGE-allpairs-noexchange',True)]:proposals[method]=greedy(h,g,u,cost,config['budget'],paired)[0]
                proposals['Random-feasible']=frozen.random_fill(cost,config['budget'],np.random.default_rng(seed+121000));proposals['Exhaustive-Gaussian-opt']=opt
                sid=f'seed{seed}_b{int(b)}_{price}'
                write_json(private/'selection_locks'/f'{sid}.json',{'state_id':sid,'candidate_semantic_ids':perm,'selections':proposals,'candidate_x':u,'cost':cost,'budget':config['budget'],'locked_before_labels':True})
                controls.append({'state_id':sid,'seed':seed,'ordinary_b':b,'price_case':price,'core_pair_gamma':gamma,'core_pair_gamma_analytic':analytic,
                    'core_pair_feasible':bool(cost[np.flatnonzero(np.isin(perm,[0,1]))].sum()<=config['budget']+1e-10),
                    'feasible_subset_count':nfeasible,'exact_best_gaussian_variance_decrease':value})
    # All 210 new endpoints are fixed before generating any new hidden outcomes.
    for seed,state in shared.items():
        truth,u,perm=state['truth'],state['u'],state['perm'];test=np.c_[norm.ppf((np.arange(1500)+.5)/1500),np.zeros(1500)];pt=expit(test@truth)
        rng=np.random.default_rng(seed+122000);yc=(rng.random(len(u))<expit(u@truth)).astype(int);gy=u@truth+rng.normal(size=len(u))
        ou=rng.random(4);on=rng.normal(size=4);yt=(rng.random(1500)<pt).astype(int);d=np.zeros((2,2));yd=np.array([0,1])
        np.savez_compressed(private/f'seed{seed}_environment.npz',truth=truth,candidate_x=u,candidate_semantic_ids=perm,candidate_y=yc,
            candidate_gaussian_y=gy,ordinary_uniform=ou,ordinary_gaussian_noise=on,test_x=test,test_y=yt,test_true_probability=pt)
        for b in config['ordinary_b']:
            ox=np.c_[np.zeros(4),np.array([1,-1,1,-1])*np.sqrt(b/4)];yo=(ou<expit(ox@truth)).astype(int);ogy=ox@truth+on
            h=np.eye(2)+ox.T@ox;g=np.diag([1.,0.]);baseline=float(np.mean((.5-pt)**2))
            for price,cost in state['costs'].items():
                sid=f'seed{seed}_b{int(b)}_{price}';lock=json.loads((private/'selection_locks'/f'{sid}.json').read_text())
                optimum=next(x['exact_best_gaussian_variance_decrease'] for x in controls if x['state_id']==sid)
                for method,raw in lock['selections'].items():
                    ids=np.asarray(raw,int);semantic_ids=np.sort(perm[ids]);cov=frozen.inverse(h+u[ids].T@u[ids]);risk=float(np.trace(g@cov));gain=float(np.trace(g@frozen.inverse(h)))-risk
                    updated=frozen.inverse(h)
                    for i in ids:updated=frozen.update(updated,u[i])
                    inverse_error=max(inverse_error,float(np.max(np.abs(updated-cov))))
                    gaussian_mean=cov@(ox.T@ogy+u[ids].T@gy[ids]);ordered=np.asarray(sorted(ids,key=lambda i:int(perm[i])),int)
                    x=np.vstack([d,ox,u[ordered]]);y=np.r_[yd,yo,yc[ordered]];key=digest({'x':digest(x),'y':y,'ridge':1.})
                    if key in cache:theta=cache[key];hits+=1
                    else:theta=frozen.fit_logistic(x,y,1.);cache[key]=theta
                    pred=expit(test@theta);evkey=digest({'fit_key':key,'test_probability':digest(pt),'test_y':digest(yt)})
                    dest=private/'predictions'/f'{evkey}.npz';dest.parent.mkdir(exist_ok=True)
                    np.savez_compressed(dest,prediction=pred,y=yt,true_probability=pt,train_x=x,train_y=y,selected_semantic_ids=semantic_ids)
                    assert cost[ids].sum()<=config['budget']+1e-10 and gain<=optimum+1e-12
                    rows.append({'state_id':sid,'seed':seed,'ordinary_b':b,'price_case':price,'method':method,'selected_semantic_ids':semantic_ids.tolist(),
                        'count':len(ids),'cost':float(cost[ids].sum()),'budget':config['budget'],'selected_hash':digest(semantic_ids),
                        'exact_gaussian_variance_decrease':gain,'exact_gaussian_posterior_risk':risk,'gap_to_exhaustive':max(0.,optimum-gain),
                        'realized_gaussian_target_MSE':float((gaussian_mean[0]-truth[0])**2),'lr_baseline_excess_brier':baseline,
                        'lr_excess_brier':float(np.mean((pred-pt)**2)),'lr_brier':float(np.mean((pred-yt)**2)),
                        'lr_gain_vs_ordinary':baseline-float(np.mean((pred-pt)**2)),'lr_fit_key':key,'lr_evaluation_key':evkey,
                        'actual_backend':'cpu','selected_from_hidden_labels':False,'prediction_source':'e2a_full','reused_pilot_endpoint':False})
        print('MD E2a seed complete',seed,'new LR fits',len(cache),flush=True)
    frame=pd.DataFrame(rows).drop(columns=['singleton_gaussian_gain','singleton_lr_error','extra_gaussian_gain_vs_singleton','lr_extra_gain_vs_singleton'],errors='ignore')
    ref=frame[frame.method=='Vopt-single-noexchange'][['state_id','exact_gaussian_variance_decrease','lr_excess_brier']].rename(columns={'exact_gaussian_variance_decrease':'singleton_gaussian_gain','lr_excess_brier':'singleton_lr_error'})
    frame=frame.merge(ref,on='state_id',validate='many_to_one');frame['extra_gaussian_gain_vs_singleton']=frame.exact_gaussian_variance_decrease-frame.singleton_gaussian_gain;frame['lr_extra_gain_vs_singleton']=frame.singleton_lr_error-frame.lr_excess_brier
    for r in frame.itertuples():
        p=np.load(base/'private'/r.prediction_source/'predictions'/f'{r.lr_evaluation_key}.npz')
        metric_error=max(metric_error,abs(float(np.mean((p['prediction']-p['true_probability'])**2))-r.lr_excess_brier),abs(float(np.mean((p['prediction']-p['y'])**2))-r.lr_brier))
    assert len(frame)==300 and frame.seed.nunique()==10 and len(controls)==60 and metric_error<1e-12 and inverse_error<1e-10
    write_jsonl(out/'branch_outcomes.jsonl',frame.to_dict(orient='records'));pd.DataFrame(controls).to_csv(out/'mechanism_controls.csv',index=False)
    summary=frame.groupby(['ordinary_b','price_case','method'],as_index=False).agg(exact_gaussian_gain=('exact_gaussian_variance_decrease','mean'),
        gap_to_exhaustive=('gap_to_exhaustive','max'),extra_gaussian_gain_vs_singleton=('extra_gaussian_gain_vs_singleton','mean'),
        lr_extra_gain_vs_singleton=('lr_extra_gain_vs_singleton','mean'),lr_gain_vs_ordinary=('lr_gain_vs_ordinary','mean'),
        LR_positive_seed_count=('lr_extra_gain_vs_singleton',lambda x:int((x>1e-12).sum())),LR_positive_vs_ordinary_count=('lr_gain_vs_ordinary',lambda x:int((x>1e-12).sum())))
    summary.to_csv(out/'descriptive_summary.csv',index=False)
    comp={'completed':True,'full_MD_suggested_grid_completed':True,'seeds':config['seeds'],'independent_seed_clusters':10,'endpoint_slots':300,
        'reused_pilot_endpoints':90,'new_endpoints':210,'new_unique_LR_fits':len(cache),'new_LR_cache_hits':hits,'new_GPU_fits':0,
        'max_prediction_metric_error':metric_error,'max_inverse_error':inverse_error,'confirmation':False}
    write_json(out/'completion.json',comp);write_json(out/'verification.json',{'passed':True,**comp,'pilot_evidence_unchanged':True,'all_60_cells_complete':True,
        'all_source_hashes_match':all(sha(ROOT/k)==v for k,v in source.items())})
    bridge=summary[summary.method=='BRIDGE-allpairs-noexchange']
    (out/'E2a_十种子完整网格实验报告.md').write_text('''# 按Markdown协议补齐E2a十种子完整网格

执行依据为experiment_protocol_zh.md第6节。现已完成3个普通信息水平×2种成本安排×10种子×5选择器=300方法终点，独立种子为10个。保留已有2101—2103的90个端点与原预测，新增固定连续2104—2110的210个端点；未按效果挑种子，旧目录未覆盖。该扩展仍属开发正控制，不冒称独立确认。

二维生成器、成本多重集、数值预算、隐藏标签释放、Gaussian精确目标及LR同选集迁移均保持原注册版本。每池枚举4096子集。全部新动作在新标签生成前锁定；普通基准在目标上恒预测0.5，解析计分，不重复拟合。Gaussian/LR计算使用CPU，后续XGB流程使用GPU。

'''+table(bridge[['ordinary_b','price_case','extra_gaussian_gain_vs_singleton','gap_to_exhaustive','lr_extra_gain_vs_singleton','LR_positive_seed_count','LR_positive_vs_ordinary_count']],['普通信息b','成本安排','精确额外收益','最大穷举差距','LR额外收益','LR优于单点/10','LR优于普通/10'])+
        '\n\n所有60条件和不利种子均保留。精确优化成立不代表LR收益稳定，两个终点分别报告。新LR不同拟合'+str(len(cache))+'次，缓存'+str(hits)+'次；全部300个旧/新预测重新计分最大误差'+str(metric_error)+'。继续按该Markdown第9、10、12节冻结完整流程评估，不追加DOCX实验。\n')
    snap=out/'source_snapshot';snap.mkdir()
    for p in source:(snap/Path(p).name).write_bytes((ROOT/p).read_bytes())
    write_json(out/'evidence_manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json']})
    print(json.dumps(comp),flush=True);print(bridge.to_string(index=False),flush=True)


if __name__=='__main__':main()
