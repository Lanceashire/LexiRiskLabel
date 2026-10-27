"""Registered paired metrics; conditional grouped test resampling, no iid-fold t test."""
from runtime import *
import argparse


def conditional_bootstrap(case_pairs,draws,seed):
    groups=np.unique(np.concatenate([r['groups'] for r in case_pairs]));lookup={int(g):i for i,g in enumerate(groups)}
    sums=np.zeros((len(groups),len(case_pairs)));counts=np.zeros_like(sums)
    for j,case in enumerate(case_pairs):
        ix=np.asarray([lookup[int(g)] for g in case['groups']],int)
        np.add.at(sums[:,j],ix,case['difference']);np.add.at(counts[:,j],ix,1.)
    point=float((sums.sum(axis=0)/counts.sum(axis=0)).mean());rng=np.random.default_rng(seed);values=[]
    probability=np.full(len(groups),1/len(groups))
    for begin in range(0,draws,32):
        weight=rng.multinomial(len(groups),probability,size=min(32,draws-begin))
        den=weight@counts;num=weight@sums
        assert (den>0).all();values.extend((num/den).mean(axis=1).tolist())
    values=np.asarray(values);ci=np.quantile(values,[.025,.975])
    # One predeclared centered, two-sided conditional bootstrap p; not a training-level sampling guarantee.
    p=(1+int((np.abs(values-point)>=abs(point)).sum()))/(draws+1)
    return {'mean_gain':point,'conditional_L95':float(ci[0]),'conditional_U95':float(ci[1]),
        'conditional_p':float(p),'test_group_clusters':len(groups),'bootstrap_draws':draws,
        'scope':'test group resampling conditional on the fitted model collection; training uncertainty not included'}


def table(df):
    text=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):text.append('| '+' | '.join(f'{x:+.8f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(text)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--bootstrap-draws',type=int,default=2000);a=ap.parse_args()
    base=a.base;out=base/'formal_md';identity=json.loads((out/'frozen_identity.json').read_text());cfg=identity['config']
    completed=[json.loads(p.read_text()) for p in (base/'private/formal_md/cases').glob('*/completion.json')]
    assert len(completed)==60 and all(c['identity_hash']==identity['identity_hash'] for c in completed),'full registered evaluation not complete'
    rows=[r for c in completed for r in c['episodes']];frame=pd.DataFrame(rows);assert len(frame)==720
    assert frame.episode_id.is_unique and (frame.rho==.1).sum()==360 and (frame.rho!=.1).sum()==360
    source_unchanged=all(sha(ROOT/k)==h for k,h in identity['sources'].items());assert source_unchanged
    max_error=0.;rounds=[];predictions={};private=base/'private/formal_md/cases'
    for r in rows:
        ep=private/r['case_id']/r['episode_id'];trajectory=pd.read_csv(ep/'trajectory.csv');p=np.load(ep/'test_predictions.npz')
        assert len(trajectory)==10 and p['prediction'].shape==(10,len(p['ids']))
        brier=((p['prediction']-p['y'])**2).mean(axis=1)
        max_error=max(max_error,abs(float(brier.mean())-r['ALC_Brier']),abs(float(brier[-1])-r['final_Brier']),float(np.max(np.abs(trajectory.Brier.to_numpy()-brier))))
        assert (trajectory.risk<=trajectory.B+1e-7).all() and trajectory.budget_feasible.all() and (trajectory.actual_backend=='cuda:0').all()
        assert sha(ep/'trajectory.csv')==r['all_round_metrics_hash'] and sha(ep/'test_predictions.npz')==r['prediction_hash']
        rounds.extend(trajectory.to_dict(orient='records'));predictions[r['episode_id']]={k:p[k] for k in ['ids','groups','y','prediction']}
    assert max_error<1e-12
    write_jsonl(out/'confirm_episodes.jsonl',rows);frame.to_csv(out/'episode_summary.csv',index=False)
    summary=frame.groupby(['dataset','rho','method'],as_index=False).agg(ALC_Brier=('ALC_Brier','mean'),final_Brier=('final_Brier','mean'),
        target_ALC_Brier=('target_ALC_Brier','mean'),mean_selected_count=('cumulative_count','mean'),
        exploration_risk=('total_exploration_cost','mean'),realized_exploration_loss=('total_realized_loss','mean'),
        ordinary_risk=('ordinary_cost','mean'),ordinary_loss=('ordinary_loss','mean'),episodes=('episode_id','count'))
    summary.to_csv(out/'method_summary.csv',index=False)
    paired=[];primary_boot=[]
    for (dataset,rho),group in frame.groupby(['dataset','rho']):
        for reference in ['MinRisk','ParBaLS']:
            cases=[]
            for case_id,g in group.groupby('case_id'):
                q=g.set_index('method');candidate=q.loc['FantasyCandidate'];baseline=q.loc[reference]
                left=predictions[baseline.episode_id];right=predictions[candidate.episode_id]
                assert np.array_equal(left['ids'],right['ids']) and np.array_equal(left['y'],right['y'])
                diff=((left['prediction']-left['y'])**2-(right['prediction']-right['y'])**2).mean(axis=0)
                gain=float(baseline.ALC_Brier-candidate.ALC_Brier);assert abs(gain-diff.mean())<1e-12
                paired.append({'dataset':dataset,'rho':rho,'case_id':case_id,'outer_repeat':int(candidate.outer_repeat),'outer_fold':int(candidate.outer_fold),
                    'replay_stream':int(candidate.replay_stream),'comparison':'FantasyCandidate_vs_'+reference,'gain':gain,
                    'risk_difference':float(candidate.total_exploration_cost-baseline.total_exploration_cost),
                    'count_difference':int(candidate.cumulative_count-baseline.cumulative_count)})
                cases.append({'groups':left['groups'],'difference':diff})
            if rho==cfg['main_budget']:
                boot=conditional_bootstrap(cases,a.bootstrap_draws,780001+cfg['datasets'].index(dataset)*100+(reference=='ParBaLS'))
                boot.update({'dataset':dataset,'comparison':'FantasyCandidate_vs_'+reference,'delta':identity['delta'][dataset]});primary_boot.append(boot)
    pairs=pd.DataFrame(paired);pairs.to_csv(out/'paired_episode_contrasts.csv',index=False)
    pairsummary=pairs.groupby(['dataset','rho','comparison'],as_index=False).agg(mean_gain=('gain','mean'),
        positive_replay_cases=('gain',lambda x:int((x>1e-12).sum())),negative_replay_cases=('gain',lambda x:int((x< -1e-12).sum())),
        mean_risk_difference=('risk_difference','mean'),mean_count_difference=('count_difference','mean'))
    pairsummary.to_csv(out/'paired_summary.csv',index=False)
    # Holm on the six registered task x primary-reference conditional tests only.
    order=sorted(range(len(primary_boot)),key=lambda i:primary_boot[i]['conditional_p']);running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(order)-rank)*primary_boot[index]['conditional_p']);primary_boot[index]['conditional_p_Holm']=min(1.,running)
    sensitivity=[]
    for r in primary_boot:
        for factor in [.2,1.,2.]:
            delta=r['delta']*factor;lo,hi=r['conditional_L95'],r['conditional_U95']
            status='uncertain'
            if lo>delta:status='conditional gain exceeds delta'
            elif hi<0:status='conditional loss'
            elif lo>=-delta and hi<=delta:status='conditional near-equivalence'
            elif hi<delta:status='conditional interval excludes gain of delta'
            sensitivity.append({'dataset':r['dataset'],'comparison':r['comparison'],'relative_old_ordinary_threshold':.005*factor,
                'delta':delta,'conditional_classification':status,'full_training_uncertainty_assessed':False})
    pd.DataFrame(primary_boot).to_csv(out/'primary_conditional_test_intervals.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'delta_sensitivity.csv',index=False)
    total={key:sum(c[key] for c in completed) for key in ['actual_new_real_GPU_fits','actual_new_fantasy_GPU_fits','actual_new_committee_GPU_fits','initial_tuning_GPU_fits','initial_teacher_GPU_fits','internal_neighbor_LR_refits']}
    write_json(out/'verification.json',{'passed':True,'all_720_predictions_rescored':True,'max_rescore_error':max_error,
        'all_registered_cases_complete':True,'all_budgets_feasible':True,'all_actual_training_GPU_verified':True,
        'frozen_source_identity_unchanged':source_unchanged,'no_iid_t_test_on_folds':True,'test_bootstrap_conditions_on_models':True,
        'no_first_blind_or_external_claim':True,'resource_totals':total,'analysis_source_hash':sha(__file__)})
    main_summary=summary[summary.rho==.1][['dataset','method','ALC_Brier','final_Brier','mean_selected_count','exploration_risk']]
    report=f'''# Markdown第9—10节：完整采集流程内部评估

执行依据为experiment_protocol_zh.md。已完成三任务×5外层划分×2次注册重复×2重放流：主预算0.1六方法360条十轮轨迹；预算0.025/0.4各三方法共360条，合计720条。每个划分从头重新建立预处理、D0、业务模型、参考与到达流，未在旧采集训练集上交叉验证。

HMEQ/UCI为重复分组五折；LendingClub用保留的贷款ID与放款月份做两组预定起点的五个前向月份窗口。参考特征来自初始特征池，组级排除其标签与全部外层T。LendingClub只知最终结果而不知标签实际成熟时点，因此是前向月份、即时揭示的回顾性评估，不称严格在线验证。

内层三折只用允许的D0标签，每折重拟合预处理。LR三档正则、XGB深度2/3/5×树100/300的六档参数在D0内按Brier固定选择。采集参数每方法只有一个开发后冻结配置，没有用当前隐藏候选标签调采集参数。预测器采用实际CUDA，所有方法同一初始配置与价格账本。

第六个对象是唯一已尝试的幻想候选，其开发门槛未通过的事实保留；本评估固定报告它，不称它为“胜出候选”，也没有发明第二个新候选。外部近邻明确是匹配表示的Laplace/风险成本/pending适配，不冒称作者原NUTS全复现。所有旧任务已参与开发，所以本轮称内部完整流程评估，不称首次外部验证。

## 主预算结果

ALC Brier为十个重训轮次总体Brier的平均，不含初始轮；低为好。不同任务风险单位不合并：

{table(main_summary)}

## 固定候选的主要配对

正数表示参照Brier−候选Brier，完整20个相关重放案例及每个预算均保留；案例改善次数只作稳定性描述，不把20个案例当20个独立总体样本。

{table(pairsummary[pairsummary.rho==.1])}

## 不确定性

按相同申请记录/分组对每条轨迹的十轮逐人误差做配对重采样，同时保留该组在不同重复/重放中的相关出现。每次重采样先重算每个案例均值，再平均案例，所以点估计与主配对完全一致。以下区间和Holm调整p值仅条件于本批已训练模型，不能替代重新抽样整条训练—采集流程，也不能当作完整训练不确定性的确认。

{table(pd.DataFrame(primary_boot)[['dataset','comparison','mean_gain','conditional_L95','conditional_U95','conditional_p_Holm','delta']])}

研究阈值在新结果之前固定为旧开发OrdinaryOnly Brier的0.5%，0.1%/1%敏感性也完整输出。它不是商业利润门槛。六个任务×主参照条件检验使用一个Holm家族；补充预算作次要描述，不用来更换主预算或挑选方法。没有对相关交叉验证折做普通独立t检验。

## 核验和资源

全部720条预测逐轮重新计分，最大误差{max_error:.3g}；每轮预算、唯一采集、标签时序、组/参考/测试分离、实际GPU后端和冻结代码身份检查通过。真实重训、内部幻想、委员会、D0调参分别计数：{json.dumps(total,ensure_ascii=False)}。这些拟合次数不是独立基准或新数据样本。

最终结果、补充预算及不利折均保留；不在结果后修改模型网格、伪标签世界、目标预算或候选家族。完整预测、逐轮选集与标签本地保留，公开必要材料不含第三方原始行。
'''
    (out/'MD第9至10节_完整流程评估报告.md').write_text(report)
    write_json(out/'analysis_complete.json',{'complete':True,'episodes':720,'conditional_bootstrap_draws':a.bootstrap_draws,'resource_totals':total,
        'source_hash':sha(__file__),'training_uncertainty_bootstrap_claim':False,'new_method_selected_after_results':False})
    write_json(out/'evidence_manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json']})
    print(main_summary.to_string(index=False));print(pairsummary[pairsummary.rho==.1].to_string(index=False));print(json.dumps(total))


if __name__=='__main__':main()
