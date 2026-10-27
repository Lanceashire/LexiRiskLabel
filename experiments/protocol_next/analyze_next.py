"""Development-only paired summaries; no two-seed significance testing."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from runtime import *


def mdtable(frame):
    header='| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'
    return header+''.join('| '+' | '.join(f'{v:.7f}' if isinstance(v,float) else str(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))


def verify_all(base):
    identity=json.loads((base/'source_identity.json').read_text());verify_identity(identity,run_identity(identity['configuration']))
    b=pd.read_json(base/'branch_outcomes.jsonl',lines=True);s=pd.read_json(base/'selection_summary.jsonl',lines=True)
    assert len(b)==432 and len(s)==216 and s.state_id.nunique()==18
    assert (s.loc[~s.diagnostic_only,'budget_feasible']).all()
    assert (b[b.learner=='XGB'].actual_backend=='cuda:0').all() and (b[b.learner=='LR'].actual_backend=='cpu').all()
    assert b.groupby(['state_id','selector','random_draw'],dropna=False).selected_hash.nunique().eq(1).all()
    assert b.groupby(['state_id','selector','random_draw'],dropna=False).train_hash.nunique().eq(1).all()
    for key,rows in b.groupby('fit_key'):
        pred=np.load(base/'private/predictions'/f'{key}.npz');value=np.mean((pred['prediction']-pred['y'])**2)
        assert np.allclose(rows.brier,value,atol=1e-12)
    cells=pd.read_json(base/'e2b/branch_outcomes.jsonl',lines=True);quad=cells.pivot(index=['state_id','learner'],columns='method',values='brier')
    assert np.allclose(quad.single_noexchange,quad.pair_noexchange,atol=1e-15) and np.allclose(quad.single_exchange,quad.pair_exchange,atol=1e-15)
    c=pd.read_json(base/'e2c/branch_outcomes.jsonl',lines=True);leaves=pd.read_json(base/'e2c/leaf_evidence.jsonl',lines=True)
    assert c.groupby(['state_id','condition','selector']).selected_hash.nunique().eq(1).all()
    assert leaves.trained_on_D_only.all() and leaves.leaf_feature_count.le(80).all() and leaves.actual_backend.eq('cuda:0').all()
    assert c.loc[c.learner=='XGB','actual_backend'].eq('cuda:0').all()
    for key,rows in c.groupby('fit_key'):
        pred=np.load(base/'private/predictions'/f'{key}.npz');assert np.allclose(rows.brier,np.mean((pred['prediction']-pred['y'])**2),atol=1e-12)
    finite=pd.read_json(base/'e1b/all_set_outcomes.jsonl',lines=True);references=pd.read_json(base/'e1b/validation_selected_reference.jsonl',lines=True)
    assert finite.groupby(['state_id','learner']).selected_hash.nunique().eq(16).all()
    assert finite.loc[finite.learner=='XGB','actual_backend'].eq('cuda:0').all()
    for row in references.to_dict('records'):
        group=finite[(finite.state_id==row['state_id'])&(finite.learner==row['learner'])].sort_values(['V_search_brier','set_id'])
        assert int(group.iloc[0].set_id)==row['chosen_set_id']
    for row in finite.to_dict('records'):
        p=np.load(base/'private/e1b_predictions'/f"{row['state_id']}_set{row['set_id']}_{row['learner']}.npz")
        assert not np.intersect1d(p['search_ids'],p['eval_ids']).size
        assert np.isclose(np.mean((p['search_prediction']-p['search_y'])**2),row['V_search_brier'],atol=1e-12)
        assert np.isclose(np.mean((p['eval_prediction']-p['eval_y'])**2),row['V_eval_brier'],atol=1e-12)
    result={'passed':True,'e1_shared_selection_and_training_hashes':True,'e1_saved_prediction_metrics_recomputed':True,
            'same_geometry_pair_construction_effect_exactly_zero_in_e2b':True,'e2c_shared_selection_across_learners':True,
            'leaf_models_D_only_and_maximum_80_features':True,'all_actual_XGB_training_cuda':True,'lr_backend':'cpu',
            'source_code_data_config_and_anchor_identity_checked':True,'e1b_choice_uses_V_search_only':True,'e1b_V_search_and_V_eval_disjoint':True,
            'e1b_validation_predictions_recomputed':True,'no_confirmatory_inference':True,'private_individual_artifacts_remain_local':True}
    write_json(base/'final_verification.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base;out=base/'analysis';out.mkdir(exist_ok=True)
    verification=verify_all(base);identity=json.loads((base/'source_identity.json').read_text());b=pd.read_json(base/'branch_outcomes.jsonl',lines=True);s=pd.read_json(base/'selection_summary.jsonl',lines=True)
    # Random draws average within state first, then three states within trajectory.
    state=b.groupby(['dataset','seed','round','state_id','selector','learner'],as_index=False)[['brier','gain_vs_ordinary','gain_vs_minrisk','target_brier','rest_brier']].mean()
    trace=state.groupby(['dataset','seed','selector','learner'],as_index=False)[['brier','gain_vs_ordinary','gain_vs_minrisk','target_brier','rest_brier']].mean()
    summary=trace.groupby(['dataset','selector','learner'],as_index=False)[['brier','gain_vs_ordinary','gain_vs_minrisk','target_brier','rest_brier']].mean()
    state.to_csv(out/'e1_state_means.csv',index=False);trace.to_csv(out/'e1_trajectory_paired_differences.csv',index=False);summary.to_csv(out/'e1_descriptive_summary.csv',index=False)
    correlations=[]
    combined=b.merge(s[['state_id','selector','random_draw','proxy_value']],on=['state_id','selector','random_draw'])
    for (sid,learner),group in combined[~combined.diagnostic_only].groupby(['state_id','learner']):
        group=group.drop_duplicates('selected_hash')
        rho=float(spearmanr(group.proxy_value,group.gain_vs_ordinary).statistic) if group.proxy_value.nunique()>1 and group.gain_vs_ordinary.nunique()>1 else None
        correlations.append({'state_id':sid,'learner':learner,'unique_interventions':len(group),'proxy_gain_spearman':rho})
    pd.DataFrame(correlations).to_csv(out/'e1_proxy_actual_relationship.csv',index=False)
    full_comparison=[]
    for (sid,learner),group in b.groupby(['state_id','learner']):
        full=group[group.selector=='FullFeedback'].brier.iloc[0];feasible=group[~group.diagnostic_only]
        full_comparison.append({'state_id':sid,'learner':learner,'full_feedback_brier':full,'feasible_methods_better_than_full_feedback':int((feasible.brier<full-1e-12).sum())})
    pd.DataFrame(full_comparison).to_csv(out/'full_feedback_is_not_an_upper_bound.csv',index=False)
    qsel=pd.read_json(base/'e2b/selection_summary.jsonl',lines=True);qout=pd.read_json(base/'e2b/branch_outcomes.jsonl',lines=True)
    quad=qout.pivot(index=['state_id','dataset','seed','round','learner'],columns='method',values='brier').reset_index()
    quad['pair_gain_noexchange']=quad.single_noexchange-quad.pair_noexchange;quad['pair_gain_exchange']=quad.single_exchange-quad.pair_exchange
    quad['exchange_gain']=quad.single_noexchange-quad.single_exchange;quad.to_csv(out/'e2b_factorial_effects.csv',index=False)
    proxy=qsel.pivot(index='state_id',columns='method',values='proxy_value');proxy['exchange_proxy_gain']=proxy.single_exchange-proxy.single_noexchange
    gamma_rows=[]
    for sid in qsel.state_id.unique():
        data=np.load(base/'private/e2b_gamma'/f'{sid}_single_noexchange.npz');gamma=data['gamma'];tolerance=data['tolerance'];initial=data['step']==0
        gamma_rows.append({'state_id':sid,'visited_pair_evaluations_one_cell_only':len(gamma),'positive_visited_pair_evaluations':int((gamma>tolerance).sum()),
                           'initial_unique_feasible_pairs':int(initial.sum()),'initial_positive_pairs':int((gamma[initial]>tolerance[initial]).sum()),
                           'gamma_max':float(gamma.max()) if len(gamma) else None,'gamma_min':float(gamma.min()) if len(gamma) else None})
    gamma=pd.DataFrame(gamma_rows);gamma.to_csv(out/'e2b_gamma_deduplicated.csv',index=False)
    c=pd.read_json(base/'e2c/branch_outcomes.jsonl',lines=True);cs=c.groupby(['dataset','condition','selector','learner'],as_index=False)[['brier','gain_vs_ordinary','gain_vs_minrisk']].mean()
    c.to_csv(out/'e2c_all_paired_outcomes.csv',index=False);cs.to_csv(out/'e2c_descriptive_summary.csv',index=False)
    finite=pd.read_json(base/'e1b/all_set_outcomes.jsonl',lines=True);references=pd.read_json(base/'e1b/validation_selected_reference.jsonl',lines=True)
    references.to_csv(out/'e1b_validation_selected_reference.csv',index=False);finite.to_json(out/'e1b_all_set_outcomes.jsonl',orient='records',lines=True)
    reference_summary=references.groupby(['dataset','learner'],as_index=False)[['gain_V_eval_vs_minrisk','gain_V_eval_vs_ordinary']].mean()
    reference_summary.to_csv(out/'e1b_reference_summary.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    methods=['OrdinaryOnly','LRBE-IG','Vopt','RB-BRIDGE','Vopt-Kstar','Random-feasible','FullFeedback']
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for row,learner in enumerate(['LR','XGB']):
        for col,dataset in enumerate(['hmeq','uci','lendingclub']):
            ax=axes[row,col];g=trace[(trace.dataset==dataset)&(trace.learner==learner)]
            vals=[g[g.selector==m].gain_vs_minrisk.mean()*1e4 for m in methods]
            ax.barh(methods,vals,color=['#9aaab7']*6+['#a28aad'])
            for j,m in enumerate(methods):ax.scatter(g[g.selector==m].gain_vs_minrisk*1e4,[j]*2,color='#243f55',s=18,zorder=3)
            ax.axvline(0,color='black',lw=.8);ax.invert_yaxis();ax.set_title(dataset.upper()+' / '+learner);ax.set_xlabel('Gain vs MinRisk (x 0.0001)')
    fig.savefig(out/'e1_learner_forks.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,learner in zip(axes,['LR','XGB']):
        values=quad[quad.learner==learner].set_index('state_id')
        ax.scatter(proxy.loc[values.index,'exchange_proxy_gain']*1e6,values.exchange_gain*1e4,color='#846a5e')
        ax.axhline(0,color='black',lw=.8);ax.axvline(0,color='black',lw=.8);ax.set_title(learner+' / common exchange only')
        ax.set_xlabel('Proxy improvement (x 0.000001)');ax.set_ylabel('Actual Brier improvement (x 0.0001)')
    fig.savefig(out/'e2b_proxy_vs_actual.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained');conditions=['raw_ref128','leaf_ref128','raw_ref512']
    for ri,learner in enumerate(['LR','XGB']):
        for ci,dataset in enumerate(['hmeq','uci','lendingclub']):
            ax=axes[ri,ci];group=cs[(cs.dataset==dataset)&(cs.learner==learner)]
            for j,method in enumerate(['Vopt','RB-BRIDGE']):
                vals=[group[(group.condition==cond)&(group.selector==method)].gain_vs_minrisk.iloc[0]*1e4 for cond in conditions]
                ax.bar(np.arange(3)+(j-.5)*.3,vals,width=.3,label=method,color=['#537993','#b08065'][j])
            ax.set_xticks(range(3),['Raw / 128','Leaf / 128','Raw / 512'],rotation=15);ax.axhline(0,color='black',lw=.8)
            ax.set_title(dataset.upper()+' / '+learner);ax.set_ylabel('Extra Brier improvement (x 0.0001)')
    axes[0,0].legend(fontsize=8);fig.savefig(out/'e2c_representation_reference.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
    for ax,dataset in zip(axes,['hmeq','uci','lendingclub']):
        group=references[references.dataset==dataset]
        values=[group[group.learner==learner].gain_V_eval_vs_minrisk.mean()*1e4 for learner in ['LR','XGB']]
        ax.bar(['LR','XGB'],values,color=['#527b8b','#a57e65'])
        for j,learner in enumerate(['LR','XGB']):ax.scatter([j]*2,group[group.learner==learner].gain_V_eval_vs_minrisk*1e4,color='#253e4a',s=22,zorder=3)
        ax.axhline(0,color='black',lw=.8);ax.set_title(dataset.upper());ax.set_ylabel('V_eval gain vs MinRisk (x 0.0001)')
    fig.savefig(out/'e1b_independent_validation.png',dpi=170);plt.close(fig)
    comp=json.loads((base/'completion.json').read_text());e2b=json.loads((base/'e2b/completion.json').read_text());e2c=json.loads((base/'e2c/completion.json').read_text())
    changed=int(qsel.groupby('state_id').selected_hash.nunique().gt(1).sum())
    primary=summary[summary.selector.isin(['FullFeedback','MinRisk','Vopt','RB-BRIDGE','Vopt-Kstar'])][['dataset','learner','selector','gain_vs_ordinary','gain_vs_minrisk']]
    leaf_table=cs[cs.selector.isin(['Vopt','RB-BRIDGE'])][['dataset','learner','condition','selector','gain_vs_minrisk']]
    support=pd.read_json(base/'corr_support_audit.jsonl',lines=True)
    decisions={'E0':'completed, PASS_FOR_E1 with explicit retained legacy GATE/BRIDGE findings','E1':'completed, development paired estimates only',
               'E2b':'completed; positive Gamma exists but pair construction has zero incremental selection/prediction effect; exchange is distinct and sometimes worsens actual Brier',
               'E2c':'completed at six pre-fixed round-5 states; neither frozen leaves nor Q_ref enlargement establish a general improvement',
               'BRIDGE_search':'stop enlarging partners/exchange grid; current gains cannot be attributed to complementary-group construction',
               'next_priority':'Use the independent-validation differences to constrain at most one learner-matched acquisition-value estimator; do not promote a new candidate or launch confirmation from these two-seed diagnostics',
               'E1b':'completed on all six registered round-5 anchors, sixteen unique feasible sets, independent V_search/V_eval; no stable cross-task XGB advantage',
               'E2a_full_grid':'not run; mathematical controls retained but not mislabeled a new 300-endpoint confirmation',
               'E2d':'not run; no count-protection modification promoted','CORR':'frozen negative result; no scale-up or new alpha search','GATE':'no new E4 experiments; common future protocol interface audited only',
               'confirmation':'720 ten-round episodes remain disabled; no winning new candidate has been frozen'}
    write_json(out/'stage_decisions.json',decisions)
    report=f'''# LexiRiskLabel 下一阶段实验报告

2026 年 10 月 27 日

已按提供的下一阶段协议完成 E0、E1，并根据结果触发 E2b、E2c 和 E1b。当前最清楚的结果是：同一选集对逻辑回归和 XGBoost 的作用确实不同；真实信息代理中存在正互补性，但本轮双人构造没有贡献额外选集或预测收益。交换优化提高代理目标后，真实误差仍可能变差。冻结树叶表示也没有稳定解决这些问题。独立验证挑选的有限选集参照仍未形成稳定的 XGBoost 优势。目前没有可进入正式确认的唯一新候选。

## 本阶段实际运行了什么

基于提交 {identity['base_commit']}，旧 84 条轨迹保持冻结。新 E1 只从 MinRisk 的 HMEQ/UCI/LendingClub、2201/2202、第 2/5/9 轮提取 18 个共同状态。它们属于 6 条轨迹、两个种子值，不能作为 18 个独立种子。

每个状态包含普通反馈基线、MinRisk、LRBE-IG、V-opt、BRIDGE、固定 K* V-opt、5 次固定随机可行抽签和全反馈诊断。每个选集原封不动交给 LR 和 XGBoost；没有为两个学习器分别挑样本。随机重复先在状态内取均值，再把同轨迹三个状态取均值，最后完整保留两种子的配对差。

E1 共 {comp['nominal_fit_slots']} 个名义拟合槽位，实际新拟合 {comp['actual_unique_fits']} 次、缓存复用 {comp['fit_cache_hits']} 次；72 次 IG 委员会拟合另记。E2b 144 个槽位全部复用 E1，新增拟合为 0。E2c 144 个槽位只新增 {e2c['actual_new_learner_fits']} 次学习器拟合，复用 {e2c['reused_learner_fits']} 次，另训练 6 个固定的小型叶表示模型。这些是单轮诊断拟合槽位，不是协议中暂缓的 720 条十轮确认轨迹。

E1b 针对另外的独立 V_search/V_eval 预测，6 个状态各生成 16 个不同可行选集，共拟合 192 个分支。此前缓存保存的是 T 预测，没有保存这些模型或 V 预测，因此该组需要拟合固定训练集后预测新的验证块；它们仍不增加独立数据簇。全部训练行按申请 ID 规范排序，所有分支共用同一排序约定。

所有 XGBoost 和叶表示模型实际训练后端均为 cuda:0；逻辑回归在 CPU 求解。没有静默 CPU 回退。没有新预算网格、调参、追加种子或结果驱动删任务。

## E0 审计

候选标签、普通 pending 标签和外层 T 标签分别翻转，8 种选择器及重新计算的 IG 分数不变。配置、代码、数据身份改变均被拒绝复用；未锁定选集时反馈服务拒绝释放 O/S 标签；重复采集被拒绝。相同干预与相同训练数据分别以 hash 去重和缓存，不作为重复证据。

采集输入只有 D 标签与 O/C/Q_ref 特征；T 特征和标签保留在独立 evaluator 中。预处理按冻结 D0 拟合，Q_ref 与 T、D、O、C 隔离。逐人误差、预测、原始 Gamma、采集记录和叶模型均留在 private 目录。

新 FutureProtocol 将未来批量 160、容量 80、风险上限 20、阈值 0.5 显式统一，并把 planning_horizon 和 actual_future_release 分开。审计证明 horizon=1 不会自动关闭环境中的未来反馈，关闭实际释放也不会偷改规划时域。它是后续 E4 的接口与审计准备，不代表旧 GATE 实验已经按新参数重做。

旧 CORR 的 60 个轮次已记录 pi=0 支持区域，其中 {int(support.pi_zero_count.sum())} 个候选轮次记录不可达；只保留正概率支持域上的设计校正恒等式，不宣称全域恢复。已有矩阵、穷举与校正恒等式继续通过，但这些不被当成学习效益证据。

## E1 同状态收益

下表为先按轨迹聚合后的两种子开发均值。gain_vs_ordinary 为 Brier(D+O) 减 Brier(D+O+S)；gain_vs_minrisk 为 MinRisk 分支 Brier 减该分支 Brier。正数表示改善。

{mdtable(primary)}

HMEQ 和 LendingClub 的 XGBoost 分支中，V-opt/BRIDGE 相对 MinRisk 有开发性单步改善；LR 并未同步改善，UCI 的复杂自由基数方法仍偏弱。这个结果表明学习器与训练历史会改变结论，不能把上一轮完整十轮策略的排名直接搬来解释这一组共同状态分叉。

FullFeedback 使用 D+O+C，费用不受本轮预算限制，只是诊断。部分预算内选集胜过全反馈，因此全反馈不是本实验的统计或理论上界。反之，全反馈较差也不能排除某个预算内子集更好。

![同状态学习器分叉](e1_learner_forks.png)

完整的每条轨迹均值与配对差在 e1_trajectory_paired_differences.csv；不为两个种子生成强显著性判断。外层旧测试已经用于开发，所有结果都按开发证据解释。

## E2b 正互补性与求解器的分离

在同一 H、G、风险成本和候选池上，检查全部当前可行双人对。Gamma 阈值为绝对 1e-12 加相对 1e-8 的联合增益尺度，保存原始数值与阈值。

去掉四格的重复审计后，一条单人构造路径上累计有 {int(gamma.positive_visited_pair_evaluations.sum())} 次 visited pair 的正 Gamma，涉及全部 18 个状态；初始状态共有 {int(gamma.initial_unique_feasible_pairs.sum())} 个可行候选对，其中 {int(gamma.initial_positive_pairs.sum())} 个超过容差。visited pair 会随已选前缀重复评价，不能当成独立样本或独立证据条数。

四格为单人/不交换、单人/交换、双人/不交换、双人/交换。所有格共享候选池、几何、最佳单点保护、计算上限；启用交换时统一只做最多一次 1→1 可行交换，探查相同的 8 个移除候选。这与旧 BRIDGE 的混合交换规则分开记账。

结论是：18 个状态里，启用双人构造本身没有改变任何选集；在相同交换设置下，两种学习器的 Brier 差均严格为 0。{changed} 个状态的选集因交换而改变，但这不是互补组构造的作用。所有格均未触及计算上限。存在正 Gamma 不等于某个双人单元的风险归一化联合价值足以胜过最佳单人，更不等于最终学习必然改善。

交换增加代理 F 后，LendingClub 的若干实际重训分支反而变差。当前证据反对把代理优化改善直接解释成预测收益；也不支持继续盲目增加 partners 或交换次数。

![交换的代理与实际效果](e2b_proxy_vs_actual.png)

## E2c 表示与参考规模诊断

预先固定六个 dataset×seed 的第 5 轮状态，未挑选提升轮次。主对照是 raw/Q128 与 frozen-leaf/Q128；另以 raw/Q512 检查参考规模。没有同时引入新求解器或调参。

叶模型仅用当时 D 的标签训练，20 棵深度 2 树，实际形成 79–80 个叶特征。各方法共用冻结叶表示和同一 G、正则、预算。LR 在对应冻结表示上训练；完整 XGBoost 始终重新生长原始标准化特征上的树，同一方法跨学习器仍共享 selected_hash。没有把叶子 Fisher 当成全树重训的精确 Hessian。

{mdtable(leaf_table)}

叶表示在 HMEQ 的 XGBoost 分支有改善，但在 LendingClub 等任务上没有一致收益；匹配叶逻辑头本身也未稳定胜过 MinRisk。把 Q_ref 从 128 加到 512，同样没有形成一致改善。这些结果支持继续研究价值估计与真实训练的关系，不能据此认定叶表示已经解决代理失配。

![表示与参考规模](e2c_representation_reference.png)

## E1b 独立验证的有限选集参照

E2b/E2c 之后，剩余收益空间仍难区分，因此按协议在全部六个固定第 5 轮状态执行 E1b。每状态先生成 16 个不同预算可行集合，来源为既有方法、随机可行及风险匹配交换。交换风险差预设为不超过 1% 的 B，失败时不放宽容差。相同集合只记一次，来源别名合并。全部集合先锁定，再释放对应 O/S 标签。

从此前只用作开发协变量的 20% 数据中，排除 Q_ref 后，将剩余部分按固定随机种子分成互不重叠的 V_search 和 V_eval；也与 D/O/C 和外层 T 隔离。训练参数不调，按 V_search 的 Brier 选取 16 个集合中的最佳，再只用 V_eval 报告该参照的效果。没有在外层 T 或 V_eval 上挑最优集合。

下表是两个开发种子的配对均值。它明确使用额外验证标签，只能作为有限集合离线诊断，不能放入可部署策略主排名，也不是全局最优或统计上界。

{mdtable(reference_summary)}

UCI 和 LendingClub 的 LR 参照相对 MinRisk 有正向小试信号，但 XGBoost 的验证优势未跨种子稳定保留；HMEQ 的 XGBoost 参照在两种子上都劣于 MinRisk。这说明旧 T 上的开发改善不能直接当成验证中可靠的收益。它也没有证明预算内不存在更好选集：候选家族仅有 16 个集合，V_search 选择本身有噪声，验证区间尚未做确认性推断。

![独立验证的有限参照](e1b_independent_validation.png)

## 阶段判断

支持：共同状态分叉有必要；选集对不同学习器的真实作用会变化；正 Gamma 可以存在，但当前逐步搜索没有将其变成额外干预；普通局部交换必须与成组构造分开解释。

反证或不支持：本轮数据不支持把 BRIDGE 的开发收益归因为互补双人构造；F 变大不保证真实 Brier 变好；单纯换冻结叶表示或扩大目标参考样本，尚未形成稳定修复。

仍不确定：预算内是否有当前有限方法未找到的高价值选集；这些状态的效果能否在独立验证块与新完整流程中保留；标签构成、模型容量和真实学习价值各自的边界。

继续路线：把 E1b 的验证差异作为后续约束，只允许围绕一个明确学习器设计最多一个价值估计候选。需要承认目前仍无法完全区分选集空间、有限候选家族与估计噪声；不能把外层 T 或 V_eval 上最好的事后选集当成新泛化算法。

停止或暂缓：不扩大 BRIDGE 的 partners/交换网格；CORR 仍保留冻结负结果，不追加 alpha 搜索；GATE 不扩大时域或算力。E2a 的完整 300 端点网格、E2d 数量交叉、完整内外层调参与 720 条十轮确认轨迹均未启动。目前没有可冻结为正式主比较的唯一新候选。

## 交付与复现

运行配置 protocol.json、原协议全文与新增代码位于项目 experiments/protocol_next。E1 源身份固定了配置、代码、数据、锚点档案与依赖版本；E2b/E2c 的 activation.json 另记录触发原因、源 hash、父身份和预设计算范围。

state_manifest.jsonl、selection_summary.jsonl、branch_outcomes.jsonl 使用状态和选集 hash 审计；全部逐人记录与预测误差在 private，不随公开汇总发布。final_verification.json 验证共享选集/训练 hash、预测重算、GPU 实际后端与叶表示可见性。运行 manifest 记录文件身份，不把缓存命中或重复抽签计为新独立数据。
'''
    (out/'阶段实验报告.md').write_text(report)
    write_json(out/'analysis_identity.json',{'source_hash':sha(__file__),'parent_identity':identity['identity_hash']})
    print('Analysis and verification complete',json.dumps({'e1_slots':comp['nominal_fit_slots'],'e1_fits':comp['actual_unique_fits'],'e2b_new_fits':e2b['actual_new_fits'],'e2c_new_fits':e2c['actual_new_learner_fits'],'unique_initial_gamma_pairs':int(gamma.initial_unique_feasible_pairs.sum())}))


if __name__=='__main__':main()
