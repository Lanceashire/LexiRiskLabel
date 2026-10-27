"""Finalize the two subsequent registered blocks, without fitting models."""
from runtime import *
import argparse


def table(df):
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:+.8f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def nearest_report(base):
    out=base/'nearest_neighbor_small';private=base/'private/nearest_neighbor_small'
    rows=pd.read_csv(out/'branch_outcomes.csv');summary=pd.read_csv(out/'descriptive_summary.csv');locks=pd.read_json(out/'locked_actions.jsonl',lines=True)
    oldv=pd.read_json(base/'e1b/all_set_outcomes.jsonl',lines=True);oldt=pd.read_json(base/'branch_outcomes.jsonl',lines=True)
    error=0.
    for r in rows.itertuples():
        if r.reused:
            v=oldv[(oldv.state_id==r.state_id)&(oldv.learner==r.learner)&(oldv.selected_hash==r.selected_hash)].iloc[0]
            t=oldt[(oldt.state_id==r.state_id)&(oldt.learner==r.learner)&(oldt.selected_hash==r.selected_hash)].iloc[0]
            vs=np.load(base/'private/e1b_predictions'/f'{r.state_id}_set{int(v.set_id)}_{r.learner}.npz')
            ts=np.load(base/'private/predictions'/f'{t.fit_key}.npz')
            vb=float(np.mean((vs['eval_prediction']-vs['eval_y'])**2));tb=float(np.mean((ts['prediction']-ts['y'])**2))
        else:
            p=np.load(private/f'{r.state_id}_{r.learner}_predictions.npz')
            vb=float(np.mean((p['eval_prediction']-p['eval_y'])**2));tb=float(np.mean((p['T_prediction']-p['T_y'])**2))
        error=max(error,abs(vb-r.V_eval_brier),abs(tb-r.T_brier))
    assert error<1e-12
    new=rows[rows.method=='ParBaLS-EPIG-Laplace-cost-pending']
    same=[]
    for r in new[new.learner=='XGB'].itertuples():
        v=rows[(rows.state_id==r.state_id)&(rows.learner=='XGB')&(rows.method=='Vopt')].iloc[0]
        same.append({'state_id':r.state_id,'dataset':r.dataset,'seed':r.seed,'same_as_Vopt':r.selected_hash==v.selected_hash})
    pd.DataFrame(same).to_csv(out/'same_action_vs_Vopt.csv',index=False)
    comp=json.loads((out/'completion.json').read_text());ver=json.loads((out/'verification.json').read_text());assert ver['passed']
    report=f'''# 真实任务近邻比较：ParBaLS-EPIG 的成本与 pending 适配

## 在计划中的位置

本轮继续原重构DOCX第8.3节的近邻比较，先执行六个已经固定的真实round-5共同状态：HMEQ/UCI/LendingClub各两个开发种子。并非重新扫描全部轨迹、预算或超参数。全部采集动作先锁定，再读取V/T结果。

按[论文](https://arxiv.org/abs/2510.09877v4)及[作者代码](https://github.com/ADDAPT-ML/ParBaLS/blob/b3dcc7a76528c311a73d753e4385aca973ffac65/LabelBench/strategy/strategy_impl/eer.py)核对部分批伪标签和EPIG计算。官方当前代码的10个pseudo universes及每个参数样本内argmax伪标签约定予以保留；联合为O与C构造每个世界，所有世界只来自可见D。每次加入候选后，用D+幻想O+幻想已选C重新拟合并平均下一步EPIG。

与本地Vopt共享固定特征、ridge=1、Q_ref及普通反馈特征，目标参考不另外截断、候选不预筛。异质风险成本下用完整预测互信息除以风险价格。作者源码在等成本argmax时可省略目标熵常数；除以不同成本后不能省略该项，因此本成本适配补回它。独立互信息公式复核及等成本排序一致检查通过。

本实现用32个Laplace后验样本，不是作者完整NUTS后验或原论文整套复现。名称明确为ParBaLS-EPIG-Laplace-cost-pending，不把本结果当作对原版算法优劣的最终结论。上游eer.py与bayesian.py按Git blob SHA核验并保留MIT许可，区别可审阅。

## XGBoost 主要开发终点

正增益表示对照误差减所比较方法误差。neighbor_vs_MinRisk为近邻胜过MinRisk的幅度；Vopt_vs_neighbor/BRIDGE_vs_neighbor为本地方法胜过近邻的幅度。V_eval为主要开发终点，T为次要结果。

{table(summary[(summary.learner=='XGB')&summary.contrast.isin(['neighbor_vs_MinRisk','Vopt_vs_neighbor','BRIDGE_vs_neighbor'])][['dataset','contrast','V_eval_gain','T_gain','positive_seeds','negative_seeds']])}

在HMEQ及LendingClub的四个状态，近邻与Vopt是完全相同的选集；只有UCI两个状态不同。不能把相同动作下多个方法名当成独立方法胜负证据。近邻相对MinRisk的XGB验证均值HMEQ略正、另两个任务为负；每任务都是一正一负。Vopt与近邻在UCI也各一正一负，没有建立跨任务稳定优越。

全部LR结果、每个种子的采集人数及风险差保存在CSV。风险差不能跨数据集货币单位直接相加；同预算不是同花费。

## 复用和核验

实际只新增{comp['new_unique_LR_fits']}个LR与{comp['new_unique_GPU_fits']}个GPU真实结果模型，复用{comp['reused_actual_reference_slots']}个既有评价槽。另有{comp['internal_hypothetical_LR_refits']}个内部幻想LR重拟合，是采集算法计算，不是330个基准、数据重复或独立样本。复用/新预测全部重新计分，最大误差{error:.3g}。

普通/探索标签始终对采集函数隐藏；选集哈希与原方法引用核验通过；实际新XGB训练为CUDA。V_eval和T都曾用于开发，不称新盲测或确认性统计。该比较补上有限状态的公平信息对照；完整轨迹、公平算力配置及原版MCMC近邻的确认仍未完成。

本轮没有按验证结果改后验样本数、伪标签数或候选池。继续原计划中的普通反馈规模、成本关联与非线性错设控制；不以本次有没有新方法获胜作为执行这些实验的前提。
'''
    (out/'真实任务近邻适配比较_实验报告.md').write_text(report)
    write_json(out/'analysis_verification.json',{'passed':True,'all_reused_and_new_predictions_rescored':True,'maximum_error':error,
        'same_as_Vopt_states':sum(r['same_as_Vopt'] for r in same),'analyzer_sha256':sha(__file__)})
    write_json(out/'decision.json',{'continue_original_plan':True,'new_candidate_promoted':False,'full_trajectory_comparison_completed':False,
        'original_NUTS_baseline_reproduction_completed':False,'next':'ordinary volume and cost-association controls'})


def volume_report(base):
    out=base/'approval_volume_cost';private=base/'private/approval_volume_cost'
    rows=pd.read_csv(out/'new_branch_outcomes.csv');summary=pd.read_csv(out/'descriptive_summary.csv')
    error=0.
    for r in rows.itertuples():
        p=np.load(private/'predictions'/f'{r.evaluation_key}.npz')
        error=max(error,abs(float(np.mean((p['prediction']-p['pTest'])**2))-r.excess_brier),
            abs(float(np.mean((p['prediction']-p['yTest'])**2))-r.brier))
    assert error<1e-12
    diag=pd.read_json(out/'cost_marginal_diagnostics.jsonl',lines=True)
    for r in diag.itertuples():assert r.positive_abs_target_exposure_correlation>r.neutral_abs_target_exposure_correlation
    comp=json.loads((out/'completion.json').read_text());ver=json.loads((out/'verification.json').read_text());assert ver['passed']
    view=summary[(summary.method=='C1B1')|((summary.contrast=='ordinary_4000_vs_500')&(summary.method=='OrdinaryOnly'))]
    report=f'''# 普通反馈规模与成本关联：原计划后续机制对照

继续原DOCX第8.2节，不换数据集或另起算法。复用已完成的500条普通反馈低/高覆盖条件，补4000条对应条件；500条是4000条的逐行嵌套前缀，标签也严格相同。固定D、C、Q、测试和预算。数量和方向形成2×2关键对照。

成本关联只在事先选定的500条、低目标覆盖状态追加一组，未展开全部笛卡尔积。在五个固定业务PD排序分层内，将原暴露多重集按|候选目标坐标|排序分配，保持暴露多重集、候选、标签及数值预算完全相同。成本为PD×新暴露，所以最终成本多重集不严格相同；完整均值、分位数及总成本变化在cost_marginal_diagnostics.jsonl披露，没有重新按总成本缩放B。

方向重要性由预定义生成坐标给定，不用待比较算法的分数或隐藏标签构造。本实验保持所有普通样本满足同一审批门、候选位于拒绝区域。

## 结果

ordinary_4000_vs_500：500条状态误差−4000条状态误差，正数表示更多普通反馈改善。
exploration_gain_4000_minus_500：相对对应参照的探索增益，在4000条与500条间的差，负数表示额外探索收益缩小。
cost_positive_vs_neutral：中性成本状态误差−正关联成本状态误差，负数表示正关联时表现变差。

{table(view[['learner','coverage','contrast','method','mean_gain','positive_seeds','negative_seeds']])}

当普通反馈直接覆盖目标方向时，从500增加至4000后，完整方法相对OrdinaryOnly的探索增益平均缩小：LR约0.000502，XGB约0.003438。LR三个种子方向一致，XGB只有一个种子缩小、两个反向，均值并不代表多数种子的同方向结果；逐种子表保留。

普通反馈位于低目标覆盖方向时，更多普通反馈并不保证实际模型更好：完整方法LR平均改善约0.000907，XGB反而平均变差约0.007485。不能把普通数量增加一概解释成信息充分或把异质结果删掉。

代表性成本配对中，将较大暴露分配给目标方向幅度更大的样本后，完整方法LR与XGB平均误差分别增加约0.001531、0.003253，两种学习器均1/3种子改善、2/3变差。由于PD分层只近似固定，最终成本边际有变化，这支持该价格安排下的有限机制结果，不是信息—价格关联的普遍因果定律。

## 核验与范围

本阶段新选择器槽{comp['new_selector_slots']}、新拟合槽{comp['new_fit_slots']}，实际{comp['actual_unique_LR_fits']}次LR、{comp['actual_unique_GPU_fits']}次CUDA XGB，内部缓存{comp['within_stage_cache_hits']}次，并复用此前{comp['inherited_actual_reference_slots']}个参考评价槽。独立种子仍为3个。逐预测重新计分最大误差{error:.3g}，嵌套标签、审批门、暴露多重集、固定预算、选择提前锁定与GPU后端检查通过。

本次未把两档方向、代表性价格控制说成完整14条件×30种子确认。中间覆盖水平、非线性错设子集及独立确认仍在计划中；后续正常按队列执行，已有正负结果均保留。没有新增方法或根据结果扩大网格。
'''
    # Verify the direction counts in prose against the actual paired rows.
    paired=pd.read_csv(out/'paired_contrasts.csv')
    x=paired[(paired.learner=='XGB')&(paired.coverage=='high_target_overlap')&
             (paired.contrast=='exploration_gain_4000_minus_500_vs_OrdinaryOnly')]
    assert (x.gain<0).sum()==1 and (x.gain>0).sum()==2
    (out/'普通反馈规模与成本关联_实验报告.md').write_text(report)
    write_json(out/'analysis_verification.json',{'passed':True,'all_new_predictions_rescored':True,'maximum_error':error,
        'positive_cost_association_all_seeds':True,'analyzer_sha256':sha(__file__)})
    write_json(out/'decision.json',{'continue_original_plan':True,'new_candidate_promoted':False,
        'next':'nonlinear misspecification subset and remaining scheduled development controls','full_14_condition_confirmation_completed':False})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);a=ap.parse_args()
    nearest_report(a.base);volume_report(a.base)
    for stage in ['nearest_neighbor_small','approval_volume_cost']:
        out=a.base/stage;(out/'source_snapshot/analyze_plan_continuation.py').write_bytes(Path(__file__).read_bytes())
        write_json(out/'evidence_manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*'))
            if p.is_file() and p.name!='evidence_manifest.json']})
    print('Reports completed; all saved predictions rescored; zero new fitting in analysis.')


if __name__=='__main__':main()
