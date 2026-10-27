"""Aggregate E2d paired outcomes without selecting a winning configuration on T."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from runtime import write_json,sha


def table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:+.8f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base
    out=base/'e2d';analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    d=pd.read_json(out/'branch_outcomes.jsonl',lines=True)
    sel=pd.read_json(out/'selection_summary.jsonl',lines=True)
    matched=pd.read_json(out/'matching_status.jsonl',lines=True)
    state=pd.read_json(out/'state_manifest.jsonl',lines=True)
    completion=json.loads((out/'completion.json').read_text())
    mr=d[d.method=='MinRisk-k'][['state_id','learner','k','brier']].rename(columns={'brier':'minrisk_k_brier'})
    within=d[d.method.isin(['IG-k','Vopt-k'])].merge(mr,on=['state_id','learner','k'],validate='many_to_one')
    within['gain_vs_minrisk_same_k']=within.minrisk_k_brier-within.brier
    within.to_csv(analysis/'same_count_paired.csv',index=False)
    within_seed=within.groupby(['learner','dataset','seed','method'],as_index=False).agg(
        gain=('gain_vs_minrisk_same_k','mean'),cells=('k','size'),wins=('gain_vs_minrisk_same_k',lambda x:int((x>1e-12).sum())))
    summary=within_seed.groupby(['learner','dataset','method'],as_index=False).agg(
        gain=('gain','mean'),seed_clusters=('seed','size'),cells=('cells','sum'),wins=('wins','sum'))
    summary.to_csv(analysis/'same_count_summary.csv',index=False);within_seed.to_csv(analysis/'same_count_by_seed.csv',index=False)
    swaps=[]
    for method in ['IG-k','Vopt-k']:
        t=d[d.method==method];r=d[d.method=='RiskMatched-'+method]
        z=t.merge(r,on=['state_id','dataset','seed','learner','k'],suffixes=('_target','_random'),validate='one_to_one')
        z['target_method']=method;z['gain_vs_matched_swap']=z.brier_random-z.brier_target
        z['risk_difference']=z.cost_target-z.cost_random;swaps.append(z)
    swaps=pd.concat(swaps,ignore_index=True);swaps.to_csv(analysis/'risk_matched_paired.csv',index=False)
    # Local matched-cell summaries explicitly retain missing states; no imputation.
    swap_summary=swaps.groupby(['learner','dataset','target_method'],as_index=False).agg(
        gain=('gain_vs_matched_swap','mean'),cells=('k','size'),seed_clusters=('seed','nunique'),
        wins=('gain_vs_matched_swap',lambda x:int((x>1e-12).sum())))
    swap_summary.to_csv(analysis/'risk_matched_summary.csv',index=False)
    gradient=[]
    for (sid,learner,method),v in d[d.method.isin(['MinRisk-k','IG-k','Vopt-k'])].groupby(['state_id','learner','method']):
        v=v.sort_values('k');gradient.append({'state_id':sid,'dataset':v.dataset.iloc[0],'seed':int(v.seed.iloc[0]),
              'learner':learner,'method':method,'low_k':int(v.k.iloc[0]),'Kstar':int(v.k.iloc[-1]),
              'low_brier':float(v.brier.iloc[0]),'Kstar_brier':float(v.brier.iloc[-1]),
              'gain_low_to_Kstar':float(v.brier.iloc[0]-v.brier.iloc[-1]),
              'monotone_nonincreasing_brier':bool(np.all(np.diff(v.brier)<=1e-12))})
    gradient=pd.DataFrame(gradient);gradient.to_csv(analysis/'quantity_gradient.csv',index=False)
    resource=sel[sel.method.isin(['MinRisk-k','IG-k','Vopt-k'])].copy()
    resource['risk_budget_fraction']=resource.cost/resource.budget
    resource['premium_budget_fraction']=resource.composition_premium/resource.budget
    resource.to_csv(analysis/'resource_costs.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for j,dataset in enumerate(['hmeq','uci','lendingclub']):
        for i,seed in enumerate([2201,2202]):
            ax=axes[i,j];v=d[(d.dataset==dataset)&(d.seed==seed)&(d.learner=='XGB')]
            baseline=float(v[v.method=='OrdinaryOnly'].brier.iloc[0])
            for method in ['MinRisk-k','IG-k','Vopt-k']:
                q=v[v.method==method].sort_values('k');ax.plot(q.k,1000*(baseline-q.brier),marker='o',label=method)
            ax.axhline(0,c='grey',lw=.7,ls='--');ax.set_title(f'{dataset.upper()}, seed {seed}')
            ax.set_xlabel('Acquired label count k');ax.set_ylabel('Brier decrease vs OrdinaryOnly (x1000)')
    axes[0,0].legend(fontsize=8);fig.savefig(analysis/'quantity_composition_xgb.png',dpi=160);plt.close(fig)
    xgb=summary[summary.learner=='XGB'][['dataset','method','gain','wins','cells']]
    lr=summary[summary.learner=='LR'][['dataset','method','gain','wins','cells']]
    sw=swap_summary[swap_summary.learner=='XGB'][['dataset','target_method','gain','wins','cells','seed_clusters']]
    ks=state[['dataset','seed','Kstar','unique_k']]
    gx=gradient[gradient.learner=='XGB'];improved=int((gx.gain_low_to_Kstar>1e-12).sum())
    mono=int(gx.monotone_nonincreasing_brier.sum())
    fail=matched[matched.status=='not_estimable'][['state_id','k','target_method']]
    report=f'''# E2d：数量与构成的固定状态诊断

结论：本轮没有支持“只要增加探索标签数量就能修复第一轮结果”的一致证据，也没有确认一种稳定优于 MinRisk 的标签构成规则。固定数量后，Vopt 的 XGB 平均 Brier 略有改善，但各数据集都只有 3/6 个状态—数量组合改善；局部近似同风险交换比较也随数据集、种子变化。暂不更换数据集、不扩大 BRIDGE 搜索、不按这次测试结果挑选最优 k。

## 范围与执行

严格使用协议 E2d，3 个原有数据集、2 个原有种子，只取第 5 轮的 6 个共同状态。每个状态比较 floor(0.6Kstar)、floor(0.8Kstar)、Kstar，去重后共 18 个状态—数量组合，另保留 OrdinaryOnly 基准。

{table(ks,['数据集','种子','Kstar','实际 k'])}

MinRisk-k 取最小风险的 k 人；IG-k 在共同预算内以等式约束强制恰好 k 人，MILP 均达到最优；Vopt-k 使用原有条件几何、单位风险边际价值及剩余数量可行性约束，不启用交换。IG 复用同一状态已有的 D-only 委员会分数。所有选集均先锁定，再释放标签；LR 与 XGB 用同一选集。

共有 {completion['selection_slots']} 个方法动作槽、{completion['nominal_fit_slots']} 个学习器拟合槽，实际新增 {completion['actual_new_fits']} 次拟合、复用 {completion['reused_fits']} 次；不同训练干预共 {completion['unique_training_interventions']} 个。这些都不是独立统计样本。XGB 实际后端为 CUDA，LR 使用 CPU。代码、原始输入、锚点和 E1 身份核验通过，保存预测重新计算的 Brier 一致。

## 同数量比较

下表增益 = MinRisk-k 的 Brier − 该方法 Brier，正数表示改善。先在每条种子轨迹内平均 3 个 k，再平均 2 条轨迹；改善次数仅作描述。完整风险、预算使用率、相对同数量最小风险集合的构成溢价保存在 resource_costs.csv。同数量不代表同风险。

XGBoost：

{table(xgb,['数据集','方法','Brier 增益','改善次数','组合数'])}

逻辑回归：

{table(lr,['数据集','方法','Brier 增益','改善次数','组合数'])}

在 XGB 的 18 条“状态×方法”数量曲线中，从最低 k 增至 Kstar 后有 {improved} 条改善，{18-improved} 条变差；完整三个点单调改善的只有 {mono} 条。因此不能把第一轮结果统一归因为探索标签太少。这里数量变化伴随具体选集和实际风险变化，仍不是纯粹数量的因果效应。

![数量—构成曲线](quantity_composition_xgb.png)

## 近似同风险的随机交换

在看到本轮终点前固定容差为每状态预算 B 的 1%。对 IG-k、Vopt-k 各自枚举同数量的一换一集合，仅保留预算可行且与目标选集实际风险差不超过 0.01B 的交换，使用固定种子均匀抽取一个。只做一次局部对照，不扩大为随机重复网格，不根据标签或测试结果选交换。

36 个目标中 {completion['matched_controls']} 个成功、{completion['not_estimable_controls']} 个不可估计。下表仅对成功匹配的状态—数量组合取平均，缺失不插补；这是局部交换结果，不能推广为全空间随机策略的优劣。部分行只覆盖一个种子，不能与其他行直接比较总体大小。

下表增益 = 随机交换集合 Brier − 目标集合 Brier，正数表示目标构成较好（XGB）：

{table(sw,['数据集','目标方法','相对交换增益','改善次数','匹配组合数','覆盖种子数'])}

无法匹配的具体组合：

{table(fail,['状态','k','目标方法'])}

LendingClub 的 Vopt 局部匹配只覆盖种子 2201，种子 2202 全部匹配失败。不能据此声称在 LendingClub 已确认构成优势。

## 对下一步的约束

这些结果收窄了原因：部分状态下标签构成确实改变了结果，但数量增加和代理价值更高都没有一致转化为 XGB 改善。数据规模可能限制可检测性，却不能作为全部失败的解释。

本阶段只完成 E2d，不启动 720 条确认轨迹、不更换数据集、不增大搭档或交换次数。当前不足以冻结一个新 BRIDGE 候选；保留旧负结果和本轮全部组合，后续若继续协议，应进入已规定的 CORR/GATE 小型机制诊断，不能把这轮测试集上最好的 k 包装成新的推荐策略。

每数据集只有两个开发种子；同一状态内各分支共享外部测试集，三个 k 也共享同一锚点，不能按 18 或 91 个独立样本计算显著性。本报告不提供确认性 p 值或宣称稳定胜出。个体 ID、标签、逐条预测和选集仅保存在本地 private/，交付压缩包只有代码与汇总证据。
'''
    (analysis/'E2d_数量与构成实验报告.md').write_text(report)
    write_json(analysis/'decision.json',{'E2d_complete':True,'new_candidate_frozen':False,'switch_dataset_now':False,
       'expand_pair_search':False,'confirmation_started':False,'two_seed_development_only':True,
       'quantity_improved_curves_XGB':improved,'quantity_total_curves_XGB':18,'monotone_improving_curves_XGB':mono,
       'analyzer_hash':sha(__file__)})
    print(report)


if __name__=='__main__':main()
