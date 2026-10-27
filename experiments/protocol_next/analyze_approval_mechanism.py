from runtime import *
import argparse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();out=args.base/'approval_mechanism_small'
    frame=pd.read_csv(out/'branch_outcomes.csv');pairs=[]
    for (seed,coverage,learner),g in frame.groupby(['seed','coverage','learner']):
        q=g.set_index('method')
        for name,left,right in [('conditioning_static','C0B0','C1B0'),('conditioning_batch','C0B1','C1B1'),
            ('batch_unconditional','C0B0','C0B1'),('batch_conditional','C1B0','C1B1'),
            ('combined_vs_OrdinaryOnly','OrdinaryOnly','C1B1'),('combined_vs_MinRisk','MinRisk','C1B1')]:
            pairs.append({'seed':seed,'coverage':coverage,'learner':learner,'contrast':name,
                'excess_gain':q.loc[left,'excess_brier']-q.loc[right,'excess_brier'],
                'sampled_brier_gain':q.loc[left,'brier']-q.loc[right,'brier'],
                'changed_selected_set':q.loc[left,'selected_hash']!=q.loc[right,'selected_hash'],
                'count_difference':int(q.loc[right,'count']-q.loc[left,'count']),
                'cost_difference':q.loc[right,'cost']-q.loc[left,'cost'],
                'research_delta':.01*q.loc['OrdinaryOnly','brier']})
        pairs.append({'seed':seed,'coverage':coverage,'learner':learner,'contrast':'interaction',
            'excess_gain':(q.loc['C1B0','excess_brier']-q.loc['C1B1','excess_brier'])-(q.loc['C0B0','excess_brier']-q.loc['C0B1','excess_brier']),
            'sampled_brier_gain':(q.loc['C1B0','brier']-q.loc['C1B1','brier'])-(q.loc['C0B0','brier']-q.loc['C0B1','brier']),
            'research_delta':.01*q.loc['OrdinaryOnly','brier']})
    p=pd.DataFrame(pairs);p.to_csv(out/'paired_contrasts.csv',index=False)
    summary=p.groupby(['coverage','learner','contrast'],as_index=False).agg(mean_excess_gain=('excess_gain','mean'),
        positive_seeds=('excess_gain',lambda x:int((x>1e-12).sum())),negative_seeds=('excess_gain',lambda x:int((x< -1e-12).sum())),
        mean_sampled_brier_gain=('sampled_brier_gain','mean'),mean_count_difference=('count_difference','mean'),mean_cost_difference=('cost_difference','mean'))
    summary.to_csv(out/'descriptive_summary.csv',index=False)
    differences=[]
    for (seed,learner,contrast),g in p.groupby(['seed','learner','contrast']):
        q=g.set_index('coverage');differences.append({'seed':seed,'learner':learner,'contrast':contrast,
            'high_minus_low_excess_gain':q.loc['high_target_overlap','excess_gain']-q.loc['low_target_overlap','excess_gain']})
    pd.DataFrame(differences).to_csv(out/'coverage_direction_contrasts.csv',index=False)
    comp=json.loads((out/'completion.json').read_text());ver=json.loads((out/'verification.json').read_text());assert ver['passed']
    def table(q):
        lines=['| 条件 | 学习器 | 对照 | 平均改善 | 改善种子/3 |','| --- | --- | --- | --- | --- |']
        for r in q.itertuples():lines.append(f'| {r.coverage} | {r.learner} | {r.contrast} | {r.mean_excess_gain:+.8f} | {r.positive_seeds} |')
        return '\n'.join(lines)
    report=f'''# 审批约束下的普通覆盖方向机制实验

本轮按原重构DOCX第8.2节继续执行，修补上一小试没有业务审批门的范围缺口。固定三个既有开发种子，初始可见800、候选400、普通反馈500、参考1024、测试5000；每种子两个相同数量的普通覆盖条件，四格加OrdinaryOnly/MinRisk。新增{comp['actual_unique_LR_fits']}次LR、{comp['actual_unique_GPU_fits']}次实际CUDA XGB拟合，缓存{comp['cache_hits']}次；独立生成种子只有3个。

## 设计与审批

预定义固定业务评分为sigmoid(-1+审批坐标)，它同时决定普通/拒绝区域及候选风险价格。该评分不是已知真还款概率；真概率来自另一个预先固定的逻辑模型，只供环境生成和评价。可见与普通样本的审批坐标位于[-2,0.8]，候选位于[1.2,2.2]，因此所有普通样本通过0.5门、所有候选被拒。成本为业务PD×暴露，预算为候选总预测风险的10%。

两个普通条件保持审批坐标、业务PD、数量、随机方向幅度完全相同，仅将方向从无关轴旋转到目标轴。D、C、Q、测试、价格、暴露、预算相同。普通标签通过共同Bernoulli随机数生成；因真实条件概率随方向改变，标签允许不同，不强行固定隐藏结果。目标/无关方向区别预先由生成器定义，不按算法分数或结果挑方向。

四格仍共用同一个单点贪心求解器，只切换是否计入O与是否批内更新。全部选集锁定后才训练评价；真概率不进入选择。主机制终点是LR对真概率的excess Brier，同选集GPU XGB检验迁移，采样标签Brier完整保留。

## 对普通反馈以外的新增收益

正数表示参照误差减C1B1误差：

{table(summary[summary.contrast.isin(['combined_vs_OrdinaryOnly','combined_vs_MinRisk'])])}

低目标覆盖时，新增探索相对OrdinaryOnly的收益较大，但LR在三个种子都不及MinRisk。高目标覆盖时，普通反馈已提供较多目标信息，探索相对OrdinaryOnly的增益缩小；相对MinRisk的LR均值略正、XGB均值也略正，均仅2/3种子改善。这是本生成条件的开发结果，不是稳定优越或等效结论。

## 组件拆分

{table(summary[summary.contrast.isin(['conditioning_static','conditioning_batch','batch_conditional','interaction'])])}

条件化与批更新的效果随学习器、覆盖条件和种子变化；不能仅因正平均值就宣布两个组件总是互补。每个组件对照的数量与实际花费差保存在paired_contrasts.csv；同预算不冒充同花费。

## 完成范围与后续队列

审批门、共同价格、方向干预、空O零对照、预算及实际GPU后端检查通过。本次补的是500条普通反馈、低/高覆盖的关键配对，并非原14条件×30种子的完整确认；4000普通量、中间覆盖、成本关联置换、非线性错设子集与公平近邻比较仍分别登记为后续工作。

当前正常进入计划中的近邻比较，不要求先在这三个开发种子上获胜。旧CORR/GATE不扩模块，不据本结果添加新算法候选。完整本地环境、选集和预测保留，必要汇总加入审阅包。
'''
    (out/'审批约束普通反馈方向_实验报告.md').write_text(report)
    write_json(out/'decision.json',{'stage_completed':True,'next_stage':'same-state real-data cost/pending-adapted nearest-neighbor comparison',
        'continue_original_plan':True,'new_algorithm_candidate_promoted':False,'full_DOCX_confirmation_completed':False})
    (out/'source_snapshot/analyze_approval_mechanism.py').write_bytes((HERE/'analyze_approval_mechanism.py').read_bytes())
    write_json(out/'evidence_manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='evidence_manifest.json']})
    print(summary[summary.contrast.isin(['combined_vs_OrdinaryOnly','combined_vs_MinRisk'])].to_string(index=False))


if __name__=='__main__':main()
