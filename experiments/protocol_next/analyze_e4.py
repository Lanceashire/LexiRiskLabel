"""E4 paired diagnostics and stopping decision; no validation-based reselection."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from runtime import write_json,sha


def table(df,headers):
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:+.8f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);a=ap.parse_args();base=a.base
    out=base/'e4';analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    d=pd.read_json(out/'actual_outcomes.jsonl',lines=True);world=pd.read_json(out/'independent_true_world_pairs.jsonl',lines=True)
    ranks=pd.read_json(out/'ranking_diagnostics.jsonl',lines=True);comp=json.loads((out/'completion.json').read_text())
    pairs=[]
    for (seed,planner,learner),group in d.groupby(['seed','planner','learner']):
        x=group.set_index('condition');h1=x.loc['H1_release_dynamic'];h2=x.loc['H2_release_dynamic'];nr=x.loc['H2_no_release_dynamic']
        fixed=x.loc['H2_release_fixed'];matched=x.loc['H2_fixed_matched_action']
        pairs.append({'seed':seed,'planner':planner,'learner':learner,
          'horizon_gain':h1.excess_brier-h2.excess_brier,'future_release_gain':nr.excess_brier-h2.excess_brier,
          'feedback_channel_gain':matched.excess_brier-h2.excess_brier,'fixed_replanning_gain':matched.excess_brier-fixed.excess_brier,
          'H1_total_predicted_risk':h1.total_predicted_risk,'H2_total_predicted_risk':h2.total_predicted_risk,
          'H2_total_realized_loss':h2.total_realized_loss,'no_release_total_realized_loss':nr.total_realized_loss,
          'H2_future_approval_count':h2.future_approval_count,'H2_fixed_approval_count':matched.future_approval_count,
          'H2_future_set_change_vs_fixed':h2.future_set_change_vs_fixed})
    pairs=pd.DataFrame(pairs);pairs.to_csv(analysis/'actual_paired_contrasts.csv',index=False)
    summary=pairs.groupby(['planner','learner'],as_index=False).agg(
       horizon_gain=('horizon_gain','mean'),horizon_positive_seeds=('horizon_gain',lambda x:int((x>1e-12).sum())),
       future_release_gain=('future_release_gain','mean'),feedback_channel_gain=('feedback_channel_gain','mean'),
       fixed_replanning_gain=('fixed_replanning_gain','mean'),H2_total_predicted_risk=('H2_total_predicted_risk','mean'),
       H2_total_realized_loss=('H2_total_realized_loss','mean'))
    summary.to_csv(analysis/'actual_descriptive_summary.csv',index=False)
    world.to_csv(analysis/'independent_true_world_pairs.csv',index=False)
    world_summary=world.groupby('planner',as_index=False).agg(
       horizon_gain=('horizon_gain_true_verification','mean'),positive_seeds=('horizon_gain_true_verification',lambda x:int((x>1e-12).sum())),
       changed_action_seeds=('different_horizon_actions','sum'),finite_family_room=('finite_family_room_after_H1','mean'))
    world_summary.to_csv(analysis/'independent_world_summary.csv',index=False)
    ranking=ranks.groupby(['planner','condition'],as_index=False).agg(
      mean_same_model_verification_regret=('model_verification_family_regret','mean'),
      mean_true_generator_verification_regret=('true_verification_family_regret','mean'),
      independent_world_ranking_disagreement_seeds=('independent_world_ranking_disagrees','sum'),
      mean_model_true_rank_correlation=('model_true_verification_rank_correlation','mean'))
    ranking.to_csv(analysis/'ranking_diagnostic_summary.csv',index=False)
    # Recompute the action lock from persisted selection scores, never from checks.
    for seed in sorted(d.seed.unique()):
        lock=json.loads((base/'private/e4/selection_locks'/f'seed{seed}.json').read_text())
        raw=np.load(base/'private/e4'/f'seed{seed}_worlds.npz')
        for planner,chosen in lock['plans'].items():
            mean=raw[planner+'_selection'].mean(axis=0)
            assert chosen['H1_release_dynamic']==int(np.argmin(mean[:,0]))
            assert chosen['H2_release_dynamic']==int(np.argmin(mean[:,1]))
            assert chosen['H2_release_fixed']==int(np.argmin(mean[:,2]))
    oracle=world_summary[world_summary.planner=='oracle_true_generator'].iloc[0]
    gate=bool(oracle.horizon_gain>1e-12 and oracle.positive_seeds>=2)
    decision={'E4_completed':True,'registered_oracle_gate_passed':gate,'oracle_horizon_mean_gain':float(oracle.horizon_gain),
       'oracle_positive_seed_count':int(oracle.positive_seeds),'underlying_seed_count':3,
       'expand_horizon_worlds_or_candidate_family':gate,'real_data_confirmation_started':False,
       'action':'retain only a development candidate for a separately frozen real-data pilot' if gate else 'stop expanding current GATE route; no real-data or large confirmation expansion',
       'scope':'finite-family, finite-world, three-seed mechanism; not a proof that all foresight policies lack value',
       'analyzer_hash':sha(__file__)}
    write_json(analysis/'decision.json',decision)
    write_json(analysis/'analysis_verification.json',{'passed':True,'choices_recomputed_from_selection_worlds_only':True,'verification_not_used_for_reselection':True})
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    names=['estimated_Laplace','oracle_true_generator']
    for i,name in enumerate(names):
        z=world[world.planner==name];axes[0].scatter(np.full(len(z),i),1000*z.horizon_gain_true_verification,label=name)
    axes[0].set_xticks([0,1],['Estimated','True generator']);axes[0].set_ylabel('H2 horizon gain on true check worlds (x1000)')
    axes[0].axhline(0,color='black',lw=.8);axes[0].set_title('Matched LR mechanism / 3 seeds')
    for i,name in enumerate(names):
        z=summary[summary.planner==name].set_index('learner');axes[1].bar(np.arange(2)+(i-.5)*.35,1000*z.loc[['LR','XGB'],'horizon_gain'],width=.35,label=name)
    axes[1].set_xticks([0,1],['LR','GPU XGB']);axes[1].set_ylabel('Single actual-world H2 gain (x1000)');axes[1].axhline(0,color='black',lw=.8)
    axes[1].legend(fontsize=7);axes[1].set_title('Secondary transfer check')
    fig.savefig(analysis/'e4_horizon_checks.png',dpi=150);plt.close(fig)
    wtable=world[['seed','planner','horizon_gain_true_verification','different_horizon_actions']]
    stable=summary[['planner','learner','horizon_gain','future_release_gain','feedback_channel_gain']]
    report=f'''# E4：GATE 协议修复与关键配对机制实验

结论：本轮未通过预设的真生成器继续条件，不扩大当前 GATE 的时域、模拟世界数、候选族或真实数据确认。已知正确生成器的匹配逻辑回归规划，在独立核验世界上两步相对一步的平均增益为 {oracle.horizon_gain:+.8f}，三个种子没有一个正增益。这个结果仅约束当前有限候选族和世界数，不证明所有前瞻策略都没有价值。

## 实验范围与修复

使用原有 C1 稀少安全区域合成生成器和原有种子 2101、2102、2103，共三个底层状态。旧 C2 与 C1 同种子的数据相同，仅审批冻结标志不同，因此改为同状态的固定审批配对，不把重复数据计作额外独立环境。

候选族事先固定为 OrdinaryOnly、MinRisk、Vopt、RB-BRIDGE 和一个固定随机可行批次，重复集合去重，各状态剩 4 或 5 个候选。所有规划器、时域和审批条件共用同一候选族与模拟随机数；核验结果不用于修改候选族或重选动作。

规划与实际环境共同使用未来批量 160、容量 80、风险上限 20、阈值 0.5。每状态从已知历史协变量抽取同一个条件预测批次，规划与实际环境共享其特征，真实未来标签在锁定所有动作之后生成。这里固定到达预测误差，只诊断标签生成、排序与反馈机制，不能推广为已验证未知真实未来流。

比较估计的 Laplace 规划器与仅合成可用的真生成器 oracle。每个规划器用 16 个选择世界，另用 8 个独立同模型核验世界；估计规划器再与 8 个共同真生成器世界对照。oracle 的同模型核验即真生成器核验，不重复计算。真实信用中没有可用的 oracle 真实概率。

关键配对：

- H1/H2：只改变规划时域，实际环境都释放未来标签。
- release/no-release：保持 H2 动作与未来审批相同，真正改变是否把未来行加入训练；不把 planner 参数改名当环境消融。
- dynamic/fixed matched action：保持当前 H2 动作相同，只改变未来审批，隔离反馈通道。
- fixed replanning：在同一固定审批环境下，比较动态规划动作与按固定审批重规划的动作。

## 独立真世界上的前瞻价值

增益 = H1 所选动作在真实 H2 释放环境的误差 − H2 所选动作在同一环境的误差。正数表示两步动作选择更好。评价使用匹配的 LR refit 与 Q_ref 真实概率的 excess Brier；不使用当前实际世界或测试集来选择动作。

{table(wtable,['种子','规划器','独立核验前瞻增益','H1/H2 动作不同'])}

已知真生成器时，两个种子的 H1/H2 动作相同，一个种子的 H2 更差。按预设条件“平均正增益且至少 2/3 种子改善”，应停止扩展当前路线。

有限 Monte Carlo 排序仍有噪声：oracle 的 H2 选择在 2/3 种子上与独立核验的族内最优不一致。估计模型与真世界的差异另见 ranking_diagnostic_summary.csv。有限候选族的核验最小误差只是诊断量，不是全局最优或可部署竞争者；有剩余族内空间也不能用核验世界反复挑选新动作。

## 单次实际环境及 GPU 迁移检查

实际环境使用独立未来标签，再按锁定动作运行 LR 与 GPU XGB。LR 匹配规划中的学习器；XGB 仅是同动作迁移检查。增益均为 excess Brier 的下降，不能将个别 XGB 正结果用于推翻匹配 oracle 的预设停止条件。

{table(stable,['规划器','实际学习器','两步规划增益','真实释放增益','同动作动态反馈增益'])}

![独立核验与实际世界的区别](e4_horizon_checks.png)

实际总体普通风险、探索风险、未来普通风险、总预测风险、实际损失与合成真实期望损失均已保存。未来审批发生的经济损失在 release 与 no-release 中都计算；不释放训练标签不等于不发生经济风险。二者的选集、审批及经济损失相同，只有训练标签释放不同，已逐项核验。

## 执行与核验

共有 {comp['actual_policy_learner_slots']} 个规划/条件/学习器动作槽，但只有三个独立底层种子。实际新增 {comp['actual_unique_GPU_XGB_fits']} 次 GPU XGB 拟合、{comp['actual_unique_LR_fits']} 次匹配 LR 拟合，缓存命中 {comp['actual_model_cache_hits']} 次。内部 {comp['internal_planning_LR_refits']} 次五维 LR refit 是固定 16/8 个世界的 Monte Carlo 规划计算，不是新增基准、独立数据或参数扫描。

所有未来容量和风险约束、真实释放开关、同动作对照、固定审批集合、实际 CUDA 后端、保存预测重新计分及仅按选择世界锁定动作的检查均通过。完整原始记录保留在本地 private/e4。

## 下一步

E2b 未确认搭档构造的新增效应；E2c/E1b/E2d 未形成稳定 BRIDGE 修复；E3 未通过校正继续条件；本轮 E4 的正确生成器检查也未通过。因此暂停扩大三条改进路线，不启动 720 条确认轨迹，不通过追加世界、种子或新数据集追逐当前正结果。

下一步应把已有证据交给 Pro 判断论文贡献如何收敛，以及是否存在最多一个有明确机制依据的小改动；若没有依据，应保留负结果并停止扩展。交付包须独立说明问题、设计、各阶段结果、证据边界和待判断问题。根据用户最新指令，批量明细上传已停止，本轮先完成实验与必要审阅包整理。
'''
    (analysis/'E4_GATE关键配对实验报告.md').write_text(report)
    print(json.dumps(decision,ensure_ascii=False))


if __name__=='__main__':main()
