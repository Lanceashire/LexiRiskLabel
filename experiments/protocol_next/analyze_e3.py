"""E3 descriptive analysis; public summaries are dataset-level only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from runtime import write_json,sha


def markdown(frame,headers):
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:+.8f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base
    out=base/'e3';analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    d=pd.read_json(out/'branch_outcomes.jsonl',lines=True);diag=pd.read_json(out/'target_diagnostics.jsonl',lines=True)
    state=pd.read_json(out/'state_manifest.jsonl',lines=True);res=pd.read_json(out/'resources.jsonl',lines=True)
    variance=pd.read_json(out/'offline_design_variance.jsonl',lines=True)
    comp=json.loads((out/'completion.json').read_text());ver=json.loads((out/'verification.json').read_text())
    baseline=d[d.branch=='Observed'][['state_id','draw','V_eval_brier','brier','fit_key']].rename(
        columns={'V_eval_brier':'observed_V_eval','brier':'observed_T','fit_key':'observed_fit_key'})
    paired=d.merge(baseline,on=['state_id','draw'],validate='many_to_one')
    paired['V_eval_gain_vs_observed']=paired.observed_V_eval-paired.V_eval_brier
    paired['T_gain_vs_observed']=paired.observed_T-paired.brier
    # Separate selection variability from paired validation-example variability.
    private=base/'private/e3/predictions';conditional_se=[]
    for row in paired.itertuples():
        p=np.load(private/f'{row.fit_key}.npz');b=np.load(private/f'{row.observed_fit_key}.npz')
        assert np.array_equal(p['eval_ids'],b['eval_ids'])
        delta=(b['eval_prediction']-b['eval_y'])**2-(p['eval_prediction']-p['eval_y'])**2
        conditional_se.append(float(np.std(delta,ddof=1)/np.sqrt(len(delta))))
    paired['conditional_V_sample_SE']=conditional_se
    paired.to_csv(analysis/'local_paired_draws.csv',index=False)
    within=paired.groupby(['dataset','seed','branch'],as_index=False).agg(
      V_eval_brier=('V_eval_brier','mean'),T_brier=('brier','mean'),V_eval_gain=('V_eval_gain_vs_observed','mean'),
      T_gain=('T_gain_vs_observed','mean'),acquisition_draw_SD_V=('V_eval_brier','std'),
      mean_conditional_V_sample_SE=('conditional_V_sample_SE','mean'))
    within.to_csv(analysis/'local_seed_means.csv',index=False)
    summary=within.groupby(['dataset','branch'],as_index=False).agg(
      V_eval_brier=('V_eval_brier','mean'),T_brier=('T_brier','mean'),V_eval_gain=('V_eval_gain','mean'),
      T_gain=('T_gain','mean'),positive_seed_count_V=('V_eval_gain',lambda v:int((v>1e-12).sum())),
      mean_acquisition_draw_SD_V=('acquisition_draw_SD_V','mean'),
      mean_conditional_V_sample_SE=('mean_conditional_V_sample_SE','mean'),seed_clusters=('seed','size'))
    summary.to_csv(analysis/'dataset_summary.csv',index=False)
    target=diag.groupby(['dataset','branch'],as_index=False).agg(
      train_target_min=('train_target_min','min'),train_target_max=('train_target_max','max'),
      mean_train_outside_fraction=('train_target_outside_fraction','mean'),
      mean_candidate_pseudo_outside_fraction=('candidate_pseudo_outside_fraction','mean'),
      gradient_min=('gradient_min_at_frozen_g','min'),gradient_max=('gradient_max_at_frozen_g','max'),
      mean_gradient_abs_q99=('gradient_abs_q99_at_frozen_g','mean'))
    target.to_csv(analysis/'dataset_target_diagnostics.csv',index=False)
    support=state.groupby('dataset',as_index=False).agg(
      reachable_fraction=('reachable_candidate_fraction','mean'),unreachable_candidates=('unreachable_n','sum'),
      min_positive_pi=('pi_min_positive','min'),max_inverse_pi=('inverse_pi_max','max'),
      mean_visible_oof_brier=('oof_brier','mean'))
    ess=res.groupby('dataset',as_index=False).agg(mean_selected_count=('count','mean'),
       mean_round_ESS=('round_correction_ESS','mean'),min_round_ESS=('round_correction_ESS','min'),
       mean_observed_plus_current_ipw_ESS=('observed_plus_current_ipw_ESS','mean'))
    support=support.merge(ess,on='dataset');support.to_csv(analysis/'dataset_support_ess.csv',index=False)
    varsummary=variance.groupby(['dataset','alpha'],as_index=False).agg(
      mean_target_bias_squared=('mean_target_bias_squared','mean'),mean_target_variance=('mean_target_variance','mean'),
      mean_target_mse=('mean_target_mse','mean'),full_design_variance=('full_pi_ij_variance','mean'),
      diagonal_proxy=('diagonal_only_proxy','mean'),fixed_g_brier_estimator_bias=('fixed_g_brier_estimator_bias','mean'))
    varsummary.to_csv(analysis/'dataset_offline_design_variance.csv',index=False)
    # Gate was declared in e3_config.json before outcomes; no outer-T criterion.
    matrix=summary.pivot(index='dataset',columns='branch',values='V_eval_brier')
    full_beats_plugin=int((matrix.Plugin-matrix.FullCorrection>1e-12).sum())
    gates={}
    for method in ['Shrink025','Shrink050']:
        z=summary[summary.branch==method]
        all_means=bool((z.V_eval_gain>1e-12).all());both_seeds=int((z.positive_seed_count_V==2).sum())
        gates[method]={'positive_dataset_means':int((z.V_eval_gain>1e-12).sum()),
          'datasets_with_both_seeds_positive':both_seeds,'full_correction_beats_plugin_datasets':full_beats_plugin,
          'passed':all_means and both_seeds>=2 and full_beats_plugin>=2}
    passed=any(v['passed'] for v in gates.values())
    decision={'E3_completed':True,'continuation_gate':gates,'correction_mainline_continue':passed,
      'action':'retain development candidate only; independent confirmation still required' if passed else 'close current correction improvement mainline; retain finite-mixture design and audit tools',
      'next_protocol_stage':'E4 small paired planning diagnostic; no scale-up','E4_started':False,
      'alpha_selected_on_outer_T':False,'fresh_validation_claim':False,'analyzer_hash':sha(__file__)}
    write_json(analysis/'decision.json',decision)
    methods=['Plugin','FullCorrection','Shrink025','Shrink050'];datasets=['hmeq','uci','lendingclub']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,col,title in zip(axes,['V_eval_gain','T_gain'],['Development validation V_eval','Outer T (secondary only)']):
        for j,dataset in enumerate(datasets):
            vals=[summary[(summary.dataset==dataset)&(summary.branch==m)][col].iloc[0]*1000 for m in methods]
            ax.bar(np.arange(len(methods))+(j-1)*.24,vals,width=.24,label=dataset.upper())
        ax.set_xticks(range(4),['Plugin','Full','Shrink .25','Shrink .50']);ax.axhline(0,color='black',lw=.8)
        ax.set_title(title);ax.set_ylabel('Brier decrease vs Observed (x1000)')
    axes[0].legend(fontsize=8);fig.savefig(analysis/'e3_dataset_gains.png',dpi=160);plt.close(fig)
    reporttable=summary[summary.branch!='Observed'][['dataset','branch','V_eval_gain','T_gain','positive_seed_count_V']]
    range_table=target[target.branch.isin(['FullCorrection','Shrink025','Shrink050'])][['dataset','branch','train_target_min','train_target_max','mean_candidate_pseudo_outside_fraction']]
    report=f'''# E3：CORR 校正与收缩的小型机制诊断

结论：本轮未通过预注册继续条件，关闭当前 CORR 改进主线，保留可行混合概率设计和审计工具。完整校正在三个数据集的验证均值都弱于普通观测标签训练。收缩能降低设计方差并缓解完整校正的损害，但尚未形成跨数据集稳定收益。这个停止决定只针对当前配置，不是校正方法普遍无效的证明。

## 设计与执行

沿用三个数据集、两个种子、第 5 轮的六个 MinRisk 共同状态。每个状态固定一次原有优化混合设计及精确 π、πij，预设三次随机采集；五个目标分支共用每次选集、π、g 和同一 XGBRegressor。没有新增数据集、时域或参数网格。

- Observed：D、O 和实际采得 S 的真标签训练。
- Plugin：D、O 真标签加全部可达候选的 g 伪目标，不使用 S 的残差修正。
- FullCorrection：g + I/π × (Y−g)。
- Shrink025 / Shrink050：g + αI/π × (Y−g)，α 预先固定为 0.25 / 0.5。

Observed 不等于 α=0。Plugin 与校正分支使用完全相同的扩展训练行；Observed 按定义只使用真实已观测行，训练规模不同。所有分支的回归器、正则化与单行权重约定相同，训练伪目标不截断；评价预测统一截断到 [0,1]。g 是固定的 D-only 逻辑回归。历史可见残差使用三折交叉拟合，仅描述可见区域，无法验证不可达区域。

共 {comp['acquisition_draws']} 次采集动作槽、{comp['unique_sampled_sets']} 个不同采集集合、{comp['nominal_fit_slots']} 个回归拟合槽，实际新增 {comp['actual_unique_regressor_fits']} 次 GPU 回归拟合、复用 {comp['fit_cache_hits']} 次，另有 {comp['historical_crossfit_LR_fits']} 次 CPU 历史残差交叉拟合。三次随机采集重复衡量同状态的选样波动，不增加独立数据种子数量。

## 验证结果

主要终点是固定的 E1b V_eval：与训练、Q_ref、外部 T 不相交，但曾用于前阶段研究诊断，所以是重复使用的开发验证块，不是新确认集。本轮未在其上反复调参；两种 α 全部报告。外部 T 仅作次要结果，不参与分支或 α 选择。

下表增益 = Observed 的 Brier − 对应分支 Brier，正数表示改善。每种子先平均三次采集，再平均两个种子，避免把重复采集当作独立任务。

{markdown(reporttable,['数据集','分支','V_eval 增益','外部 T 增益','验证改善种子数/2'])}

完整校正在验证均值上三项均为负。α=0.25 仅 UCI 验证均值为正；α=0.5 的三个数据集验证均值均为负。部分外部 T 结果为正，不能据此推翻主要验证终点或改选 α。

![数据集级验证与次要测试增益](e3_dataset_gains.png)

## 校正方差、目标范围与覆盖

所有有限混合原子的预算均核验通过。使用完整 πij 核验固定函数 Brier 校正估计量的设计方差，而不是把仅含对角项的代理称为完整方差；精确枚举与公式最大绝对误差为 {ver['max_exact_design_formula_error']:.3g}。对各 α 同时核验了 E[tildeY]−Y=(1−α)(g−Y) 和 Var(tildeY)=α²(1/π−1)(Y−g)²。

真实隐藏残差仅在全部模型训练完成后的离线公式审计中使用，没有进入采集、g 或回归训练。收缩把条件设计方差缩至完整校正的 1/16 或 1/4，这是公式关系，不是泛化改善保证；它同时引入偏差。完整设计方差及对角代理分别保存在 dataset_offline_design_variance.csv。

{markdown(support[['dataset','reachable_fraction','unreachable_candidates','min_positive_pi','max_inverse_pi','mean_round_ESS']],['数据集','平均可达比例','不可达候选总数','最小正 π','最大 1/π','平均当轮 ESS'])}

目标人群覆盖在这里指当前初始价格拒绝候选的正概率支持比例，不是所有未来人群的覆盖。HMEQ 存在一个不可达候选；不能对它声称无偏反馈恢复。本轮是单状态分叉，当轮与单次分叉累计校正 ESS 相同；完整多轮历史校正 ESS 不可由 MinRisk 历史重建，明确记为不可用，未冒充十轮重放结果。

伪目标范围与越界比例：

{markdown(range_table,['数据集','分支','训练目标最小值','训练目标最大值','平均可达伪目标越界比例'])}

梯度极值在固定 g 处按平方损失导数记录，详见 dataset_target_diagnostics.csv，不声称它是逐步追踪的实际 boosting 梯度。dataset_summary.csv 分别列出固定验证样本上的采集重复标准差，以及固定采集下逐验证样本配对差异的条件标准误描述；二者不合并成确认性置信区间。

## 启停决定

实验前固定的继续门槛是：同一预设收缩 α 在三个数据集的验证均值均优于 Observed，且至少两个数据集的两个种子都改善；完整校正还应在至少两个数据集的验证均值优于 Plugin。当前两个 α 均未通过，完整校正优于 Plugin 的数据集数为 {full_beats_plugin}/3。

因此关闭当前 CORR 改进主线，不扩充 α、采集重复或数据集来追逐正结果，不启动大规模确认。下一步可按原协议做 E4 的小型关键配对诊断，先统一规划与评价配置、真实未来标签释放和独立模拟世界；本报告阶段尚未启动 E4。

本轮每数据集仍只有两个开发种子。全部分支、失败与停止条件都保留在本地。用户最新要求停止批量明细上传，先继续实验，后续上传前整理 Pro 能独立理解的必要报告和证据包。原始明细不默认逐项上传；本轮科学结果不因交付范围变化而改变。
'''
    (analysis/'E3_CORR校正机制实验报告.md').write_text(report)
    print(json.dumps({'continuation_gate':gates,'continue_CORR':passed,'report':str(analysis/'E3_CORR校正机制实验报告.md')},ensure_ascii=False))


if __name__=='__main__':main()
