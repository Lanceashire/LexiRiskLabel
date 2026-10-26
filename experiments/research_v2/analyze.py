"""Descriptive development analysis. No post-hoc seed expansion or significance search."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/research_v2'


def table(df):
    head='| '+' | '.join(map(str,df.columns))+' |\n'
    sep='| '+' | '.join(['---']*len(df.columns))+' |\n'
    body=''.join('| '+' | '.join(str(v) if not isinstance(v,float) else f'{v:.7f}' for v in row)+' |\n' for row in df.itertuples(index=False,name=None))
    return head+sep+body


def main():
    global BASE
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--base',type=Path,default=BASE);a=ap.parse_args();BASE=a.base;a.output.mkdir(parents=True,exist_ok=True)
    allrows=pd.concat([pd.read_csv(p) for p in (BASE/'final_real').glob('*/trajectory.csv')],ignore_index=True)
    assert len(allrows)==840
    episodes=allrows.groupby(['dataset','seed','protocol','method','learner'],as_index=False).agg(
        alc_brier=('brier','mean'),final_brier=('brier','last'),target_alc_brier=('target_brier','mean'),
        selected_labels=('selected_count','sum'),predicted_risk=('predicted_exploration_risk','sum'),
        realized_loss=('realized_exploration_loss','sum'),runtime_seconds=('runtime_seconds','max'),
        selection_seconds=('selection_seconds','sum'),max_weight=('max_weight','max'),
        minimum_effective_sample_size=('effective_sample_size','min'))
    ordinary=episodes[episodes.method=='OrdinaryOnly'][['dataset','seed','protocol','alc_brier']].rename(columns={'alc_brier':'ordinary_alc_brier'})
    episodes=episodes.merge(ordinary,on=['dataset','seed','protocol']);episodes['improvement_vs_ordinary']=episodes.ordinary_alc_brier-episodes.alc_brier
    episodes.to_csv(a.output/'episode_summary.csv',index=False)
    summary=episodes.groupby(['dataset','protocol','method','learner'],as_index=False).agg(
        independent_seeds=('seed','count'),mean_alc_brier=('alc_brier','mean'),
        mean_improvement_vs_ordinary=('improvement_vs_ordinary','mean'),mean_selected_labels=('selected_labels','mean'),
        mean_predicted_risk=('predicted_risk','mean'),mean_realized_loss=('realized_loss','mean'),
        mean_runtime_seconds=('runtime_seconds','mean'),max_inverse_probability_weight=('max_weight','max'))
    summary.to_csv(a.output/'method_summary.csv',index=False)
    # Within-protocol, within-seed contrasts. Positive delta means the new method is worse.
    contrasts=[]
    for protocol,new,comparators in [('fixed','RB-BRIDGE',['OrdinaryOnly','MinRisk','Vopt','BAIT-cost','ParBaLS-cost']),
                                      ('correction','RB-CORR',['OrdinaryOnly','CORR-opt-plain','CORR-base-corrected','CORR-vopt-corrected'])]:
        for dataset in ['hmeq','uci','lendingclub']:
            group=episodes[(episodes.protocol==protocol)&(episodes.dataset==dataset)].pivot(index='seed',columns='method',values='alc_brier')
            for comparator in comparators:
                delta=group[new]-group[comparator]
                contrasts.append({'dataset':dataset,'protocol':protocol,'method':new,'comparator':comparator,'n_paired_seeds':len(delta),
                                  'mean_delta_brier':delta.mean(),'min_seed_delta':delta.min(),'max_seed_delta':delta.max(),
                                  'seed_wins':int((delta<0).sum()),'seed_losses':int((delta>0).sum())})
    contrast=pd.DataFrame(contrasts);contrast.to_csv(a.output/'paired_descriptive_contrasts.csv',index=False)
    mechanisms=pd.read_csv(BASE/'final_mechanisms/mechanism_rows.csv')
    mechanisms.to_csv(a.output/'mechanism_rows.csv',index=False)
    msummary=mechanisms.groupby(['environment','protocol','learner','method'],as_index=False).mean(numeric_only=True)
    msummary.to_csv(a.output/'mechanism_summary.csv',index=False)
    focus=pd.read_csv(BASE/'focused/focused_rows.csv');focus.to_csv(a.output/'focused_rows.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    methods=['RB-BRIDGE','Vopt','BAIT-cost','ParBaLS-cost','LRBE-IG','ScoreFirst-IG']
    for ax,dataset in zip(axes,['hmeq','uci','lendingclub']):
        g=episodes[(episodes.protocol=='fixed')&(episodes.dataset==dataset)].pivot(index='seed',columns='method',values='alc_brier')
        deltas=np.array([(g[m]-g['MinRisk']).values for m in methods])*1e4;ax.barh(methods,deltas.mean(axis=1),color=['#466c91']+['#adb9c4']*5)
        for k,values in enumerate(deltas):ax.scatter(values,np.full(len(values),k),s=22,color='#20394d',zorder=3)
        ax.axvline(0,color='black',lw=.8);ax.set_title(dataset.upper());ax.invert_yaxis();ax.set_xlabel('ALC Brier minus MinRisk (x 0.0001)')
    fig.savefig(a.output/'fixed_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    cm=['CORR-base-plain','CORR-base-corrected','CORR-opt-plain','RB-CORR','CORR-vopt-corrected']
    labels=['Base / plain','Base / corrected','Optimized / plain','RB-CORR','Vopt / corrected']
    for ax,dataset in zip(axes,['hmeq','uci','lendingclub']):
        g=episodes[(episodes.protocol=='correction')&(episodes.dataset==dataset)].pivot(index='seed',columns='method',values='alc_brier')
        delta=np.array([(g[m]-g['OrdinaryOnly']).values for m in cm])*1e3;ax.barh(labels,delta.mean(axis=1),color=['#b4c2cb','#bd896b','#b4c2cb','#a7523c','#bd896b'])
        for k,values in enumerate(delta):ax.scatter(values,np.full(len(values),k),s=22,color='#382a26',zorder=3)
        ax.axvline(0,color='black',lw=.8);ax.set_title(dataset.upper());ax.invert_yaxis();ax.set_xlabel('ALC Brier minus OrdinaryOnly (x 0.001)')
    fig.savefig(a.output/'correction_factorial.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,env in zip(axes,['C1','C2']):
        g=msummary[msummary.environment==env];labels=['MinRisk','RB-BRIDGE','GATE-one','RB-GATE']
        vals=[g[g.method==m].feedback_channel_value.iloc[0]*1e4 for m in labels]
        ax.bar(labels,vals,color='#6f8f83');ax.axhline(0,color='black',lw=.8);ax.set_title(env+(' dynamic approval' if env=='C1' else ' frozen approval'))
        ax.set_ylabel('Feedback-channel reduction (x 0.0001)');ax.set_ylim(0,6.5);ax.tick_params(axis='x',rotation=20)
        if env=='C2':ax.text(.5,.5,'Exactly zero in every branch',ha='center',va='center',transform=ax.transAxes)
    fig.savefig(a.output/'gate_feedback_channel.png',dpi=180);plt.close(fig)
    fixed=summary[summary.protocol=='fixed'].pivot(index='dataset',columns='method',values='mean_alc_brier').reset_index()
    fixed=fixed[['dataset','OrdinaryOnly','MinRisk','LRBE-IG','Vopt','BAIT-cost','ParBaLS-cost','RB-BRIDGE']]
    corr=summary[summary.protocol=='correction'].pivot(index='dataset',columns='method',values='mean_alc_brier').reset_index()
    corr=corr[['dataset','OrdinaryOnly','CORR-base-plain','CORR-base-corrected','CORR-opt-plain','RB-CORR','CORR-vopt-corrected']]
    bridge_delta=contrast[(contrast.protocol=='fixed')&(contrast.comparator=='MinRisk')][['dataset','mean_delta_brier','seed_wins','seed_losses']]
    verification=json.loads((BASE/'verification.json').read_text())
    mechanism_count=len(mechanisms)
    report=f'''# LexiRiskLabel 小规模重做实验结果

2026 年 10 月 26 日

三种新方法均已实现并实际运行。当前开发结果没有支持把其中任何一种写成稳定优于强基线的新算法。RB-BRIDGE 在三份真实数据上平均未胜过 MinRisk；RB-CORR 的残差校正在本轮真实任务中增大 Brier；RB-GATE 的小型闭环试验没有显示其前瞻机制超越短视选择的稳定优势。全部正负结果均保留。

## 执行范围

按“论文改进建议”聊天的三条新算法路线执行，并完整读取补充的《LexiRiskLabel 研究重构与实验方案》。采用其中共同风险账本、普通反馈基线、独立目标参考、隐藏标签隔离、配置身份、完整成本及预测留存要求。

真实数据固定 rho=0.1，每个数据两个预定开发种子 2201、2202，十轮；分类器比较 48 条轨迹，平方损失交叉实验 36 条轨迹，共 84 条、840 个逐轮记录。十轮记录不作为十个独立重复。

六类机制环境 A1/A2、B1/B2、C1/C2 各用 2101–2103 三个开发种子，共 {mechanism_count} 个方法终点；另在两个代表状态完成文档要求的条件化×批更新及数量下限诊断。它们属于实际重训分支，并非新增一批无关基准测试。

本轮没有启动聊天和文档举例的上千条正式轨迹、14×30 条件确认、额外大预算网格或 CVaR、公平性、延迟模块。RB-GATE 依聊天要求先限于小型闭环环境。当前结论是开发筛查，不能作为确认、显著性或投稿等级结论。

## RB-BRIDGE 与真实分类器比较

下表为十轮平均 Brier 的两种子均值，越低越好。所有方法共享固定业务审批、候选、数值预算、初始数据、目标参考及特征表示。

{table(fixed)}

RB-BRIDGE 减去 MinRisk 的配对差如下。正数表示 RB-BRIDGE 更差；“胜/负”仅表示这两个开发种子的方向。

{table(bridge_delta)}

HMEQ 中 RB-BRIDGE 优于普通反馈基线，但没有超过 MinRisk 和 LRBE-IG；UCI 与 LendingClub 的差异也很小。可以确认代码改变了选样，不能据此确认新的互补组搜索具有实际优势。

![固定审批结果](fixed_comparison.png)

## RB-CORR 的采集与训练交叉实验

所有分支均使用相同平方损失 XGBRegressor，输出限制到 [0,1]；本表不与上表的分类器混排。同一采集方式的普通训练与校正训练实际获得相同选集、相同采集概率，因而可以直接观察校正的影响。

{table(corr)}

本轮完整 RB-CORR 在三份数据上均比对应的优化采集普通训练更差，也比普通反馈基线更差。数学上的固定函数风险估计无偏，未转化为训练后泛化改善。日志保存了精确边际概率、混合组成、历史伪目标、最大权重与有效样本量，没有通过截断权重隐藏方差代价。

这支持暂不把该校正版本直接用于论文主结果，不能仅凭无偏恒等式宣称学习收益。局部影响量采用平方损失线性头的曲率；迁移到树回归器仍属于代理，不能解释为精确树模型影响函数。

![校正交叉结果](correction_factorial.png)

## 六类机制与 RB-GATE

- A1：普通反馈未覆盖干扰方向。探索相对 OrdinaryOnly 有明显改善，但 MinRisk 的平均 excess Brier 更低，结构化配对未形成优势。
- A2：保持普通反馈数量不变，改为覆盖干扰方向。OrdinaryOnly 自身的误差明显下降，付费探索的额外空间收缩；这支持覆盖方向会影响探索价值，但尚不证明 BRIDGE 搜索更好。
- B1：非线性概率与线性平方损失学习器错设。完整 CORR 未稳定胜过其不校正对照。
- B2：有界线性概率与线性平方损失学习器匹配。仍未出现校正必然有效的规律，保留其方差代价。
- C1：允许后续审批改变。RB-GATE 的平均 excess Brier 约为 {msummary[(msummary.environment=='C1')&(msummary.method=='RB-GATE')].excess_brier.iloc[0]:.7f}，RB-BRIDGE 约为 {msummary[(msummary.environment=='C1')&(msummary.method=='RB-BRIDGE')].excess_brier.iloc[0]:.7f}。当前前瞻没有超过更简单的 BRIDGE。
- C2：冻结下一轮审批。每个分支的反馈渠道价值均严格为零，负控制成立。冻结审批后仍可能存在标签的直接价值，不能把它误称为渠道价值。

GATE 使用同一候选批次族、共同随机数和一致的潜在参数世界，16 个世界选择、另 8 个世界核验；未来申请特征从已经观察到的历史重采样。实际未来标签只在采集决定锁定后释放。普通与探索风险分别记账，未来普通审批受共同容量和预测风险额度约束。

![反馈渠道分解](gate_feedback_channel.png)

## 文档中的组件与数量诊断

两个代表状态完成了有/无普通反馈条件化、有/无批内更新的四格对照，以及固定 K* 版本。它们保存在 focused_rows.csv，未扩展成额外的全因子大网格。

同一状态、同一预算、同一 IG 代理扫描 epsilon={{0,0.05,0.10,0.20,1}}。这两个状态的 K*=8，前三个 epsilon 因取整具有相同数量下限；即使放松至 0 或 7，求解器仍选择相同八人集合。因此这次小试未显示数量放松带来收益；没有把参数不同但选集相同伪装成不同发现。此处 IG 为机制模型的 Laplace 后验信息增益，不是信用数据上旧委员会 IG 的正式数量扫描。

## 正确性与原始证据核验

二维反例得到互补对价值 0.6666667，逐点贪心得到 0.0099990；普通方向信息增加到 b=9999 时互补性归零。秩二联合增益与直接求解的最大误差约 1.5e-15。12 人小候选池穷举中，该状态的 V-opt 和 BRIDGE 都达到代理最优；这只描述该小池，不是全局算法保证。

CORR 对有限采集分布逐项求和，风险估计期望误差约 5.6e-17。这个检查没有用几千次随机抽样制造更多“实验”。

对最终 84 个 episode 的 840 条逐轮记录，核验了实际 GPU 训练 device 与 updater、预算、选集子集、无重复、测试隔离、共同 O/C/成本/预算、有限混合概率及保存预测重算 Brier；预测预算违约为 {verification['predicted_budget_violations']}。改变隐藏普通、候选和测试标签的边界检查没有改变三种选择器输出。配置阈值改为 0.51 的伪续跑被明确拒绝，没有混入旧结果。

宿主机 RTX 5060 8GB 已通过真实训练验证；逐轮实际 device 为 cuda:0。逻辑代理与矩阵/背包优化在 CPU 上执行。NumPy 输入预测使用 DMatrix 路径，训练后端与输入位置分别记录，没有把预测警告误报为训练回退。

## 数据与解释边界

新获取 HMEQ 5,960 行、UCI 30,000 行、LendingClub 成熟队列 16,566 行（3,321 正类）。来源分别为 SAS 官方 GitHub、UCI 官方下载和仓库记载的 LendingClub 2014/2015 样本。每份新文件均有来源与 SHA-256，但没有与旧登记文件达成字节一致，因此新实验独立记录，未调用覆盖旧数据的强制选项。

LendingClub 采用预先列明的 21 个紧凑字段，排除 last_fico 等更新结果字段；HMEQ 排除暴露 LOAN 作为预测特征。该 schema、依赖版本、审批协议和划分不同于旧论文，不能把本轮差值直接拼到旧稿的均值或显著性表里。

两种子真实结果和三种子机制结果只够做开发判断。没有按结果追加种子、删任务或改成另一套获胜指标。BAIT/ParBaLS 为公开原理与作者代码结构上的透明成本适配；ParBaLS 使用 4 条幻想、32 个后验样本，不能宣称直接复现作者默认配置，也不能据此宣称战胜原版 SOTA。通用双人对照匹配初始候选对数，实际墙钟与后续可行组合数仍有差异。

## 当前决定

保留 MinRisk 与原 IG 作为有效强基线。RB-BRIDGE 目前只验证了机制起点；RB-CORR 的精确概率与校正实现可复用，但当前训练版本的真实负结果必须保留；RB-GATE 暂停在小型闭环开发结论，不投入大规模确认。

若继续研究，应针对本轮已经暴露的代理失配、校正方差和前瞻收益不足分别改方法，再冻结一个新版本。当前证据不支持把三个名字包装成已经成立的强创新算法。

## 复现入口

项目代码在 LexiRiskLabel/experiments/research_v2，最终原始目录为 results/research_v2/final_real、final_mechanisms 和 focused；其他原型目录不纳入汇总。每条真实轨迹包含 identity.json、十轮选样日志、trajectory.csv、test_predictions.npz，校正分支额外保存历史概率与伪目标。

本次依赖锁定、GPU 证据、来源对照、完整种子结果、配对描述差异和运行日志一并交付。运行命令在复现脚本中，已有结果的配置、源代码或数据身份不一致时拒绝续跑。

参考：[BAIT 作者论文](https://proceedings.neurips.cc/paper/2021/file/4afe044911ed2c247005912512ace23b-Paper.pdf)，[ParBaLS 作者实现](https://github.com/ADDAPT-ML/ParBaLS)，[SAS HMEQ](https://github.com/sassoftware/sas-viya-dmml-pipelines/blob/master/data/hmeq.csv)，[UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)，[LendingClub 样本来源](https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/)。
'''
    (a.output/'实验汇总.md').write_text(report)
    summary.to_csv(a.output/'results.csv',index=False)
    print('analysis complete',len(episodes),'episodes',mechanism_count,'mechanism rows')


if __name__=='__main__':main()
