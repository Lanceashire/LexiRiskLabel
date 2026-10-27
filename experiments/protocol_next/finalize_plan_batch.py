"""Close the executed four-block batch; no new experiments in reporting."""
from runtime import *
import argparse


def mdtable(df):
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:+.8f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base
    out=base/'nonlinear_mechanism_small';private=base/'private/nonlinear_mechanism_small'
    comp=json.loads((out/'completion.json').read_text());ver=json.loads((out/'verification.json').read_text());assert ver['passed']
    rows=pd.read_csv(out/'branch_outcomes.csv');summary=pd.read_csv(out/'descriptive_summary.csv');error=0.
    for r in rows.itertuples():
        p=np.load(private/'predictions'/f'{r.evaluation_key}.npz')
        error=max(error,abs(float(np.mean((p['prediction']-p['pTest'])**2))-r.excess_brier),
            abs(float(np.mean((p['prediction']-p['yTest'])**2))-r.brier))
    assert error<1e-12
    view=summary[summary.contrast.isin(['combined_vs_OrdinaryOnly','combined_vs_MinRisk'])]
    assert (view[(view.learner=='XGB')&(view.contrast=='combined_vs_MinRisk')].negative_seeds==3).all()
    report=f'''# 非线性错设子集：审批约束下的后续机制实验

按原DOCX第8.2节继续执行非线性错设检查。三个既有开发种子；复用审批约束实验的全部特征池、暴露、业务审批模型及预算。真实logit从线性项变为“原线性logit + 0.8×(目标坐标²−1)”，二次项及系数在运行前锁定。全部标签用原共同Bernoulli随机数按新真概率重新生成，可见D的逻辑代理只用新D标签重新拟合。

预测器和采集器保留原五个特征，不加入平方项；LR因此错设，GPU XGB仍按原100树深度3程序训练。每个种子仍有500条普通反馈的低/高目标覆盖两个条件，四格加OrdinaryOnly和MinRisk。所有普通样本通过相同业务审批门，候选全部在拒绝区域。选集锁定后才训练及评价，真实概率不进入采集器。

## 结果

主机制指标是对已知真概率的excess Brier，正数表示参照误差减完整C1B1误差；样本标签Brier为辅助指标：

{mdtable(view[['coverage','learner','contrast','mean_excess_gain','positive_seeds','negative_seeds','mean_sampled_brier_gain']])}

XGB在两个覆盖条件的三个种子中，完整方法均劣于MinRisk（六个配对状态、三个独立种子簇）。低覆盖下LR相对MinRisk平均也更差；高覆盖下LR虽然2/3种子改善，均值仍为负。不删去与平均方向不一致的种子。

相对OrdinaryOnly，XGB低覆盖三个种子均有改善，但仍未超过便宜标签基线；高覆盖只有1/3种子改善却得到正均值。LR的两个覆盖条件相对OrdinaryOnly均值都负，采样标签Brier与真概率误差也有方向差异，全部保留。

因此，“探索能改善不探索”与“复杂选择比MinRisk值得”仍是两个命题。不能仅用正的代理价值或相对OrdinaryOnly的均值，推断新增方法有优势。

## 核验与解释范围

新增{comp['actual_unique_LR_fits']}次不同LR和{comp['actual_unique_GPU_fits']}次实际CUDA XGB拟合，缓存{comp['cache_hits']}次；72个拟合槽不是72个独立实验。全部保存预测重新计分，最大误差{error:.3g}；信息边界、审批门、预算、空O零对照及源码哈希检查通过。

线性与非线性世界的标签规律不同，初始可见标签与代理拟合也会随之改变。这是预设错设边界子集，不是只改变学习器而保持所有标签相同的因果干预；跨世界原始Brier差也混合不可约误差变化。主要结论来自各个非线性共同状态内的配对结果，不把它写成普遍不可能性。

本项不触发换方法、加平方特征、调参或寻找新候选，继续原实验队列。完整14条件与独立确认未因本子集而自动完成。
'''
    (out/'非线性错设子集_实验报告.md').write_text(report)
    write_json(out/'analysis_verification.json',{'passed':True,'all_saved_predictions_rescored':True,'max_error':error,'analyzer_sha256':sha(__file__)})
    write_json(out/'decision.json',{'continue_original_plan':True,'new_candidate_promoted':False,'misspecification_subset_completed':True,
        'full_mechanism_confirmation_completed':False,'next':'remaining scheduled coverage controls and frozen-method full-trajectory nearest-neighbor comparison'})
    (out/'source_snapshot/finalize_plan_batch.py').write_bytes(Path(__file__).read_bytes())
    write_json(out/'evidence_manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='evidence_manifest.json']})
    stages=['approval_mechanism_small','nearest_neighbor_small','approval_volume_cost','nonlinear_mechanism_small']
    totals={'LR':0,'GPU':0};completed=[]
    for stage in stages:
        c=json.loads((base/stage/'completion.json').read_text())
        totals['LR']+=c.get('actual_unique_LR_fits',c.get('new_unique_LR_fits',0));totals['GPU']+=c.get('actual_unique_GPU_fits',c.get('new_unique_GPU_fits',0))
        completed.append({'stage':stage,'completion_sha256':sha(base/stage/'completion.json'),'verification_sha256':sha(base/stage/'verification.json')})
    state={'instruction':'continue original experiment plan; completed-stage reports are not a pause request',
        'completed_this_batch':completed,'new_actual_LR_fits':totals['LR'],'new_actual_GPU_fits':totals['GPU'],
        'algorithm_candidate_search_expanded':False,'confirmation_claim':False,
        'next_queue':['remaining scheduled middle-coverage controls','frozen-method ten-round comparison including adapted nearest neighbor',
                      'independent confirmation under its registered information and statistical protocol'],
        'scope':'continue scheduled comparisons, retaining negative results; no requirement to first discover a winning new method; do not invent new modules'}
    write_json(base/'plan_execution_state.json',state)
    brief=f'''# 原实验计划继续执行：本轮四项结果

本轮根据用户“按照实验计划正常推进”的指示继续实验，没有把前面候选的阴性结果当作停止执行既定比较的理由。完成以下四项，实际新增{totals['GPU']}次GPU和{totals['LR']}次LR拟合；它们是有限配对设计中的训练次数，独立性仍按原数据/生成种子计算。

| 原计划问题 | 本轮执行 | 结果 |
| --- | --- | --- |
| 普通反馈覆盖方向，且满足审批门 | D800/O500/C400/Q1024/T5000，三个种子，两覆盖方向，四格加两参照 | 低覆盖时比不探索好但两学习器均在3/3种子输给MinRisk；高覆盖时相对MinRisk均值略正，均2/3改善 |
| 公平近邻比较 | 三真实任务各两个round-5状态；10伪标签世界、32后验样本、全Q/全候选、相同pending信息 | 近邻与Vopt在4/6状态同选集；XGB相对MinRisk的验证均值仅HMEQ正。是显式Laplace适配，不冒称作者原NUTS完整复现 |
| 普通数量与价格关联 | O500严格嵌套O4000；五PD层内暴露置换、固定数值B；复用原500结果 | 更多普通反馈不是所有学习器都更好；代表性高成本关联使完整方法两学习器均值变差，均2/3种子变差 |
| 非线性错设边界 | 原特征池与审批模型，真logit加预设二次项；三个种子两覆盖方向 | XGB完整方法在两条件全部三个种子均不及MinRisk；相对OrdinaryOnly的收益不能替代强参照比较 |

每项配置、源码、逐种子配对、均值与核验都已保存，完整生成数据、选集和预测本地保留。细表同时报告数量、花费及采样标签Brier，不把同预算写成同风险，也不删去与均值反向的种子。

本轮未运行全14条件×30种子或720条确认轨迹，未把旧种子换生成器当作新外部验证。剩余队列是中间覆盖控制、冻结方法的完整十轮近邻比较及按协议独立确认。当前没有新增候选，不以临时造出一个赢家作为继续这些比较的前提。

## 阅读顺序

1. approval_mechanism_small/审批约束普通反馈方向_实验报告.md。
2. nearest_neighbor_small/真实任务近邻适配比较_实验报告.md。
3. approval_volume_cost/普通反馈规模与成本关联_实验报告.md。
4. nonlinear_mechanism_small/非线性错设子集_实验报告.md。

所有结果均属于开发或机制诊断，未给少量开发种子套确认性显著性。本文件是必要审阅入口，不依赖先前聊天。
'''
    (base/'继续按原计划_本轮四项实验报告.md').write_text(brief)
    print(json.dumps({'new_actual_fits':totals,'completed_stages':4,'reports_ready':True},ensure_ascii=False))


if __name__=='__main__':main()
