"""Close every executable item in the supplied Markdown, retaining data limits."""
from runtime import *
import argparse


def table(df):
    out=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):out.append('| '+' | '.join(f'{x:+.8f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(out)


def supplemental(base,mode):
    folder=base/'completion_checks'/mode;comp=json.loads((folder/'completion.json').read_text());ver=json.loads((folder/'verification.json').read_text())
    assert comp['completed'] and ver['passed']
    summary=pd.read_csv(folder/'method_summary.csv');pairs=pd.read_csv(folder/'paired_contrasts.csv')
    paired=pairs.groupby(['dataset','reference'],as_index=False).agg(ALC_gain=('ALC_gain','mean'),final_gain=('final_gain','mean'),
        positive_streams=('ALC_gain',lambda x:int((x>1e-12).sum())),negative_streams=('ALC_gain',lambda x:int((x< -1e-12).sum())),
        risk_difference=('risk_difference','mean'))
    paired.to_csv(folder/'paired_summary.csv',index=False)
    if mode=='transfer':
        identity=json.loads((folder/'frozen_identity.json').read_text());config_summary=[]
        for target,p in identity['source_parameters'].items():
            assert target not in p['source_tasks'] and not p['source_or_target_outer_test_used_for_selection']
            config_summary.append({'held_out_task':target,'source_tasks':p['source_tasks'],'ridge':p['ridge'],'XGB':p['xgb'],
                'source_D0_tuning_cases':len(p['source_tuning_records']),'source_tuning_commitment':digest(p['source_tuning_records'])})
        write_json(folder/'configuration_transfer_summary.json',config_summary)
        title='MD第9节：三任务内部迁移轮换'
        text='''依照清单的跨数据集项，完成三次“两个源任务固定配置、第三任务只拟合本地D0”的轮换。每个目标取事前指定的第1重复、第1外层划分的两条重放流，六方法、主预算0.1、完整十轮，共36条轨迹。不是再次展开整套预算网格。

模型配置由另外两个任务的40个D0内层调参结果取众数，平局按原网格顺序；不读源/目标外层T分数，不在目标任务搜索超参数。采集配置沿用冻结版本，目标只拟合本地D0并按允许反馈更新。选中的源配置和输入承诺保存在configuration_transfer_summary.json。

HMEQ、UCI中候选相对MinRisk两个流均有正收益，LendingClub两个流均负；结果异质。两条流不是两个新人群，旧任务均参与过开发，因此只称内部迁移稳健性，不能据此推翻主实验20个相关重放案例的结果。
'''
    else:
        title='MD第9节：未参与开发的2016后续队列回顾性检查'
        provenance=json.loads((folder/'data_provenance.json').read_text())
        text=f'''按清单的后续时间队列项，在获取目标结果前固定数据源、21个决策时特征、标签与筛选规则、方法、预算和两条流。来源为[DePaul公开2016样本]({provenance['url']})，原始字节SHA256为`{provenance['download_sha256']}`。符合固定规则的目标记录为{provenance['external_eligible_rows']}条，全部只作为T，不进入训练、调参、参考或采集。训练与模拟反馈只使用旧2014/2015的{provenance['historical_rows']}条记录；特征角色由历史schema决定，不让目标值改变预处理类型。

状态只保留Fully Paid/Charged Off，与旧数据一致。六方法×两条流×十轮，共12条轨迹。候选相对MinRisk的ALC收益两流均正，均值约0.00009103；相对近邻均值约0.00003876，但只有一流为正。候选终轮误差仍略高于MinRisk/近邻，不能只挑ALC或只挑某条流报告成功。

这是未参与本次开发的后续放款年份的静态外推检查。只有最终状态、没有结果实际可得日期，所以历史2014/2015标签也不能被假称在2016年初全部可得；按清单允许的即时揭示回顾性假设报告，不能宣称严格在线时间因果验证。贷款ID/完全重复特征用于分组，但没有真实借款人跨贷款关联，实际历史拒绝人的反事实仍不可识别。
'''
    report='# '+title+'\n\n'+text+'\n\n## 全方法结果\n\n'+table(summary)+'\n\n## 配对结果\n\n正数为参照Brier−候选Brier。两流仅描述，不新增确认性显著性检验；风险单位仅在任务内比较。\n\n'+table(paired)+'\n\n## 核验\n\n全部预测按原引擎复核，最大误差为0；每轮预算、标签时序、参考与测试分离及实际CUDA后端检查通过。没有修改原720条主实验的模型、分数或配置，也没有引入第二个新候选。完整预测与选集本地保留。\n'
    filename='MD第9节_三任务迁移报告.md' if mode=='transfer' else 'MD第9节_后续时间队列报告.md'
    (folder/filename).write_text(report)
    write_json(folder/'evidence_manifest.json',{'files':[{'path':str(p.relative_to(folder)),'sha256':sha(p)} for p in sorted(folder.rglob('*'))
        if p.is_file() and p.name!='evidence_manifest.json' and 'private' not in p.relative_to(folder).parts]})
    return paired


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);a=ap.parse_args();base=a.base
    out=base/'md_execution_complete_20261029';out.mkdir(exist_ok=True)
    identity=json.loads((base/'formal_md/frozen_identity.json').read_text());assert sha(HERE/'experiment_protocol_zh.md')==identity['config']['controlling_protocol_sha256']
    assert json.loads((base/'formal_md/completion.json').read_text())['completed_episodes']==720
    assert json.loads((base/'formal_md/verification.json').read_text())['passed']
    assert json.loads((base/'formal_md/round_record_audit.json').read_text())['passed']
    assert json.loads((base/'formal_md/formal_selector_visibility_audit.json').read_text())['passed']
    transfer=supplemental(base,'transfer');later=supplemental(base,'later')
    intervals=pd.read_csv(base/'formal_md/primary_conditional_test_intervals.csv');effects=pd.read_csv(base/'formal_md/paired_summary.csv')
    summary=pd.read_csv(base/'formal_md/method_summary.csv');main_summary=summary[summary.rho==.1]
    flags=intervals[['dataset','comparison','mean_gain','conditional_L95','conditional_U95','conditional_p_Holm','delta']].copy()
    flags['conditional_CI_below_zero']=flags.conditional_U95<0
    flags['conditional_CI_within_delta_band']=(flags.conditional_L95>=-flags.delta)&(flags.conditional_U95<=flags.delta)
    flags['Holm_rejects_zero_at_005']=flags.conditional_p_Holm<.05
    flags['conditional_L95_exceeds_delta']=flags.conditional_L95>flags.delta
    flags.to_csv(out/'主比较完整判据.csv',index=False)
    assert not flags.Holm_rejects_zero_at_005.any() and flags.conditional_CI_within_delta_band.all()
    main_effect=effects[effects.rho==.1].merge(intervals[['dataset','comparison','conditional_L95','conditional_U95','conditional_p_Holm','delta']],on=['dataset','comparison'],validate='one_to_one')
    refs=main_summary.set_index(['dataset','method']).ALC_Brier
    main_effect['relative_gain_percent']=[100*r.mean_gain/refs.loc[(r.dataset,r.comparison.replace('FantasyCandidate_vs_',''))] for r in main_effect.itertuples()]
    main_effect.to_csv(out/'主预算配对结论.csv',index=False)
    pd.read_csv(base/'formal_md/paired_episode_contrasts.csv').query('rho == 0.1').to_csv(out/'主预算逐划分配对.csv',index=False)
    sections=[
        ('1','唯一主问题','完成','所有主要效果以相同协议下相对MinRisk/近邻的真实Brier和风险报告'),
        ('2','信息边界','完成','D/O/C/R/T分离；先锁定再释放；跨期只作允许的回顾性假设'),
        ('3/E0','实现与协议审计','完成','原分标签扰动、身份拒绝及新六方法定点扰动全部通过'),
        ('4/E1','共同状态与学习器分叉','完成','18状态，432拟合槽；193不同干预，386实际模型；不把轮次当独立种子'),
        ('5/E1b','有限可行选集诊断','完成','6固定状态×16不同集合；Vsearch选择、Veval评价，额外研究标签单列'),
        ('6/E2a','精确正控制','完成','十种子、300终点；保留原90端点，逐池穷举；代理与LR收益分开'),
        ('6/E2b','成对搜索与交换拆分','完成','18状态四格，保存全部Gamma；不把交换效果归给成对构造'),
        ('6/E2c','表示匹配与参考规模','完成','D-only冻结20棵深度2叶表示；Q128/512单列'),
        ('6/E2d','数量与构成','完成','18去重数量状态；31风险匹配成功，5不可估计，未放宽容差'),
        ('7/E3','校正标签使用消融','完成并按门槛关闭主线','固定S/pi/g/回归器，含插件和两种收缩；完整piij方差核验'),
        ('8/E4','GATE关键配对','完成并按门槛停止扩展','规划/环境时序统一，真假释放及审批控制、独立世界和真生成器核验'),
        ('9','完整采集流程内部评估','完成','60注册外层/重放单元，720条十轮轨迹，D0内层调参和月度前向划分'),
        ('9/迁移','三任务轮换','完成','另外两任务固定配置，目标不调参；36条主预算轨迹，称内部迁移'),
        ('9/时间队列','后续年份检查','按数据条件完成回顾性版本','2016预定义9168条目标记录仅评估，12轨迹；无标签可得时点，不称严格在线'),
        ('10','指标与统计','按可支持范围完成','主预算固定；绝对/相对效果、delta敏感性、模型条件重采样与Holm；未估计完整训练不确定性'),
        ('11','输出和证据','完成','确认表768轨迹；主实验状态/选集/结果各7200轮，原始选集/预测留本地'),
        ('12','最多一个候选修改','完成','叶表示和唯一幻想候选已诊断；候选开发失败，固定保留作比较，不称胜出'),
        ('13','支持/否定/不确定和停止决定','完成','无稳定新候选优势，不继续扩大CORR/GATE或新目标；正负结果全部保留')]
    checklist=pd.DataFrame(sections,columns=['清单章节','要求','状态','证据与范围']);checklist.to_csv(out/'实验清单逐项完成表.csv',index=False)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.8),sharey=True)
    lo=intervals.conditional_L95.min();hi=intervals.conditional_U95.max();span=hi-lo
    for ax,dataset in zip(axes,identity['config']['datasets']):
        q=intervals[intervals.dataset==dataset].set_index('comparison')
        for j,ref in enumerate(['MinRisk','ParBaLS']):
            r=q.loc['FantasyCandidate_vs_'+ref];x=r.mean_gain*1e4
            ax.errorbar(x,j,xerr=[[(r.mean_gain-r.conditional_L95)*1e4],[(r.conditional_U95-r.mean_gain)*1e4]],fmt='o',color='#b65e29' if x<0 else '#216994',capsize=4)
        ax.axvline(0,color='#555555',lw=1);ax.set_xlim((lo-.12*span)*1e4,(hi+.12*span)*1e4);ax.set_title(dataset.upper() if dataset!='lendingclub' else 'LendingClub')
        ax.set_yticks([0,1],['vs MinRisk','vs ParBaLS']);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Brier gain × 10⁻⁴ (positive = better)')
    axes[0].invert_yaxis();fig.suptitle('Fixed candidate: main-budget paired effects',fontsize=13)
    fig.text(.5,.015,'95% conditional test-group intervals; unadjusted intervals. All six Holm-adjusted p values > 0.05. Training uncertainty excluded.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(figs/'main_paired_gain_conditional.svg');fig.savefig(figs/'main_paired_gain_conditional.png',dpi=160);plt.close(fig)
    compact=main_effect[['dataset','comparison','mean_gain','relative_gain_percent','positive_replay_cases','negative_replay_cases','conditional_p_Holm']]
    report='''# LexiRiskLabel：按指定Markdown实验清单的完成报告

执行依据：用户提供的experiment_protocol_zh.md，SHA256为052b064834a804f613e016aee007cee79b4fb201205d4135c7d3517886d8c9e8。当前报告只将这份Markdown列为执行清单；旧DOCX追加实验保留为历史补充，不计入本清单的完成依据。

## 完成了什么

E0/E1/E1b/E2a—E2d/E3/E4以及唯一候选诊断均已完成。E2a现为十种子300终点完整网格。第9节完成360条主预算和360条补充预算的十轮轨迹，共720条；另按该节完成三任务配置轮换36条、后续2016队列12条，共768条完整轨迹。后两项未扩展补充预算或增加方法。

外层每次重新划分、拟合预处理、建立初始可见集、训练业务定价模型、构造独立参考、在D0内调参，再完整采集与重训。HMEQ/UCI采用重复分组五折，LendingClub采用注册月份前向窗口。现有7200轮可见历史、采集去重、预算与实际损失账本已重建核验；所有保存预测重新计分误差为0。第11节三种逐轮表各7200行；未记录的同状态边际基线或逐模型耗时保留为空，未从不同历史的普通轨迹伪造。

## 主结果

正数为参照Brier−候选Brier，负数表示候选更差。主预算0.1保持不变；下表的20个重放案例相互相关，不作为20个独立总体样本：

'''+table(compact)+'''

![主预算配对效果](figures/main_paired_gain_conditional.svg)

候选相对MinRisk的三个任务平均收益全部为负；相对近邻只有LendingClub均值微正。条件重采样给出的六个Holm调整p均大于0.05；CV重叠训练造成的依赖没有通过重训重采样覆盖，因此这些p与区间只作为固定预测集合的条件敏感性，不能当作整个流程已校准的确认检验。主实验描述结果也未形成跨任务稳定优势。没有在结果出来后更换主预算、模型网格、伪标签世界或新候选。

候选同时通常少买标签、少花探索风险，不能把预测排序简单当作完整经济排序；风险、数量和损失逐任务单列，不跨货币单位相加。0.5%研究阈值下，模型条件区间均落入等效带，但这些区间只条件于本批已训练模型，既不是整个训练过程的等效证明，也不是利润结论。区间低于零和落入等效带可以同时成立，完整布尔判据另列，不以互斥标签隐藏重叠。

## 迁移与后续队列

三任务内部迁移：

'''+table(transfer)+'''

2016后续队列：

'''+table(later)+'''

迁移中HMEQ、UCI相对MinRisk出现正均值，LendingClub负；这是两个重放流的内部稳健性，不推翻主实验。2016固定合格队列9168条只用于评价，候选相对MinRisk的ALC收益约+0.00009103，两流均正；相对近邻约+0.00003876但只有一流正，终轮指标也并未一致胜出。正结果完整保留，不能挑它替代主要结果。

只有最终贷款状态、没有结果实际可得时点，因此2016检查按清单允许的即时标签回顾性假设报告，不声称严格as-of在线成熟标签验证、机构现场因果效果或真实拒绝人的反事实。贷款ID/精确重复特征分组也不等于拥有真实跨贷款借款人关联。

## 清单逐项完成情况

'''+table(checklist)+'''

## 结论与交付

清单中可执行的实验及核验已完成。当前证据不支持把唯一新候选描述为稳定优于MinRisk的算法；开发失败和正式比较负结果均保留。E3/E4按原门槛停止扩展，不添加CVaR、公平性或生存模型。候选“胜出”不是被人为制造的完成条件。

完整训练抽样不确定性、标签真实成熟时点、借款人跨贷款关联与真实拒绝标签不由这些数据识别，未假称完成这类不具数据条件的推断。必要代码、冻结配置、报告、汇总和核验整理为独立审阅材料；完整第三方输入、选集、标签与逐样本预测保留本地。
'''
    (out/'实验清单完成总报告.md').write_text(report)
    checks={'formal_MD':base/'formal_md/verification.json','round_audit':base/'formal_md/round_record_audit.json',
        'selector_audit':base/'formal_md/formal_selector_visibility_audit.json','transfer':base/'completion_checks/transfer/verification.json',
        'later':base/'completion_checks/later/verification.json','E2a_full':base/'e2a_full/verification.json'}
    verification={'completed_executable_checklist':True,'main_episodes':360,'supplementary_episodes':360,'transfer_episodes':36,'later_episodes':12,
        'formal_rounds':7200,'total_complete_episodes':768,'no_false_strict_online_or_training_uncertainty_claim':True,
        'checks':{name:{'path':str(p.relative_to(base)),'sha256':sha(p)} for name,p in checks.items()},'finalizer_sha256':sha(__file__)}
    write_json(out/'checklist_verification.json',verification)
    state=json.loads((base/'plan_execution_state.json').read_text());state.update({'status':'completed_executable_MD_checklist_with_stated_data_limits',
        'current_MD_section':13,'current_action':'completed experiments, analysis and verification; publishing necessary self-contained materials',
        'remaining_experiment_queue':[],'next_queue':[],'final_checklist_report':str((out/'实验清单完成总报告.md').relative_to(base))})
    write_json(base/'plan_execution_state.json',state)
    write_json(out/'evidence_manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json']})
    print(json.dumps(verification,ensure_ascii=False));print(compact.to_string(index=False))


if __name__=='__main__':main()
