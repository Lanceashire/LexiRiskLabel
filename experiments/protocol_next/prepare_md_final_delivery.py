"""Necessary standalone review material only; never include raw individual arrays."""
from pathlib import Path
import json,hashlib,shutil,zipfile,re,platform

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];TASK=REPO.parents[1]
BASE=TASK/'outputs/next_protocol_20261027';DEST=REPO/'results/protocol_next_20261027'
PACK=TASK/'outputs/LexiRiskLabel_MD清单完成_必要审阅包_20261029'
PACK.mkdir(exist_ok=True);paths=[]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def publish_copy(source,relative,strip_images=False):
    target=REPO/relative;target.parent.mkdir(parents=True,exist_ok=True)
    if source.resolve()!=target.resolve():
        if strip_images:
            text=re.sub(r'^!\[[^\n]*\]\([^\n]*\)\s*$','',source.read_text(),flags=re.MULTILINE)
            target.write_text(text+'\n\n精简交付说明：原图和逐样本记录保留本地；本文数值表及对应公开汇总为审阅依据。\n')
        else:shutil.copy2(source,target)
    paths.append(relative)


def main():
    final=BASE/'md_execution_complete_20261029';assert json.loads((final/'checklist_verification.json').read_text())['completed_executable_checklist']
    for name in ['实验清单完成总报告.md','实验清单逐项完成表.csv','主比较完整判据.csv','主预算配对结论.csv','主预算逐划分配对.csv','checklist_verification.json','figures/main_paired_gain_conditional.svg']:
        publish_copy(final/name,'results/protocol_next_20261027/md_execution_complete_20261029/'+name)
    formal=['MD第9至10节_完整流程评估报告.md','method_summary.csv','paired_summary.csv','primary_conditional_test_intervals.csv',
        'delta_sensitivity.csv','completion.json','verification.json','round_record_audit.json','formal_selector_visibility_audit.json',
        'registration.json','frozen_identity.json','analysis_complete.json','resume_audit.json']
    for name in formal:publish_copy(BASE/'formal_md'/name,'results/protocol_next_20261027/formal_md/'+name)
    for mode,report in [('transfer','MD第9节_三任务迁移报告.md'),('later','MD第9节_后续时间队列报告.md')]:
        names=[report,'method_summary.csv','paired_summary.csv','completion.json','verification.json']
        names+=['configuration_transfer_summary.json'] if mode=='transfer' else ['data_provenance.json']
        for name in names:publish_copy(BASE/'completion_checks'/mode/name,'results/protocol_next_20261027/completion_checks/'+mode+'/'+name)
    publish_copy(BASE/'completion_checks/registration.json','results/protocol_next_20261027/completion_checks/registration.json')
    # Include previously completed MD stages that were local while bulk publication was stopped.
    old_reports=[('e3/analysis/E3_CORR校正机制实验报告.md','e3/analysis/E3_CORR校正机制实验报告.md'),
        ('e4/analysis/E4_GATE关键配对实验报告.md','e4/analysis/E4_GATE关键配对实验报告.md'),
        ('fantasy_small/XGBoost_幻想重训小实验报告.md','fantasy_small/XGBoost_幻想重训小实验报告.md')]
    for src,dst in old_reports:publish_copy(BASE/src,'results/protocol_next_20261027/'+dst,True)
    for stage in ['e3','e4','fantasy_small']:
        for name in ['completion.json','verification.json']:
            publish_copy(BASE/stage/name,'results/protocol_next_20261027/'+stage+'/'+name)
    for src in ['e3/analysis/dataset_summary.csv','e3/analysis/decision.json','e4/analysis/independent_world_summary.csv','e4/analysis/decision.json',
        'fantasy_small/dataset_summary.csv','fantasy_small/decision.json']:
        publish_copy(BASE/src,'results/protocol_next_20261027/'+src)
    source_names=['experiment_protocol_zh.md','runtime.py','run_formal_md.py','formal_md_config.json','analyze_formal_md.py',
        'export_md_formal_tables.py','audit_md_selector_visibility.py','md_completion_config.json','run_md_completion_checks.py',
        'finalize_md_checklist.py','prepare_md_final_delivery.py','run_fantasy.py','fantasy_config.json',
        'run_nearest_neighbor.py','nearest_neighbor_config.json','run_e3.py','analyze_e3.py','e3_config.json',
        'run_e4.py','analyze_e4.py','e4_config.json']
    for name in source_names:publish_copy(HERE/name,'experiments/protocol_next/'+name)
    status=HERE/'MD_EXECUTION_STATUS.md'
    status.write_text('''# 指定Markdown清单执行状态\n\n以experiment_protocol_zh.md为唯一执行清单。E0—E4及唯一候选诊断完成；E2a为十种子300端点；第9节720条十轮主/补充预算轨迹完成，三任务内部迁移36条、2016后续队列12条，合计768条。第10节统计、第11节日志和核验已完成。\n\n主入口：[清单完成总报告](../../results/protocol_next_20261027/md_execution_complete_20261029/实验清单完成总报告.md)。\n\n候选相对MinRisk的主预算任务均值均负，没有被改称为“胜出算法”。条件预测重采样不等于整个训练流程的不确定性；2016结果是允许的即时标签回顾性敏感性，非严格在线成熟标签验证。完整输入、逐人标签/选集/预测留本地，未恢复明细分片上传。\n''')
    publish_copy(status,'experiments/protocol_next/MD_EXECUTION_STATUS.md')
    for rel in paths:
        src=REPO/rel;dst=PACK/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    # Review context for already-published diagnostics and executable dependencies.
    extra=[(BASE/'analysis/阶段实验报告.md','reports/早期E0_E1_E2b_E2c_E1b.md'),
        (BASE/'e2d/analysis/E2d_数量与构成实验报告.md','reports/E2d_数量构成.md'),
        (BASE/'e2a_full/E2a_十种子完整网格实验报告.md','reports/E2a_十种子完整网格.md'),
        (BASE/'e2a_full/descriptive_summary.csv','evidence/E2a_十种子汇总.csv')]
    for source,rel in extra:
        dst=PACK/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if source.suffix=='.md':dst.write_text(re.sub(r'^!\[[^\n]*\]\([^\n]*\)\s*$','',source.read_text(),flags=re.MULTILINE))
        else:shutil.copy2(source,dst)
    for rel in ['experiments/research_v2/core.py','experiments/research_v2/run_real.py','experiments/research_v2/run_mechanisms.py',
                'experiments/research_v2/config.json','experiments/research_v2/prepare_data.py','experiments/protocol_next/gate_protocol.py',
                'experiments/protocol_next/run_e1b.py','results/research_v2/analysis/method_summary.csv','data/research_v2/provenance.json']:
        if (REPO/rel).exists():
            dst=PACK/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,dst)
    pub=TASK/'work/next_protocol/publication/MD_FINAL_PUBLISHED.json'
    if pub.exists():shutil.copy2(pub,PACK/'publication_receipt.json')
    (PACK/'README_给Pro.md').write_text('''# LexiRiskLabel：指定Markdown实验清单完成交接\n\n先读 `results/protocol_next_20261027/md_execution_complete_20261029/实验清单完成总报告.md` 与逐项完成表，再读formal_md的完整流程报告、条件重采样表，以及completion_checks中迁移和后续年份报告。无需先前聊天。\n\n768条轨迹均已完成，原7200轮主实验日志与预测复核通过。候选主预算相对MinRisk三任务均值为负；正的迁移/后续队列结果也完整保留。统计区间仅为已训练预测的条件敏感性，不证明完整训练流程等效或显著；后续队列缺标签可得时点，不称严格在线。\n\n目录保留仓库相对结构。source配置和代码、关键汇总、核验以及必要旧阶段背景已包含。第三方输入、逐人标签、选集和预测数组仍在本地，因此不声称本包无数据解压即能重跑。逐轮正式记录未采集的同状态边际基线或拟合耗时保留为空，未用不同历史的普通策略填造。\n\n审阅重点：信息和风险账本是否合理；没有新候选稳定优势时哪些贡献可保留；主实验与小量迁移/后续队列的方向差异如何界定。不要把方法名、矩阵恒等式或失败后的扩模块当作创新，也不要默认追加实验网格。\n\nGitHub发布只有在publication_receipt.json存在并标为远端哈希核验通过后才称已完成。原始数据明细未打包或发布。\n''')
    # Text-only public allowlist, explicit hashes; no individual arrays or raw inputs.
    assert len(paths)==len(set(paths))
    assert not any(p.endswith(('.npz','.part')) or '/private/' in p for p in paths)
    manifest={'purpose':'necessary self-contained MD execution review, no raw individual evidence',
        'files':[{'path':p,'sha256':sha(REPO/p),'git_blob_sha1':hashlib.sha1(b'blob '+str((REPO/p).stat().st_size).encode()+b'\0'+(REPO/p).read_bytes()).hexdigest()} for p in paths]}
    mf=REPO/'results/protocol_next_20261027/MD_FINAL_DELIVERY_MANIFEST.json';mf.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    paths.append(str(mf.relative_to(REPO)));dst=PACK/mf.relative_to(REPO);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(mf,dst)
    payload=[{'path':p,'mode':'100644','type':'blob','content':(REPO/p).read_text()} for p in paths]
    publication=TASK/'work/next_protocol/publication';publication.mkdir(exist_ok=True)
    (publication/'MD_FINAL_PAYLOAD.json').write_text(json.dumps(payload,ensure_ascii=False))
    allfiles=sorted(p for p in PACK.rglob('*') if p.is_file() and p.name!='MANIFEST.json')
    pm={'files':[{'path':str(p.relative_to(PACK)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in allfiles]}
    (PACK/'MANIFEST.json').write_text(json.dumps(pm,ensure_ascii=False,indent=2)+'\n')
    archive=TASK/'outputs/LexiRiskLabel_MD实验清单_完成审阅包_20261029.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACK.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(PACK)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(n.endswith(('.npz','.part')) or '/private/' in n for n in z.namelist())
        assert all(hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'] for r in pm['files'])
    print(json.dumps({'public_files':len(paths),'public_UTF8_bytes':sum(len(p['content'].encode()) for p in payload),
        'zip':str(archive),'compressed_bytes':archive.stat().st_size,'package_files':len(allfiles)+1,'integrity_verified':True},ensure_ascii=False))


if __name__=='__main__':main()
