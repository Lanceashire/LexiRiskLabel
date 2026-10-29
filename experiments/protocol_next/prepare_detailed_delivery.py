"""Package retained experiment evidence, without source credit-feature datasets.

This exports existing results only. It never calls a learner or changes the protocol.
Run from the original workspace; archives preserve paths relative to the output root.
"""
from pathlib import Path
import csv, hashlib, json, re, zipfile
from collections import defaultdict
import io, subprocess, tarfile

REPO = Path(__file__).resolve().parents[2]
TASK = REPO.parents[1]
BASE = TASK / 'outputs/next_protocol_20261027'
DEST = REPO / 'results/protocol_next_20261027/details_20261029'
ARCH = DEST / 'archives'
PUB = TASK / 'work/next_protocol/publication'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_record(p):
    data = p.read_bytes()
    return {'path': str(p.relative_to(REPO)), 'bytes': len(data),
            'sha256': digest(data),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ARCH.mkdir(parents=True, exist_ok=True)
    profiles = [('formal', BASE/'formal_md', BASE/'private/formal_md/cases')]
    for task in ['hmeq', 'uci', 'lendingclub']:
        profiles.append(('transfer_' + task, BASE/'completion_checks/transfer',
                         BASE/'completion_checks/transfer'/task/'private/formal_md/cases'))
    profiles.append(('later_2016', BASE/'completion_checks/later',
                     BASE/'completion_checks/later/lendingclub_2016/private/formal_md/cases'))
    episodes, rounds, assignments = [], [], defaultdict(list)
    counts = {}
    for profile, summary, cases in profiles:
        eps = list(csv.DictReader((summary/'episode_summary.csv').open()))
        if profile.startswith('transfer_'):
            eps = [r for r in eps if r['dataset'] == profile.removeprefix('transfer_')]
        counts[profile] = len(eps)
        for r in eps:
            r = dict(r, experiment_profile=profile, record_id=profile+'::'+r['episode_id'])
            episodes.append(r)
        for case in sorted(cases.iterdir()):
            if not case.is_dir():
                continue
            if profile == 'formal':
                m = re.fullmatch(r'(.*)_repeat\d+_fold(\d+)_stream\d+', case.name)
                assert m, case.name
                bundle = f'formal_{m[1]}_fold{m[2]}'
            else:
                bundle = profile
            for p in sorted(case.rglob('*')):
                if p.is_file():
                    assert p.suffix in {'.json', '.npz', '.csv'}, p
                    assignments[bundle].append(p)
                    if p.name == 'trajectory.csv':
                        rr = list(csv.DictReader(p.open()))
                        assert len(rr) == 10
                        for r in rr:
                            rounds.append(dict(r, experiment_profile=profile,
                                               episode_id=p.parent.name,
                                               record_id=profile+'::'+p.parent.name+'::round'+r['round']))
    assert len(episodes) == 768 and len(rounds) == 7680
    assert len({r['record_id'] for r in episodes}) == 768
    assert len({r['record_id'] for r in rounds}) == 7680
    write_csv(DEST/'all_768_episodes.csv', episodes)
    write_csv(DEST/'all_7680_rounds.csv', rounds)

    # Earlier shared prediction cache is stored once, in four stable hash buckets.
    # No byte fragments: each archive is an independently extractable ZIP.
    for directory in sorted((BASE/'private').iterdir()):
        if directory.name == 'formal_md' or not directory.is_dir():
            continue
        for p in sorted(directory.rglob('*')):
            if not p.is_file():
                continue
            if directory.name == 'predictions':
                bucket = int(p.stem[:2], 16) % 4
                bundle = f'earlier_shared_predictions_cache{bucket+1}'
            else:
                bundle = 'earlier_' + directory.name
            assignments[bundle].append(p)

    # Detailed metric tables and fixed configurations, excluding any nested private
    # directories already archived above. Supplementary DOCX-era evidence is retained
    # and explicitly labelled historical, not an additional MD experiment requirement.
    for p in sorted(BASE.rglob('*')):
        if not p.is_file() or 'private' in p.relative_to(BASE).parts:
            continue
        if p.suffix not in {'.csv', '.json', '.jsonl', '.md', '.py', '.txt', '.svg'}:
            continue
        assignments['all_stage_tables_configs_and_audits'].append(p)

    archive_records, member_count, source_bytes = [], 0, 0
    coverage = set()
    for name, files in sorted(assignments.items()):
        target = ARCH/(name+'.zip')
        manifest = []
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in files:
                rel = p.relative_to(BASE).as_posix()
                assert rel not in coverage, rel
                coverage.add(rel)
                data = p.read_bytes()
                # The approved evidence roots contain indices, costs, labels,
                # predictions, probability/weight arrays, and synthetic inputs.
                # Real source credit matrices are outside BASE, and never included.
                assert not p.name.startswith('LC_Loan_sample_')
                manifest.append({'path': rel, 'bytes': len(data), 'sha256': digest(data)})
                info = zipfile.ZipInfo(rel, date_time=(2026, 10, 29, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                z.writestr(info, data, compresslevel=9)
                source_bytes += len(data)
            z.writestr('ARCHIVE_MANIFEST.json', json.dumps({'members': manifest}, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None
            assert all(digest(z.read(r['path'])) == r['sha256'] for r in manifest)
        r = file_record(target)
        r.update(member_files=len(manifest), original_bytes=sum(x['bytes'] for x in manifest),
                 integrity_verified=True)
        archive_records.append(r)
        member_count += len(manifest)
        print(json.dumps({'archive': name, 'bytes': r['bytes'], 'files': len(manifest)}), flush=True)

    index = {'created': '2026-10-29', 'purpose': 'retained detailed experiment evidence; no new fitting',
             'formal_and_completion_episodes': len(episodes), 'rounds': len(rounds),
             'profile_counts': counts, 'archived_original_files': member_count,
             'original_bytes': source_bytes, 'archives': archive_records,
             'source_root': 'outputs/next_protocol_20261027',
             'exclusions': ['third-party original credit-feature datasets', 'credentials', 'Python environments'],
             'limits': ['No unmeasured fit durations or same-state counterfactual outcomes were invented.',
                        'Historical DOCX supplementary folders are retained evidence, not part of the MD checklist.',
                        'Labels and predictions are retained experimental outcomes, not online label-availability evidence.']}
    (DEST/'DETAILS_INDEX.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
    (DEST/'README.md').write_text('''# 实验详细数据补传（2026-10-29）

本目录补充已经完成实验的明细，不新增实验，也不改变原结论。先读[实验清单完成总报告](../md_execution_complete_20261029/实验清单完成总报告.md)。

- [全部768条轨迹](all_768_episodes.csv)：主/补充预算720条、三任务配置迁移36条、2016后续队列12条。
- [全部7680轮记录](all_7680_rounds.csv)：每轮Brier、分组指标、选取数量、预算与风险账本、后端和选集哈希。
- [完整文件索引与SHA-256](DETAILS_INDEX.json)：列出所有独立ZIP、原文件数量和压缩前后大小。每个ZIP内还包含逐文件校验清单。
- `archives/formal_<任务>_fold<折号>.zip`：两次划分重复和两次回放，包含选集、D/O/C/R行号、候选成本、划分、初始调参记录、逐轮日志、测试标签及预测。
- `archives/transfer_*.zip`、`later_2016.zip`：迁移及后续队列的同类明细。
- `archives/earlier_*.zip`：早期诊断的逐集合结果、选集、π/Γ、伪目标、预测和模拟世界参数等已保存证据。共享预测缓存按文件哈希分四组，均可独立解压，无二进制分片。
- `archives/all_stage_tables_configs_and_audits.zip`：各阶段完整表格、配对明细、配置、审计、源码快照和报告。

`experiment_profile`与`record_id`区分不同阶段中重用的case/episode名称，合表时请保留。归档路径相对于`outputs/next_protocol_20261027`，全部ZIP可解压到同一目录，路径不重叠。用Python标准库zipfile即可读取，预测数组用NumPy读取。

第三方原始信用特征输入、环境和凭据未上传。历史DOCX补充文件夹（ordinary_factorial_small、approval_mechanism_small、nearest_neighbor_small、approval_volume_cost、nonlinear_mechanism_small）仅作为已做工作的历史记录，不计入指定Markdown清单的768条轨迹。所有不利结果完整保留。未采集的同状态边际基线和单次拟合耗时保持缺失；不同策略历史下的差值不能当成同状态边际收益。统计区间仍只描述已训练预测的条件敏感性，2016队列仍受标签可得时点缺失限制。

下载和校验：将本目录整体下载后运行`python verify_details.py`。也可单独下载需要的ZIP及DETAILS_INDEX.json，运行`python verify_details.py --allow-missing`；缺失包会明确列出，不会当成完整交付。
''')
    (DEST/'verify_details.py').write_text('''"""Verify downloaded detailed archives and all original members, using stdlib."""
from pathlib import Path
import argparse, hashlib, json, zipfile
p = argparse.ArgumentParser(); p.add_argument('--allow-missing', action='store_true'); a = p.parse_args()
root = Path(__file__).resolve().parent
index = json.loads((root/'DETAILS_INDEX.json').read_text())
missing = []; verified = 0
for r in index['archives']:
    path = root/'archives'/Path(r['path']).name
    if not path.exists(): missing.append(path.name); continue
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r['sha256'], path
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('ARCHIVE_MANIFEST.json'))['members']
        assert len(manifest) == r['member_files']
        for item in manifest:
            assert hashlib.sha256(z.read(item['path'])).hexdigest() == item['sha256'], item['path']
    verified += 1
print(json.dumps({'verified_archives': verified, 'missing': missing, 'complete': not missing}))
if missing and not a.allow_missing: raise SystemExit(1)
''')
    paths = [p for p in sorted(DEST.rglob('*')) if p.is_file()]
    paths += [Path(__file__).resolve()]
    (PUB/'DETAILS_UPLOAD_MANIFEST.json').write_text(json.dumps({'files': [file_record(p) for p in paths]}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'archives': len(archive_records), 'bytes': sum(r['bytes'] for r in archive_records),
                      'original_files': member_count, 'profiles': counts, 'verified': True}))


def convert_to_zstd():
    """Preserve every member byte, compress across files to reduce download size."""
    index = json.loads((DEST/'DETAILS_INDEX.json').read_text())
    for record in index['archives']:
        source = REPO/record['path']
        target = source.with_suffix('.tar.zst')
        with target.open('wb') as out:
            process = subprocess.Popen(['zstd', '-q', '-T2', '-10', '-c'],
                                       stdin=subprocess.PIPE, stdout=out)
            with zipfile.ZipFile(source) as z, tarfile.open(fileobj=process.stdin, mode='w|') as tar:
                for name in z.namelist():
                    data = z.read(name)
                    info = tarfile.TarInfo(name); info.size = len(data); info.mtime = 0
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(data))
            process.stdin.close()
            assert process.wait() == 0
        # Check every member after actual decompression, not just archive headers.
        with zipfile.ZipFile(source) as z:
            manifest = json.loads(z.read('ARCHIVE_MANIFEST.json'))['members']
        expected = {r['path']: r for r in manifest}
        process = subprocess.Popen(['zstd', '-q', '-d', '-c', str(target)], stdout=subprocess.PIPE)
        verified = set()
        with tarfile.open(fileobj=process.stdout, mode='r|') as tar:
            for member in tar:
                data = tar.extractfile(member).read()
                if member.name == 'ARCHIVE_MANIFEST.json':
                    continue
                assert digest(data) == expected[member.name]['sha256']
                verified.add(member.name)
        assert process.wait() == 0 and verified == set(expected)
        record.update(file_record(target))
        print(json.dumps({'verified_zstd': target.name, 'bytes': record['bytes']}), flush=True)
    index['archive_format'] = 'tar.zst; member bytes identical to locally verified ZIP originals'
    (DEST/'DETAILS_INDEX.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
    readme = (DEST/'README.md').read_text()
    readme = readme.replace('.zip', '.tar.zst').replace('ZIP', 'tar.zst归档')
    readme = readme.replace('用Python标准库zipfile即可读取，预测数组用NumPy读取。',
                            '用tar --zstd -xf 文件.tar.zst解压（需要zstd），预测数组用NumPy读取。')
    readme += '\n用户已明确授权公开逐样本标签、选集行号与预测等全部上述派生明细。为减少下载体积，公开归档改用跨文件压缩的tar.zst；各成员字节与原ZIP完全一致，已解压逐文件核验。\n'
    (DEST/'README.md').write_text(readme)
    (DEST/'verify_details.py').write_text('''"""Verify detailed archives and every member (requires zstd executable)."""
from pathlib import Path
import argparse, hashlib, json, subprocess, tarfile
p = argparse.ArgumentParser(); p.add_argument('--allow-missing', action='store_true'); a = p.parse_args()
root = Path(__file__).resolve().parent; index = json.loads((root/'DETAILS_INDEX.json').read_text())
missing = []; verified = 0
for r in index['archives']:
    path = root/'archives'/Path(r['path']).name
    if not path.exists(): missing.append(path.name); continue
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r['sha256'], path
    process = subprocess.Popen(['zstd', '-q', '-d', '-c', str(path)], stdout=subprocess.PIPE)
    hashes = {}; manifest = None
    with tarfile.open(fileobj=process.stdout, mode='r|') as tar:
        for member in tar:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            data = tar.extractfile(member).read()
            if member.name == 'ARCHIVE_MANIFEST.json': manifest = json.loads(data)['members']
            else: hashes[member.name] = hashlib.sha256(data).hexdigest()
    assert process.wait() == 0 and manifest is not None
    assert len(manifest) == r['member_files'] == len(hashes)
    assert all(hashes[m['path']] == m['sha256'] for m in manifest)
    verified += 1
print(json.dumps({'verified_archives': verified, 'missing': missing, 'complete': not missing}))
if missing and not a.allow_missing: raise SystemExit(1)
''')
    public = [DEST/'README.md', DEST/'DETAILS_INDEX.json', DEST/'verify_details.py',
              DEST/'all_768_episodes.csv', DEST/'all_7680_rounds.csv', Path(__file__).resolve()]
    public += [REPO/r['path'] for r in index['archives']]
    (PUB/'DETAILS_UPLOAD_MANIFEST.json').write_text(json.dumps({'files': [file_record(p) for p in public]}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'archives': len(index['archives']), 'bytes': sum(r['bytes'] for r in index['archives']),
                      'member_files': index['archived_original_files'], 'verified': True}))


if __name__ == '__main__':
    import sys
    if '--convert-existing' not in sys.argv:
        main()
    convert_to_zstd()
