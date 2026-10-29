"""Verify detailed archives and every member (requires zstd executable)."""
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
