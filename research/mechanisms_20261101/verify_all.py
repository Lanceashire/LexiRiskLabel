"""Check archive bytes, extract, and recompute all published stage evidence."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tempfile,zipfile
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--examples',action='store_true');args=parser.parse_args()
    manifest=json.loads((ROOT/'CHECKSUMS.json').read_text())
    for rel,h in manifest['files'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    jobs=[('M1_M2_synthesis_20261101.zip','verify_delivery.py'),('M2_original_completion_20261031.zip','code/verify_m2_package.py'),('M2_reversal_20261031.zip','code/verify_package.py'),('M3_cost_pairing_20261101.zip','verify_portable.py')]
    reports={};env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'m1m3-verify-mpl'))
    with tempfile.TemporaryDirectory(prefix='lexirisklabel_m1_m3_') as temp:
        temp=Path(temp)
        for name,entry in jobs:
            with zipfile.ZipFile(ROOT/'archives'/name) as z:
                assert z.testzip() is None
                assert all(not Path(x).is_absolute() and '..' not in Path(x).parts for x in z.namelist())
                tops={Path(x).parts[0] for x in z.namelist()};assert len(tops)==1
                z.extractall(temp)
            target=temp/next(iter(tops));cmd=[sys.executable,'-B',str(target/entry)]
            if name.startswith('M3_') and args.examples:cmd.append('--examples')
            result=subprocess.run(cmd,capture_output=True,text=True,env=env)
            if result.returncode:raise RuntimeError(name+'\n'+result.stdout+'\n'+result.stderr)
            reports[name]=json.loads(result.stdout)
            assert reports[name]['passed']
    output={'passed':True,'all_four_archives_extracted_and_recomputed':True,'checksummed_files':len(manifest['files']),'stages':reports,'new_training_performed':False}
    (ROOT/'verification_result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
