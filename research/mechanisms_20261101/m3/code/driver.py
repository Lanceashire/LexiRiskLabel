"""Only scheduling varies across workers; frozen world/label/fit identities do not."""
from pathlib import Path
import argparse, concurrent.futures, json, os, subprocess, sys
from study import sha,write
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=True)
    def run(i):
        with (a.out/f'worker_{i}.log').open('a') as log:
            cmd=[sys.executable,'-B',str(HERE/'study.py'),'--plan',str(a.plan.resolve()),'--out',str((a.out/f'worker_{i}').resolve()),'--worker',str(i),'--workers',str(a.workers)]
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        if result.returncode:raise RuntimeError(f'worker {i} exited {result.returncode}; see log')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run,range(a.workers)))
    reports=[]
    for i in range(a.workers):reports.append(json.loads((a.out/f'worker_{i}/completion.json').read_text()))
    assert sum(r['worlds'] for r in reports)==len(plan['world_ids'])
    write(a.out/'completion.json',{'completed':True,'worlds':len(plan['world_ids']),'unique_fits_per_learner':sum(r['fits_per_learner'] for r in reports),'role':plan['role'],'plan_sha256':sha(a.plan),'all_workers_completed':True})
    print(json.dumps(json.loads((a.out/'completion.json').read_text())),flush=True)
if __name__=='__main__':main()
