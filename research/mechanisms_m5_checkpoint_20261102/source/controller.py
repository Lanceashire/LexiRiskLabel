"""One local checkpointed service; fixed 40 tasks, then audit and one evaluation."""
from common import *
from replay import task,verify_code,CFG
from concurrent.futures import ProcessPoolExecutor,as_completed
import datetime,fcntl,subprocess,traceback,multiprocessing
def status(stage,**kw):
    row={'stage':stage,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),**kw};write(HERE/'progress.json',row);print(json.dumps(row),flush=True)
def main():
    lock=(HERE/'service.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_code()
    tasks=[(name,seed) for name in ['LendingClub2017','SouthGermanCredit'] for seed in CFG['seeds']]
    completed=0;status('training',tasks_total=len(tasks),tasks_complete=0,workers=8)
    try:
        with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('fork')) as pool:
            futures={pool.submit(task,*job):job for job in tasks}
            for future in as_completed(futures):
                result=future.result();completed+=1;status('training',tasks_total=40,tasks_complete=completed,last=result)
        status('raw_prediction_and_lock_audit',tasks_complete=40)
        subprocess.run([sys.executable,'-B',str(HERE/'raw_audit.py'),'--workers','8'],check=True)
        assert json.loads((HERE/'raw_audit.json').read_text())['passed']
        status('all_models_audited_evaluating_sealed_T_once',tasks_complete=40)
        subprocess.run([sys.executable,'-B',str(HERE/'analysis.py')],check=True)
        status('analysis_complete_report_pending',tasks_complete=40)
    except BaseException:
        status('failed',tasks_complete=completed,error=traceback.format_exc());raise
if __name__=='__main__':main()
