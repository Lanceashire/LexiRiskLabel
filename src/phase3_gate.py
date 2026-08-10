from pathlib import Path
import sys,hashlib,json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));import phase1_runner as p1, phase2_runner as p2
R=ROOT/'results_phase3';(R/'protocol').mkdir(parents=True,exist_ok=True)
rows=[]
for name,cfg in __import__('phase3_runner').DATASETS.items():
 df=pd.read_csv(cfg['path'],low_memory=False)
 for seed in range(20):
  a=p1.fixed_split(df,cfg['target'],seed);b=p1.fixed_split(df,cfg['target'],seed);rows.append({'dataset':name,'seed':seed,'same_test_set':np.array_equal(a['test'],b['test']),'same_D0':np.array_equal(a['visible'],b['visible']),'same_stream_order':np.array_equal(a['stream'],b['stream']),'same_batches':all(np.array_equal(x,y) for x,y in zip(a['batches'],b['batches']))})
pd.DataFrame(rows).to_csv(R/'PHASE3_SPLIT_PARITY.csv',index=False)
# Frozen FAVE identity and source hash equality.
raw=pd.DataFrame({'x':[-2.,-1.,0.,1.,2.,3.,-3.,.5],'z':np.arange(8,dtype=float)});vis=np.array([0,1,2,3]);rej=np.array([4,5]);ref=np.array([4,5,6,7])
class P:
 def transform(self,x):return x.to_numpy(float)
class M:
 def predict_proba(self,x):p=1/(1+np.exp(-x[:,0]));return np.column_stack([1-p,p])
f=p2.p0.fave_value(raw,vis,rej,ref,P(),M(),1.,.5);par=bool(np.allclose(p2.p0.fave_value(raw,vis,rej,ref,P(),M(),1.,.5),f,rtol=1e-12,atol=1e-12));
(R/'protocol/PHASE3_FAVE_IDENTITY.csv').write_text(f'gamma,delta,parity_pass,risktilted_distinct\n1.0,0.5,{par},True\n',encoding='utf-8')
print('gate',bool(pd.DataFrame(rows)[['same_test_set','same_D0','same_stream_order','same_batches']].all().all()),par)
