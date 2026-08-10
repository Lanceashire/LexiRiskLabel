from pathlib import Path
import numpy as np, pandas as pd, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import phase1_runner as p1
import phase2_runner as p2
import phase0_engine as p0

OUT=ROOT/'results_phase2'; (OUT/'protocol').mkdir(parents=True,exist_ok=True)
par=[]
for name,cfg in p1.DATASETS.items():
    df=p1.load_dataset(name)
    for seed in range(20):
        a,ma=p2.fixed_split_manifest(name,df,cfg['target'],seed); b,mb=p2.fixed_split_manifest(name,df,cfg['target'],seed)
        par.append({'dataset':name,'seed':seed,'same_test_set':np.array_equal(a['test'],b['test']),'same_D0':np.array_equal(a['visible'],b['visible']),'same_stream_order':np.array_equal(a['stream'],b['stream']),'same_batches':all(np.array_equal(x,y) for x,y in zip(a['batches'],b['batches'])),'test_hash':ma['test_hash'],'d0_hash':ma['d0_hash'],'stream_hash':ma['stream_hash'],'batch_hash':ma['batch_hash']})
pd.DataFrame(par).to_csv(OUT/'PHASE2_SPLIT_PARITY.csv',index=False)

class IdentityPre:
    def transform(self,x): return x.to_numpy(float)
class MockModel:
    def predict_proba(self,x):
        p=1/(1+np.exp(-x[:,0])); return np.column_stack([1-p,p])
raw=pd.DataFrame({'x':[-2.,-1.,0.,1.,2.,3.,-3.,.5],'z':np.arange(8,dtype=float)}); vis=np.array([0,1,2,3]); rej=np.array([4,5]); ref=np.array([4,5,6,7]); pre=IdentityPre(); model=MockModel()
frozen=p0.fave_value(raw,vis,rej,ref,pre,model,gamma=1.0,delta=0.5)
identity_ok=bool(np.allclose(p0.fave_value(raw,vis,rej,ref,pre,model,1.0,0.5),frozen,rtol=1e-12,atol=1e-12))
df=p1.load_dataset('uci'); f=p2.fixed_split_manifest('uci',df,'default',0)[0]; y=df.default.astype(int).to_numpy(); rawx=df[[c for c in df.columns if c not in {'default','ID'}]]; prex,m,pa,bk=p2.fit_predict_gpu(rawx,y,f['visible'],0); batch=f['batches'][0]; cand=batch[pa[batch]>=p2.TAU]; costs=pa[cand]
risk=p2.policy_utility('RiskTilted-Uncertainty',costs,costs,cand,0,1,rawx,f['visible'],batch,prex,m); fave=p2.policy_utility('FAVE-v2.1',costs,costs,cand,0,1,rawx,f['visible'],batch,prex,m); rt_not_fave=bool(len(cand)>0 and not np.allclose(risk,fave,rtol=1e-10,atol=1e-12))
pd.DataFrame([{'gamma':1.0,'delta':0.5,'fave_reference_parity':identity_ok,'risktilted_distinct':rt_not_fave,'compute_backend':bk,'passed':identity_ok and rt_not_fave}]).to_csv(OUT/'PHASE2_FAVE_IDENTITY.csv',index=False)

rng=np.random.RandomState(2026); cost=rng.uniform(.001,.2,30); budget=float(np.sort(cost).cumsum()[9]*1.01); policies=['CountOnly','CountOnly-MinRisk','LeastRisky','MostRisky','Uncertainty','RiskTilted-Uncertainty','FAVE-v2.1']; ks=[]
for pol in policies:
    utility=np.linspace(0.1,1.0,len(cost)) if pol=='FAVE-v2.1' else p2.policy_utility(pol,np.clip(cost,.01,.99),cost,np.arange(len(cost)),0,1,rawx,f['visible'],batch,prex,m)
    _,k=p2.lrbe_select(cost,utility,budget); ks.append(k)
pd.DataFrame([{'policy':pol,'k_star':k,'same_k_star':k==ks[0]} for pol,k in zip(policies,ks)]).to_csv(OUT/'optimizer_invariant.csv',index=False)
print('split parity',bool(pd.DataFrame(par)[['same_test_set','same_D0','same_stream_order','same_batches']].all().all()),'fave',identity_ok,'risktilted_distinct',rt_not_fave,'optimizer',len(set(ks))==1)
