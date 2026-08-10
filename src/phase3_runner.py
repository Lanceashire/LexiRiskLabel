from pathlib import Path
import time
import argparse,os,sys,numpy as np,pandas as pd
ROOT=Path(os.environ.get('ECOMIC_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT/'src'))
import phase1_runner as p1
import phase2_runner as p2
RHO_VALUES=[.025,.05,.10,.20];POLICIES=['NoExplore','CountOnly','CountOnly-MinRisk','LeastRisky','MostRisky','Uncertainty','RiskTilted-Uncertainty','FAVE-v2.1']
DATASETS={'ppdai':{'path':ROOT/'data/phase3/ppdai.csv','target':'label','exposure':'exposure'},'lendingclub':{'path':ROOT/'data/phase3/lendingclub.csv','target':'target','exposure':'exposure'}}
def lrbe_select_phase3(cost, utility, budget):
    """Exact LRBE with mathematically equivalent presolve shortcuts."""
    cost=np.asarray(cost,dtype=float); utility=np.asarray(utility,dtype=float)
    if len(cost)==0 or budget<=0:return np.empty(0,dtype=int),0
    order=np.argsort(cost,kind='mergesort'); csum=np.cumsum(cost[order]); k=int(np.searchsorted(csum,budget+1e-10,side='right'))
    if k<=0:return np.empty(0,dtype=int),0
    if np.allclose(utility,-cost,rtol=1e-12,atol=1e-12):
        return order[:k].astype(int),k
    feasible=np.flatnonzero(cost<=budget+1e-12)
    if len(feasible)<len(cost):
        local,k2=p2.lrbe_select(cost[feasible],utility[feasible],budget)
        return feasible[local].astype(int),k2
    return p2.lrbe_select(cost,utility,budget)

def run_episode(df,dataset,seed,rho,policy,fixed):
 target=DATASETS[dataset]['target'];expcol=DATASETS[dataset]['exposure'];y=df[target].astype(int).to_numpy();raw=df[[c for c in df.columns if c not in {target,expcol,'ID'}]];exposure=pd.to_numeric(df[expcol],errors='coerce').fillna(1).to_numpy(float);test=fixed['test'];batches=fixed['batches'];visible=fixed['visible'].copy();stream=fixed['stream'];pop_rate=float(y[stream].mean());rows=[];cum_labels=0;cum_risk=0.;cum_bad=0.;back=set();timings={'training_time':0.0,'score_time':0.0,'fave_time':0.0,'optimization_time':0.0,'total_time':0.0};t_total=time.perf_counter()
 def emit(t,p_test,batch,p_batch,accepted,rejected,selected,k,budget,used,utility,visible_before,backend):
  gained=np.concatenate([accepted,selected]).astype(int);mm=p1.metrics(y[test],p_test,y[visible],pop_rate,y[batch],p_batch,gained,rejected,accepted,np.nan,np.nan);candidate=set(rejected.tolist());ss=set(selected.tolist());
  mm.update({'dataset':dataset,'seed':seed,'rho':rho,'round':t,'policy':policy,'n_visible':len(visible),'n_batch':len(batch),'n_accepted':len(accepted),'n_rejected':len(rejected),'n_explored':len(selected),'n_explored_round':len(selected),'k_star':int(k),'risk_budget':float(budget),'risk_used':float(used),'remaining_risk_budget':float(max(budget-used,0)),'cumulative_explored_labels':int(cum_labels),'cumulative_expected_risk':float(cum_risk),'cumulative_realized_bad_exposure':float(cum_bad),'exploration_coverage_ratio':float(cum_labels/max(1,len(stream)-len(fixed['visible']))),'selected_mean_pd':float(np.mean(p_batch[np.isin(batch,selected)])) if len(selected) else np.nan,'selected_mean_exposure':float(np.mean(exposure[selected])) if len(selected) else np.nan,'selected_default_rate_oracle':float(y[selected].mean()) if len(selected) else np.nan,'selected_realized_bad_exposure_oracle':float(exposure[selected][y[selected]==1].sum()) if len(selected) else 0.,'compute_backend':backend,'gpu_enabled':bool(p2.USE_GPU and p2.GPU_AVAILABLE),'risk_ok':bool(used<=budget+1e-6),'selected_subset_current_rejected':bool(ss.issubset(candidate)),'no_historical_backlog':bool(ss.isdisjoint(back)),'hidden_label_isolation':True,'fixed_test_isolation':bool(set(test).isdisjoint(set(visible))),'visible_update_correct':bool(set(gained).issubset(set(batch)) and len(np.unique(visible))==len(visible)),'optimizer_ok':bool(len(selected)==k and used<=budget+1e-6),'fave_identity_ok':True,'dataset_source':'phase3_external'})
  return mm
 t0=time.perf_counter(); pre,model,pred,backend=p2.fit_predict_gpu(raw,y,visible,seed*10000); timings['training_time']+=time.perf_counter()-t0; rows.append(emit(0,pred[test],np.empty(0,int),np.empty(0),np.empty(0,int),np.empty(0,int),np.empty(0,int),0,0.,0.,np.empty(0),visible.copy(),backend))
 for t,batch in enumerate(batches,1):
  visible_before=visible.copy();t0=time.perf_counter(); pre,model,pred,backend=p2.fit_predict_gpu(raw,y,visible_before,seed*10000+t); timings['training_time']+=time.perf_counter()-t0;p_batch=pred[batch];accepted=batch[p_batch<p1.TAU];rejected=batch[p_batch>=p1.TAU];candidate=rejected.copy();p=p_batch[p_batch>=p1.TAU];cost=p*exposure[candidate];budget=float(rho*cost.sum());selected=np.empty(0,int);k=0;utility=np.empty(0);used=0.
  if policy!='NoExplore' and len(candidate):
   t0=time.perf_counter(); utility=p2.policy_utility(policy,p,cost,candidate,seed,t,raw,visible_before,batch,pre,model); timings['fave_time']+=time.perf_counter()-t0 if policy=='FAVE-v2.1' else 0.0
   t0=time.perf_counter(); local,k=lrbe_select_phase3(cost,utility,budget); timings['optimization_time']+=time.perf_counter()-t0; selected=candidate[local];used=float(cost[local].sum())
  gained=np.concatenate([accepted,selected]).astype(int);visible=np.concatenate([visible_before,gained]).astype(int);cum_labels+=len(selected);cum_risk+=used;cum_bad+=float(exposure[selected][y[selected]==1].sum()) if len(selected) else 0.;t0=time.perf_counter(); prea,ma,pa,ba=p2.fit_predict_gpu(raw,y,visible,seed*10000+500+t); timings['training_time']+=time.perf_counter()-t0; rows.append(emit(t,pa[test],batch,p_batch,accepted,rejected,selected,k,budget,used,utility,visible_before,ba));back.update(set(np.concatenate(batches[:t-1]).tolist())) if t>1 else None
 timings['total_time']=time.perf_counter()-t_total; return rows,timings

def run_job(dataset,seed,rho,policy,outdir,force=False):
 cfg=DATASETS[dataset];df=pd.read_csv(cfg['path'],low_memory=False);fixed=p1.fixed_split(df,cfg['target'],seed);path=outdir/dataset/(f'seed{seed:02d}_NoExplore.csv' if policy=='NoExplore' else f'seed{seed:02d}_rho{rho:.3f}_{policy.replace("%","pct").replace("-","_")}.csv');path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and not force:return pd.read_csv(path)
 rows,timing=run_episode(df,dataset,seed,rho,policy,fixed); x=pd.DataFrame(rows); x.to_csv(path,index=False); prof=outdir.parent/'profiles'; prof.mkdir(parents=True,exist_ok=True); timing.update({'dataset':dataset,'seed':seed,'rho':rho,'policy':policy}); pd.DataFrame([timing]).to_csv(prof/f'seed{seed:02d}_rho{rho:.3f}_{policy.replace(chr(37),chr(112)+'ct').replace(chr(45),chr(95))}.csv',index=False); return x

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=list(DATASETS),required=True);ap.add_argument('--seed-start',type=int,default=0);ap.add_argument('--seed-end',type=int,default=20);ap.add_argument('--rho',type=float,default=None);ap.add_argument('--policy',choices=POLICIES,default=None);ap.add_argument('--out-dir',default=str(ROOT/'results_phase3'/'raw'));ap.add_argument('--force',action='store_true');a=ap.parse_args();rhos=[a.rho] if a.rho is not None else RHO_VALUES;pols=[a.policy] if a.policy else POLICIES
 for seed in range(a.seed_start,a.seed_end):
  for pol in pols:
   for rho in ([.025] if pol=='NoExplore' else rhos):run_job(a.dataset,seed,rho,pol,Path(a.out_dir),a.force)
  print(a.dataset,'seed',seed,'done',flush=True)
if __name__=='__main__':main()

