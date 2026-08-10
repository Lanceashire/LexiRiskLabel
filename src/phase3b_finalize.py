from pathlib import Path
import pandas as pd,numpy as np,hashlib,json,zipfile,shutil
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results_phase3'; S=R/'summary'; RAW=R/'raw'
RHO=[.025,.05,.1,.2]; POL=['NoExplore','CountOnly','CountOnly-MinRisk','LeastRisky','MostRisky','Uncertainty','RiskTilted-Uncertainty','FAVE-v2.1']
def load_raw(ds):
 rows=[]; files=sorted((RAW/ds).glob('*.csv'))
 for f in files:
  x=pd.read_csv(f); x['source_file']=f.name; rows.append(x)
 return (pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()),files
def expand_ne(x):
 ne=x[x.policy.eq('NoExplore')].copy(); rest=x[~x.policy.eq('NoExplore')].copy(); out=[rest]
 for rho in RHO:
  z=ne.copy();z['rho']=rho;z['noexplore_reused']=True;out.append(z)
 y=pd.concat(out,ignore_index=True);y['noexplore_reused']=y.get('noexplore_reused',False).fillna(False);return y
def stats(x,ds):
 final=x[x['round'].eq(x['round'].max())].copy(); pairs=[('FAVE-v2.1','CountOnly-MinRisk','FAVE_vs_CountOnlyMinRisk'),('FAVE-v2.1','RiskTilted-Uncertainty','FAVE_vs_RiskTilted'),('RiskTilted-Uncertainty','CountOnly-MinRisk','RiskTilted_vs_CountOnlyMinRisk')]; rec=[]
 for rho in RHO:
  for a,b,label in pairs:
   aa=final[(final.rho==rho)&(final.policy==a)].set_index('seed');bb=final[(final.rho==rho)&(final.policy==b)].set_index('seed');ids=sorted(set(aa.index)&set(bb.index))
   for m in ['recall','fd_ratio','brier']:
    d=(aa.loc[ids,m]-bb.loc[ids,m]).to_numpy(float) if ids else np.array([]);d=d[np.isfinite(d)]
    try: p=float(wilcoxon(d,zero_method='wilcox').pvalue) if len(d)>=2 and np.any(np.abs(d)>1e-15) else 1.0
    except Exception:p=1.0
    rng=np.random.default_rng(3+int(rho*1000));boot=rng.choice(d,(10000,len(d)),replace=True).mean(1) if len(d) else np.array([np.nan]);rec.append({'dataset':ds,'rho':rho,'comparison':label,'metric':m,'n_pairs':len(d),'mean_difference':float(d.mean()) if len(d) else np.nan,'median_difference':float(np.median(d)) if len(d) else np.nan,'wilcoxon_raw_p':p,'bootstrap_ci95_low':float(np.nanpercentile(boot,2.5)),'bootstrap_ci95_high':float(np.nanpercentile(boot,97.5)),'win':int((d>1e-15).sum()),'tie':int((np.abs(d)<=1e-15).sum()),'loss':int((d<-1e-15).sum()),'practical_magnitude_abs':float(abs(d.mean())) if len(d) else np.nan})
 out=pd.DataFrame(rec)
 if len(out):
  out['holm_adjusted_p']=np.nan
  for m in ['recall','fd_ratio','brier']:
   q=out.metric.eq(m);out.loc[q,'holm_adjusted_p']=multipletests(out.loc[q,'wilcoxon_raw_p'],method='holm')[1]
 return out,final
# completed datasets
pp=expand_ne(pd.read_csv(S/'all_rounds.csv')); lc,lcfiles=load_raw('lendingclub'); lc=expand_ne(lc) if len(lc) else lc
if len(lc): lc.to_csv(R/'LENDINGCLUB_ALL_ROUNDS.csv',index=False); lc_final=lc[lc['round'].eq(lc['round'].max())]; lc_final.to_csv(R/'LENDINGCLUB_FINAL_BY_SEED.csv',index=False); lc_final.groupby(['dataset','rho','policy'],as_index=False)[['recall','balanced_accuracy','roc_auc','pr_auc','brier','ece','fd_ratio','n_explored','risk_used','cumulative_explored_labels','cumulative_expected_risk','cumulative_realized_bad_exposure','exploration_coverage_ratio']].mean().to_csv(R/'LENDINGCLUB_FINAL_SUMMARY.csv',index=False)
else: lc_final=pd.DataFrame()
allstats=[]
for ds,x in [('ppdai',pp),('lendingclub',lc)]:
 if len(x):
  st,fin=stats(x,ds);allstats.append(st); st.to_csv(R/(ds.upper()+'_PRIMARY_STATS.csv'),index=False)
  sev=x.groupby(['dataset','seed','rho','policy','round'],as_index=False).agg(n_rejected=('n_rejected','sum'),n_batch=('n_batch','sum'),hidden_default_fraction=('hidden_default_fraction','mean'),fd_ratio=('fd_ratio','mean'),n_explored=('n_explored','sum'),k_star=('k_star','sum'));sev['selection_rate']=sev.n_rejected/sev.n_batch.replace(0,np.nan);sev['feedback_gap']=1-sev.fd_ratio;sev.to_csv(R/(ds.upper()+'_SELECTION_SEVERITY.csv'),index=False)
if allstats: pd.concat(allstats,ignore_index=True).to_csv(R/'PHASE3B_PRIMARY_STATS_ALL.csv',index=False)
# compute profile merge
profiles=sorted((R/'profiles').glob('*.csv')); pd.concat([pd.read_csv(f) for f in profiles],ignore_index=True).to_csv(R/'LENDINGCLUB_COMPUTE_PROFILE.csv',index=False) if profiles else pd.DataFrame().to_csv(R/'LENDINGCLUB_COMPUTE_PROFILE.csv',index=False)
# risk heterogeneity and policy dispersion
heter=[];disp=[]
for ds,path,exposure_type in [('uci',ROOT/'data/raw/uci_default_credit_card.csv','unit'),('hmeq',ROOT/'data/phase2/hmeq.csv','monetary_proxy'),('ppdai',ROOT/'data/phase3/ppdai.csv','unit'),('lendingclub',ROOT/'data/phase3/lendingclub.csv','loan_amnt')]:
 try:
  d=pd.read_csv(path,low_memory=False); e=d['exposure'] if 'exposure' in d else (d['LOAN'] if 'LOAN' in d else pd.Series(np.ones(len(d)))); y=d['target'] if 'target' in d else (d['label'] if 'label' in d else pd.Series(np.zeros(len(d))));
  heter.append({'dataset':ds,'exposure_type':exposure_type,'mean_EAD':float(e.mean()),'std_EAD':float(e.std()),'cv_EAD':float(e.std()/e.mean()) if e.mean() else np.nan,'iqr_EAD':float(e.quantile(.75)-e.quantile(.25)),'corr_target_EAD':float(np.corrcoef(y,e)[0,1]) if e.std() and y.std() else np.nan})
 except Exception: pass
for ds,x in [('ppdai',pp),('lendingclub',lc)]:
 if len(x):
  f=x[x["round"].eq(x["round"].max())]
  for rho in RHO:
   z=f[f.rho.eq(rho)];disp.append({'dataset':ds,'rho':rho,'recall_range':float(z.groupby('policy').recall.mean().max()-z.groupby('policy').recall.mean().min()),'fd_ratio_range':float(z.groupby('policy').fd_ratio.mean().max()-z.groupby('policy').fd_ratio.mean().min()),'brier_range':float(z.groupby('policy').brier.mean().max()-z.groupby('policy').brier.mean().min()),'selected_set_jaccard':'NA_ID_NOT_RETAINED'})
pd.DataFrame(heter).to_csv(R/'RISK_HETEROGENEITY_SUMMARY.csv',index=False);pd.DataFrame(disp).to_csv(R/'POLICY_DIFFERENTIATION_SUMMARY.csv',index=False)
# cross-dataset final table
rows=[]
for ds,x,kind,etype in [('uci',None,'development benchmark','unit'),('hmeq',None,'development monetary validation','monetary proxy'),('ppdai',pp,'external weak-selection boundary','unit'),('lendingclub',lc,'external heterogeneous exposure' if len(lc) else 'external unavailable', 'loan_amnt')]:
 if x is not None and len(x):
  f=x[x["round"].eq(x["round"].max())];g=f.groupby(['dataset','rho','policy'],as_index=False).agg(Recall=('recall','mean'),FDRatio=('fd_ratio','mean'),Brier=('brier','mean'),ExpectedRisk=('cumulative_expected_risk','mean'),Labels=('cumulative_explored_labels','mean'),Coverage=('exploration_coverage_ratio','mean'));g['external_or_development']=kind;g['exposure_type']=etype;rows.append(g)
 else:
  old=pd.read_csv(ROOT/'results_phase2/summary/final_summary.csv');old=old[old.dataset.eq(ds)].rename(columns={'recall_mean':'Recall','fd_ratio_mean':'FDRatio','brier_mean':'Brier','cumulative_expected_risk_mean':'ExpectedRisk','cumulative_explored_labels_mean':'Labels','exploration_coverage_ratio_mean':'Coverage'});old['external_or_development']=kind;old['exposure_type']=etype;rows.append(old[['dataset','rho','policy','Recall','FDRatio','Brier','ExpectedRisk','Labels','Coverage','external_or_development','exposure_type']])
if rows: pd.concat(rows,ignore_index=True).to_csv(R/'CROSS_DATASET_GENERALIZATION_FINAL.csv',index=False)
print(json.dumps({'lc_files':len(lcfiles),'lc_rows':len(lc),'stats_rows':sum(len(z) for z in allstats),'profiles':len(profiles)},indent=2))