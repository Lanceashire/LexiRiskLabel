from pathlib import Path
import hashlib, json, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT=Path(os.environ.get('ECOMIC_ROOT',Path(__file__).resolve().parents[1])); R=ROOT/'results_phase2'; RAW=R/'raw'; SUM=R/'summary'; SUM.mkdir(parents=True,exist_ok=True)
METRICS=['recall','balanced_accuracy','roc_auc','pr_auc','brier','ece','visible_tdr','fd_ratio']
def load_raw():
    fs=list(RAW.rglob('*.csv')); return pd.concat([pd.read_csv(f) for f in fs],ignore_index=True) if fs else pd.DataFrame()
def boot(x,seed):
    x=np.asarray(x,float); rng=np.random.RandomState(seed); b=np.asarray([rng.choice(x,len(x),replace=True).mean() for _ in range(10000)]); return float(np.quantile(b,.025)),float(np.quantile(b,.975))
def wl(x):
    x=np.asarray(x,float); return float(wilcoxon(x).pvalue) if np.any(np.abs(x)>0) else 1.0
def pareto(x,y):
    out=[]
    for i in range(len(x)):
        dominated=False
        for j in range(len(x)):
            if i!=j and x[j]<=x[i] and y[j]>=y[i] and (x[j]<x[i] or y[j]>y[i]): dominated=True; break
        out.append(not dominated)
    return out
d=load_raw();
if d.empty: raise SystemExit('no raw Phase 2 checkpoint files')
d.to_csv(SUM/'all_rounds.csv',index=False); final=d[d['round']==10].copy(); final.to_csv(SUM/'final_by_seed.csv',index=False)
group_cols=['dataset','rho','policy']; rows=[]
for key,g in final.groupby(group_cols):
    row=dict(zip(group_cols,key));
    for m in METRICS+['cumulative_explored_labels','cumulative_expected_risk','cumulative_realized_bad_exposure','risk_used','risk_budget','exploration_coverage_ratio']:
        row[m+'_mean']=float(g[m].mean()) if m in g else np.nan; row[m+'_sd']=float(g[m].std(ddof=1)) if m in g and len(g)>1 else np.nan
    rows.append(row)
pd.DataFrame(rows).to_csv(SUM/'final_summary.csv',index=False)
acq=final.groupby(group_cols)[['cumulative_explored_labels','cumulative_expected_risk','cumulative_realized_bad_exposure','risk_budget','risk_used','remaining_risk_budget','exploration_coverage_ratio']].mean().reset_index(); acq['budget_utilization']=acq['risk_used']/acq['risk_budget'].replace(0,np.nan); acq.to_csv(SUM/'acquisition_summary.csv',index=False)
coverage=final.groupby(['dataset','rho','policy'])[['exploration_coverage_ratio','n_rejected','k_star']].mean().reset_index(); coverage.rename(columns={'exploration_coverage_ratio':'coverage_ratio','n_rejected':'mean_final_candidate_count','k_star':'mean_final_k_star'},inplace=True); coverage.to_csv(SUM/'coverage_vs_budget.csv',index=False)
diag=final.groupby(group_cols)[['selected_mean_pd','selected_median_pd','selected_mean_risk_cost','selected_mean_exposure','selected_mean_uncertainty','selected_default_rate_oracle','selected_realized_bad_exposure_oracle']].mean().reset_index(); diag.to_csv(SUM/'policy_selection_diagnostics.csv',index=False)

# Quantity versus quality descriptive table and correlations against NoExplore.
qq=[]
for (ds,rho,pol),g in final.groupby(group_cols):
    row={'dataset':ds,'rho':rho,'policy':pol,'cum_labels':float(g.cumulative_explored_labels.mean()),'cum_expected_risk':float(g.cumulative_expected_risk.mean()),'final_recall':float(g.recall.mean()),'final_fdratio':float(g.fd_ratio.mean()),'final_brier':float(g.brier.mean())}
    base=final[(final.dataset==ds)&(final.rho==rho)&(final.policy=='NoExplore')].set_index('seed'); q=g.set_index('seed').join(base[['recall','fd_ratio']],rsuffix='_no')
    row['corr_labels_recall_recovery']=float(np.corrcoef(q.cumulative_explored_labels,q.recall-q.recall_no)[0,1]) if len(q)>1 and q.cumulative_explored_labels.std()>0 else np.nan
    row['corr_labels_fdratio_recovery']=float(np.corrcoef(q.cumulative_explored_labels,q.fd_ratio-q.fd_ratio_no)[0,1]) if len(q)>1 and q.cumulative_explored_labels.std()>0 else np.nan
    qq.append(row)
pd.DataFrame(qq).to_csv(SUM/'quantity_quality_summary.csv',index=False)

# Paired seed comparisons; all primary families are corrected within dataset/metric.
comparisons=[('FAVE-v2.1','CountOnly-MinRisk','C1'),('FAVE-v2.1','RiskTilted-Uncertainty','C2'),('RiskTilted-Uncertainty','CountOnly-MinRisk','C3'),('Uncertainty','CountOnly-MinRisk','C4')]
pt=[]
for ds in final.dataset.unique():
  for rho in sorted(final.rho.unique()):
    for a,b,cid in comparisons:
      aa=final[(final.dataset==ds)&(final.rho==rho)&(final.policy==a)].set_index('seed'); bb=final[(final.dataset==ds)&(final.rho==rho)&(final.policy==b)].set_index('seed')
      for m in METRICS:
        q=aa[[m]].join(bb[[m]],lsuffix='_a',rsuffix='_b').dropna(); diff=(q[m+'_a']-q[m+'_b']).to_numpy(float); lo,hi=boot(diff,710000+len(pt)); tol=1e-10; pt.append({'dataset':ds,'rho':rho,'comparison':cid,'policy_a':a,'policy_b':b,'metric':m,'n':len(diff),'mean_diff':float(diff.mean()),'median_diff':float(np.median(diff)),'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'wilcoxon_p':wl(diff),'wins':int((diff>tol).sum()),'ties':int((np.abs(diff)<=tol).sum()),'losses':int((diff<-tol).sum())})
pt=pd.DataFrame(pt); pt['holm_p']=pt.groupby(['dataset','metric'])['wilcoxon_p'].transform(lambda s:multipletests(s,method='holm')[1]); pt.to_csv(SUM/'paired_tests.csv',index=False); pt.to_csv(SUM/'paired_tests_holm.csv',index=False)

# Scarcity contrasts: low budget mean minus high budget mean, paired by seed.
sc=[]
for ds in final.dataset.unique():
  for a,b,cid in [('FAVE-v2.1','CountOnly-MinRisk','FAVE_vs_CountOnlyMinRisk'),('FAVE-v2.1','RiskTilted-Uncertainty','FAVE_vs_RiskTilted')]:
    for m in ['recall','fd_ratio','brier_improvement']:
      vals=[]
      for seed in sorted(final.seed.unique()):
        diffs=[]
        for rho in [0.025,0.05,0.10,0.20]:
          x=final[(final.dataset==ds)&(final.seed==seed)&(final.rho==rho)&(final.policy==a)].iloc[0]; y=final[(final.dataset==ds)&(final.seed==seed)&(final.rho==rho)&(final.policy==b)].iloc[0]
          diffs.append(float(x[m.replace('_improvement','')] - y[m.replace('_improvement','')]) if m!='brier_improvement' else float(y.brier-x.brier))
        vals.append(np.mean(diffs[:2])-np.mean(diffs[2:]))
      lo,hi=boot(vals,770000+len(sc)); sc.append({'dataset':ds,'comparison':cid,'metric':m,'n':len(vals),'mean_scarcity_contrast':float(np.mean(vals)),'median_scarcity_contrast':float(np.median(vals)),'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'wilcoxon_p':wl(vals),'positive_count':int((np.asarray(vals)>0).sum())})
sc=pd.DataFrame(sc); sc['holm_p']=sc.groupby('dataset')['wilcoxon_p'].transform(lambda s:multipletests(s,method='holm')[1]); sc.to_csv(SUM/'scarcity_tests.csv',index=False)

# Pareto points at final round, relative to NoExplore at the same rho.
pp=[]
for ds in final.dataset.unique():
  for rho in sorted(final.rho.unique()):
    base=final[(final.dataset==ds)&(final.rho==rho)&(final.policy=='NoExplore')][['seed','recall','fd_ratio','brier']].set_index('seed')
    for pol in final.policy.unique():
      if pol=='NoExplore': continue
      g=final[(final.dataset==ds)&(final.rho==rho)&(final.policy==pol)].set_index('seed').join(base,rsuffix='_no')
      pp.append({'dataset':ds,'rho':rho,'policy':pol,'x_cumulative_expected_risk':float(g.cumulative_expected_risk.mean()),'x_cumulative_realized_bad_exposure':float(g.cumulative_realized_bad_exposure.mean()),'recall_gain':float((g.recall-g.recall_no).mean()),'fdratio_gain':float((g.fd_ratio-g.fd_ratio_no).mean()),'brier_improvement':float((g.brier_no-g.brier).mean())})
pp=pd.DataFrame(pp)
for metric in ['recall_gain','fdratio_gain','brier_improvement']:
  pp[metric+'_pareto_efficient']=False
  for ds in pp.dataset.unique():
    q=pp[pp.dataset==ds]; pp.loc[q.index,metric+'_pareto_efficient']=pareto(q.x_cumulative_expected_risk.to_numpy(),q[metric].to_numpy())
pp.to_csv(SUM/'pareto_points.csv',index=False)
print('analysis complete',len(d),'rows',len(final),'final rows')
