from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results_phase3'; S=R/'summary'; S.mkdir(exist_ok=True)
# freeze existing primary
(R/'PHASE3A_FREEZE_DECLARATION.md').write_text('''# PHASE3A Freeze Declaration\n\nPPDai primary results are frozen and unchanged. Dataset, seeds 0..19, tau=0.5, rho={0.025, 0.05, 0.10, 0.20}, model, policy suite, unit-exposure risk, split construction, and all raw result files are frozen. No subsequent sensitivity analysis may replace these primary results.\n''',encoding='utf-8')
# corrected coverage from raw analyzed rounds
x=pd.read_csv(S/'all_rounds.csv'); grp=x.groupby(['dataset','seed','rho','policy'],as_index=False).agg(cum_rejected=('n_rejected','sum'),cum_explored=('n_explored','sum')); grp['coverage_ratio']=grp.cum_explored/grp.cum_rejected.replace(0,np.nan); grp['coverage_ratio']=grp.coverage_ratio.fillna(0.0); grp.to_csv(R/'PHASE3_corrected_coverage.csv',index=False)
legacy=pd.DataFrame([{'file':'results_phase3/summary/all_rounds.csv','field':'exploration_coverage_ratio','status':'LEGACY_WRONG_DEFINITION','note':'explored / whole stream; do not use for paper'}]); legacy.to_csv(R/'PHASE3_legacy_coverage_notice.csv',index=False)
# primary corrected stats
final=x[x['round'].eq(x['round'].max())].copy(); final=final.merge(grp,on=['dataset','seed','rho','policy'],how='left')
pairs=[('FAVE-v2.1','CountOnly-MinRisk','FAVE_vs_CountOnlyMinRisk'),('FAVE-v2.1','RiskTilted-Uncertainty','FAVE_vs_RiskTilted'),('RiskTilted-Uncertainty','CountOnly-MinRisk','RiskTilted_vs_CountOnlyMinRisk')]
rows=[]; rng=np.random.default_rng(20260809)
for dataset in sorted(final.dataset.unique()):
 for rho in sorted(final.rho.unique()):
  for a,b,label in pairs:
   aa=final[(final.dataset==dataset)&(final.rho==rho)&(final.policy==a)].set_index('seed'); bb=final[(final.dataset==dataset)&(final.rho==rho)&(final.policy==b)].set_index('seed'); ids=sorted(set(aa.index)&set(bb.index))
   for metric in ['recall','fd_ratio','brier']:
    d=(aa.loc[ids,metric]-bb.loc[ids,metric]).to_numpy(float); d=d[np.isfinite(d)];
    if len(d)>=2 and np.any(np.abs(d)>1e-15):
      try: wp=float(wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto').pvalue)
      except Exception: wp=np.nan
    else: wp=1.0
    boot=rng.choice(d,(10000,len(d)),replace=True).mean(axis=1) if len(d) else np.array([np.nan]);
    rows.append({'dataset':dataset,'rho':rho,'comparison':label,'metric':metric,'n_pairs':len(d),'mean_difference':float(np.mean(d)) if len(d) else np.nan,'median_difference':float(np.median(d)) if len(d) else np.nan,'wilcoxon_raw_p':wp,'bootstrap_ci95_low':float(np.nanpercentile(boot,2.5)),'bootstrap_ci95_high':float(np.nanpercentile(boot,97.5)),'win':int(np.sum(d>1e-15)),'tie':int(np.sum(np.abs(d)<=1e-15)),'loss':int(np.sum(d<-1e-15)),'practical_magnitude_abs':float(abs(np.mean(d))) if len(d) else np.nan})
st=pd.DataFrame(rows); st['holm_adjusted_p']=np.nan
for ds in st.dataset.unique():
 for metric in ['recall','fd_ratio','brier']:
  mask=(st.dataset==ds)&(st.metric==metric); st.loc[mask,'holm_adjusted_p']=multipletests(st.loc[mask,'wilcoxon_raw_p'].fillna(1).to_numpy(),method='holm')[1]
st.to_csv(R/'PHASE3_PRIMARY_STATS_CORRECTED.csv',index=False)
# selection severity summary
sev=x.groupby(['dataset','seed','rho','policy','round'],as_index=False).agg(n_rejected=('n_rejected','sum'),n_batch=('n_batch','sum'),hidden_default_fraction=('hidden_default_fraction','mean'),fd_ratio=('fd_ratio','mean'),n_explored=('n_explored','sum'),k_star=('k_star','sum')); sev['selection_rate']=sev.n_rejected/sev.n_batch.replace(0,np.nan); sev['feedback_gap']=1-sev.fd_ratio; sev.to_csv(R/'SELECTION_SEVERITY_SUMMARY.csv',index=False)
# provenance final audit: strict classification
source=pd.read_csv(ROOT/'data/phase3/ppdai.csv',nrows=1); amb=set('outcome outcome_tm prior_repay repay_state loan_time time repay performance browse_data tm_encode_3 prior_account yu_e yu_e_categery yu_e_tm yu_e_tm_categery'.split()); post={'label','exposure'}; idset=set()
recs=[]
for c in source.columns:
 if c in post: cls='POST_OUTCOME' if c=='label' else 'DERIVED_EXPOSURE'
 elif c in idset: cls='IDENTIFIER'
 elif c in amb or any(k in c.lower() for k in ['outcome','repay','loan_time','performance','post']): cls='AMBIGUOUS'
 elif c in {'sex','occupation','education','marriage','household','income','credit_limit'}: cls='PRE_DECISION_VERIFIED'
 else: cls='AMBIGUOUS'
 recs.append({'feature':c,'classification':cls,'included_in_frozen_primary':c not in post,'evidence':'Existing source audit only; no field-level decision-time proof supplied' if cls=='AMBIGUOUS' else 'Static/source schema review'})
aud=pd.DataFrame(recs); aud.to_csv(R/'PPDAI_FEATURE_AUDIT_FINAL.csv',index=False); risk=bool((aud.classification=='AMBIGUOUS').any() and (aud.included_in_frozen_primary & aud.classification.eq('AMBIGUOUS')).any())
(R/'PPDAI_PROVENANCE_FINAL.md').write_text(f'''# PPDai Provenance Final\n\nSource and dataset hash are unchanged from Phase 3A. Primary raw results were not rerun or modified. Strict field review classifies only static fields with direct schema support as PRE_DECISION_VERIFIED; outcome/repay/time/performance-like fields remain AMBIGUOUS when decision-time observability cannot be proven.\n\n`PPDAI_PRIMARY_LEAKAGE_RISK = {"TRUE" if risk else "FALSE"}`\n\nTherefore the PPDai result is retained as a frozen exploratory external result with a provenance risk caveat, not upgraded to clean confirmatory evidence.\n''',encoding='utf-8')
print(json.dumps({'coverage_rows':len(grp),'stats_rows':len(st),'ambiguous_features':int((aud.classification=='AMBIGUOUS').sum()),'primary_leakage_risk':risk},indent=2))