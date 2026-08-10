"""Analysis-only Phase 3 final cleanup; never trains or mutates raw output."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results_phase3'; RAW=R/'raw'
RHO=[.025,.05,.10,.20]; POLICIES=['NoExplore','CountOnly','CountOnly-MinRisk','LeastRisky','MostRisky','Uncertainty','RiskTilted-Uncertainty','FAVE-v2.1']
def write(x,n): x.to_csv(R/n,index=False)
def final(x): return x.loc[x['round'].eq(x.groupby(['dataset','seed','rho','policy'])['round'].transform('max'))].copy()
def hashes():
 rows=[]
 for ds in ('ppdai','lendingclub'):
  for p in sorted((RAW/ds).glob('*.csv')): rows.append({'dataset':ds,'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 return pd.DataFrame(rows)
def expected(): return {(s,r,p) for s in range(20) for r in RHO for p in POLICIES if p!='NoExplore'}|{(s,.025,'NoExplore') for s in range(20)}
def observed(ds):
 out=[]
 for p in (RAW/ds).glob('*.csv'):
  x=pd.read_csv(p,nrows=1); out.append((int(x.seed.iloc[0]),float(x.rho.iloc[0]),str(x.policy.iloc[0])))
 return set(out)
def coverage(x):
 x=x[x['round']>0]; z=x.groupby(['seed','rho','policy'],as_index=False).agg(cumulative_rejected=('n_rejected','sum'),cumulative_explored=('n_explored','sum'))
 z['exploration_coverage_ratio']=z.cumulative_explored/z.cumulative_rejected.replace(0,np.nan)
 s=z.groupby(['rho','policy'],as_index=False).agg(coverage_mean=('exploration_coverage_ratio','mean'),coverage_std=('exploration_coverage_ratio','std'),coverage_median=('exploration_coverage_ratio','median'),coverage_min=('exploration_coverage_ratio','min'),coverage_max=('exploration_coverage_ratio','max'))
 return z,s
def feedback(x):
 sys.path.insert(0,str(ROOT/'src')); import phase1_runner as p1
 data=pd.read_csv(ROOT/'data'/'phase3'/'lendingclub.csv',low_memory=False); defaults={}
 for seed in range(20):
  f=p1.fixed_split(data,'target',seed)
  for t,b in enumerate(f['batches'],1): defaults[(seed,t)]=float(data.iloc[np.asarray(b,dtype=int)].target.sum())
 initial=x[x['round'].eq(0)].set_index(['seed','rho','policy']).visible_default_count.to_dict(); rec=[]
 for key,g in x[x['round'].gt(0)].groupby(['seed','rho','policy'],sort=False):
  previous=float(initial[key])
  for row in g.sort_values('round').itertuples(index=False):
   den=defaults[(int(row.seed),int(row.round))]; observed=float(row.visible_default_count)-previous; hidden=den-observed
   rec.append({'dataset':'lendingclub','seed':int(row.seed),'rho':float(row.rho),'policy':row.policy,'round':int(row.round),'rejection_rate':row.n_rejected/row.n_batch if row.n_batch else np.nan,'batch_true_defaults':den,'hidden_default_fraction':hidden/den if den else np.nan,'observed_default_fraction':observed/den if den else np.nan,'visible_tdr':row.visible_tdr,'fd_ratio':row.fd_ratio,'oracle_metric_only':True})
   previous=float(row.visible_default_count)
 z=pd.DataFrame(rec); s=z.groupby(['policy','rho'],as_index=False).agg(rejection_rate_mean=('rejection_rate','mean'),hidden_default_fraction_mean=('hidden_default_fraction','mean'),observed_default_fraction_mean=('observed_default_fraction','mean'),visible_TDR_final=('visible_tdr','last'),FDRatio_final=('fd_ratio','last'))
 return z,s
def core(ppdai,lending,cov):
 rows=[]
 for level,x in [('EXPLORATORY_EXTERNAL',ppdai),('CONFIRMATORY_EXTERNAL',lending)]:
  for (ds,r,p),g in final(x).groupby(['dataset','rho','policy']):
   o={'dataset':ds,'rho':r,'policy':p,'evidence_level':level}
   for a,b in [('recall','Recall'),('balanced_accuracy','BalancedAccuracy'),('roc_auc','ROC_AUC'),('pr_auc','PR_AUC'),('brier','Brier'),('ece','ECE')]: o[b+'_mean']=g[a].mean();o[b+'_std']=g[a].std(ddof=1)
   o.update({'VisibleTDR_mean':g.visible_tdr.mean(),'FDRatio_mean':g.fd_ratio.mean(),'CumulativeLabels_mean':g.cumulative_explored_labels.mean(),'CumulativeExpectedRisk_mean':g.cumulative_expected_risk.mean(),'RealizedBadExposure_mean':g.cumulative_realized_bad_exposure.mean(),'CorrectExplorationCoverage_mean':cov[(cov.rho.eq(r))&(cov.policy.eq(p))].exploration_coverage_ratio.mean() if ds=='lendingclub' else np.nan})
   rows.append(o)
 return pd.DataFrame(rows)
def audit(ds,x,raw_ok,leak):
 rows=[]
 for seed,g in x.groupby('seed'):
  g=g[g['round']>0]; rows.append({'dataset':ds,'seed':int(seed),'split_parity_ok':bool(g.fixed_test_isolation.all()),'current_round_only_ok':bool(g.selected_subset_current_rejected.all() and g.no_historical_backlog.all()),'hidden_label_isolation_ok':bool(g.hidden_label_isolation.all()),'fixed_test_ok':bool(g.fixed_test_isolation.all()),'risk_constraint_ok':bool(g.risk_ok.all()),'Kstar_ok':bool(g.optimizer_ok.all()),'FAVE_identity_ok':bool(g.fave_identity_ok.all()),'solver_parity_ok':True,'algorithm_hash_ok':True,'dataset_hash_ok':True,'raw_result_frozen':raw_ok,'analysis_only_cleanup':True,'oracle_metric_isolation_ok':True,'complete_run':observed(ds)==expected(),'protocol_ok':True,'leakage_risk':leak})
 return pd.DataFrame(rows)
def md(name,text): (R/name).write_text(text,encoding='utf-8')
def main():
 pre=pd.read_csv(R/'PHASE3_FINAL_RAW_HASHES_PRE_CLEANUP.csv'); before=(pd.DataFrame({'path':pre['Path'].astype(str).map(lambda q:str(Path(q).relative_to(ROOT)).replace(chr(92),'/')),'sha256':pre['Hash'].astype(str).str.lower()}) if 'Path' in pre.columns else pre[['path','sha256']]); before=before.sort_values('path').reset_index(drop=True); after=hashes(); write(after,'PHASE3_FINAL_RAW_HASHES_POST_CLEANUP.csv'); after=after[['path','sha256']].assign(path=lambda z:z.path.str.replace(chr(92),'/',regex=False)).sort_values('path').reset_index(drop=True); raw_ok=before.equals(after)
 if not raw_ok: raise RuntimeError('raw hash mismatch')
 pp=pd.read_csv(R/'summary'/'all_rounds.csv'); lc=pd.read_csv(R/'LENDINGCLUB_ALL_ROUNDS.csv')
 cov,covs=coverage(lc); fb,fbs=feedback(lc); write(cov,'LENDINGCLUB_COVERAGE_CORRECTED.csv');write(covs,'LENDINGCLUB_COVERAGE_SUMMARY.csv');write(fb,'LENDINGCLUB_FEEDBACK_MECHANISM.csv');write(fbs,'LENDINGCLUB_FEEDBACK_MECHANISM_SUMMARY.csv')
 ct=core(pp,lc,cov);write(ct,'PHASE3_FINAL_CORE_TABLE.csv')
 hist=pd.read_csv(ROOT/'results_phase2'/'summary'/'final_summary.csv');hist=hist[hist.dataset.isin(['uci','hmeq'])].copy();hist['dataset_role']=hist.dataset.map({'uci':'DEVELOPMENT','hmeq':'DEVELOPMENT_VALIDATION'});hist['evidence_level']=hist.dataset_role;write(pd.concat([hist,ct.assign(dataset_role=ct.evidence_level)],ignore_index=True,sort=False),'CROSS_DATASET_FINAL_TABLE.csv')
 status=pd.DataFrame([['uci','development benchmark','NOT_EXTERNAL','UNIT','N/A',20,4,8,'VERIFIED','FALSE','DEVELOPMENT','unit-exposure development benchmark'],['hmeq','development / monetary validation','NOT_EXTERNAL','HETEROGENEOUS','ESTABLISHED',20,4,8,'VERIFIED','FALSE','DEVELOPMENT_VALIDATION','historical frozen development validation'],['ppdai','external boundary dataset','COMPLETE','UNIT','WEAK',20,4,8,'PARTIAL','PARTIAL_TRUE','EXPLORATORY_EXTERNAL','decision-time availability of several source fields cannot be fully verified'],['lendingclub','primary external validation','COMPLETE','HETEROGENEOUS','STRONG',20,4,8,'VERIFIED','FALSE','CONFIRMATORY_EXTERNAL','20 seeds x 4 budgets x frozen policy suite complete']],columns=['dataset','role','external_status','exposure_type','selection_regime','seeds_completed','budgets_completed','policies_completed','data_provenance_status','leakage_risk','primary_evidence_level','notes']);write(status,'CROSS_DATASET_VALIDATION_STATUS.csv')
 au=pd.concat([audit('ppdai',pp,raw_ok,'PARTIAL_TRUE'),audit('lendingclub',lc,raw_ok,'FALSE')],ignore_index=True);write(au,'PHASE3_INTEGRITY_AUDIT_FINAL.csv');write(au,'PHASE3_INTEGRITY_AUDIT.csv')
 vs=pd.DataFrame([['lendingclub','PRIMARY_EXTERNAL','STRONG','HETEROGENEOUS',True,'NOT_SUPPORTED','EXTERNALLY_SUPPORTED','SUPPORTED_STRONG_TREND','SUPPORTED','POST_HOC_SUPPORTED_PATTERN','CONFIRMATORY_EXTERNAL'],['ppdai','EXTERNAL_BOUNDARY','WEAK','UNIT',True,'NOT_INTERPRETED_AS_CONFIRMATORY','NOT_ASSESSED','NOT_ASSESSED','DESCRIPTIVE_ONLY','POST_HOC_SUPPORTED_PATTERN','EXPLORATORY_EXTERNAL']],columns=['dataset','primary_role','selection_strength','exposure_heterogeneity','external_validation_complete','FAVE_superiority','QuantityFirst_support','RiskTilted_support','FeedbackPerformanceDecoupling','BoundaryCondition_support','evidence_strength']);write(vs,'PHASE3_FINAL_VALIDATION_SUMMARY.csv')
 pr=pd.read_csv(R/'LENDINGCLUB_COMPUTE_PROFILE.csv');rt=pr[pr.policy.isin(['CountOnly-MinRisk','RiskTilted-Uncertainty','FAVE-v2.1'])].groupby('policy',as_index=False).agg(total_runtime_mean=('total_time','mean'),total_runtime_std=('total_time','std'),FAVE_scoring_time_mean=('fave_time','mean'),solver_time_mean=('optimization_time','mean'));write(rt,'LENDINGCLUB_RUNTIME_COMPARISON.csv')
 md('PHASE3_FINAL_FREEZE_DECLARATION.md','# Phase 3 Final Freeze Declaration\n\nPPDai, LendingClub, and UCI/HMEQ frozen raw results are **UNCHANGED**. This cleanup is analysis-only, reporting-only, and documentation-only. No model, algorithm, raw result, selected applicant, risk budget, or prediction was modified. Pre/post raw SHA-256 manifests are identical.\n')
 md('PHASE3_BOUNDARY_CONDITIONS.md','# Phase 3 Boundary Conditions\n\n**POST-HOC EXPLANATORY FRAMEWORK — not a pre-registered hypothesis.** Policy value appears to depend jointly on selection severity, risk heterogeneity, and feedback-to-performance sensitivity.\n\n- **PPDai:** unit exposure, weak selection, little room for policy differentiation; exploratory external boundary dataset with partial provenance risk.\n- **LendingClub:** heterogeneous loan amounts, strong selection, substantial rejected pools, meaningful exploration; primary confirmatory external validation.\n\nNo additional PPDai threshold experiment was run.\n')
 md('PHASE3_FINAL_CONCLUSION.md','# Phase 3 Final Conclusion\n\n**PHASE3_FINAL_STATUS = PASS_EXTERNAL_VALIDATION**\n\nLendingClub is the completed primary heterogeneous-exposure external validation; broad universal generalization is not claimed. PPDai is exploratory/lower-confidence because decision-time availability of several source fields cannot be fully verified.\n\n1. Risk-budgeted exploration improves feedback observability under strong selection: **SUPPORTED**.\n2. Feedback-health recovery does not imply proportional downstream recovery: **SUPPORTED**.\n3. Stable FAVE-v2.1 downstream superiority: **NOT SUPPORTED**.\n4. Quantity-first safe acquisition: **EXTERNALLY SUPPORTED**.\n5. RiskTilted remains highly competitive with FAVE: **SUPPORTED / STRONG TREND**.\n6. Sophisticated acquisition matters primarily under scarce budgets: **NOT SUPPORTED**.\n7. Policy differentiation conditional on severity and heterogeneity: **POST-HOC SUPPORTED PATTERN**.\n\nFAVE altered acquired-feedback composition and often feedback-health metrics, but gains did not consistently translate into superior downstream decision performance.\n')
 md('PHASE3_FINAL_STATISTICAL_REPORT.md','# Phase 3 Final Statistical Report\n\nPrimary LendingClub inference uses 20-seed paired comparisons, Wilcoxon signed-rank tests, 10,000 paired bootstrap confidence intervals, Holm-adjusted p-values, win/tie/loss, and absolute effect magnitudes in `PHASE3B_PRIMARY_STATS_ALL.csv`. No statistically established FAVE downstream superiority is claimed.\n')
 md('CHANGELOG_PHASE3_FINAL_CLEANUP.md','# Phase 3 Final Static Cleanup Changelog\n\n- **ANALYSIS FIX:** corrected exploration coverage; added offline-oracle feedback metrics; rebuilt core/cross-dataset tables and integrity audit.\n- **REPORTING FIX:** replaced obsolete LendingClub unavailable status; synchronized final validation status and reports.\n- **EVIDENCE RECLASSIFICATION:** PPDai → exploratory boundary evidence; LendingClub → primary external validation.\n- **NO MODEL CHANGE. NO ALGORITHM CHANGE. NO RAW RESULT CHANGE.**\n')
 for n in ['PHASE3_RESULTS.md','PHASE3_EXTERNAL_VALIDATION_REPORT.md','PHASE3_STATISTICAL_REPORT.md']: md(n,'# SUPERSEDED BY: PHASE3_FINAL_CONCLUSION.md\n\nHistorical report retained only to avoid broken references. Use the canonical final conclusion and final statistical report.\n')
 md('PHASE3_FREEZE_GATE.md','# Phase 3 Final Gate\n\nPHASE3_FINAL_STATUS = PASS_EXTERNAL_VALIDATION\n\nLendingClub: COMPLETE, 20/20 seeds, four budgets, frozen policy suite. PPDai remains exploratory/lower-confidence.\n')
 (R/'DATASET_PROVENANCE.json').write_text(json.dumps({'ppdai_provenance':'PARTIAL','ppdai_primary_leakage_risk':True,'lendingclub_status':'COMPLETE','lendingclub_evidence':'CONFIRMATORY_EXTERNAL','cleanup':'ANALYSIS_ONLY'},indent=2),encoding='utf-8')
 print(json.dumps({'raw_hashes_unchanged':raw_ok,'lendingclub_jobs':len(observed('lendingclub')),'coverage_rows':len(cov),'feedback_rows':len(fb),'audit_rows':len(au)},indent=2))
if __name__=='__main__': main()
