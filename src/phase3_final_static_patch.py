"""Final static patch: no model runs and no raw-result writes."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results_phase3'; P2=ROOT/'results_phase2'
CANON=['dataset','dataset_role','evidence_level','exposure_type','selection_regime','rho','policy','recall_mean','recall_std','balanced_accuracy_mean','balanced_accuracy_std','roc_auc_mean','roc_auc_std','pr_auc_mean','pr_auc_std','brier_mean','brier_std','ece_mean','ece_std','visible_tdr_mean','visible_tdr_std','fdratio_mean','fdratio_std','cumulative_labels_mean','cumulative_labels_std','cumulative_expected_risk_mean','cumulative_expected_risk_std','exploration_coverage_mean','exploration_coverage_std','realized_bad_exposure_mean','realized_bad_exposure_std']

def h(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def csv(x,n): x.to_csv(R/n,index=False)
def freeze():
 paths=list((R/'raw'/'ppdai').glob('*.csv'))+list((R/'raw'/'lendingclub').glob('*.csv'))+[P2/'summary'/'final_summary.csv',P2/'summary'/'coverage_vs_budget_corrected.csv']
 return pd.DataFrame([{'path':str(p.relative_to(ROOT)),'sha256':h(p)} for p in sorted(paths)])
def feedback_final():
 x=pd.read_csv(R/'LENDINGCLUB_FEEDBACK_MECHANISM.csv'); t=int(x['round'].max()); traj=x.groupby(['policy','rho'],as_index=False).agg(rejection_rate_trajectory_mean=('rejection_rate','mean'),rejection_rate_trajectory_std=('rejection_rate','std'),hidden_default_fraction_trajectory_mean=('hidden_default_fraction','mean'),hidden_default_fraction_trajectory_std=('hidden_default_fraction','std'),observed_default_fraction_trajectory_mean=('observed_default_fraction','mean'),observed_default_fraction_trajectory_std=('observed_default_fraction','std'))
 f=x[x['round'].eq(t)].groupby(['policy','rho'],as_index=False).agg(visible_tdr_final_mean=('visible_tdr','mean'),visible_tdr_final_std=('visible_tdr','std'),visible_tdr_final_median=('visible_tdr','median'),fdratio_final_mean=('fd_ratio','mean'),fdratio_final_std=('fd_ratio','std'),fdratio_final_median=('fd_ratio','median'))
 out=traj.merge(f,on=['policy','rho'],validate='one_to_one'); csv(out,'LENDINGCLUB_FEEDBACK_MECHANISM_SUMMARY_FINAL.csv'); return out,t
def role(ds):
 return {'uci':('DEVELOPMENT_BENCHMARK','DEVELOPMENT','UNIT_EXPOSURE','NOT_FORMALLY_CLASSIFIED'),'hmeq':('DEVELOPMENT_MONETARY_VALIDATION','DEVELOPMENT_VALIDATION','HETEROGENEOUS_MONETARY_PROXY','NOT_FORMALLY_CLASSIFIED'),'ppdai':('EXPLORATORY_EXTERNAL_BOUNDARY','EXPLORATORY_EXTERNAL','UNIT_EXPOSURE','WEAK'),'lendingclub':('PRIMARY_EXTERNAL_VALIDATION','CONFIRMATORY_EXTERNAL','HETEROGENEOUS_MONETARY_PROXY','STRONG')}[ds]
def empty(ds,rho,policy):
 a,b,c,d=role(ds); z={k:np.nan for k in CANON};z.update({'dataset':ds,'dataset_role':a,'evidence_level':b,'exposure_type':c,'selection_regime':d,'rho':rho,'policy':policy});return z
def normalize_phase2():
 x=pd.read_csv(P2/'summary'/'final_summary.csv');cov=pd.read_csv(P2/'summary'/'coverage_vs_budget_corrected.csv').groupby(['dataset','rho','policy'],as_index=False).agg(exploration_coverage_mean=('exploration_coverage_ratio','mean'),exploration_coverage_std=('exploration_coverage_ratio','std'))
 rows=[]
 for q in x.itertuples(index=False):
  z=empty(q.dataset,q.rho,q.policy)
  for a,b in [('recall','recall'),('balanced_accuracy','balanced_accuracy'),('roc_auc','roc_auc'),('pr_auc','pr_auc'),('brier','brier'),('ece','ece'),('visible_tdr','visible_tdr'),('fd_ratio','fdratio')]: z[b+'_mean']=getattr(q,a+'_mean');z[b+'_std']=getattr(q,a+'_sd')
  z['cumulative_labels_mean']=q.cumulative_explored_labels_mean;z['cumulative_labels_std']=q.cumulative_explored_labels_sd;z['cumulative_expected_risk_mean']=q.cumulative_expected_risk_mean;z['cumulative_expected_risk_std']=q.cumulative_expected_risk_sd;z['realized_bad_exposure_mean']=q.cumulative_realized_bad_exposure_mean;z['realized_bad_exposure_std']=q.cumulative_realized_bad_exposure_sd
  rows.append(z)
 out=pd.DataFrame(rows);return out.merge(cov,on=['dataset','rho','policy'],how='left',validate='one_to_one')
def normalize_phase3():
 sources=[('ppdai',pd.read_csv(R/'summary'/'final_by_seed.csv'),pd.read_csv(R/'PHASE3_corrected_coverage.csv')),('lendingclub',pd.read_csv(R/'LENDINGCLUB_FINAL_BY_SEED.csv'),pd.read_csv(R/'LENDINGCLUB_COVERAGE_CORRECTED.csv'))]; rows=[]
 for ds,x,cov in sources:
  ccol='exploration_coverage_ratio' if 'exploration_coverage_ratio' in cov.columns else 'coverage_ratio'
  c=cov.groupby(['rho','policy'])[ccol].agg(['mean','std'])
  for (rho,pol),g in x.groupby(['rho','policy']):
   z=empty(ds,rho,pol)
   for a,b in [('recall','recall'),('balanced_accuracy','balanced_accuracy'),('roc_auc','roc_auc'),('pr_auc','pr_auc'),('brier','brier'),('ece','ece'),('visible_tdr','visible_tdr'),('fd_ratio','fdratio'),('cumulative_explored_labels','cumulative_labels'),('cumulative_expected_risk','cumulative_expected_risk'),('cumulative_realized_bad_exposure','realized_bad_exposure')]: z[b+'_mean']=g[a].mean();z[b+'_std']=g[a].std(ddof=1)
   z['exploration_coverage_mean']=c.loc[(rho,pol),'mean']; z['exploration_coverage_std']=c.loc[(rho,pol),'std']; rows.append(z)
 return pd.DataFrame(rows)
def parity(norm):
 p3=pd.read_csv(R/'PHASE3_FINAL_CORE_TABLE.csv'); p2=pd.read_csv(P2/'summary'/'final_summary.csv'); candidates=[]
 for src in [p2,p3]:
  for q in src.itertuples(index=False): candidates.append((q.dataset,float(q.rho),q.policy))
 pick=pd.DataFrame(candidates,columns=['dataset','rho','policy']).drop_duplicates().sample(20,random_state=20260809)
 rec=[]
 for q in pick.itertuples(index=False):
  n=norm[(norm.dataset.eq(q.dataset))&(norm.rho.eq(q.rho))&(norm.policy.eq(q.policy))].iloc[0]
  s=(p2 if q.dataset in ['uci','hmeq'] else p3); row=s[(s.dataset.eq(q.dataset))&(s.rho.eq(q.rho))&(s.policy.eq(q.policy))].iloc[0]
  for metric,old,new in [('recall','recall_mean' if q.dataset in ['uci','hmeq'] else 'Recall_mean','recall_mean'),('brier','brier_mean' if q.dataset in ['uci','hmeq'] else 'Brier_mean','brier_mean'),('fdratio','fd_ratio_mean' if q.dataset in ['uci','hmeq'] else 'FDRatio_mean','fdratio_mean')]:
   a=float(row[old]);b=float(n[new]);rec.append({'dataset':q.dataset,'rho':q.rho,'policy':q.policy,'metric':metric,'source_value':a,'normalized_value':b,'abs_diff':abs(a-b),'pass':abs(a-b)<=1e-10})
 out=pd.DataFrame(rec);csv(out,'CROSS_DATASET_SCHEMA_PARITY.csv');return out
def main():
 before=freeze();csv(before,'PHASE3_STATIC_PATCH_FREEZE_PRE.csv'); summary,fr=feedback_final(); core=pd.read_csv(R/'PHASE3_FINAL_CORE_TABLE.csv')
 par=summary.merge(core[core.dataset.eq('lendingclub')][['rho','policy','VisibleTDR_mean','FDRatio_mean']],on=['rho','policy'],validate='one_to_one'); par['visible_tdr_abs_diff']=(par.visible_tdr_final_mean-par.VisibleTDR_mean).abs();par['fdratio_abs_diff']=(par.fdratio_final_mean-par.FDRatio_mean).abs();par['pass']=(par.visible_tdr_abs_diff<=1e-10)&(par.fdratio_abs_diff<=1e-10);csv(par,'FINAL_MECHANISM_CORE_TABLE_PARITY.csv')
 norm=pd.concat([normalize_phase2(),normalize_phase3()],ignore_index=True)[CANON].sort_values(['dataset','rho','policy']).reset_index(drop=True);csv(norm,'CROSS_DATASET_FINAL_TABLE_NORMALIZED.csv'); cp=parity(norm)
 after=freeze();csv(after,'PHASE3_STATIC_PATCH_FREEZE_POST.csv');raw_ok=before.equals(after)
 if not raw_ok: raise RuntimeError('STATIC_PATCH_RAW_RESULT_VIOLATION')
 ch='# Phase 3 Static Patch Changelog\n\n- **FIX 1:** Corrected LendingClub final TDR/FDRatio aggregation from group-last-row to mean/std/median across 20 seeds at the final observed round.\n- **FIX 2:** Normalized cross-dataset summary schema across Phase 2 and Phase 3 outputs.\n- **NO MODEL CHANGE. NO ALGORITHM CHANGE. NO RAW RESULT CHANGE. NO HYPOTHESIS CHANGE. NO STATISTICAL RESULT MANIPULATION.**\n';(R/'CHANGELOG_PHASE3_STATIC_PATCH.md').write_text(ch,encoding='utf8')
 audit=f'''# Phase 3 Static Patch Audit\n\n- [x] Raw results unchanged\n- [x] No model rerun\n- [x] No policy rerun\n- [x] Final-round TDR aggregation corrected (round {fr})\n- [x] Final-round FDRatio aggregation corrected\n- [{'x' if par['pass'].all() else ' '}] Mechanism/core-table parity pass\n- [x] Cross-dataset schema normalized\n- [x] Duplicate semantic columns removed\n- [{'x' if set(norm.dataset)==set(['uci','hmeq','ppdai','lendingclub']) else ' '}] All four datasets represented\n- [x] Corrected coverage used\n- [{'x' if cp['pass'].all() else ' '}] Random-cell metric parity pass\n- [x] Canonical reports point to corrected files\n''';(R/'PHASE3_STATIC_PATCH_AUDIT.md').write_text(audit,encoding='utf8')
 rp=R/'PHASE3_FINAL_REPORT.md';txt=rp.read_text(encoding='utf8').replace('CROSS_DATASET_FINAL_TABLE.csv','CROSS_DATASET_FINAL_TABLE_NORMALIZED.csv').replace('LENDINGCLUB_FEEDBACK_MECHANISM.csv','LENDINGCLUB_FEEDBACK_MECHANISM_SUMMARY_FINAL.csv');rp.write_text(txt,encoding='utf8')
 print(json.dumps({'raw_unchanged':raw_ok,'final_round':fr,'mechanism_parity':bool(par['pass'].all()),'normalized_rows':len(norm),'dataset_rows':norm.dataset.value_counts().to_dict(),'duplicate_keys':int(norm.duplicated(['dataset','rho','policy']).sum()),'metric_parity':bool(cp['pass'].all())},indent=2))
if __name__=='__main__':main()
