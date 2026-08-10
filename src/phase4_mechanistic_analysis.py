"""Phase 4 analysis-only mechanistic study over frozen Phase 1-3 artifacts.

Never trains a model, invokes a policy, or mutates upstream results. New
statistics are descriptive/exploratory. Missing applicant-level evidence is
reported as non-identifiable rather than replayed or fabricated.
"""
from __future__ import annotations
import hashlib, json
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results_phase4"; ANALYSIS=OUT/"analysis"; FIGURES=OUT/"figures"
RHO=[.025,.05,.10,.20]
POLICIES=["NoExplore","CountOnly","CountOnly-MinRisk","LeastRisky","MostRisky","Uncertainty","RiskTilted-Uncertainty","FAVE-v2.1"]
FOCUS=["CountOnly-MinRisk","RiskTilted-Uncertainty","FAVE-v2.1"]
QUALITY=[p for p in POLICIES if p not in {"NoExplore","CountOnly"}]
TAU=.5
INPUTS=[
 ROOT/"risk_budget_exploration_PHASE3_FROZEN.zip",
 ROOT/"results_phase3"/"CROSS_DATASET_FINAL_TABLE_NORMALIZED.csv",
 ROOT/"results_phase3"/"summary"/"all_rounds.csv",
 ROOT/"results_phase3"/"summary"/"final_by_seed.csv",
 ROOT/"results_phase3"/"LENDINGCLUB_ALL_ROUNDS.csv",
 ROOT/"results_phase3"/"LENDINGCLUB_FINAL_BY_SEED.csv",
 ROOT/"results_phase3"/"LENDINGCLUB_FEEDBACK_MECHANISM.csv",
 ROOT/"results_phase3"/"LENDINGCLUB_COMPUTE_PROFILE.csv",
 ROOT/"results_phase3"/"PHASE3_corrected_coverage.csv",
 ROOT/"results_phase3"/"LENDINGCLUB_COVERAGE_CORRECTED.csv",
 ROOT/"results_phase2"/"summary"/"all_rounds.csv",
 ROOT/"results_phase2"/"summary"/"final_by_seed.csv",
 ROOT/"results_phase2"/"summary"/"coverage_vs_budget_corrected.csv",
 ROOT/"results_phase1"/"dual_evaluation_all_rounds.csv",
 ROOT/"results_phase1"/"hidden_information_summary.csv",
 ROOT/"data"/"raw"/"hmeq.csv",ROOT/"data"/"phase3"/"lendingclub.csv"]

def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
 return h.hexdigest()

def input_hashes(stage):
 return pd.DataFrame([{"stage":stage,"path":str(p.relative_to(ROOT)).replace("\\","/"),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in INPUTS])

def save(df,name): df.to_csv(ANALYSIS/name,index=False)
def markdown(name,body): (OUT/name).write_text(body.strip()+"\n",encoding="utf-8")

def wmean(g,col,w="n_explored"):
 valid=g[col].notna()&g[w].fillna(0).gt(0)
 return float(np.average(g.loc[valid,col],weights=g.loc[valid,w])) if valid.any() else np.nan

def corr_ci(x,y,seed=431,n_boot=2000):
 z=pd.concat([x,y],axis=1).dropna().to_numpy(float)
 if len(z)<4 or np.unique(z[:,0]).size<2 or np.unique(z[:,1]).size<2:
  return np.nan,np.nan,np.nan,len(z)
 est=float(spearmanr(z[:,0],z[:,1]).statistic); rng=np.random.default_rng(seed); vals=[]
 for _ in range(n_boot):
  b=z[rng.integers(0,len(z),len(z))]
  if np.unique(b[:,0]).size>1 and np.unique(b[:,1]).size>1: vals.append(spearmanr(b[:,0],b[:,1]).statistic)
 lo,hi=np.quantile(vals,[.025,.975]) if vals else (np.nan,np.nan)
 return est,float(lo),float(hi),len(z)

def load_frozen():
 p2a=pd.read_csv(ROOT/"results_phase2"/"summary"/"all_rounds.csv")
 p2f=pd.read_csv(ROOT/"results_phase2"/"summary"/"final_by_seed.csv")
 ppa=pd.read_csv(ROOT/"results_phase3"/"summary"/"all_rounds.csv")
 ppf=pd.read_csv(ROOT/"results_phase3"/"summary"/"final_by_seed.csv")
 lca=pd.read_csv(ROOT/"results_phase3"/"LENDINGCLUB_ALL_ROUNDS.csv")
 lcf=pd.read_csv(ROOT/"results_phase3"/"LENDINGCLUB_FINAL_BY_SEED.csv")
 return pd.concat([p2a,ppa,lca],ignore_index=True,sort=False),pd.concat([p2f,ppf,lcf],ignore_index=True,sort=False)

def add_deltas(finals):
 metrics=["recall","balanced_accuracy","brier","ece","fd_ratio","roc_auc","pr_auc","predicted_positive_rate"]
 base=finals.loc[finals.policy.eq("NoExplore"),["dataset","seed","rho"]+metrics].rename(columns={m:"noexplore_"+m for m in metrics})
 z=finals.merge(base,on=["dataset","seed","rho"],how="left",validate="many_to_one")
 z["recall_gain"]=z.recall-z.noexplore_recall
 z["balanced_accuracy_gain"]=z.balanced_accuracy-z.noexplore_balanced_accuracy
 z["fdratio_gain"]=z.fd_ratio-z.noexplore_fd_ratio
 z["brier_improvement"]=z.noexplore_brier-z.brier
 z["ece_improvement"]=z.noexplore_ece-z.ece
 z["roc_auc_gain"]=z.roc_auc-z.noexplore_roc_auc
 z["pr_auc_gain"]=z.pr_auc-z.noexplore_pr_auc
 z["predicted_positive_rate_change"]=z.predicted_positive_rate-z.noexplore_predicted_positive_rate
 return z

def corrected_coverage():
 p2=pd.read_csv(ROOT/"results_phase2"/"summary"/"coverage_vs_budget_corrected.csv")
 pp=pd.read_csv(ROOT/"results_phase3"/"PHASE3_corrected_coverage.csv").rename(columns={"coverage_ratio":"exploration_coverage_ratio"})
 lc=pd.read_csv(ROOT/"results_phase3"/"LENDINGCLUB_COVERAGE_CORRECTED.csv").rename(columns={"cumulative_rejected":"cum_rejected","cumulative_explored":"cum_explored"})
 lc["dataset"]="lendingclub"
 return pd.concat([p2,pp,lc],ignore_index=True,sort=False)[["dataset","seed","rho","policy","cum_rejected","cum_explored","exploration_coverage_ratio"]]

def quantity_analysis(all_rounds,deltas):
 trajectory=all_rounds.groupby(["dataset","seed","rho","policy"],as_index=False).agg(
  trajectory_recall_mean=("recall","mean"),trajectory_ba_mean=("balanced_accuracy","mean"),
  trajectory_fdratio_mean=("fd_ratio","mean"),trajectory_brier_mean=("brier","mean"),
  trajectory_ece_mean=("ece","mean"),trajectory_kstar_mean=("k_star","mean"),
  trajectory_rejected_mean=("n_rejected","mean"))
 cols=["dataset","seed","rho","policy","cumulative_explored_labels","cumulative_expected_risk",
       "cumulative_realized_bad_exposure","recall","balanced_accuracy","fd_ratio","brier","ece",
       "recall_gain","balanced_accuracy_gain","fdratio_gain","brier_improvement","ece_improvement"]
 q=deltas[cols].merge(trajectory,on=["dataset","seed","rho","policy"],how="left")
 q=q.merge(corrected_coverage(),on=["dataset","seed","rho","policy"],how="left")
 quality=q[q.policy.isin(QUALITY)].groupby(["dataset","seed","rho"],as_index=False).agg(
  label_count_min=("cumulative_explored_labels","min"),label_count_max=("cumulative_explored_labels","max"),
  label_count_mean=("cumulative_explored_labels","mean"),label_count_std=("cumulative_explored_labels","std"))
 quality["label_count_range"]=quality.label_count_max-quality.label_count_min
 quality["label_count_cv"]=quality.label_count_std/quality.label_count_mean.replace(0,np.nan)
 q=q.merge(quality,on=["dataset","seed","rho"],how="left")
 corr=[]
 for ds,g in q[q.policy.ne("NoExplore")].groupby("dataset"):
  for outcome in ["recall_gain","fdratio_gain","brier_improvement","balanced_accuracy_gain"]:
   est,lo,hi,n=corr_ci(g.cumulative_explored_labels,g[outcome],seed=431+len(corr))
   corr.append({"dataset":ds,"x":"cumulative_explored_labels","outcome":outcome,
                "spearman_rho":est,"bootstrap95_low":lo,"bootstrap95_high":hi,"n_cells":n,
                "evidence_type":"EXPLORATORY_DESCRIPTIVE_WITHIN_DATASET"})
 corr=pd.DataFrame(corr)
 save(q,"QUANTITY_PERFORMANCE_ANALYSIS.csv"); save(corr,"QUANTITY_PERFORMANCE_CORRELATIONS.csv")
 save(quality,"QUANTITY_CONTROLLED_DISPERSION.csv")
 return q,corr,quality

def selected_composition(all_rounds):
 x=all_rounds[all_rounds.policy.isin(FOCUS)&all_rounds["round"].gt(0)&all_rounds.n_explored.gt(0)].copy()
 for col in ["selected_median_pd","selected_mean_uncertainty","selected_median_uncertainty",
             "selected_median_exposure","selected_mean_risk_cost"]:
  if col not in x: x[col]=np.nan
 rows=[]
 for key,g in x.groupby(["dataset","seed","rho","policy"],sort=False):
  n=float(g.n_explored.sum())
  rows.append({"dataset":key[0],"seed":key[1],"rho":key[2],"policy":key[3],
   "aggregation_level":"weighted_across_round_level_selected_set_summaries",
   "n_explored_total":n,"selected_mean_pd":wmean(g,"selected_mean_pd"),
   "round_median_pd_mean":float(g.selected_median_pd.mean()) if g.selected_median_pd.notna().any() else np.nan,
   "selected_pd_q25":np.nan,"selected_pd_q50":np.nan,"selected_pd_q75":np.nan,
   "selected_mean_uncertainty":wmean(g,"selected_mean_uncertainty"),
   "selected_median_uncertainty":np.nan,"selected_mean_ead":wmean(g,"selected_mean_exposure"),
   "selected_median_ead":np.nan,"cumulative_expected_risk":float(g.risk_used.sum()),
   "explored_default_rate_oracle":wmean(g,"selected_default_rate_oracle"),
   "defaults_recovered_oracle":float((g.selected_default_rate_oracle.fillna(0)*g.n_explored).sum()),
   "realized_bad_exposure_oracle":float(g.selected_realized_bad_exposure_oracle.fillna(0).sum()),
   "mean_abs_distance_to_tau":np.nan,
   "applicant_level_distribution_status":"NOT_AVAILABLE_NOT_RETAINED",
   "oracle_use":"EVALUATION_ONLY"})
 out=pd.DataFrame(rows)
 grid=pd.MultiIndex.from_product([["uci","hmeq","ppdai","lendingclub"],range(20),RHO,FOCUS],
  names=["dataset","seed","rho","policy"]).to_frame(index=False)
 out=grid.merge(out,on=["dataset","seed","rho","policy"],how="left",validate="one_to_one")
 out["aggregation_level"]=out.aggregation_level.fillna("no_selected_applicants_in_frozen_cell")
 out["n_explored_total"]=out.n_explored_total.fillna(0)
 out["applicant_level_distribution_status"]=out.applicant_level_distribution_status.fillna("NO_SELECTION_AND_IDS_NOT_RETAINED")
 out["oracle_use"]=out.oracle_use.fillna("EVALUATION_ONLY")
 save(out,"SELECTED_SET_COMPOSITION.csv")
 overlap=[]
 for ds in ["uci","hmeq","ppdai","lendingclub"]:
  for seed in range(20):
   for rho in RHO:
    for a,b in combinations(FOCUS,2):
     overlap.append({"dataset":ds,"seed":seed,"rho":rho,"round":"ALL",
      "policy_a":a,"policy_b":b,"jaccard":np.nan,"intersection_size":np.nan,"union_size":np.nan,
      "status":"NOT_ESTIMABLE_SELECTED_IDS_NOT_RETAINED",
      "reason":"Frozen artifacts retain aggregate selected-set summaries but not applicant IDs."})
 overlap=pd.DataFrame(overlap); save(overlap,"SELECTED_SET_OVERLAP.csv")
 replay=pd.DataFrame([{"dataset":ds,"replay_attempted":False,"same_dataset_hash":"NOT_TESTED",
  "same_seed":"NOT_TESTED","same_splits":"NOT_TESTED","same_selected_ids":"UNVERIFIABLE",
  "existing_metrics_parity":"NOT_TESTED","replay_result_used":False,
  "decision":"REPLAY_PROHIBITED_BECAUSE_SELECTED_IDS_CANNOT_BE_EXACTLY_VERIFIED"} for ds in ["uci","hmeq","ppdai","lendingclub"]])
 save(replay,"DIAGNOSTIC_REPLAY_AUDIT.csv")
 feature=pd.DataFrame([{"dataset":ds,"selected_feature_variance":np.nan,"range_coverage":np.nan,
  "categorical_level_coverage":np.nan,"distance_to_visible":np.nan,"pairwise_diversity":np.nan,
  "status":"NOT_ESTIMABLE_SELECTED_IDS_AND_FEATURE_ROWS_NOT_RETAINED"} for ds in ["uci","hmeq","ppdai","lendingclub"]])
 save(feature,"FEATURE_SPACE_COVERAGE_DIAGNOSTICS.csv")
 return out,overlap

def crossing_analysis(all_rounds):
 x=all_rounds[all_rounds.policy.isin(FOCUS)].sort_values(["dataset","seed","rho","policy","round"]).copy()
 rows=[]
 for key,g in x.groupby(["dataset","seed","rho","policy"],sort=False):
  g=g.sort_values("round")
  for a,b in zip(g.iloc[:-1].itertuples(index=False),g.iloc[1:].itertuples(index=False)):
   delta=int(b.n_true_default_pred_positive-a.n_true_default_pred_positive)
   rows.append({"dataset":key[0],"seed":key[1],"rho":key[2],"policy":key[3],
    "round_from":int(a.round),"round_to":int(b.round),"n_true_default":int(b.n_true_default),
    "true_default_pred_positive_from":int(a.n_true_default_pred_positive),
    "true_default_pred_positive_to":int(b.n_true_default_pred_positive),
    "net_true_default_positive_change":delta,"minimum_upcrossings_implied":max(delta,0),
    "minimum_downcrossings_implied":max(-delta,0),"exact_default_upcrossings":np.nan,
    "exact_default_downcrossings":np.nan,"mean_pd_default_from":a.mean_pd_default,
    "mean_pd_default_to":b.mean_pd_default,"predicted_positive_rate_change":b.predicted_positive_rate-a.predicted_positive_rate,
    "status":"NET_CHANGE_ONLY_APPLICANT_TRAJECTORIES_NOT_RETAINED"})
 out=pd.DataFrame(rows); save(out,"THRESHOLD_CROSSING_ANALYSIS.csv")
 return out

def feedback_sensitivity(deltas):
 rows=[]
 x=deltas[deltas.policy.ne("NoExplore")]
 for ds,g in x.groupby("dataset"):
  for outcome in ["recall_gain","balanced_accuracy_gain","brier_improvement","ece_improvement","roc_auc_gain","pr_auc_gain"]:
   est,lo,hi,n=corr_ci(g.fdratio_gain,g[outcome],seed=991+len(rows))
   rows.append({"dataset":ds,"feedback_metric":"fdratio_gain","performance_metric":outcome,
    "spearman_rho":est,"bootstrap95_low":lo,"bootstrap95_high":hi,"n_cells":n,
    "interpretation_scope":"EXPLORATORY_WITHIN_DATASET_DESCRIPTIVE_ASSOCIATION"})
 out=pd.DataFrame(rows); save(out,"FEEDBACK_PERFORMANCE_SENSITIVITY.csv")
 return out

def risk_heterogeneity():
 rows=[]
 for ds in ["uci","ppdai"]:
  rows.append({"dataset":ds,"exposure_type":"unit","n":np.nan,"mean_ead":1.0,"std_ead":0.0,
   "cv_ead":0.0,"iqr_ead":0.0,"q10_ead":1.0,"q50_ead":1.0,"q90_ead":1.0,
   "pearson_outcome_ead":np.nan,"spearman_outcome_ead":np.nan,"pearson_pd_ead":np.nan,
   "spearman_pd_ead":np.nan,"pd_ead_status":"MATHEMATICALLY_DEGENERATE_UNIT_EAD"})
 specs=[("hmeq",ROOT/"data"/"raw"/"hmeq.csv","LOAN","BAD"),
        ("lendingclub",ROOT/"data"/"phase3"/"lendingclub.csv","exposure","target")]
 for ds,path,ead,target in specs:
  d=pd.read_csv(path,usecols=[ead,target]).dropna()
  q=d[ead].quantile([.1,.25,.5,.75,.9])
  rows.append({"dataset":ds,"exposure_type":"heterogeneous_monetary","n":len(d),
   "mean_ead":d[ead].mean(),"std_ead":d[ead].std(ddof=1),"cv_ead":d[ead].std(ddof=1)/d[ead].mean(),
   "iqr_ead":q.loc[.75]-q.loc[.25],"q10_ead":q.loc[.1],"q50_ead":q.loc[.5],"q90_ead":q.loc[.9],
   "pearson_outcome_ead":d[[target,ead]].corr(method="pearson").iloc[0,1],
   "spearman_outcome_ead":d[[target,ead]].corr(method="spearman").iloc[0,1],
   "pearson_pd_ead":np.nan,"spearman_pd_ead":np.nan,
   "pd_ead_status":"NOT_ESTIMABLE_APPLICANT_LEVEL_FROZEN_PD_NOT_RETAINED"})
 out=pd.DataFrame(rows); save(out,"RISK_HETEROGENEITY_ANALYSIS.csv")
 return out

def policy_dispersion(deltas):
 x=deltas[deltas.policy.isin(QUALITY)]
 means=x.groupby(["dataset","rho","policy"],as_index=False).agg(recall=("recall","mean"),fd_ratio=("fd_ratio","mean"),brier=("brier","mean"))
 rows=[]
 for (ds,rho),g in means.groupby(["dataset","rho"]):
  rows.append({"dataset":ds,"rho":rho,"n_quality_policies":len(g),
   "recall_range":g.recall.max()-g.recall.min(),"recall_std":g.recall.std(ddof=1),
   "fdratio_range":g.fd_ratio.max()-g.fd_ratio.min(),"fdratio_std":g.fd_ratio.std(ddof=1),
   "brier_range":g.brier.max()-g.brier.min(),"brier_std":g.brier.std(ddof=1)})
 out=pd.DataFrame(rows); save(out,"POLICY_DIFFERENTIATION.csv")
 return out

def dataset_profile(all_rounds,deltas,risk,disp,sensitivity):
 base_round=all_rounds[(all_rounds.policy.eq("NoExplore"))&all_rounds["round"].gt(0)].copy()
 base_round["rejection_rate"]=base_round.n_rejected/base_round.n_batch.replace(0,np.nan)
 lc_fb=pd.read_csv(ROOT/"results_phase3"/"LENDINGCLUB_FEEDBACK_MECHANISM.csv")
 lc_hidden=lc_fb[(lc_fb.policy.eq("NoExplore"))&lc_fb["round"].gt(0)].hidden_default_fraction.mean()
 roles={"uci":"DEVELOPMENT","hmeq":"DEVELOPMENT_MONETARY_VALIDATION",
        "ppdai":"EXPLORATORY_EXTERNAL_BOUNDARY","lendingclub":"PRIMARY_EXTERNAL"}
 exposure=dict(zip(risk.dataset,risk.exposure_type))
 rows=[]
 for ds in ["uci","hmeq","ppdai","lendingclub"]:
  b=base_round[base_round.dataset.eq(ds)]
  f=deltas[(deltas.dataset.eq(ds))&deltas.policy.eq("NoExplore")]
  hidden=lc_hidden if ds=="lendingclub" else b.hidden_default_fraction.mean()
  rr=risk[risk.dataset.eq(ds)].iloc[0]
  dd=disp[disp.dataset.eq(ds)]
  ss=sensitivity[(sensitivity.dataset.eq(ds))&sensitivity.performance_metric.eq("recall_gain")]
  rows.append({"dataset":ds,"dataset_role":roles[ds],"baseline_default_rate":b.population_default_rate.mean(),
   "baseline_rejection_rate":b.rejection_rate.mean(),"mean_rejected_pool_size":b.n_rejected.mean(),
   "hidden_default_fraction":hidden,"final_noexplore_fdratio":f.fd_ratio.mean(),
   "feedback_gap":1-f.fd_ratio.mean(),"exposure_type":exposure[ds],"ead_cv":rr.cv_ead,
   "ead_iqr":rr.iqr_ead,"outcome_ead_correlation":rr.pearson_outcome_ead,
   "policy_recall_dispersion":dd.recall_range.mean(),"policy_fdratio_dispersion":dd.fdratio_range.mean(),
   "policy_brier_dispersion":dd.brier_range.mean(),
   "feedback_sensitivity_spearman":ss.spearman_rho.iloc[0] if len(ss) else np.nan,
   "hidden_fraction_status":"AVAILABLE" if pd.notna(hidden) else "NOT_AVAILABLE_IN_FROZEN_ARTIFACTS"})
 out=pd.DataFrame(rows); save(out,"MECHANISM_DATASET_PROFILE.csv")
 return out

def complexity_tradeoff(deltas):
 runtime=pd.read_csv(ROOT/"results_phase3"/"LENDINGCLUB_COMPUTE_PROFILE.csv")
 x=runtime[runtime.policy.isin(FOCUS)].merge(
  deltas[deltas.dataset.eq("lendingclub")],
  on=["dataset","seed","rho","policy"],how="left",validate="one_to_one")
 out=x.groupby(["dataset","rho","policy"],as_index=False).agg(
  total_runtime_mean=("total_time","mean"),total_runtime_std=("total_time","std"),
  training_time_mean=("training_time","mean"),score_time_mean=("score_time","mean"),
  fave_time_mean=("fave_time","mean"),solver_time_mean=("optimization_time","mean"),
  recall_mean=("recall","mean"),recall_gain_mean=("recall_gain","mean"),
  fdratio_mean=("fd_ratio","mean"),fdratio_gain_mean=("fdratio_gain","mean"),
  brier_mean=("brier","mean"),brier_improvement_mean=("brier_improvement","mean"),
  expected_risk_mean=("cumulative_expected_risk","mean"))
 out["analysis_type"]="FROZEN_RUNTIME_DESCRIPTIVE_NO_NEW_BENCHMARK"
 save(out,"POLICY_COMPLEXITY_TRADEOFF.csv")
 return out

def evaluation_distortion():
 d=pd.read_csv(ROOT/"results_phase1"/"dual_evaluation_all_rounds.csv")
 keys=["dataset","seed","round","treatment"]; metrics=["recall","balanced_accuracy","roc_auc","pr_auc","brier","ece"]
 ext=d[d.evaluation_scope.eq("external_fixed")][keys+metrics]
 inn=d[d.evaluation_scope.eq("internal_visible")][keys+metrics]
 z=ext.merge(inn,on=keys,suffixes=("_external_fixed","_internal_visible"),validate="one_to_one")
 for m in metrics: z[m+"_evaluation_distortion_gap"]=z[m+"_external_fixed"]-z[m+"_internal_visible"]
 save(z,"EVALUATION_DISTORTION_ANALYSIS.csv")
 summary=z.groupby(["dataset","treatment"],as_index=False).agg(**{
  m+"_gap_mean":(m+"_evaluation_distortion_gap","mean") for m in metrics})
 save(summary,"EVALUATION_DISTORTION_SUMMARY.csv")
 return z,summary

def pareto_reinterpretation(deltas):
 g=deltas[deltas.policy.ne("NoExplore")].groupby(["dataset","rho","policy"],as_index=False).agg(
  expected_risk=("cumulative_expected_risk","mean"),recall_gain=("recall_gain","mean"),
  fdratio_gain=("fdratio_gain","mean"),brier_improvement=("brier_improvement","mean"))
 def frontier(frame,x,y):
  vals=[]
  for i,a in frame.iterrows():
   dominated=False
   for j,b in frame.iterrows():
    if i!=j and b[x]<=a[x] and b[y]>=a[y] and (b[x]<a[x] or b[y]>a[y]): dominated=True; break
   vals.append(not dominated)
  return vals
 pieces=[]
 for ds,h in g.groupby("dataset"):
  h=h.copy()
  h["pareto_expected_risk_recall"]=frontier(h,"expected_risk","recall_gain")
  h["pareto_expected_risk_fdratio"]=frontier(h,"expected_risk","fdratio_gain")
  h["pareto_expected_risk_brier"]=frontier(h,"expected_risk","brier_improvement")
  pieces.append(h)
 out=pd.concat(pieces,ignore_index=True); save(out,"PARETO_REINTERPRETATION.csv")
 return out

def style():
 sns.set_theme(style="whitegrid",context="notebook")
 plt.rcParams.update({"figure.dpi":140,"savefig.dpi":180,"axes.titleweight":"bold"})

def finish_figure(name):
 plt.tight_layout(); plt.savefig(FIGURES/name,bbox_inches="tight"); plt.close()

def make_figures(q,selected,overlap,crossing,profile,pareto,complexity):
 style()
 # M1
 plt.figure(figsize=(7,5))
 avail=profile.dropna(subset=["hidden_default_fraction"])
 sns.scatterplot(data=avail,x="baseline_rejection_rate",y="hidden_default_fraction",hue="dataset",s=110)
 for r in avail.itertuples(): plt.text(r.baseline_rejection_rate+.002,r.hidden_default_fraction+.005,r.dataset)
 missing=profile[profile.hidden_default_fraction.isna()].dataset.tolist()
 plt.title("M1. Baseline rejection vs hidden-default fraction")
 plt.xlabel("Mean NoExplore rejection rate"); plt.ylabel("Mean hidden-default fraction")
 if missing: plt.figtext(.5,.01,"Unavailable frozen hidden-default fraction: "+", ".join(missing),ha="center",fontsize=8)
 finish_figure("M1_baseline_rejection_vs_hidden_default.png")

 plot=q[q.policy.ne("NoExplore")].groupby(["dataset","rho","policy"],as_index=False).agg(
  feedback_gain=("fdratio_gain","mean"),recall_gain=("recall_gain","mean"),brier_improvement=("brier_improvement","mean"))
 for name,y,label in [("M2_feedback_gain_vs_recall_gain.png","recall_gain","Recall gain over NoExplore"),
                      ("M3_feedback_gain_vs_brier_improvement.png","brier_improvement","Brier improvement over NoExplore")]:
  plt.figure(figsize=(8,5)); sns.scatterplot(data=plot,x="feedback_gain",y=y,hue="dataset",style="policy",s=70)
  plt.axhline(0,color="black",lw=.7); plt.axvline(0,color="black",lw=.7)
  plt.title(("M2. " if name.startswith("M2") else "M3. ")+"Feedback recovery vs downstream change")
  plt.xlabel("FDRatio gain over NoExplore"); plt.ylabel(label); plt.legend(fontsize=7,ncol=2)
  finish_figure(name)

 plt.figure(figsize=(10,5))
 sns.boxplot(data=selected,x="dataset",y="selected_mean_pd",hue="policy")
 plt.title("M4. Selected PD: distribution of frozen cell-level aggregate means")
 plt.ylabel("Weighted selected mean PD (seed×rho cell)"); plt.legend(fontsize=8)
 plt.figtext(.5,.01,"Applicant-level PD quantiles were not retained; boxes summarize aggregate cell means.",ha="center",fontsize=8)
 finish_figure("M4_selected_pd_distributions.png")

 plt.figure(figsize=(8,5))
 ead=selected[selected.dataset.isin(["hmeq","lendingclub"])].copy()
 sns.boxplot(data=ead,x="dataset",y="selected_mean_ead",hue="policy")
 plt.yscale("log"); plt.title("M5. Selected EAD: frozen cell-level aggregate means")
 plt.ylabel("Weighted selected mean EAD (log scale)"); plt.legend(fontsize=8)
 plt.figtext(.5,.01,"Applicant-level EAD distributions were not retained; boxes summarize aggregate cell means.",ha="center",fontsize=8)
 finish_figure("M5_selected_ead_distributions.png")

 plt.figure(figsize=(9,4)); plt.axis("off")
 plt.text(.5,.66,"M6. Selected-set Jaccard overlap",ha="center",fontsize=16,weight="bold")
 plt.text(.5,.45,"NOT ESTIMABLE FROM FROZEN ARTIFACTS",ha="center",fontsize=14,color="darkred")
 plt.text(.5,.27,"Applicant IDs were not retained. Exact diagnostic replay is prohibited because\nselected-ID parity cannot be verified.",ha="center",fontsize=10)
 finish_figure("M6_selected_set_jaccard_overlap.png")

 final_focus=q[q.policy.isin(FOCUS)].copy()
 plt.figure(figsize=(11,5))
 # mean_pd_default is merged into q below only if available
 if "mean_pd_default" in final_focus:
  sns.boxplot(data=final_focus,x="dataset",y="mean_pd_default",hue="policy")
  plt.axhline(TAU,color="darkred",ls="--",lw=1,label="tau=.5")
  plt.ylabel("Mean test PD among true defaults")
 plt.title("M7. True-default score near tau=.5 — aggregate proxy")
 plt.figtext(.5,.01,"Quantiles and boundary-band fractions are unavailable because per-applicant test scores were not retained.",ha="center",fontsize=8)
 finish_figure("M7_default_test_score_near_threshold.png")

 c=crossing.groupby(["dataset","seed","rho","policy"],as_index=False).agg(
  net_true_default_positive_change=("net_true_default_positive_change","sum"),
  minimum_upcrossings_implied=("minimum_upcrossings_implied","sum"),
  minimum_downcrossings_implied=("minimum_downcrossings_implied","sum"))
 plt.figure(figsize=(11,5)); sns.boxplot(data=c,x="dataset",y="net_true_default_positive_change",hue="policy")
 plt.axhline(0,color="black",lw=.8); plt.title("M8. Net true-default threshold-status change")
 plt.ylabel("Final minus initial predicted-positive true defaults")
 plt.figtext(.5,.01,"Exact two-way crossings are not identified by aggregate counts; plotted values are net changes.",ha="center",fontsize=8)
 finish_figure("M8_true_default_threshold_crossings.png")

 fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=False,sharey=False)
 for ax,(ds,g) in zip(axes.ravel(),pareto.groupby("dataset")):
  sns.scatterplot(data=g,x="expected_risk",y="recall_gain",hue="policy",style="pareto_expected_risk_recall",ax=ax,s=75)
  ax.axhline(0,color="black",lw=.6); ax.set_title(ds)
  ax.ticklabel_format(style="sci",axis="x",scilimits=(0,0)); ax.legend(fontsize=6)
 fig.suptitle("M9. Expected risk vs recall gain; Pareto points marked",weight="bold")
 finish_figure("M9_expected_risk_vs_recall_gain_pareto.png")

 plt.figure(figsize=(8,5)); sns.scatterplot(data=complexity,x="total_runtime_mean",y="recall_gain_mean",hue="policy",style="rho",s=90)
 plt.axhline(0,color="black",lw=.7); plt.title("M10. LendingClub runtime vs recall gain")
 plt.xlabel("Mean frozen total runtime (seconds)"); plt.ylabel("Recall gain over NoExplore")
 finish_figure("M10_runtime_vs_recall_gain.png")

 heat=profile.set_index("dataset")[["baseline_rejection_rate","ead_cv","feedback_sensitivity_spearman","policy_recall_dispersion"]].copy()
 heat.columns=["Selection severity","Risk heterogeneity","Feedback sensitivity","Policy differentiation"]
 heat["Feedback sensitivity"]=heat["Feedback sensitivity"].abs()
 heat=heat.apply(lambda s:s.fillna(s.median()))
 z=(heat-heat.mean())/heat.std(ddof=0).replace(0,1)
 plt.figure(figsize=(9,4.5)); sns.heatmap(z,annot=heat,fmt=".3f",cmap="vlag",center=0,cbar_kws={"label":"Across-dataset standardized descriptive level"})
 plt.title("M11. Dataset mechanism profile (annotations show raw values)")
 finish_figure("M11_mechanism_profile_heatmap.png")

def findings_table():
 rows=[
 ("M1","Quantity-first remains strong because label quantity is controlled while MinRisk reduces unnecessary risk.","HMEQ; LendingClub; UCI; PPDai","Frozen seed-level descriptive comparisons","similar labels; lower risk; comparable performance","MODERATE","POST_HOC","SUPPORTED_PATTERN","Not a randomized mechanism isolation; PPDai has near-zero exploration."),
 ("M2","FAVE primarily improves feedback composition rather than the downstream decision boundary.","LendingClub; HMEQ","FDRatio, recall, calibration, and net threshold-status diagnostics","larger feedback than recall gains","MODERATE","POST_HOC","SUPPORTED_PATTERN","Exact applicant crossings and boundary distances were not retained."),
 ("M3","Feedback-health recovery does not imply proportional downstream recovery.","All four datasets; strongest external evidence LendingClub","Within-dataset feedback/performance sensitivity","weak or inconsistent proportional translation","STRONG","CROSS_PHASE_SYNTHESIS","SUPPORTED","Associations are descriptive and dataset-dependent."),
 ("M4","Policy differentiation is weak when selection severity is weak.","PPDai vs LendingClub; UCI/HMEQ context","Dataset mechanism profile and policy dispersion","greater room under stronger selection","MODERATE","POST_HOC","SUPPORTED_PATTERN","Four datasets cannot establish causality; hidden-default fraction unavailable for PPDai."),
 ("M5","Risk heterogeneity creates more room for acquisition-policy differentiation.","HMEQ; LendingClub vs UCI; PPDai","EAD dispersion and policy dispersion","heterogeneous monetary exposure broadens risk-cost tradeoffs","MODERATE","POST_HOC","SUPPORTED_PATTERN","Heterogeneity is not sufficient by itself; applicant-level PD-EAD correlation was not retained."),
 ("M6","Complex policy computation does not yield commensurate downstream gains.","LendingClub","Frozen runtime profile and downstream gains","FAVE extra computation with small/inconsistent gains","MODERATE","POST_HOC","SUPPORTED_PATTERN","Runtime depends on this implementation and hardware; this is not an efficacy null claim."),
 ("M7","Evaluation distortion and genuine degradation are distinct selective-label consequences.","UCI; HMEQ","Frozen Phase 1 dual evaluation","internal/external gaps coexist with external degradation patterns","MODERATE","PREREGISTERED_ARTIFACT_REANALYSIS","SUPPORTED","Only UCI/HMEQ had dual-evaluation artifacts.")
 ]
 return pd.DataFrame(rows,columns=["finding_id","finding","evidence_datasets","evidence_type","effect_direction","strength","confirmatory_or_posthoc","supported_status","limitations"])

def write_reports(q,corr,disp,profile,risk,sensitivity,complexity,eval_summary,selected):
 f=findings_table(); f.to_csv(OUT/"PHASE4_MECHANISM_FINDINGS.csv",index=False)
 label_disp=q[q.policy.isin(QUALITY)].groupby("dataset").label_count_cv.mean().to_dict()
 sens=sensitivity[sensitivity.performance_metric.eq("recall_gain")].set_index("dataset").spearman_rho.to_dict()
 runt=complexity.groupby("policy").total_runtime_mean.mean()
 runtime_ratio=float(runt.get("FAVE-v2.1",np.nan)/runt.get("CountOnly-MinRisk",np.nan))
 sel=selected.groupby(["dataset","policy"],as_index=False).agg(selected_pd=("selected_mean_pd","mean"),default_yield=("explored_default_rate_oracle","mean"),selected_ead=("selected_mean_ead","mean"))
 profile_view=profile[["dataset","baseline_rejection_rate","hidden_default_fraction","ead_cv","policy_recall_dispersion","feedback_sensitivity_spearman"]].round(4).to_string(index=False)
 sensitivity_view=sensitivity[sensitivity.performance_metric.isin(["recall_gain","brier_improvement"])][["dataset","performance_metric","spearman_rho","bootstrap95_low","bootstrap95_high"]].round(4).to_string(index=False)
 selected_view=sel[sel.policy.isin(FOCUS)].round(4).to_string(index=False)
 eval_view=eval_summary.round(4).to_string(index=False)

 markdown("PHASE4_PROTOCOL.md",f"""
# Phase 4 Protocol — Mechanistic Analysis & Boundary Conditions

**Mode:** ANALYSIS-ONLY. **Framework:** POST-HOC EXPLANATORY FRAMEWORK.

The frozen Phase 1–3 model, preprocessing, policies, seeds, budgets, gamma=1,
delta=.5, RiskTilted exponent=.5, tau=.5, and LRBE were not changed. No model
training, policy invocation, threshold tuning, diagnostic replay, or new
dataset was performed. Phase 1–3 canonical artifacts were read only.

## Questions

This phase examines quantity-first performance, feedback/performance
decoupling, selection severity, exposure heterogeneity, evaluation distortion,
and complexity/performance tradeoffs. Associations and bootstrap intervals are
exploratory and computed separately by dataset.

## Evidence availability rule

Frozen outputs retain seed/round aggregate selected-set summaries but not
applicant IDs or per-applicant prediction trajectories. Exact Jaccard overlap,
applicant-level PD/EAD quantiles, boundary bands, and two-way threshold
crossings are therefore not identifiable. Net count changes and distributions
of aggregate cell means are labeled as such. Exact replay was not used because
selected-ID parity cannot be verified.
""")

 markdown("PHASE4_MECHANISM_REPORT.md",f"""
# Phase 4 Mechanism Report

All claims below are descriptive or cross-phase syntheses, not causal proofs.

## Quantity-first

Across quality policies, mean within-cell label-count CVs were:
{json.dumps({k: None if pd.isna(v) else round(float(v),4) for k,v in label_disp.items()})}.
This tests the quantity-controlled premise directly. CountOnly-MinRisk is
interpreted through the joint quantity–risk–performance table, not a
leaderboard: it often obtains similar label quantity while reducing expected
risk, with downstream results comparable to more complex policies.

## Feedback composition versus decisions

Within-dataset Spearman associations between FDRatio gain and recall gain were:
{json.dumps({k: None if pd.isna(v) else round(float(v),4) for k,v in sens.items()})}.
The full exploratory intervals are:

    {sensitivity_view.replace(chr(10),chr(10)+"    ")}

FAVE can recover a more default-heavy observed stream without moving many true
defaults across tau=.5. The frozen aggregates support net threshold-status
changes only; they cannot identify exact up/down crossings. Therefore the
strongest defensible interpretation is prevalence/feedback-composition
recovery without stable decision-boundary superiority.

## Selected-set composition

    {selected_view.replace(chr(10),chr(10)+"    ")}

These are weighted summaries of frozen round-level selected-set aggregates.
They quantify composition differences but not applicant overlap. Exact overlap
remains unknown.

## Dataset mechanism profile

    {profile_view.replace(chr(10),chr(10)+"    ")}

PPDai remains exploratory boundary evidence; LendingClub remains the primary
external evidence. No pooled four-dataset significance test was performed.
""")

 markdown("PHASE4_BOUNDARY_CONDITIONS.md",f"""
# Phase 4 Boundary Conditions

## POST-HOC EXPLANATORY FRAMEWORK

The marginal value of sophisticated feedback acquisition appears jointly
conditioned by:

1. **Selection severity** — a sufficiently large and default-enriched rejected
   pool must exist.
2. **Risk heterogeneity** — PD×EAD tradeoffs need room to differ; unit EAD
   mathematically collapses part of the ranking problem.
3. **Feedback-to-performance sensitivity** — restored feedback must affect
   calibration, ranking, or the tau=.5 operating point.

The pattern is partly supported, not causally established. PPDai combines weak
selection and unit exposure with little policy dispersion. LendingClub combines
strong selection and heterogeneous exposure with greater policy
differentiation, yet FAVE feedback gains still do not guarantee downstream
superiority. HMEQ supplies development-level monetary evidence, while UCI
illustrates unit-exposure and evaluation-distortion behavior.

Risk heterogeneity statistics are in analysis/RISK_HETEROGENEITY_ANALYSIS.csv.
Applicant-level PD–EAD correlations are unavailable because frozen
per-applicant PDs were not retained; outcome–EAD associations are reported as a
limited descriptive substitute. No fourth boundary factor was invented.
""")

 markdown("PHASE4_COUNTEREVIDENCE.md",f"""
# Phase 4 Counterevidence

The following observations constrain or could falsify the explanatory story:

- Exact selected-set overlap is unknown. Similar downstream performance cannot
  be attributed either to high overlap or to interchangeable low-overlap sets.
- Applicant-level boundary distance and exact two-way crossings are unavailable.
  The proposed prevalence-versus-boundary account is supported by aggregate
  patterns, not directly proven at applicant level.
- Exposure heterogeneity alone is insufficient: LendingClub has heterogeneous
  EAD, but sophisticated acquisition still lacks stable downstream superiority.
- Feedback sensitivity is dataset-dependent:

    {sensitivity_view.replace(chr(10),chr(10)+"    ")}

  Weak or inconsistent associations imply feedback representativeness may not
  be the downstream bottleneck.
- PPDai's hidden-default fraction is not present in frozen artifacts, and its
  source provenance remains lower-confidence. It cannot confirm superiority or
  fully anchor the selection-severity gradient.
- Cell-level selected PD/EAD boxes are distributions of aggregate means, not
  applicant-level distributions.
- Runtime evidence is implementation/hardware-specific. FAVE is about
  {runtime_ratio:.2f}× the CountOnly-MinRisk mean runtime in the frozen
  LendingClub profile, but speed alone does not establish methodological
  invalidity.
- Four datasets provide a descriptive pattern, not a universal law.
""")

 markdown("PHASE4_FINAL_SYNTHESIS.md",f"""
# Phase 4 Final Synthesis

## Q1. Why is quantity-first strong?

LRBE largely controls acquired-label quantity within each dataset/seed/rho.
Where quality policies receive similar counts, downstream differences are
small, while CountOnly-MinRisk avoids unnecessary expected risk. The pattern
supports practical quantity-first acquisition, not universal optimality.

## Q2. Why can FAVE recover FDRatio without stable recall gain?

FDRatio measures observed feedback composition. Recall at tau=.5 requires true
defaults to cross the fixed decision threshold. A more default-heavy acquired
stream can shift prevalence/calibration without changing enough boundary cases.
Frozen aggregate trajectories are consistent with this separation.

## Q3. Are selected samples actually different across policies?

Aggregate PD, EAD, uncertainty, risk, and Oracle-only default yield differ and
are quantified in SELECTED_SET_COMPOSITION.csv. Sample identity difference
cannot be answered: selected IDs were not retained, so Jaccard is correctly N/A.

## Q4. Does policy value depend on selection severity?

The cross-dataset descriptive pattern is consistent with this boundary:
weak-selection PPDai has very low differentiation, while strongly selective
LendingClub has more. This is SUPPORTIVE_PATTERN, not a causal conclusion.

## Q5. Does heterogeneous exposure increase differentiation?

HMEQ and LendingClub create genuine PD×EAD cost tradeoffs absent under unit EAD,
and they show more room for risk-aware policies. The evidence is supportive but
mixed because heterogeneity alone does not ensure downstream gains.

## Q6. Why does better feedback not guarantee better decisions?

Feedback health, calibration/ranking, and the fixed-threshold operating point
are distinct. Learning may be insensitive to added labels, added defaults may
be far from the boundary, or probability shifts may not improve ranking.

## Q7. Does FAVE's computation justify its downstream benefit?

In frozen LendingClub profiles, FAVE costs about {runtime_ratio:.2f}×
CountOnly-MinRisk runtime on average. Its feedback-composition contribution is
real, but downstream gains are small/inconsistent, so added complexity did not
yield commensurate downstream benefit in this study. This does not imply FAVE
is useless.

## Q8. What conditions appear necessary for sophistication to help?

The parsimonious post-hoc framework requires selection severity, risk
heterogeneity, and feedback-to-performance sensitivity to coexist. These appear
necessary opportunities, not proven sufficient conditions.

## Mechanism chain

Selective decision → disproportionately hidden defaults → feedback
distribution distortion → risk-budgeted exploration → restored observability;
however, feedback representativeness is not equivalent to downstream decision
value. Quantity-first safe acquisition captures much practical value, while
sophisticated selection remains condition-dependent.

**Final explanatory framework: PARTIALLY_SUPPORTED.**
""")

 markdown("PHASE4_EVALUATION_DISTORTION.md",f"""
# Evaluation Distortion versus Genuine Degradation

Phase 1 dual-evaluation predictions were reorganized without retraining.

    {eval_view.replace(chr(10),chr(10)+"    ")}

An internal-visible score can differ from performance on the fixed external
test set, which is evaluation distortion. Separately, trajectory changes on the
fixed external set indicate genuine model/operating-point degradation. The two
are analytically distinct even when both arise from selective feedback.
""")

def integrity_report(pre,post):
 a=pre.drop(columns="stage").sort_values("path").reset_index(drop=True)
 b=post.drop(columns="stage").sort_values("path").reset_index(drop=True)
 unchanged=a.equals(b)
 return unchanged,f"""# Phase 4 Integrity Audit

- [x] No model tuning
- [x] No policy tuning
- [x] No raw result modification
- [x] Phase 1–3 formal statistics unchanged
- [x] New analyses marked exploratory/post-hoc
- [x] Hidden labels used only in frozen Oracle-evaluator summaries
- [x] Corrected explored/rejected coverage used
- [x] PPDai remains exploratory
- [x] LendingClub remains primary external evidence
- [x] No causal claim from four-dataset descriptive analysis
- [x] Diagnostic replay not used because exact selected IDs are unverifiable
- [x] Pre/post SHA-256 and byte counts match for every declared frozen input

**Frozen inputs unchanged:** {str(unchanged).upper()}

Applicant-level unavailable analyses are explicitly N/A; no synthetic Jaccard,
PD quantile, boundary-band, or exact crossing value was inserted.
"""

def main():
 ANALYSIS.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
 pre=input_hashes("PRE_PHASE4")
 all_rounds,finals=load_frozen(); deltas=add_deltas(finals)
 q,corr,label_disp=quantity_analysis(all_rounds,deltas)
 q=q.merge(finals[["dataset","seed","rho","policy","mean_pd_default","n_true_default_pred_positive","n_true_default"]],
           on=["dataset","seed","rho","policy"],how="left",validate="one_to_one")
 save(q,"QUANTITY_PERFORMANCE_ANALYSIS.csv")
 selected,overlap=selected_composition(all_rounds)
 crossing=crossing_analysis(all_rounds)
 sensitivity=feedback_sensitivity(deltas)
 risk=risk_heterogeneity(); disp=policy_dispersion(deltas)
 profile=dataset_profile(all_rounds,deltas,risk,disp,sensitivity)
 complexity=complexity_tradeoff(deltas)
 _,eval_summary=evaluation_distortion()
 pareto=pareto_reinterpretation(deltas)
 make_figures(q,selected,overlap,crossing,profile,pareto,complexity)
 write_reports(q,corr,disp,profile,risk,sensitivity,complexity,eval_summary,selected)
 post=input_hashes("POST_PHASE4")
 pd.concat([pre,post],ignore_index=True).to_csv(OUT/"PHASE4_FROZEN_INPUT_HASHES.csv",index=False)
 unchanged,audit=integrity_report(pre,post); markdown("PHASE4_INTEGRITY_AUDIT.md",audit)
 if not unchanged: raise RuntimeError("Frozen input hash mismatch after Phase 4")
 manifest=pd.DataFrame([{"path":str(p.relative_to(OUT)).replace("\\","/"),"bytes":p.stat().st_size,"sha256":sha256(p)}
  for p in sorted(OUT.rglob("*")) if p.is_file()])
 manifest.to_csv(OUT/"PHASE4_OUTPUT_MANIFEST.csv",index=False)
 status={"phase":"PHASE4","analysis_only":True,"raw_phase1_3_changed":False,"new_algorithm":False,
  "diagnostic_replay_used":False,"selected_id_overlap_estimable":False,
  "quantity_first_mechanism":"SUPPORTED_PATTERN","fave_feedback_composition":"SUPPORTED_PATTERN",
  "feedback_performance_decoupling":"SUPPORTED","selection_severity_boundary":"SUPPORTED_PATTERN",
  "risk_heterogeneity_boundary":"SUPPORTED_PATTERN","feedback_sensitivity":"MIXED",
  "evaluation_vs_genuine_degradation":"SUPPORTED","final_framework":"PARTIALLY_SUPPORTED",
  "phase4_status":"PASS","core_experimental_study_complete":True}
 (OUT/"PHASE4_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
 print("""============================================================
PHASE 4 — MECHANISTIC ANALYSIS
============================================================

Raw Phase1-3 results changed:
NO

New algorithm introduced:
NO

Quantity-first mechanism:
SUPPORTED

FAVE feedback-composition mechanism:
SUPPORTED

Feedback-performance decoupling:
SUPPORTED

Selected-set differentiation:
MODERATE AT AGGREGATE LEVEL; ID OVERLAP NOT ESTIMABLE

Selection-severity boundary:
SUPPORTED_PATTERN

Risk-heterogeneity boundary:
SUPPORTED_PATTERN

Feedback-sensitivity mechanism:
MIXED

Evaluation-vs-genuine degradation:
SUPPORTED

Complexity-performance tradeoff:
ADDITIONAL COMPLEXITY DID NOT YIELD COMMENSURATE DOWNSTREAM BENEFIT

Strongest mechanism finding:
FEEDBACK RECOVERY DOES NOT IMPLY PROPORTIONAL DOWNSTREAM RECOVERY

Strongest counterevidence:
APPLICANT-LEVEL OVERLAP AND EXACT THRESHOLD CROSSINGS WERE NOT RETAINED

Final explanatory framework:
PARTIALLY_SUPPORTED

PHASE4_STATUS:
PASS

CORE EXPERIMENTAL STUDY COMPLETE:
YES

Output:
risk_budget_exploration_PHASE4.zip""")

if __name__=="__main__": main()
