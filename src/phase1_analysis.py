from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/"results_phase1"; FIG=RES/"figures"; PROT=RES/"protocol"
FIG.mkdir(exist_ok=True); PROT.mkdir(exist_ok=True)
DATASETS=["uci","hmeq"]
TREATMENTS=["Oracle","NoExplore","RandomExplore-2%","RandomExplore-5%","RandomExplore-10%"]
COMPARISONS=[("NoExplore","Oracle"),("RandomExplore-2%","NoExplore"),("RandomExplore-5%","NoExplore"),("RandomExplore-10%","NoExplore")]
METRICS=["recall","balanced_accuracy","visible_tdr","fd_ratio","brier","ece","roc_auc","pr_auc","accuracy","specificity","gmean"]
HIGHER={"recall","balanced_accuracy","visible_tdr","fd_ratio","roc_auc","pr_auc","accuracy","specificity","gmean"}

def bootstrap_ci(d, seed, B=10000):
    d=np.asarray(d,float); d=d[np.isfinite(d)]
    if len(d)==0: return np.nan,np.nan
    rng=np.random.RandomState(seed); means=np.empty(B)
    for i in range(B): means[i]=rng.choice(d,size=len(d),replace=True).mean()
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))

def process_dataset(name):
    d=pd.read_csv(RES/f"{name}_all_rounds.csv"); ds=RES/name; ds.mkdir(exist_ok=True)
    d.to_csv(ds/"all_rounds.csv",index=False)
    final=d[d["round"]==10].copy(); final.to_csv(ds/"final_by_seed.csv",index=False)
    traj=[]
    for (seed,treat),g in d.groupby(["seed","treatment"]):
        for metric in METRICS:
            vals=g.sort_values("round")[metric].to_numpy(float)
            traj.append({"dataset":name,"seed":seed,"treatment":treat,"metric":metric,"initial":vals[0],"final":vals[-1],"delta":vals[-1]-vals[0],"trajectory_mean":np.nanmean(vals)})
    tr=pd.DataFrame(traj)
    tsum=tr.groupby(["dataset","treatment","metric"]).agg(initial_mean=("initial","mean"),initial_std=("initial","std"),final_mean=("final","mean"),final_std=("final","std"),delta_mean=("delta","mean"),trajectory_mean=("trajectory_mean","mean"),trajectory_std=("trajectory_mean","std")).reset_index()
    tsum.to_csv(ds/"trajectory_summary.csv",index=False)
    tests=[]
    for a,b in COMPARISONS:
        for metric in METRICS:
            xa=final[final.treatment==a].set_index("seed")[metric]; xb=final[final.treatment==b].set_index("seed")[metric]
            joined=pd.concat([xa.rename("a"),xb.rename("b")],axis=1).dropna(); diff=(joined["a"]-joined["b"]).to_numpy(float)
            try: p=float(wilcoxon(diff,zero_method="wilcox",alternative="two-sided",method="auto").pvalue) if np.any(np.abs(diff)>0) else 1.0
            except Exception: p=1.0
            lo,hi=bootstrap_ci(diff,700000+len(tests),10000)
            tests.append({"dataset":name,"comparison":f"{a}_vs_{b}","metric":metric,"n":len(diff),"mean_diff":float(np.mean(diff)),"median_diff":float(np.median(diff)),"bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"wilcoxon_p":p,"wins":int(np.sum(diff>0)),"ties":int(np.sum(diff==0)),"losses":int(np.sum(diff<0))})
    tests=pd.DataFrame(tests); tests["holm_p"]=multipletests(tests["wilcoxon_p"].fillna(1.0),method="holm")[1]
    tests.to_csv(ds/"paired_tests.csv",index=False)
    rec=[]
    for rate in TREATMENTS[2:]:
        for metric in METRICS:
            o=final[final.treatment=="Oracle"].set_index("seed")[metric]; n=final[final.treatment=="NoExplore"].set_index("seed")[metric]; r=final[final.treatment==rate].set_index("seed")[metric]
            gap=(o-n).to_numpy(float); recovery=(r-n).to_numpy(float)
            if metric not in HIGHER: gap=-gap; recovery=-recovery
            ratio=np.where(np.abs(gap)>1e-12,recovery/gap,np.nan)
            rec.append({"dataset":name,"treatment":rate,"metric":metric,"oracle_minus_noexplore_mean":float(np.nanmean(gap)),"random_minus_noexplore_mean":float(np.nanmean(recovery)),"recovery_ratio_mean":float(np.nanmean(ratio)),"recovery_ratio_median":float(np.nanmedian(ratio))})
    pd.DataFrame(rec).to_csv(ds/"recovery_ratios.csv",index=False)
    d[["dataset","seed","treatment","round","n_true_default","n_true_default_pred_positive","n_true_default_below_threshold","n_true_nondefault","n_true_nondefault_above_threshold","mean_pd_default","mean_pd_nondefault","mean_test_pd","predicted_positive_rate"]].to_csv(ds/"threshold_crossing_summary.csv",index=False)
    rel=[]
    for _,row in d.iterrows():
        for item in json.loads(row["reliability_bins"]): rel.append({"dataset":name,"seed":int(row.seed),"treatment":row.treatment,"round":int(row["round"]),**item})
    pd.DataFrame(rel).to_csv(ds/"reliability_bins.csv",index=False)
    return d,tsum,tests

all_summaries=[]; all_tests=[]
for name in DATASETS:
    d,ts,tests=process_dataset(name); all_summaries.append(ts); all_tests.append(tests)
pd.concat(all_summaries,ignore_index=True).to_csv(RES/"trajectory_summary.csv",index=False)
pd.concat(all_tests,ignore_index=True).to_csv(RES/"paired_tests.csv",index=False)
checks=[]
for name in DATASETS:
    d=pd.read_csv(RES/f"{name}_all_rounds.csv")
    checks += [
        {"dataset":name,"check":"rows_20_seeds_x_5_treatments_x_11_rounds","passed":len(d)==1100,"detail":len(d)},
        {"dataset":name,"check":"seeds_0_to_19","passed":sorted(d.seed.unique().tolist())==list(range(20)),"detail":sorted(d.seed.unique().tolist())},
        {"dataset":name,"check":"treatments_exact","passed":sorted(d.treatment.unique().tolist())==sorted(TREATMENTS),"detail":sorted(d.treatment.unique().tolist())},
        {"dataset":name,"check":"rounds_0_to_10","passed":sorted(d["round"].unique().tolist())==list(range(11)),"detail":sorted(d["round"].unique().tolist())},
        {"dataset":name,"check":"runtime_invariants","passed":bool(d[["fixed_test_set_ok","current_round_only_ok","visible_update_ok","no_duplicate_ok","preprocessing_visible_only_ok"]].all().all()),"detail":"all rows"},
        {"dataset":name,"check":"same_batch_sizes_across_treatments","passed":bool(d.groupby(["seed","treatment"]).n_batch.first().unstack().nunique(axis=1).eq(1).all()),"detail":"per seed"},
        {"dataset":name,"check":"forbidden_fave_treatments_absent","passed":not d.treatment.isin(["CountOnly","CountOnly-MinRisk","LeastRisky","MostRisky","Uncertainty","RiskTilted-Uncertainty","FAVE-v2.1"]).any(),"detail":"Phase 1 treatment isolation"},
    ]
audit=pd.DataFrame(checks); audit.to_csv(RES/"PHASE1_INTEGRITY_AUDIT.csv",index=False)
(RES/"PHASE1_INTEGRITY_AUDIT.md").write_text("# Phase 1 integrity audit\n\n"+audit.to_string(index=False)+"\n\nAll checks passed: "+str(bool(audit.passed.all()))+"\n",encoding="utf-8")
config={"phase":"Phase 1 - Selective-label Feedback Mechanism Validation","datasets":{"uci":"data/raw/uci_default_credit_card.csv","hmeq":"data/raw/hmeq.csv"},"seeds":list(range(20)),"rounds":10,"test_fraction":0.25,"initial_visible_fraction":0.20,"tau":0.5,"treatments":TREATMENTS,"random_rates":[0.02,0.05,0.10],"random_k":"max(1, round(rate * current_round_rejected_count)) when count > 0","batch_partition":"exact_10_way_np_array_split","candidate_scope":"current_round_rejected_only","model":{"type":"XGBClassifier","n_estimators":100,"max_depth":3,"learning_rate":0.1,"tree_method":"hist","n_jobs":1},"preprocessing":"fit visible only","stop_after_phase1":True}
(PROT/"phase1_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
(PROT/"preregistration.json").write_text(json.dumps({"hypotheses":["H1 selective-label feedback distorts visible TDR/FDRatio","H2 default-class recall/BA may degrade versus Oracle","H3 accuracy/AUC may remain stable while recall/TDR degrade","H4 calibration may deteriorate without ranking collapse","H5 simple random exploration may recover part of the Oracle gap"],"primary_endpoints":["recall","balanced_accuracy","visible_tdr","fd_ratio","brier","ece"],"secondary_endpoints":["roc_auc","pr_auc","accuracy","specificity","calibration_slope","calibration_intercept"],"formal_seeds":"0..19","post_phase1_stop":True},indent=2),encoding="utf-8")
(PROT/"PHASE1_PROTOCOL.md").write_text("""# Phase 1 protocol

Treatments are exactly Oracle, NoExplore, RandomExplore-2%, RandomExplore-5%, and RandomExplore-10%. No FAVE, CountOnly, or risk-budget optimization is used.

Each dataset/seed shares one fixed 25% oracle test set, one D0 (20% of the non-test stream), one applicant ordering, and one exact ten-way batch partition. Models and preprocessing are fit only on currently visible labels. Random exploration samples only the current round rejected set with deterministic seed seed*100000 + round*100 + rate_code. Unselected rejected applicants leave the episode permanently.

Formal seeds are 0..19 for both UCI Default of Credit Card Clients and HMEQ. Phase 1 stops after mechanism validation; no Phase 2 execution is authorized by this protocol.
""",encoding="utf-8")
for name in DATASETS:
    d=pd.read_csv(RES/f"{name}_all_rounds.csv")
    for metric in ["visible_tdr","fd_ratio","recall","balanced_accuracy","roc_auc","brier","ece"]:
        plt.figure(figsize=(7,4))
        for treatment in TREATMENTS:
            g=d[d.treatment==treatment].groupby("round")[metric].agg(["mean","std","count"]).reset_index(); ci=1.96*g["std"]/np.sqrt(g["count"])
            plt.plot(g["round"],g["mean"],label=treatment); plt.fill_between(g["round"],g["mean"]-ci,g["mean"]+ci,alpha=.12)
        plt.xlabel("Round"); plt.ylabel(metric); plt.title(f"{name.upper()} {metric} trajectory"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG/f"{name}_{metric}_trajectory.png",dpi=160); plt.close()
print("analysis complete")


