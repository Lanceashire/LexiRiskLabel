from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"results_phase1"
HIGHER={"recall","balanced_accuracy","visible_tdr","fd_ratio","roc_auc","pr_auc","accuracy","specificity","gmean"}
# Hidden-information summary based on frozen NoExplore trajectory
hidden=[]; hidden_agg=[]
for name in ["uci","hmeq"]:
    d=pd.read_csv(R/name/"all_rounds.csv"); d=d[d.treatment=="NoExplore"].copy(); d=d[d["round"]>0]
    x=d[["dataset","seed","round","rejection_rate_current_batch","hidden_default_fraction","observed_default_fraction"]].copy()
    hidden.append(x)
    z=x[["rejection_rate_current_batch","hidden_default_fraction","observed_default_fraction"]].agg(["mean","std"]).T.reset_index().rename(columns={"index":"metric","mean":"mean_value","std":"sd_value"})
    z.insert(0,"dataset",name); hidden_agg.append(z)
pd.concat(hidden,ignore_index=True).to_csv(R/"hidden_information_summary.csv",index=False)
pd.concat(hidden_agg,ignore_index=True).to_csv(R/"hidden_information_summary_dataset_mean_sd.csv",index=False)

# Correct aggregate recovery ratios, preserving old per-seed statistic under legacy name
rows=[]
for name in ["uci","hmeq"]:
    f=pd.read_csv(R/name/"all_rounds.csv"); f=f[f["round"]==10]
    old=pd.read_csv(R/name/"recovery_ratios.csv") if (R/name/"recovery_ratios.csv").exists() else pd.DataFrame()
    for rate in ["RandomExplore-2%","RandomExplore-5%","RandomExplore-10%"]:
        for metric in ["recall","balanced_accuracy","visible_tdr","fd_ratio","brier","ece","roc_auc","pr_auc"]:
            means=f.groupby("treatment")[metric].mean()
            mo,mn,mr=means["Oracle"],means["NoExplore"],means[rate]
            if metric in HIGHER: gap=mo-mn; rec=mr-mn
            else: gap=mn-mo; rec=mn-mr
            o=f[f.treatment=="Oracle"].set_index("seed")[metric]; n=f[f.treatment=="NoExplore"].set_index("seed")[metric]; r=f[f.treatment==rate].set_index("seed")[metric]
            if abs(gap)>=1e-8:
                ratio=float(rec/gap)
                if metric in HIGHER: per=(r-n)/(o-n)
                else: per=(n-r)/(n-o)
                median_ratio=float(np.nanmedian(per))
                ratio_status="DEFINED"
            else:
                ratio="N/A"
                median_ratio="N/A"
                ratio_status="N/A_no_oracle_gap"
            o=f[f.treatment=="Oracle"].set_index("seed")[metric]; n=f[f.treatment=="NoExplore"].set_index("seed")[metric]; r=f[f.treatment==rate].set_index("seed")[metric]
            if metric in HIGHER: per=(r-n)/(o-n)
            else: per=(n-r)/(n-o)
            legacy=np.nan
            if len(old):
                q=old[(old.treatment==rate)&(old.metric==metric)]
                if len(q): legacy=float(q.iloc[0].recovery_ratio_mean)
            rows.append({"dataset":name,"exploration_rate":rate,"metric":metric,"mean_oracle":mo,"mean_noexplore":mn,"mean_random":mr,"oracle_gap":gap,"random_recovery":rec,"aggregate_recovery_ratio":ratio,"median_seed_recovery_ratio":median_ratio,"recovery_ratio_status":ratio_status,"legacy_per_seed_ratio_mean":legacy,"epsilon":1e-8})
pd.DataFrame(rows).to_csv(R/"recovery_ratios_corrected.csv",index=False)

# Calibration extended paired tests
rows=[]
for name in ["uci","hmeq"]:
    f=pd.read_csv(R/name/"all_rounds.csv"); f=f[f["round"]==10]
    for metric in ["calibration_slope","calibration_intercept"]:
        for a,b in [("NoExplore","Oracle"),("RandomExplore-2%","NoExplore"),("RandomExplore-5%","NoExplore"),("RandomExplore-10%","NoExplore")]:
            xa=f[f.treatment==a].set_index("seed")[metric]; xb=f[f.treatment==b].set_index("seed")[metric]
            q=pd.concat([xa.rename("a"),xb.rename("b")],axis=1).dropna(); diff=(q.a-q.b).to_numpy(float)
            try: p=float(wilcoxon(diff).pvalue) if np.any(np.abs(diff)>0) else 1.0
            except Exception: p=1.0
            rng=np.random.RandomState(900000+len(rows)); boots=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(10000)]) if len(diff) else np.array([np.nan])
            rows.append({"dataset":name,"comparison":f"{a}_vs_{b}","metric":metric,"n_valid":len(diff),"mean_diff":float(np.nanmean(diff)),"sd_diff":float(np.nanstd(diff,ddof=1)) if len(diff)>1 else np.nan,"median_diff":float(np.nanmedian(diff)) if len(diff) else np.nan,"bootstrap_ci_low":float(np.nanquantile(boots,.025)),"bootstrap_ci_high":float(np.nanquantile(boots,.975)),"wilcoxon_p":p,"wins":int(np.sum(diff>0)),"ties":int(np.sum(diff==0)),"losses":int(np.sum(diff<0))})
cal=pd.DataFrame(rows); cal["holm_p"]=cal.groupby("dataset")["wilcoxon_p"].transform(lambda s:multipletests(s,method="holm")[1]); cal.to_csv(R/"paired_tests_calibration_extended.csv",index=False)
print("closeout analysis artifacts generated")



