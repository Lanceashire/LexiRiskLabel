from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"src"))
import phase1_runner as p1

ROOT=p1.ROOT; RES=ROOT/"results_phase1"; rows=[]; parity=[]

def eval_scope(raw,y,fit_idx,eval_idx,pop_rate,visible,seed,scope,batch=None,p_batch=None):
    pre,model=p1.fit_predict(raw,y,fit_idx,seed)
    pe=model.predict_proba(pre.transform(raw.iloc[eval_idx]))[:,1]
    mm=p1.metrics(y[eval_idx],pe,y[visible],pop_rate,[],np.array([]),np.array([]),np.array([]),np.array([]))
    mm.update({"evaluation_scope":scope,"mean_test_pd":mm["mean_test_pd"]})
    return mm

for name,cfg in p1.DATASETS.items():
    df=pd.read_csv(cfg["path"]); target=cfg["target"]; y=df[target].astype(int).to_numpy()
    raw=df[[c for c in df.columns if c not in {target,"ID"}]]
    official=pd.read_csv(RES/f"{name}_all_rounds.csv")
    ext_lookup={(int(r.seed),r.treatment,int(r["round"])):r for _,r in official.iterrows()}
    stream_population=None
    for seed in range(20):
        fixed=p1.fixed_split(df,target,seed); test=fixed["test"]; batches=fixed["batches"]; visible=fixed["visible"].copy(); pop_rate=float(y[fixed["stream"]].mean())
        for treatment in ["Oracle","NoExplore"]:
            visible=fixed["visible"].copy()
            for t in range(11):
                if t==0:
                    dynamic=eval_scope(raw,y,visible,test,pop_rate,visible,seed*10000,"external_fixed")
                else:
                    dynamic=eval_scope(raw,y,visible,test,pop_rate,visible,seed*10000+500+t,"external_fixed")
                dynamic.update({"dataset":name,"seed":seed,"round":t,"treatment":treatment})
                rows.append(dynamic)
                if t==0:
                    batch=np.array([],dtype=int); p_batch=np.array([])
                else:
                    batch=batches[t-1]
                    pre_b,model_b=p1.fit_predict(raw,y,visible_before,seed*10000+t)
                    p_batch=model_b.predict_proba(pre_b.transform(raw.iloc[batch]))[:,1]
                # internal evaluator uses current visible only, fixed 80/20 split
                if len(np.unique(y[visible]))>1 and len(visible)>=20:
                    tr,va=train_test_split(visible,test_size=.2,stratify=y[visible],random_state=seed*1000+t)
                else:
                    cut=max(1,int(.8*len(visible))); tr,va=visible[:cut],visible[cut:]
                internal=eval_scope(raw,y,tr,va,pop_rate,visible,seed*20000+9000+t,"internal_visible")
                internal.update({"dataset":name,"seed":seed,"round":t,"treatment":treatment})
                rows.append(internal)
                ref=ext_lookup[(seed,treatment,t)]
                for metric in ["accuracy","precision","recall","specificity","balanced_accuracy","gmean","roc_auc","pr_auc","brier","ece","calibration_slope","calibration_intercept"]:
                    parity.append(abs(float(dynamic[metric])-float(ref[metric])) if np.isfinite(dynamic[metric]) and np.isfinite(float(ref[metric])) else 0.0)
                if t<10:
                    visible_before=visible.copy()
                    pre_b,model_b=p1.fit_predict(raw,y,visible,seed*10000+t)
                    p_batch=model_b.predict_proba(pre_b.transform(raw.iloc[batches[t]]))[:,1]
                    batch=batches[t]; accepted=batch[p_batch<p1.TAU]; rejected=batch[p_batch>=p1.TAU]
                    gained=batch.copy() if treatment=="Oracle" else accepted.copy()
                    visible=np.concatenate([visible,gained]).astype(int)
        print(name,"dual seed",seed,"done",flush=True)

out=pd.DataFrame(rows); out.to_csv(RES/"dual_evaluation_all_rounds.csv",index=False)
# parity report
pd.DataFrame([{"max_abs_diff":max(parity),"n_compared":len(parity),"passed":max(parity)<=1e-8}]).to_csv(RES/"DUAL_EVAL_PARITY.csv",index=False)
# gaps and difference-in-distortion
metrics=["accuracy","recall","specificity","balanced_accuracy","roc_auc","pr_auc","brier","ece"]
ext=out[out.evaluation_scope=="external_fixed"].copy(); inte=out[out.evaluation_scope=="internal_visible"].copy()
gap=[]
for _,r in inte.iterrows():
    q=ext[(ext.dataset==r.dataset)&(ext.seed==r.seed)&(ext["round"]==r["round"])&(ext.treatment==r.treatment)].iloc[0]
    row={"dataset":r.dataset,"seed":r.seed,"round":r["round"],"treatment":r.treatment}
    for m in metrics: row[m+"_gap_internal_minus_external"]=float(r[m]-q[m])
    gap.append(row)
pd.DataFrame(gap).to_csv(RES/"dual_evaluation_gap.csv",index=False)
stats=[]
for name in p1.DATASETS:
    for m in ["recall","balanced_accuracy","roc_auc","pr_auc"]:
        vals=[]
        for seed in range(20):
            ni=inte[(inte.dataset==name)&(inte.seed==seed)&(inte.treatment=="NoExplore")&(inte["round"]==10)][m].iloc[0]
            oi=inte[(inte.dataset==name)&(inte.seed==seed)&(inte.treatment=="Oracle")&(inte["round"]==10)][m].iloc[0]
            ne=ext[(ext.dataset==name)&(ext.seed==seed)&(ext.treatment=="NoExplore")&(ext["round"]==10)][m].iloc[0]
            oe=ext[(ext.dataset==name)&(ext.seed==seed)&(ext.treatment=="Oracle")&(ext["round"]==10)][m].iloc[0]
            vals.append((ni-oi)-(ne-oe))
        vals=np.asarray(vals,float); from scipy.stats import wilcoxon
        p=float(wilcoxon(vals).pvalue) if np.any(vals!=0) else 1.0
        rng=np.random.RandomState(880000+len(stats)); boots=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(10000)])
        stats.append({"dataset":name,"round":10,"metric":m,"mean_distortion_gap":float(vals.mean()),"median_distortion_gap":float(np.median(vals)),"bootstrap_ci_low":float(np.quantile(boots,.025)),"bootstrap_ci_high":float(np.quantile(boots,.975)),"wilcoxon_p":p})
st=pd.DataFrame(stats)
from statsmodels.stats.multitest import multipletests
st["holm_p"]=st.groupby("dataset")["wilcoxon_p"].transform(lambda s:multipletests(s,method="holm")[1])
st.to_csv(RES/"dual_evaluation_statistics.csv",index=False)
print("dual complete",len(out),"max parity",max(parity))




