from pathlib import Path
import json, inspect, os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score, roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

ROOT=Path(os.environ.get("ECOMIC_ROOT", Path(__file__).resolve().parents[1]))
DATASETS={
    "uci": {"path": ROOT/"data/raw/uci_default_credit_card.csv", "target":"default"},
    "hmeq": {"path": ROOT/"data/raw/hmeq.csv", "target":"BAD"},
}
TAU=0.5
ROUNDS=10
TEST_FRACTION=0.25
INITIAL_VISIBLE_FRACTION=0.20
DEVICE="cuda:0"
TREATMENTS=["Oracle","NoExplore","RandomExplore-2%","RandomExplore-5%","RandomExplore-10%"]
RATES={"RandomExplore-2%":0.02,"RandomExplore-5%":0.05,"RandomExplore-10%":0.10}
RATE_CODES={"RandomExplore-2%":2,"RandomExplore-5%":5,"RandomExplore-10%":10}

def make_pre(raw):
    num=[c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
    cat=[c for c in raw.columns if c not in num]
    return ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="Missing")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)
    ])

def fit_predict(raw, y, visible, seed):
    pre=make_pre(raw)
    Xt=pre.fit_transform(raw.iloc[visible])
    model=XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,objective="binary:logistic",
        eval_metric="logloss",random_state=seed,n_jobs=1,tree_method="hist",device=DEVICE)
    model.fit(Xt,y[visible])
    return pre, model

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); value=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
        if mask.any(): value += mask.mean()*abs(float(y[mask].mean())-float(p[mask].mean()))
    return float(value)

def calibration(y,p):
    if len(np.unique(y))<2 or np.std(p)<1e-12: return np.nan,np.nan
    z=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))
    b0=0.0; b1=1.0
    try:
        for _ in range(60):
            eta=np.clip(b0+b1*z,-30,30); q=1.0/(1.0+np.exp(-eta)); w=np.clip(q*(1-q),1e-9,None)
            g0=float(np.sum(y-q)); g1=float(np.sum((y-q)*z))
            h00=float(np.sum(w))+1e-8; h01=float(np.sum(w*z)); h11=float(np.sum(w*z*z))+1e-8
            det=h00*h11-h01*h01
            if abs(det)<1e-14: return np.nan,np.nan
            d0=(h11*g0-h01*g1)/det; d1=(-h01*g0+h00*g1)/det
            b0+=d0; b1+=d1
            if max(abs(d0),abs(d1))<1e-8: break
        return float(b1),float(b0)
    except Exception:
        return np.nan,np.nan

def metrics(y_test,p_test,visible_y,pop_rate,batch_y=None,p_batch=None,gained=None,rejected=None,accepted=None,hidden_fraction=np.nan,observed_fraction=np.nan):
    pred=(p_test>=TAU).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
    spec=tn/(tn+fp) if tn+fp else np.nan
    rec=recall_score(y_test,pred,zero_division=0)
    slope,intercept=calibration(y_test,p_test)
    test_pred=(p_test>=TAU).astype(int)
    reliability=[]
    for b,(lo,hi) in enumerate(zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:])):
        mask=(p_test>=lo)&(p_test<(hi if hi<1 else hi+1e-12))
        reliability.append({"bin":b,"count":int(mask.sum()),"accuracy":float(y_test[mask].mean()) if mask.any() else np.nan,"confidence":float(p_test[mask].mean()) if mask.any() else np.nan})
    batch_y=np.asarray(batch_y if batch_y is not None else [],dtype=int)
    gained=np.asarray(gained if gained is not None else [],dtype=int)
    rejected=np.asarray(rejected if rejected is not None else [],dtype=int)
    accepted=np.asarray(accepted if accepted is not None else [],dtype=int)
    n_defaults=int(batch_y.sum()) if len(batch_y) else 0
    return {
        "accuracy":accuracy_score(y_test,pred),
        "precision":precision_score(y_test,pred,zero_division=0),
        "recall":rec,
        "specificity":spec,
        "balanced_accuracy":balanced_accuracy_score(y_test,pred),
        "gmean":float(np.sqrt(max(rec*spec,0))) if np.isfinite(spec) else np.nan,
        "roc_auc":roc_auc_score(y_test,p_test) if len(np.unique(y_test))>1 else np.nan,
        "pr_auc":average_precision_score(y_test,p_test) if len(np.unique(y_test))>1 else np.nan,
        "brier":brier_score_loss(y_test,p_test),
        "ece":ece(y_test,p_test),
        "calibration_slope":slope,
        "calibration_intercept":intercept,
        "mean_test_pd":float(np.mean(p_test)),
        "median_test_pd":float(np.median(p_test)),
        "n_true_default":int(np.sum(y_test)),
        "n_true_default_pred_positive":int(np.sum((y_test==1)&(test_pred==1))),
        "n_true_default_below_threshold":int(np.sum((y_test==1)&(test_pred==0))),
        "n_true_nondefault":int(np.sum(y_test==0)),
        "n_true_nondefault_above_threshold":int(np.sum((y_test==0)&(test_pred==1))),
        "mean_pd_default":float(np.mean(p_test[y_test==1])) if np.any(y_test==1) else np.nan,
        "mean_pd_nondefault":float(np.mean(p_test[y_test==0])) if np.any(y_test==0) else np.nan,
        "reliability_bins":json.dumps(reliability,separators=(",",":")),
        "std_test_pd":float(np.std(p_test)),
        "predicted_positive_rate":float(np.mean(pred)),
        "mean_pd_accepted":float(np.mean(p_batch[p_batch<TAU])) if len(p_batch) and np.any(p_batch<TAU) else np.nan,
        "mean_pd_rejected":float(np.mean(p_batch[p_batch>=TAU])) if len(p_batch) and np.any(p_batch>=TAU) else np.nan,
        "visible_tdr":float(np.mean(visible_y)),
        "population_default_rate":float(pop_rate),
        "fd_ratio":float(np.mean(visible_y)/pop_rate) if pop_rate>0 else np.nan,
        "hidden_default_fraction":hidden_fraction,
        "observed_default_fraction":observed_fraction,
        "visible_default_count":int(np.sum(visible_y)),
        "visible_nondefault_count":int(len(visible_y)-np.sum(visible_y)),
    }

def run_treatment(df, dataset, seed, treatment, fixed):
    target=DATASETS[dataset]["target"]
    y=df[target].astype(int).to_numpy()
    feature_cols=[c for c in df.columns if c not in {target,"ID"}]
    raw=df[feature_cols]
    test=fixed["test"]; batches=fixed["batches"]; visible=fixed["visible"].copy()
    stream=fixed["stream"]; pop_rate=float(y[stream].mean())
    rows=[]
    def emit(round_id, p_test=None, batch=None, p_batch=None, accepted=None, rejected=None, gained=None, n_explored=0, n_revealed_rejected=0):
        if p_test is None:
            pre,model=fit_predict(raw,y,visible,seed*10000+round_id)
            p_test=model.predict_proba(pre.transform(raw.iloc[test]))[:,1]
        batch=np.asarray(batch if batch is not None else [],dtype=int); p_batch=np.asarray(p_batch if p_batch is not None else [],dtype=float)
        accepted=np.asarray(accepted if accepted is not None else [],dtype=int); rejected=np.asarray(rejected if rejected is not None else [],dtype=int); gained=np.asarray(gained if gained is not None else [],dtype=int)
        batch_y=y[batch] if len(batch) else np.array([],dtype=int)
        n_defaults=int(batch_y.sum()) if len(batch_y) else 0
        reject_mask=(p_batch>=TAU) if len(p_batch) else np.array([],dtype=bool)
        hidden_fraction=float(batch_y[reject_mask].sum()/n_defaults) if n_defaults else np.nan
        observed_fraction=float(y[gained].sum()/n_defaults) if n_defaults else np.nan
        mm=metrics(y[test],p_test,y[visible],pop_rate,batch_y,p_batch,gained,rejected,accepted,hidden_fraction,observed_fraction)
        mm.update({"dataset":dataset,"seed":seed,"treatment":treatment,"exploration_rate":1.0 if treatment=="Oracle" else (0.0 if treatment=="NoExplore" else RATES[treatment]),"round":round_id,
            "n_visible":len(visible),"n_batch":len(batch),"n_accepted":len(accepted),"n_rejected":len(rejected),"n_explored":n_explored,"n_revealed_rejected":n_revealed_rejected,
            "acceptance_rate_current_batch":float(len(accepted)/len(batch)) if len(batch) else np.nan,"rejection_rate_current_batch":float(len(rejected)/len(batch)) if len(batch) else np.nan,
            "test_score_q05":float(np.quantile(p_test,.05)),"test_score_q25":float(np.quantile(p_test,.25)),"test_score_q50":float(np.quantile(p_test,.50)),"test_score_q75":float(np.quantile(p_test,.75)),"test_score_q95":float(np.quantile(p_test,.95)),
            "threshold_degeneracy":bool(len(accepted)==0 or len(rejected)==0) if len(batch) else False,
            "fixed_test_set_ok":bool(set(test.tolist()).isdisjoint(set(visible.tolist()))),
            "current_round_only_ok":bool(set(rejected.tolist()).issubset(set(batch.tolist())) and set(gained.tolist()).issubset(set(batch.tolist()))),
            "visible_update_ok":bool(set(gained.tolist()).issubset(set(batch.tolist()))),
            "no_duplicate_ok":bool(len(np.unique(visible))==len(visible)),
            "preprocessing_visible_only_ok":bool("raw.iloc[visible]" in inspect.getsource(fit_predict))})
        return mm
    rows.append(emit(0))
    for t,batch in enumerate(batches,1):
        pre,model=fit_predict(raw,y,visible,seed*10000+t)
        p_batch=model.predict_proba(pre.transform(raw.iloc[batch]))[:,1]
        accepted=batch[p_batch<TAU]; rejected=batch[p_batch>=TAU]
        if treatment=="Oracle":
            explored=np.array([],dtype=int); gained=batch.copy(); n_explored=0; n_revealed=len(rejected)
        elif treatment=="NoExplore":
            explored=np.array([],dtype=int); gained=accepted.copy(); n_explored=0; n_revealed=0
        else:
            rate=RATES[treatment]; k=max(1,int(np.rint(rate*len(rejected)))) if len(rejected)>0 else 0
            rr=np.random.RandomState(seed*100000+t*100+RATE_CODES[treatment])
            explored=rr.permutation(rejected)[:k] if k else np.array([],dtype=int)
            gained=np.concatenate([accepted,explored]).astype(int); n_explored=len(explored); n_revealed=n_explored
        expected=np.concatenate([visible,gained]).astype(int)
        visible=expected
        pre_after,model_after=fit_predict(raw,y,visible,seed*10000+500+t)
        p_test=model_after.predict_proba(pre_after.transform(raw.iloc[test]))[:,1]
        rows.append(emit(t,p_test,batch,p_batch,accepted,rejected,gained,n_explored,n_revealed))
    return rows

def fixed_split(df, target, seed):
    n=len(df); idx=np.arange(n); perm=np.random.RandomState(seed).permutation(idx); ntest=int(TEST_FRACTION*n)
    test=perm[:ntest]; stream=perm[ntest:]; n0=max(100,int(INITIAL_VISIBLE_FRACTION*len(stream)))
    visible=stream[:n0].copy(); remaining=stream[n0:]; batches=np.array_split(remaining,ROUNDS)
    return {"test":test,"stream":stream,"visible":visible,"batches":batches}

def load_dataset(name):
    cfg=DATASETS[name]; return pd.read_csv(cfg["path"])

def run_dataset(name,seeds=range(20)):
    df=load_dataset(name); rows=[]
    for seed in seeds:
        fixed=fixed_split(df,DATASETS[name]["target"],seed)
        for treatment in TREATMENTS:
            rows.extend(run_treatment(df,name,seed,treatment,fixed))
        print(name,"seed",seed,"done",flush=True)
    return pd.DataFrame(rows)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",choices=list(DATASETS),required=True); ap.add_argument("--seed-start",type=int,default=0); ap.add_argument("--seed-end",type=int,default=20); ap.add_argument("--out",required=True)
    a=ap.parse_args(); d=run_dataset(a.dataset,range(a.seed_start,a.seed_end)); Path(a.out).parent.mkdir(parents=True,exist_ok=True); d.to_csv(a.out,index=False); print(a.out,len(d))


