"""GPU development replay, explicit configuration and immutable run identities.

The correction factorial uses XGBRegressor throughout. It is analyzed separately
from the classifier acquisition comparison. Dynamic GATE is a separate protocol.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time, warnings
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, recall_score
from xgboost import XGBClassifier,XGBRegressor
from core import *

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CFG=json.loads((HERE/'config.json').read_text())
CLASS_METHODS=['OrdinaryOnly','MinRisk','LRBE-IG','ScoreFirst-IG','Vopt','BAIT-cost','ParBaLS-cost','RB-BRIDGE']
CORR_METHODS=['OrdinaryOnly','CORR-base-plain','CORR-base-corrected','CORR-opt-plain','RB-CORR','CORR-vopt-corrected']
GATE_METHODS=['OrdinaryOnly','MinRisk','RB-BRIDGE','RB-GATE']


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a)));tmp.replace(path)


def load(name):
    path=ROOT/'data/research_v2'/f'{name}.csv';df=pd.read_csv(path,low_memory=False)
    target={'hmeq':'BAD','uci':'default','lendingclub':'target'}[name]
    exposure=df['LOAN' if name=='hmeq' else 'exposure'].to_numpy(float) if name!='uci' else np.ones(len(df))
    assert np.isfinite(exposure).all() and (exposure>0).all()
    if name=='lendingclub':
        # Versioned compact decision-time schema. No last_fico/post-outcome fields.
        cols=['loan_amnt','term','int_rate','grade','emp_length','home_ownership','annual_inc','purpose','dti',
              'delinq_2yrs','fico_range_low','fico_range_high','inq_last_6mths','mths_since_last_delinq',
              'mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','initial_list_status']
    else:cols=[c for c in df if c not in {target,'ID','LOAN','exposure'}]
    raw=df[cols].copy()
    for c in ['int_rate','revol_util']:
        if c in raw and not pd.api.types.is_numeric_dtype(raw[c]):raw[c]=pd.to_numeric(raw[c].astype(str).str.rstrip('%'),errors='coerce')
    return raw,df[target].to_numpy(int),exposure,hashlib.sha256(path.read_bytes()).hexdigest(),cols


def preprocessor(raw):
    num=[c for c in raw if pd.api.types.is_numeric_dtype(raw[c])];cat=[c for c in raw if c not in num]
    return ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),num),
                              ('category',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])


def split(raw,seed,name):
    rng=np.random.default_rng(seed);perm=rng.permutation(len(raw));nd=int(CFG['development_fraction']*len(raw));nt=int(CFG['test_fraction']*len(raw))
    development=perm[:nd];test=perm[nd:nd+nt];replay=perm[nd+nt:];pool=replay[:int(len(replay)*CFG['initial_pool_fraction'])]
    # Initial label selection uses covariates only, with fixed sample size.
    if name=='hmeq':score=raw.DELINQ.fillna(0).to_numpy()+.04*raw.DEBTINC.fillna(30).to_numpy()
    elif name=='uci':score=raw.PAY_0.to_numpy()+.2*raw.PAY_2.to_numpy()
    else:score=.04*raw.dti.fillna(20).to_numpy()-.01*raw.fico_range_low.fillna(680).to_numpy()
    visible=pool[np.argsort(score[pool],kind='stable')[:int(len(pool)*CFG['initial_observed_fraction'])]]
    batches=np.array_split(replay[len(pool):],CFG['rounds'])
    return visible,test,development,batches


def fit_model(x,y,seed,kind='classifier'):
    started=time.perf_counter();params={**CFG['model'],'random_state':seed,'tree_method':'hist'}
    if kind=='classifier':
        if len(np.unique(y))<2:raise RuntimeError('visible training data contains only one class')
        model=XGBClassifier(**params,objective='binary:logistic',eval_metric='logloss')
    else:model=XGBRegressor(**params,objective='reg:squarederror')
    model.fit(x.astype(np.float32),y)
    conf=json.loads(model.get_booster().save_config());device=conf['learner']['generic_param']['device']
    if not device.startswith('cuda'):raise RuntimeError('GPU requested but actual device='+device)
    backend={'actual_training_device':device,'updater':conf['learner']['gradient_booster']['gbtree_train_param']['updater'],
             'fit_seconds':time.perf_counter()-started}
    return model,backend


def predict(model,x,kind):
    if kind=='classifier':return model.predict_proba(x.astype(np.float32))[:,1]
    return np.clip(model.predict(x.astype(np.float32)),0,1)


def bootstrap_ig(x,y,candidate,seed):
    rng=np.random.default_rng(seed);pp=[]
    for member in range(4):
        ids=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ids]))<2:pp.append(np.full(len(candidate),y[ids].mean()));continue
        model,_=fit_model(x[ids],y[ids],seed+member)
        pp.append(predict(model,candidate,'classifier'))
    pp=np.asarray(pp);return np.maximum(entropy(pp.mean(axis=0))-entropy(pp).mean(axis=0),0)


def metric(y,p,mask):
    return {'brier':float(np.mean((y-p)**2)),'target_brier':float(np.mean((y[mask]-p[mask])**2)) if mask.any() else None,
            'rest_brier':float(np.mean((y[~mask]-p[~mask])**2)) if (~mask).any() else None,
            'target_share':float(mask.mean()),'recall':float(recall_score(y,p>=CFG['threshold'],zero_division=0)),
            'pr_auc':float(average_precision_score(y,p))}


def approval(batch,pred,price,exposure,base_ids,threshold):
    # Shared capacity and ordinary predicted risk cap prevent risk-funded expansion.
    capacity=len(base_ids);cap=float((price[base_ids]*exposure[base_ids]).sum());chosen=[];used=0.
    ids=batch[pred[batch]<threshold];ids=ids[np.argsort(pred[ids],kind='stable')]
    for i in ids:
        cost=price[i]*exposure[i]
        if len(chosen)>=capacity:break
        if used+cost<=cap+1e-8:chosen.append(int(i));used+=cost
    return np.asarray(chosen,int)


def episode(name,seed,method,protocol,out):
    raw,y,exposure,data_hash,cols=load(name);initial,test,development,batches=split(raw,seed,name)
    pre=preprocessor(raw);pre.fit(raw.iloc[initial]);z=np.asarray(pre.transform(raw),float)
    # Representation fixed from initial visible features for every policy.
    x=np.c_[np.ones(len(z)),z];reference=development[:CFG['reference_count']]
    baseline,base_backend=fit_model(z[initial],y[initial],seed)
    price=predict(baseline,z,'classifier');target=price[test]>=CFG['threshold']
    visible=initial.copy();pseudo_x=z[initial].copy();pseudo_y=y[initial].astype(float).copy()
    kind='regressor' if protocol=='correction' else 'classifier'
    model,backend=fit_model(z[visible],y[visible],seed,kind);prediction=predict(model,z,kind)
    episode_id=f'{name}_{seed}_{protocol}_{method}'
    identity={'dataset_sha256':data_hash,'schema':cols,'seed':seed,'method':method,'protocol':protocol,
              'config':CFG,'split_hashes':{'initial':digest(initial),'test':digest(test),'development':digest(development),'stream':digest(np.concatenate(batches))},
              'code_hash':digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'core.py',HERE/'run_real.py']})}
    run=out/episode_id
    if run.exists():raise FileExistsError('run already exists; refusing to mix identities: '+str(run))
    run.mkdir(parents=True);atomic_json(run/'identity.json',identity)
    rows=[];predictions=[];t0=time.perf_counter();cum_risk=cum_loss=cum_ordinary_risk=cum_ordinary_loss=0.;cum_count=0
    for round_id,batch in enumerate(batches,1):
        base_o=batch[price[batch]<CFG['threshold']];base_c=batch[price[batch]>=CFG['threshold']]
        ordinary_ids=approval(batch,prediction,price,exposure,base_o,CFG['threshold']) if protocol=='dynamic' else base_o
        candidate=np.setdiff1d(batch,ordinary_ids,assume_unique=True) if protocol=='dynamic' else base_c
        costs=price[candidate]*exposure[candidate];budget=CFG['rho']*float((price[base_c]*exposure[base_c]).sum())
        theta=fit_logistic(x[visible],y[visible],CFG['ridge'])
        state=State(x[visible],y[visible],x[ordinary_ids],x[candidate],x[reference],costs,budget,theta,
                    seed*100+round_id,CFG['ridge'],CFG['partners'],CFG['exchanges'],CFG['mixture_steps'],CFG['coverage_epsilon'])
        utility=bootstrap_ig(z[visible],y[visible],z[candidate],seed*100+round_id) if 'IG' in method and len(candidate) else None
        selection=select(method,state,utility);selected=candidate[selection.selected_ids]
        # Lock acquisition BEFORE exposing this round's labels.
        log={'round':round_id,'visible_ids':visible,'ordinary_ids':ordinary_ids,'candidate_ids':candidate,'selected_ids':selected,
             'candidate_costs':costs,'budget':budget,'reference_ids':reference,'diagnostics':selection.diagnostics,
             'inclusion_probabilities':selection.inclusion_probabilities,'sampling_design':selection.sampling_design}
        atomic_json(run/f'round_{round_id:02d}_selection.json',log)
        before=visible.copy();visible=np.r_[visible,ordinary_ids,selected]
        assert len(np.unique(visible))==len(visible) and not np.intersect1d(visible,test).size
        if protocol=='correction' and method in ['CORR-base-corrected','RB-CORR','CORR-vopt-corrected']:
            pi=selection.inclusion_probabilities
            eligible=np.flatnonzero(pi>0) if pi is not None else np.empty(0,int)
            pseudo=expit(x[candidate]@theta);local=selection.selected_ids
            pseudo[local]+=(y[selected]-pseudo[local])/pi[local]
            pseudo_x=np.vstack([pseudo_x,z[ordinary_ids],z[candidate[eligible]]]);pseudo_y=np.r_[pseudo_y,y[ordinary_ids],pseudo[eligible]]
            model,backend=fit_model(pseudo_x,pseudo_y,seed*1000+round_id,kind)
            atomic_json(run/f'round_{round_id:02d}_correction.json',{'reachable_ids':candidate[eligible],
                         'frozen_prediction':expit(x[candidate[eligible]]@theta),'pseudo_targets':pseudo[eligible],
                         'historical_pi':pi[eligible]})
        else:model,backend=fit_model(z[visible],y[visible],seed*1000+round_id,kind)
        prediction=predict(model,z,kind);p=prediction[test]
        loss=float(exposure[selected]@y[selected]);ordinary_loss=float(exposure[ordinary_ids]@y[ordinary_ids]);ordinary_risk=float(exposure[ordinary_ids]@price[ordinary_ids])
        cum_risk+=selection.predicted_cost;cum_loss+=loss;cum_count+=len(selected);cum_ordinary_loss+=ordinary_loss;cum_ordinary_risk+=ordinary_risk
        diag={k:v for k,v in selection.diagnostics.items() if isinstance(v,(str,int,float,bool))}
        row={'dataset':name,'seed':seed,'method':method,'protocol':protocol,'learner':'XGBRegressor' if kind=='regressor' else 'XGBClassifier',
             'round':round_id,'rho':CFG['rho'],'budget':budget,'selected_count':len(selected),'ordinary_count':len(ordinary_ids),
             'candidate_count':len(candidate),'visible_count':len(visible),'predicted_exploration_risk':selection.predicted_cost,
             'realized_exploration_loss':loss,'cumulative_predicted_risk':cum_risk,'cumulative_realized_loss':cum_loss,
             'cumulative_count':cum_count,'cumulative_ordinary_predicted_risk':cum_ordinary_risk,
             'cumulative_ordinary_realized_loss':cum_ordinary_loss,'total_predicted_risk':cum_ordinary_risk+cum_risk,
             'total_realized_loss':cum_ordinary_loss+cum_loss,'budget_feasible':selection.predicted_cost<=budget+1e-7,
             'runtime_seconds':time.perf_counter()-t0,**metric(y[test],p,target),**backend,**diag}
        rows.append(row);predictions.append(p)
        pd.DataFrame(rows).to_csv(run/'trajectory.csv',index=False)
    np.savez_compressed(run/'test_predictions.npz',ids=test,y=y[test],target=target,prediction=np.asarray(predictions))
    atomic_json(run/'completion.json',{'complete':True,'rounds':len(rows),'seconds':time.perf_counter()-t0})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--protocol',choices=['fixed','correction','dynamic'],required=True)
    ap.add_argument('--datasets',nargs='+',default=['hmeq','uci','lendingclub']);ap.add_argument('--seeds',nargs='+',type=int);ap.add_argument('--methods',nargs='+');a=ap.parse_args()
    methods=a.methods or (CLASS_METHODS if a.protocol=='fixed' else CORR_METHODS if a.protocol=='correction' else GATE_METHODS)
    seeds=a.seeds or CFG['real_seeds'];a.output.mkdir(parents=True,exist_ok=True)
    for name in a.datasets:
        for seed in seeds:
            for method in methods:
                dest=a.output/f'{name}_{seed}_{a.protocol}_{method}'
                if (dest/'completion.json').exists():
                    # Identity check before skipping, including code/config.
                    old=json.loads((dest/'identity.json').read_text())
                    code=digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'core.py',HERE/'run_real.py']})
                    if old['config']!=CFG or old['code_hash']!=code or old['dataset_sha256']!=load(name)[3]:raise RuntimeError('resume identity mismatch')
                    print('VERIFIED-SKIP',name,seed,method,flush=True);continue
                rr=episode(name,seed,method,a.protocol,a.output)
                print('DONE',a.protocol,name,seed,method,'ALC',np.mean([r['brier'] for r in rr]),'seconds',rr[-1]['runtime_seconds'],flush=True)


if __name__=='__main__':main()
