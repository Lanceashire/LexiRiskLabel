"""M5 mechanism validation helpers. No new acquisition policy."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,sys,time,os
import numpy as np
import pandas as pd
from scipy.special import expit
from deps import core,exact
HERE=Path(__file__).resolve().parent
METHODS=['OrdinaryOnly','MinRisk','D-only','Pending-aware','ParBaLS-Laplace-cost-pending']
FEATURES=['O_target_coverage','PA_proxy_advantage','count_sacrifice_fraction','composition_premium_fraction','marginal_cv','information_log_condition']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=default)+'\n');t.replace(p)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=default).encode()).hexdigest()
def four(s,ids):
    ids=np.asarray(ids);order=np.lexsort((ids,s.cost));k=core.kstar(s.cost,s.budget)
    a={'OrdinaryOnly':np.empty(0,int),'MinRisk':np.sort(order[:k])}
    for cond,name in [(False,'D-only'),(True,'Pending-aware')]:
        h,g,u=core.geometry(s,cond);a[name]=np.sort(exact.greedy(h,g,u,s.cost,s.budget,ids))
    return a

def indicators(s,sets=None):
    sets=four(s,np.arange(len(s.cost))) if sets is None else sets
    hd,g,u=core.geometry(s,False);ho,_,_=core.geometry(s,True);inv=core.inverse(hd)
    scale=max(float(np.trace(g@inv)),1e-15);coverage=float(np.trace(g@(inv-core.inverse(ho))))/scale
    pa,mr=sets['Pending-aware'],sets['MinRisk'];k=len(mr);m=len(pa)
    advantage=(core.set_value(ho,g,u,pa)-core.set_value(ho,g,u,mr))/scale
    marginal=core.marginal(core.inverse(ho),g,u)
    return np.array([coverage,advantage,(k-m)/max(k,1),
        (float(s.cost[pa].sum())-float(np.sort(s.cost)[:m].sum()))/max(s.budget,1e-15),
        float(np.std(marginal))/max(float(np.mean(marginal)),1e-15) if len(marginal) else 0.,
        float(np.log1p(np.linalg.cond(ho)))],float),scale

def treatments(s):
    # D and Q only. No O outcome is accessible to this function.
    hd,g,_=core.geometry(s,False);inv=core.inverse(hd);v=core.info(s.ordinary_x,s.theta)
    score=core.marginal(inv,g,v);order=np.lexsort((np.arange(len(score)),score));n=len(score)//2
    random=np.sort(np.random.default_rng(s.seed+712).choice(len(score),n,replace=False))
    return {'full':np.arange(len(score)), 'empty':np.empty(0,int),
            'high':np.sort(order[-n:]) if n else np.empty(0,int),'low':np.sort(order[:n]),'random':random}

class FeedbackVault:
    """Selectors have no reference to this object. Release is bound to a disk lock."""
    def __init__(self,y,test):self.__y=np.asarray(y);self.test=np.asarray(test);self.events=[]
    def initial(self,ids):
        assert not np.intersect1d(ids,self.test).size
        self.events.append({'kind':'initial','ids_hash':digest(ids)})
        return self.__y[ids].copy()
    def release(self,lockpath,expected):
        data=json.loads(Path(lockpath).read_text());assert digest(data)==expected,'lock changed after commitment'
        ids=np.r_[data['ordinary_ids'],data['selected_ids']].astype(int)
        assert len(np.unique(ids))==len(ids) and not np.intersect1d(ids,self.test).size
        self.events.append({'kind':'release','lock_sha256':sha(lockpath),'ids_hash':digest(ids)})
        return ids,self.__y[ids].copy()
