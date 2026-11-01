"""Risk-budgeted acquisition. Selectors receive visible labels only.

BRIDGE is a prototype of the conversation's structured pair heuristic.
CORR uses a finite mixture of feasible sets, hence exact inclusion probabilities.
GATE uses coherent Laplace worlds, never actual hidden/future labels.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json, time
import numpy as np
from scipy.special import expit
from scipy.optimize import milp, Bounds, LinearConstraint, minimize_scalar
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors


def digest(x):
    if isinstance(x, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def fit_logistic(x, y, ridge=1.):
    # x includes intercept; same regularization in information approximation.
    if len(np.unique(y)) < 2:
        theta=np.zeros(x.shape[1]); theta[0]=np.log((np.sum(y)+.5)/(len(y)-np.sum(y)+.5))
        return theta
    m=LogisticRegression(C=1/ridge, fit_intercept=False, solver='newton-cholesky', max_iter=100, tol=1e-7)
    m.fit(x,y)
    return m.coef_[0]


@dataclass
class State:
    visible_x: np.ndarray
    visible_y: np.ndarray
    ordinary_x: np.ndarray
    candidate_x: np.ndarray
    reference_x: np.ndarray
    cost: np.ndarray
    budget: float
    theta: np.ndarray
    seed: int
    ridge: float=1.
    partners: int=8
    exchanges: int=1
    mixture_steps: int=12
    coverage_epsilon: float=.15
    future_approval_frozen: bool=False


@dataclass
class Selection:
    selected_ids: np.ndarray
    predicted_cost: float
    diagnostics: dict=field(default_factory=dict)
    inclusion_probabilities: np.ndarray | None=None
    sampling_design: dict | None=None


def info(x,theta):
    p=expit(x@theta); w=np.maximum(p*(1-p),1e-8)
    return x*np.sqrt(w)[:,None]


def geometry(s, conditional=True, fisher=False):
    d=s.candidate_x.shape[1]
    vd=info(s.visible_x,s.theta)
    h=s.ridge*np.eye(d)+vd.T@vd
    if conditional:
        vo=info(s.ordinary_x,s.theta); h+=vo.T@vo
    p=expit(s.reference_x@s.theta); w=p*(1-p)
    # Brier sensitivity squared, as specified in the supplied document.
    g=s.reference_x.T@((w if fisher else w*w)[:,None]*s.reference_x)/max(len(p),1)
    u=info(s.candidate_x,s.theta)
    return h,g,u


def inverse(h):
    l=np.linalg.cholesky((h+h.T)/2)
    return np.linalg.solve(l.T,np.linalg.solve(l,np.eye(len(h))))


def marginal(c,g,u):
    a=u@c
    return np.maximum(np.einsum('ij,ij->i',a@g,a)/(1+np.einsum('ij,ij->i',a,u)),0)


def update(c,u):
    a=c@u
    return c-np.outer(a,a)/(1+u@a)


def group_gain(c,g,u):
    a=c@u.T
    return float(np.trace(np.linalg.solve(np.eye(len(u))+u@a,a.T@g@a)))


def set_value(h,g,u,selected):
    return float(np.trace(g@(inverse(h)-inverse(h+u[selected].T@u[selected]))))


def kstar(cost,budget):
    return int(np.searchsorted(np.cumsum(np.sort(cost)),budget+1e-10,side='right'))


def knapsack(cost,score,budget,minimum=0):
    if not len(cost) or budget<=0: return np.empty(0,int)
    score=np.asarray(score,float); scale=max(np.max(np.abs(score)),1e-20)
    constraints=[LinearConstraint(cost[None,:],-np.inf,budget)]
    if minimum: constraints.append(LinearConstraint(np.ones((1,len(cost))),minimum,np.inf))
    r=milp(-score/scale,integrality=np.ones(len(cost)),bounds=Bounds(0,1),constraints=constraints,
           options={'time_limit':15.,'mip_rel_gap':1e-8})
    if not r.success: raise RuntimeError('knapsack: '+r.message)
    return np.flatnonzero(r.x>.5)


def pair_candidates(h,g,u,partners,seed,mode='structured'):
    n=len(u)
    if n<2: return np.empty((0,2),int)
    k=min(partners,n-1); rng=np.random.default_rng(seed)
    if mode=='structured':
        ev,rot=np.linalg.eigh(inverse(h)); root=(rot*np.sqrt(np.maximum(ev,0)))@rot.T
        vals,vecs=np.linalg.eigh(root@g@root)
        order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]
        rank=min(max(1,int(np.searchsorted(np.cumsum(vals),.8*vals.sum())+1)),max(1,len(vals)-1))
        z=u@root@vecs
        # Information vectors are sign invariant: orient the target component.
        sign=np.where(z[:,0]<0,-1.,1.); z=z*sign[:,None]
        zn=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        query=zn.copy(); query[:,rank:]*=-1
        neighbors=NearestNeighbors(n_neighbors=k+1,algorithm='brute',n_jobs=1).fit(zn).kneighbors(query,return_distance=False)
        pairs={(min(i,int(j)),max(i,int(j))) for i in range(n) for j in neighbors[i] if i!=j}
        # Exactly cap the same number per anchor; no singleton-score screening.
        pairs=set()
        for i in range(n):
            for j in [int(j) for j in neighbors[i] if i!=j][:k]: pairs.add((min(i,j),max(i,j)))
    elif mode=='generic':
        count=len(pair_candidates(h,g,u,partners,seed,'structured'))
        pairs=set()
        while len(pairs)<count:
            i,j=rng.integers(0,n,2)
            if i!=j:pairs.add((min(int(i),int(j)),max(int(i),int(j))))
    else:
        pairs=set()
        for i in range(n):
            ids=rng.choice(n-1,k,replace=False); ids=ids+(ids>=i)
            pairs.update((min(i,int(j)),max(i,int(j))) for j in ids)
    return np.asarray(sorted(pairs),int).reshape(-1,2)


def pair_gains(c,g,u,pairs):
    if not len(pairs): return np.empty(0)
    a=u@c; b=a@g
    i,j=pairs.T
    aa=1+np.einsum('ij,ij->i',a[i],u[i]); dd=1+np.einsum('ij,ij->i',a[j],u[j])
    ab=np.einsum('ij,ij->i',a[i],u[j])
    ni=np.einsum('ij,ij->i',b[i],a[i]); nj=np.einsum('ij,ij->i',b[j],a[j])
    nij=np.einsum('ij,ij->i',b[i],a[j])
    return np.maximum((dd*ni+aa*nj-2*ab*nij)/(aa*dd-ab*ab),0)


def value_select(s, *, conditional=True, batch=True, pairs=False, pair_mode='structured', fixed_count=False):
    h,g,u=geometry(s,conditional); c=inverse(h); n=len(u)
    chosen=[]; remain=s.budget; steps=[]
    candidates=pair_candidates(h,g,u,s.partners,s.seed,pair_mode) if pairs else np.empty((0,2),int)
    initial=marginal(c,g,u); free=np.ones(n,bool)
    k=kstar(s.cost,s.budget)
    if not batch and not pairs:
        selected=knapsack(s.cost,initial,s.budget,k if fixed_count else 0)
        return finish(s,selected,{'proxy_value':set_value(h,g,u,selected),'static':True})
    while free.any():
        gain=marginal(c,g,u); valid=free&(s.cost<=remain+1e-10)
        if fixed_count:
            slots=k-len(chosen)
            if slots<=0: break
            for i in np.flatnonzero(valid):
                rest=s.cost[free & (np.arange(n)!=i)]
                if slots>1 and (len(rest)<slots-1 or np.partition(rest,slots-2)[:slots-1].sum()>remain-s.cost[i]+1e-10): valid[i]=False
        score=np.where(valid,gain/np.maximum(s.cost,1e-15),-np.inf)
        i=int(np.argmax(score)) if n else 0
        best=(float(score[i]),[i],float(gain[i])) if n and valid[i] else (-np.inf,[],0.)
        if len(candidates):
            feasible=free[candidates].all(axis=1)&(s.cost[candidates].sum(axis=1)<=remain+1e-10)
            pp=candidates[feasible]; pg=pair_gains(c,g,u,pp)
            if len(pp):
                ratios=pg/np.maximum(s.cost[pp].sum(axis=1),1e-15); jj=int(np.argmax(ratios))
                if ratios[jj]>best[0]: best=(float(ratios[jj]),pp[jj].tolist(),float(pg[jj]))
        if not best[1] or best[2]<=1e-18: break
        ids=best[1]; gamma=best[2]-float(gain[ids].sum()) if len(ids)==2 else 0.
        steps.append({'ids':ids,'gain':best[2],'gamma':gamma})
        for i in ids:
            chosen.append(i); free[i]=False; remain-=s.cost[i]; c=update(c,u[i])
    # Compare with best singleton (standard knapsack safeguard).
    feasible=np.flatnonzero(s.cost<=s.budget+1e-10)
    if len(feasible) and not fixed_count:
        one=int(feasible[np.argmax(initial[feasible])])
        if initial[one]>set_value(h,g,u,np.asarray(chosen,int))+1e-14: chosen=[one]
    # One accepted 1->1 or 1->2 exchange per configured pass; reconstruct after removal.
    exchange_count=0
    for _ in range(s.exchanges if pairs else 0):
        current=np.asarray(chosen,int); old=set_value(h,g,u,current); bestset=current; bestvalue=old
        # Bound removal work, using smallest exact removal loss, not addition-only update.
        losses=[]
        for j in current:
            rest=current[current!=j]; losses.append((old-set_value(h,g,u,rest),int(j)))
        for _,j in sorted(losses)[:8]:
            rest=current[current!=j]; cc=inverse(h+u[rest].T@u[rest]); available=np.ones(n,bool);available[rest]=False
            budget=s.budget-s.cost[rest].sum(); singles=np.flatnonzero(available&(s.cost<=budget+1e-10))
            base=set_value(h,g,u,rest)
            if len(singles):
                gs=marginal(cc,g,u)[singles]; idx=int(np.argmax(gs))
                if base+gs[idx]>bestvalue+1e-14:bestvalue=base+gs[idx];bestset=np.r_[rest,singles[idx]]
            pp=candidates[available[candidates].all(axis=1)&(s.cost[candidates].sum(axis=1)<=budget+1e-10)]
            if len(pp):
                gg=pair_gains(cc,g,u,pp); idx=int(np.argmax(gg))
                if base+gg[idx]>bestvalue+1e-14:bestvalue=base+gg[idx];bestset=np.r_[rest,pp[idx]]
        if bestvalue<=old+1e-14:break
        chosen=bestset.tolist();exchange_count+=1
    ids=np.asarray(chosen,int)
    return finish(s,ids,{'proxy_value':set_value(h,g,u,ids),'steps':steps,
                       'pair_candidates':len(candidates),'accepted_exchanges':exchange_count,
                       'positive_pair_steps':sum(t['gamma']>1e-12 for t in steps)})


def finish(s,ids,diagnostics=None,pi=None,design=None):
    ids=np.asarray(ids,int); spent=float(s.cost[ids].sum())
    assert len(ids)==len(np.unique(ids)) and spent<=s.budget+1e-7
    order=np.sort(s.cost);k=kstar(s.cost,s.budget);n=len(ids)
    mk=float(order[:k].sum());mn=float(order[:n].sum());slack=s.budget-mk
    maxswap=0
    for m in range(1,min(k,len(order)-k)+1):
        if order[k:k+m].sum()-order[k-m:k].sum()<=slack+1e-10:maxswap=m
    diag={'k_star':k,'count_sacrifice':k-n,'composition_premium':spent-mn,
          'maximum_count_slack':slack,'max_swap_count':maxswap,**(diagnostics or {})}
    return Selection(ids,spent,diag,pi,design)


def random_fill(cost,budget,rng):
    selected=[]
    for i in rng.permutation(len(cost)):
        if cost[i]<=budget+1e-10:selected.append(int(i));budget-=cost[i]
    return np.asarray(selected,int)


def corr_select(s,mode='optimized'):
    n=len(s.cost); eligible=np.flatnonzero(s.cost<=s.budget+1e-10)
    if not len(eligible):return finish(s,[],{},np.zeros(n),{'sets':[],'weights':[]})
    # Disjoint random-order packing covers EVERY feasible candidate once.
    rng=np.random.default_rng(s.seed); groups=[]; group=[];used=0.
    for i in rng.permutation(eligible):
        if group and used+s.cost[i]>s.budget+1e-10:groups.append(group);group=[];used=0.
        group.append(int(i));used+=s.cost[i]
    if group:groups.append(group)
    covers=np.zeros((len(groups),n))
    for j,gp in enumerate(groups):covers[j,gp]=1
    basepi=covers.mean(axis=0); eps=s.coverage_epsilon
    # Square-loss linear-head curvature, matching the correction training objective.
    # XGBRegressor transfer remains an explicitly approximate linear influence model.
    h=s.ridge*np.eye(s.visible_x.shape[1])+s.visible_x.T@s.visible_x+s.ordinary_x.T@s.ordinary_x
    g=s.reference_x.T@s.reference_x/max(len(s.reference_x),1);c=inverse(h)
    # Local residual variance times target-relevant influence. No hidden outcomes.
    a=s.candidate_x@c
    probability=expit(s.candidate_x@s.theta)
    importance=np.maximum(probability*(1-probability)*np.einsum('ij,ij->i',a@g,a),1e-15)
    sets=[list(v) for v in groups]; weights=np.full(len(groups),1/len(groups))
    if mode!='base':
        if mode=='vopt':
            sel=value_select(s).selected_ids
            sets.append(sel.tolist());weights=np.r_[eps*weights,1-eps]
        else:
            # Optimize q~=FW mixture; final q=eps*q0+(1-eps)*q~.
            tilde=basepi.copy(); tw=weights.copy()
            for _ in range(s.mixture_steps):
                pi=eps*basepi+(1-eps)*tilde
                score=np.zeros(n); score[eligible]=importance[eligible]/np.maximum(pi[eligible],1e-15)**2
                sel=knapsack(s.cost,score,s.budget); atom=np.zeros(n);atom[sel]=1
                def objective(alpha):
                    pp=eps*basepi+(1-eps)*((1-alpha)*tilde+alpha*atom)
                    return float(np.sum(importance[eligible]/pp[eligible]))
                opt=minimize_scalar(objective,bounds=(0,1),method='bounded',options={'xatol':1e-7})
                alpha=float(opt.x);tw*=1-alpha;tw=np.r_[tw,alpha];sets.append(sel.tolist());tilde=(1-alpha)*tilde+alpha*atom
            weights=(1-eps)*tw;weights[:len(groups)]+=eps/len(groups)
    weights=weights/weights.sum();pi=np.zeros(n)
    for gp,w in zip(sets,weights):pi[gp]+=w
    assert np.all(pi[eligible]>0)
    for gp in sets:assert s.cost[gp].sum()<=s.budget+1e-8
    chosen=int(rng.choice(len(sets),p=weights));ids=np.asarray(sets[chosen],int)
    w=1/pi[ids] if len(ids) else np.empty(0)
    ess=float(w.sum()**2/(w@w)) if len(w) else 0.
    objective=float(np.sum(importance[eligible]*(1/pi[eligible]-1)))
    return finish(s,ids,{'min_probability':float(pi[eligible].min()),'max_weight':float(w.max()) if len(w) else 0.,
                        'effective_sample_size':ess,'variance_proxy':objective,'unreachable_count':n-len(eligible)},
                  pi,{'sets':sets,'weights':weights.tolist(),'drawn_component':chosen,'coverage_epsilon':eps})


def bait_select(s):
    # Binary fixed-feature BAIT objective, cost-aware forward/backward adaptation.
    h,g,u=geometry(s,True,True);c=inverse(h);chosen=[];free=np.ones(len(u),bool)
    while free.any():
        valid=free&(s.cost<=2*s.budget-s.cost[chosen].sum()+1e-10)
        if not valid.any():break
        gain=marginal(c,g,u);score=np.where(valid,gain/np.maximum(s.cost,1e-15),-np.inf)
        i=int(np.argmax(score));chosen.append(i);free[i]=False;c=update(c,u[i])
    while s.cost[chosen].sum()>s.budget+1e-10:
        losses=[];old=set_value(h,g,u,np.asarray(chosen,int))
        for j in chosen:
            rem=np.asarray([i for i in chosen if i!=j],int)
            losses.append(((old-set_value(h,g,u,rem))/max(s.cost[j],1e-15),j))
        chosen.remove(min(losses)[1])
    return finish(s,chosen,{'proxy_value':set_value(h,g,u,np.asarray(chosen,int)),
                            'adaptation':'binary Fisher forward to 2B, backward loss/cost to B'})


def posterior_draws(s,members=32):
    h,_,_=geometry(s,False);cov=inverse(h)
    rng=np.random.default_rng(s.seed+81)
    return rng.multivariate_normal(s.theta,cov,size=members)


def entropy(p):
    p=np.clip(p,1e-12,1-1e-12)
    return -p*np.log(p)-(1-p)*np.log1p(-p)


def ig_scores(s):
    pp=expit(posterior_draws(s)@s.candidate_x.T)
    return np.maximum(entropy(pp.mean(axis=0))-entropy(pp).mean(axis=0),0)


def epig_scores(pc,pq):
    """EER/EPIG binary mutual information, matching official ParBaLS eer.py."""
    a=pc.mean(axis=0)[:,None];b=pq.mean(axis=0)[None,:];p11=pc.T@pq/len(pc)
    joint=np.stack([1-a-b+p11,b-p11,a-p11,p11],axis=-1)
    hj=-np.sum(np.maximum(joint,1e-15)*np.log(np.maximum(joint,1e-15)),axis=-1)
    return np.maximum((entropy(a)+entropy(b)-hj).mean(axis=1),0)


def parbals_select(s):
    # Official multi-pseudo ParBaLS with shared LR features; adapt count to risk.
    # Pending O is included by fantasizing its labels in every branch.
    rng=np.random.default_rng(s.seed+470);draws=posterior_draws(s,32)
    world=draws[rng.choice(len(draws),4,replace=False)]
    fantasy_o=rng.binomial(1,expit(world@s.ordinary_x.T))
    fantasy_c=rng.binomial(1,expit(world@s.candidate_x.T))
    chosen=[];free=np.ones(len(s.cost),bool);steps=[]
    while True:
        ids=np.flatnonzero(free&(s.cost<=s.budget-s.cost[chosen].sum()+1e-10))
        if not len(ids):break
        score=np.zeros(len(ids))
        for b in range(4):
            xx=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[chosen]])
            yy=np.r_[s.visible_y,fantasy_o[b],fantasy_c[b,chosen]]
            theta=fit_logistic(xx,yy,s.ridge);v=info(xx,theta)
            post=rng.multivariate_normal(theta,inverse(s.ridge*np.eye(xx.shape[1])+v.T@v),size=32)
            score+=epig_scores(expit(post@s.candidate_x[ids].T),expit(post@s.reference_x[:128].T))/4
        idx=int(np.argmax(score/np.maximum(s.cost[ids],1e-15)));i=int(ids[idx])
        chosen.append(i);free[i]=False;steps.append(float(score[idx]))
    return finish(s,chosen,{'adaptation':'official multi-pseudo criterion, four fantasies, 32 posterior draws; marginal EPIG/cost',
                            'epig_steps':steps})


def ordinary(x,theta,price_theta,capacity,risk_cap,threshold=.5):
    p=expit(x@theta);price=expit(x@price_theta);ids=np.flatnonzero(p<threshold)
    ids=ids[np.argsort(p[ids],kind='stable')];selected=[];used=0.
    for i in ids:
        if len(selected)>=capacity:break
        if used+price[i]<=risk_cap+1e-10:selected.append(int(i));used+=price[i]
    return np.asarray(selected,int)


def gate_select(s,worlds=16,horizon=2,frozen=False,release=True):
    proposals=[np.empty(0,int),np.argsort(s.cost)[:kstar(s.cost,s.budget)],value_select(s).selected_ids,
               value_select(s,pairs=True).selected_ids]
    rng=np.random.default_rng(s.seed+250)
    while len(proposals)<8:proposals.append(random_fill(s.cost,s.budget,rng))
    # Only observed feature history is used for forecasting future arrivals.
    historical=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x])
    future=historical[rng.integers(0,len(historical),64)]
    ref=s.reference_x[:128];cap=32; risk_cap=8.
    hs,_,_=geometry(s,False);draws=rng.multivariate_normal(s.theta,inverse(hs),size=worlds+8)
    ox=s.ordinary_x;cx=s.candidate_x
    uniform_current=rng.random((worlds+8,len(ox)+len(cx)))
    uniform_future=rng.random((worlds+8,len(future)))
    records=[];values=np.zeros((len(draws),len(proposals)))
    for w,theta in enumerate(draws):
        pc=expit(np.vstack([ox,cx])@theta);yc=(uniform_current[w]<pc).astype(int)
        yf=(uniform_future[w]<expit(future@theta)).astype(int)
        truth=expit(ref@theta)
        oldtheta=fit_logistic(np.vstack([s.visible_x,ox]),np.r_[s.visible_y,yc[:len(ox)]],s.ridge)
        o0=ordinary(future,oldtheta,s.theta,cap,risk_cap)
        for j,selected in enumerate(proposals):
            xx=np.vstack([s.visible_x,ox,cx[selected]]); yy=np.r_[s.visible_y,yc[:len(ox)],yc[len(ox)+selected]]
            direct=fit_logistic(xx,yy,s.ridge);future_ids=o0 if frozen else ordinary(future,direct,s.theta,cap,risk_cap)
            final=fit_logistic(np.vstack([xx,future[future_ids]]),np.r_[yy,yf[future_ids]],s.ridge) if horizon==2 and release else direct
            values[w,j]=np.mean((expit(ref@final)-truth)**2)
    choice=int(np.argmin(values[:worlds].mean(axis=0)))
    return finish(s,proposals[choice],{'candidate_batches':[p.tolist() for p in proposals],
                      'planning_risk':values[:worlds].mean(axis=0).tolist(),
                      'heldout_planning_risk':values[worlds:].mean(axis=0).tolist(),
                      'chosen_batch':choice,'worlds':worlds,'verification_worlds':8,'horizon':horizon,
                      'future_feature_hash':digest(future),'future_features_source':'observed_history_only'})


def select(method,s,utility=None):
    start=time.perf_counter()
    if method=='OrdinaryOnly':out=finish(s,[])
    elif method=='MinRisk':out=finish(s,np.argsort(s.cost,kind='stable')[:kstar(s.cost,s.budget)])
    elif method=='Random':out=finish(s,random_fill(s.cost,s.budget,np.random.default_rng(s.seed)))
    elif method in ('LRBE-IG','ScoreFirst-IG'):
        out=finish(s,knapsack(s.cost,ig_scores(s) if utility is None else utility,s.budget,
                           kstar(s.cost,s.budget) if method=='LRBE-IG' else 0))
    elif method=='Vopt':out=value_select(s)
    elif method=='Vopt-noO':out=value_select(s,conditional=False)
    elif method=='Vopt-static':out=value_select(s,batch=False)
    elif method=='Vopt-static-noO':out=value_select(s,conditional=False,batch=False)
    elif method=='Vopt-Kstar':out=value_select(s,fixed_count=True)
    elif method=='RB-BRIDGE':out=value_select(s,pairs=True)
    elif method=='BRIDGE-noO':out=value_select(s,pairs=True,conditional=False)
    elif method=='BRIDGE-random':out=value_select(s,pairs=True,pair_mode='random')
    elif method=='BRIDGE-generic':out=value_select(s,pairs=True,pair_mode='generic')
    elif method=='BAIT-cost':out=bait_select(s)
    elif method=='ParBaLS-cost':out=parbals_select(s)
    elif method.startswith('CORR-') or method=='RB-CORR':out=corr_select(s, 'base' if method.startswith('CORR-base') else 'vopt' if method.startswith('CORR-vopt') else 'optimized')
    elif method in ('RB-GATE','GATE-one','GATE-frozen','GATE-no-release'):
        out=gate_select(s,horizon=1 if method=='GATE-one' else 2,frozen=(method=='GATE-frozen' or s.future_approval_frozen),release=method!='GATE-no-release')
    else:raise ValueError(method)
    out.diagnostics['selection_seconds']=time.perf_counter()-start
    return out
