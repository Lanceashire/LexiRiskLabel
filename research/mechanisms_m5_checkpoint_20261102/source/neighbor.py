"""Frozen explicit Laplace/cost/pending ParBaLS-EPIG adaptation from prior development."""
from common import *
frozen=core
def select_neighbor(s,cfg):
    started=time.perf_counter();rng=np.random.default_rng(s.seed+470)
    hd,_,_=frozen.geometry(s,False);draws=rng.multivariate_normal(s.theta,frozen.inverse(hd),size=cfg['posterior_draws'])
    world=draws[rng.choice(len(draws),cfg['universes'],replace=False)]
    # This hard pseudo-label convention follows current upstream source; not a Bernoulli label sampler claim.
    yo=(world@s.ordinary_x.T>=0).astype(int);yc=(world@s.candidate_x.T>=0).astype(int)
    chosen=[];free=np.ones(len(s.cost),bool);steps=[];refits=0
    while True:
        candidates=np.flatnonzero(free&(s.cost<=s.budget-s.cost[chosen].sum()+1e-10))
        if not len(candidates):break
        score=np.zeros(len(candidates))
        for w in range(cfg['universes']):
            x=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[chosen]])
            y=np.r_[s.visible_y,yo[w],yc[w,chosen]]
            theta=frozen.fit_logistic(x,y,s.ridge);v=frozen.info(x,theta)
            post=rng.multivariate_normal(theta,frozen.inverse(s.ridge*np.eye(x.shape[1])+v.T@v),size=cfg['posterior_draws'])
            score+=frozen.epig_scores(expit(post@s.candidate_x[candidates].T),expit(post@s.reference_x.T))/cfg['universes']
            refits+=1
        ratios=score/np.maximum(s.cost[candidates],1e-15);j=int(np.argmax(ratios));i=int(candidates[j])
        chosen.append(i);free[i]=False;steps.append({'candidate':i,'full_EPIG':float(score[j]),'risk_cost':float(s.cost[i]),'ratio':float(ratios[j])})
    ids=np.sort(np.asarray(chosen,int));assert s.cost[ids].sum()<=s.budget+1e-7
    return ids,{'internal_LR_refits':refits,'selection_seconds':time.perf_counter()-started,'steps':steps,
        'universe_parameters':world,'fantasy_ordinary_labels':yo,'fantasy_candidate_labels':yc}


def criterion_check():
    rng=np.random.default_rng(64179);pc=rng.uniform(.02,.98,(32,7));pq=rng.uniform(.02,.98,(32,9))
    score=frozen.epig_scores(pc,pq);direct=[];upstream=[]
    for i in range(pc.shape[1]):
        val=[];raw=0.
        for j in range(pq.shape[1]):
            a,b=pc[:,i].mean(),pq[:,j].mean();both=np.mean(pc[:,i]*pq[:,j]);joint=np.array([[1-a-b+both,b-both],[a-both,both]])
            expected=np.outer([1-a,a],[1-b,b]);val.append(float(np.sum(joint*np.log(joint/expected))))
            raw+=float(frozen.entropy(np.asarray(a))+np.sum(joint*np.log(joint)))
        direct.append(np.mean(val));upstream.append(raw)
    error=float(np.max(np.abs(score-direct)))
    # In equal costs, restoring the target entropy constant preserves original source ranking.
    offset=score-np.asarray(upstream)/len(pq.T);constant_error=float(np.ptp(offset))
    assert error<1e-12 and constant_error<1e-12 and np.argmax(score)==np.argmax(upstream)
    return {'direct_mutual_information_max_error':error,'equal_cost_constant_range':constant_error,
            'equal_cost_upstream_argmax_matches':True,'heterogeneous_cost_uses_full_nonnegative_MI':True}


