"""Small development experiments from the referenced conversation, A1--C2."""
from __future__ import annotations
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge
from core import *

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
ROOT=HERE.parents[1]


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a)))
    temp.replace(path)


def correctness():
    # Conversation's exact counterexample, before fitting any credit data.
    h=np.eye(2);g=np.diag([1.,0.]);u=np.array([[1.,100.],[1.,-100.],[.1,0.]])
    singles=marginal(inverse(h),g,u); pair=set_value(h,g,u,np.array([0,1]));greedy=set_value(h,g,u,np.array([2,0]))
    assert np.isclose(pair,2/3) and pair>60*greedy
    changes=[]
    for b in (0.,9998.,9999.,10000.,20000.):
        hh=np.diag([1.,1+b]); gamma=set_value(hh,g,u,np.array([0,1]))-marginal(inverse(hh),g,u)[:2].sum()
        changes.append({'ordinary_information':b,'gamma':gamma})
        assert (gamma>1e-12)==(b<9999.)
    # Woodbury implementations versus direct matrix solves, including pair update.
    rng=np.random.default_rng(707);a=rng.normal(size=(5,5));hh=a.T@a+np.eye(5);gg=np.eye(5);uu=rng.normal(size=(12,5));cc=inverse(hh)
    pp=np.array(list(itertools.combinations(range(12),2)))
    direct=np.array([set_value(hh,gg,uu,p) for p in pp]);assert np.allclose(pair_gains(cc,gg,uu,pp),direct)
    costs=rng.uniform(.2,1,12);budget=2.
    best=max((set_value(hh,gg,uu,np.array(ids,int)),ids) for k in range(13) for ids in itertools.combinations(range(12),k) if costs[list(ids)].sum()<=budget)
    # A real State used for probability/visibility and budget checks.
    x=np.c_[np.ones(40),rng.normal(size=(40,4))];y=rng.binomial(1,.4,40);theta=fit_logistic(x,y)
    s=State(x,y,x[:10],np.c_[np.ones(12),uu[:,:4]],x[:20],costs,budget,theta,707)
    selections={m:select(m,s) for m in ['MinRisk','Vopt','RB-BRIDGE','BRIDGE-random','BAIT-cost','RB-CORR']}
    hs,gs,us=geometry(s)
    same_opt=max((set_value(hs,gs,us,np.array(ids,int)),ids) for k in range(13) for ids in itertools.combinations(range(12),k) if costs[list(ids)].sum()<=budget)
    gaps={m:same_opt[0]-set_value(hs,gs,us,v.selected_ids) for m,v in selections.items() if m in ['Vopt','RB-BRIDGE','BRIDGE-random']}
    assert all(v>=-1e-10 for v in gaps.values())
    cr=selections['RB-CORR'];pi=cr.inclusion_probabilities;design=cr.sampling_design
    # Exact expectation over the finite design, rather than thousands of samples.
    hidden=rng.binomial(1,.3,12);f=rng.uniform(.1,.8,12);pred=expit(s.candidate_x@s.theta)
    risks=[];pseudo_average=np.zeros(12)
    for ids,w in zip(design['sets'],design['weights']):
        z=np.zeros(12);z[ids]=1;tilde=pred.copy();eligible=pi>0;tilde[eligible]+=z[eligible]*(hidden[eligible]-pred[eligible])/pi[eligible]
        pseudo_average+=w*tilde;risks.append(w*np.mean(f*f+(1-2*f)*tilde))
    assert np.allclose(pseudo_average,hidden) and np.isclose(sum(risks),np.mean((f-hidden)**2))
    # Selector's API cannot receive pending, candidate, reference or future labels.
    assert set(State.__dataclass_fields__) & {'candidate_y','ordinary_y','test_y','future_y'}==set()
    no_o=State(**{**s.__dict__,'ordinary_x':x[:0]})
    assert np.array_equal(value_select(no_o).selected_ids,value_select(no_o,conditional=False).selected_ids)
    # Exact zero-direction and ordinary-direction ranking controls.
    hc=np.diag([1.,101.]);uc=np.array([[1.,0.],[0.,.9]])
    before=marginal(np.eye(2),np.eye(2),uc);after=marginal(inverse(hc),np.eye(2),uc)
    # Ordinary on second direction must reverse the first/second ranking if initially second wins.
    ordinary_first=np.diag([101.,1.]);after_first=marginal(inverse(ordinary_first),np.eye(2),uc)
    assert before[0]>before[1] and after_first[1]>after_first[0]
    zero=marginal(np.eye(2),np.diag([1.,0.]),np.array([[0.,5.]]))[0];assert zero==0
    return {'passed':True,'bridge_single_gains':singles.tolist(),'bridge_pair_value':pair,'point_greedy_value':greedy,
            'complementarity_transition':changes,'small_pool_exact_proxy_optimum':best[0],
            'small_pool_exact_set':list(best[1]),'pair_solve_max_error':float(np.max(np.abs(direct-pair_gains(cc,gg,uu,pp)))),
            'corr_exact_expectation_error':float(abs(sum(risks)-np.mean((f-hidden)**2))),
            'ordinary_first_ranking_before':before.tolist(),'ordinary_first_ranking_after':after_first.tolist(),
            'zero_direction_gain':float(zero),'same_pool_exact_optimum':same_opt[0],'same_pool_heuristic_gaps':gaps,
            'label_visibility':'selectors have no hidden-label fields'}


def generate(env,seed):
    rng=np.random.default_rng(seed);n=CFG['mechanism'];dim=5
    def block(size):return np.c_[np.ones(size),rng.normal(size=(size,dim-1))]
    d=block(n['visible']);o=block(n['ordinary']);c=block(n['candidate']);q=block(n['reference']);test=block(n['test'])
    if env[0]=='A':
        d[:,2]=d[:,1]+rng.normal(0,.04,len(d));d[:,3:]=0
        # Whole candidate pool has paired nuisance signs, not a score-selected subset.
        z=rng.normal(1.,.15,len(c)//2);nuisance=rng.normal(6.,.2,len(c)//2)
        c[:,1]=np.r_[z,z];c[:,2]=np.r_[nuisance,-nuisance];c[:,3:]=0
        o[:,1:]=0
        if env=='A1':o[:,3]=rng.normal(size=len(o)) # same count, irrelevant direction
        else:o[:,2]=rng.normal(0,6,len(o)) # removes nuisance ambiguity
        q[:,2:]=0;test[:,2:]=0
        truth=lambda x:expit(-.6+1.4*x[:,1])
    elif env[0]=='B':
        d[:,1]=rng.normal(-1.5,.5,len(d));o[:,1]=rng.normal(-1.,.5,len(o));c[:,1]=rng.normal(1.,.6,len(c))
        if env=='B1':truth=lambda x:expit(-1.+1.2*x[:,1]+1.1*x[:,2]**2)
        else:
            # Matched square-loss linear model, with probabilities strictly in [0,1].
            for x in [d,o,c,q,test]:x[:,1:]=np.clip(x[:,1:],-2,2)
            truth=lambda x:.4+.08*x[:,1]+.04*x[:,2]
    else:
        # Rare safe region poorly covered by the initial observational sample.
        for x in [d,o,c,q,test]:
            x[:,2]=(rng.random(len(x))<.35).astype(float);x[:,3]=x[:,1]*x[:,2];x[:,4]=0
        d[:,2]=(rng.random(len(d))<.07);d[d[:,2]>0,1]+=2.;d[:,3]=d[:,1]*d[:,2]
        o[:,2]=0;o[:,3]=0;c[:,2]=1;c[:,3]=c[:,1]
        truth=lambda x:expit(-1.+x[:,1]-.8*x[:,2]+.8*x[:,3])
    yd=rng.binomial(1,truth(d));yo=rng.binomial(1,truth(o));yc=rng.binomial(1,truth(c));yt=rng.binomial(1,truth(test))
    theta=fit_logistic(d,yd,CFG['ridge']);cost=.5+.2*expit(c[:,1])
    budget=CFG['rho']*cost.sum()
    s=State(d,yd,o,c,q,cost,budget,theta,seed,CFG['ridge'],CFG['partners'],CFG['exchanges'],CFG['mixture_steps'],CFG['coverage_epsilon'])
    s.future_approval_frozen=(env=='C2')
    return s,yo,yc,test,yt,truth(test),truth


def metric(p,y,truth):return {'brier':float(np.mean((p-y)**2)),'excess_brier':float(np.mean((p-truth)**2))}


def run_fixed(env,seed):
    s,yo,yc,test,yt,pt,truth=generate(env,seed)
    methods=['OrdinaryOnly','MinRisk','Vopt','RB-BRIDGE','BRIDGE-random','BRIDGE-generic','BRIDGE-noO','BAIT-cost','ParBaLS-cost'] if env[0]=='A' else ['OrdinaryOnly','CORR-base-plain','CORR-base-corrected','CORR-opt-plain','RB-CORR','CORR-vopt-corrected']
    rows=[];logs=[]
    for method in methods:
        selection=select(method,s);ids=selection.selected_ids
        xx=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]]);yy=np.r_[s.visible_y,yo,yc[ids]]
        if env[0]=='B':
            if method in ['CORR-base-corrected','RB-CORR','CORR-vopt-corrected']:
                pi=selection.inclusion_probabilities;eligible=np.flatnonzero(pi>0)
                yy_c=expit(s.candidate_x@s.theta);yy_c[ids]+=(yc[ids]-yy_c[ids])/pi[ids]
                xx=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[eligible]]);yy=np.r_[s.visible_y,yo,yy_c[eligible]]
            learner=Ridge(alpha=1.,fit_intercept=False).fit(xx,yy);prediction=np.clip(learner.predict(test),0,1)
        else:prediction=expit(test@fit_logistic(xx,yy))
        diag={k:v for k,v in selection.diagnostics.items() if not isinstance(v,(list,dict))}
        rows.append({'environment':env,'seed':seed,'method':method,'round':1,'protocol':'fixed_approval',
                     'learner':'square_ridge' if env[0]=='B' else 'logistic','count':len(ids),'predicted_risk':selection.predicted_cost,
                     'realized_loss':float(yc[ids].sum()),**metric(prediction,yt,pt),**diag})
        logs.append({'environment':env,'seed':seed,'method':method,'selected_ids':ids,
                     'cost':s.cost,'budget':s.budget,'diagnostics':selection.diagnostics,
                     'probabilities':selection.inclusion_probabilities,'design':selection.sampling_design,
                     'visible_hash':digest(s.visible_x),'reference_hash':digest(s.reference_x)})
    return rows,logs


def run_gate(env,seed):
    s,yo,yc,test,yt,pt,truth=generate(env,seed);rng=np.random.default_rng(seed+900)
    future=s.reference_x[rng.integers(0,len(s.reference_x),160)]
    yf=rng.binomial(1,truth(future));price_theta=s.theta.copy();cap=80;risk_cap=20.
    base=fit_logistic(np.vstack([s.visible_x,s.ordinary_x]),np.r_[s.visible_y,yo]);o0=ordinary(future,base,price_theta,cap,risk_cap)
    rows=[];logs=[]
    methods=['OrdinaryOnly','MinRisk','Vopt','RB-BRIDGE','GATE-one','GATE-no-release','RB-GATE']
    for method in methods:
        selection=select(method,s);ids=selection.selected_ids
        xx=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]]);yy=np.r_[s.visible_y,yo,yc[ids]]
        theta=fit_logistic(xx,yy);o1=o0 if env=='C2' else ordinary(future,theta,price_theta,cap,risk_cap)
        fixed=fit_logistic(np.vstack([xx,future[o0]]),np.r_[yy,yf[o0]])
        final=fit_logistic(np.vstack([xx,future[o1]]),np.r_[yy,yf[o1]])
        fp=expit(test@final);fx=expit(test@fixed)
        ordinary_risk=float(expit(s.ordinary_x@price_theta).sum()+expit(future[o1]@price_theta).sum())
        ordinary_loss=float(yo.sum()+yf[o1].sum());channel=float(np.mean((fx-pt)**2)-np.mean((fp-pt)**2))
        if env=='C2':assert np.isclose(channel,0.) and np.array_equal(o0,o1)
        rows.append({'environment':env,'seed':seed,'method':method,'round':2,
                     'protocol':'dynamic_approval' if env=='C1' else 'frozen_approval',
                     'learner':'logistic','count':len(ids),'predicted_risk':selection.predicted_cost,
                     'realized_loss':float(yc[ids].sum()),'ordinary_predicted_risk':ordinary_risk,
                     'ordinary_realized_loss':ordinary_loss,'total_predicted_risk':ordinary_risk+selection.predicted_cost,
                     'total_realized_loss':ordinary_loss+yc[ids].sum(),'next_ordinary_count':len(o1),
                     'ordinary_set_change':len(set(o1)^set(o0)),'feedback_channel_value':channel,**metric(fp,yt,pt)})
        logs.append({'environment':env,'seed':seed,'method':method,'selected_ids':ids,'future_ordinary_ids':o1,
                     'fixed_ordinary_ids':o0,'cost':s.cost,'budget':s.budget,'diagnostics':selection.diagnostics})
    return rows,logs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--correctness-only',action='store_true');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    identity={'config':CFG,'code':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    identity['hash']=digest(identity)
    path=args.output/'identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise RuntimeError('output identity differs; use a new directory')
    save_json(path,identity);save_json(args.output/'correctness.json',correctness());print('E0 PASS',flush=True)
    if args.correctness_only:return
    rows=[];started=time.perf_counter()
    for env in ['A1','A2','B1','B2','C1','C2']:
        for seed in CFG['mechanism_seeds']:
            rr,logs=run_gate(env,seed) if env[0]=='C' else run_fixed(env,seed)
            rows.extend(rr);save_json(args.output/f'{env}_{seed}_selection.json',logs)
            pd.DataFrame(rows).to_csv(args.output/'mechanism_rows.csv',index=False)
            print(env,seed,'done',round(time.perf_counter()-started,1),'seconds',flush=True)
    frame=pd.DataFrame(rows);summary=frame.groupby(['environment','protocol','learner','method'],as_index=False).mean(numeric_only=True)
    summary.to_csv(args.output/'mechanism_summary.csv',index=False)
    save_json(args.output/'completion.json',{'completed':True,'rows':len(rows),'seconds':time.perf_counter()-started,'independent_seeds':len(CFG['mechanism_seeds'])})


if __name__=='__main__':main()
