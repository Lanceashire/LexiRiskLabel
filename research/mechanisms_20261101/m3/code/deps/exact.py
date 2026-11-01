"""Enumerate every subset in the registered 18 exact Gaussian conditions."""
from pathlib import Path
import argparse, hashlib, json, platform, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def risk(h,g): return float(np.trace(g@np.linalg.inv(h)))
def gain(h,g,u):
    u=np.asarray(u).reshape(-1,len(h))
    return risk(h,g)-risk(h+u.T@u,g)

def greedy(h,g,u,c,b,ids=None):
    ids=np.arange(len(c)) if ids is None else np.asarray(ids)
    order=np.argsort(ids,kind='stable'); chosen=[]; remaining=b; inv=np.linalg.inv(h)
    for _ in c:
        free=[i for i in order if i not in chosen and c[i]<=remaining+1e-12]
        if not free:break
        v=u[free]@inv
        values=np.maximum(np.einsum('ij,jk,ik->i',v,g,v)/(1+np.einsum('ij,ij->i',v,u[free])),0.)
        ratio=np.array([v/c[i] if c[i]>0 else (np.inf if v>1e-15 else 0) for i,v in zip(free,values)])
        j=int(np.argmax(ratio))
        if values[j]<=1e-15:break
        i=free[j];chosen.append(i);remaining-=c[i]
        v=inv@u[i];inv-=np.outer(v,v)/(1+u[i]@v)
    return np.array(sorted(chosen,key=lambda i:ids[i]),int)

def fraction_bound(h,g,u,c,b):
    inv=np.linalg.inv(h);v=np.einsum('ij,jk,ik->i',u@inv,g,u@inv)
    total=float(v[c==0].sum());remaining=b
    for i in sorted(np.flatnonzero(c>0),key=lambda i:(-v[i]/c[i],i)):
        z=min(1.,max(remaining,0.)/c[i]);total+=z*v[i];remaining-=z*c[i]
    return min(risk(h,g),total)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    protocol=HERE/'mechanism_protocol.json';cfg=json.loads(protocol.read_text())['exact']
    identity={'config':cfg,'protocol_sha256':sha(protocol),'runner_sha256':sha(__file__),'python':platform.python_version(),'numpy':np.__version__,'started_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    write(a.out/'registration.json',identity)
    hd=np.array(cfg['H_D'],float);g=np.array(cfg['G'],float);u=np.array(cfg['candidate_vectors'],float)
    n=len(u);masks=((np.arange(2**n)[:,None]>>np.arange(n))&1).astype(bool)
    adds=np.einsum('si,ij,ik->sjk',masks.astype(float),u,u)
    base=gain(hd,g,u[0:1]);ranks=np.argsort([gain(hd,g,x[None]) for x in u],kind='stable')
    prices=np.sort(cfg['cost_multiset']);neutral=np.random.default_rng(cfg['cost_permutation_seed']).permutation(prices)
    expensive=np.empty(n);expensive[ranks]=prices;cheap=np.empty(n);cheap[ranks]=prices[::-1]
    rows=[];subsets=[];fixed=[];max_bound_error=0.;max_null_error=0.
    for name,d in cfg['directions'].items():
        d=np.array(d,float)
        for strength in cfg['strengths']:
            h=hd+strength*np.outer(d,d);r0=risk(h,g)
            values=r0-np.einsum('ij,sji->s',g,np.linalg.inv(h[None]+adds))
            fixed.append({'direction':name,'strength':strength,'fixed_u_gain':gain(h,g,u[0:1]),'total_risk':r0})
            if name=='irrelevant':max_null_error=max(max_null_error,abs(gain(h,g,u[0:1])-base))
            for association,c in [('neutral',neutral),('information_cheap',cheap),('information_expensive',expensive)]:
                spent=masks@c; feasible=spent<=cfg['budget']+1e-12;feas=np.flatnonzero(feasible)
                best=int(feas[np.argmax(values[feas])]);costorder=np.argsort(c,kind='stable');k=int(np.searchsorted(np.cumsum(c[costorder]),cfg['budget']+1e-12,side='right'))
                sm=costorder[:k];fm=gain(h,g,u[sm]);proxy=greedy(hd,g,u,c,cfg['budget']);fp=gain(h,g,u[proxy])
                ub=fraction_bound(h,g,u,c,cfg['budget']);head=values[best]-fm;hu=max(0.,ub-fm)
                assert values[best]>=fm-1e-12 and values[best]>=fp-1e-12
                assert ub>=values[best]-1e-12 and hu>=head-1e-12
                max_bound_error=max(max_bound_error,float(values[best]-ub))
                row={'direction':name,'strength':strength,'cost_condition':association,'budget':cfg['budget'],'cost_multiset_hash':hashlib.sha256(np.sort(c).tobytes()).hexdigest(),
                     'feasible_sets':len(feas),'Kstar':k,'optimal_gain':float(values[best]),'MinRisk_gain':fm,'proxy_gain':fp,'headroom':float(head),'proxy_regret':float(values[best]-fp),
                     'fractional_total_bound':ub,'headroom_upper_bound':hu,'bound_slack':float(ub-values[best]),'excludes_delta':bool(hu<=cfg['delta']),
                     'optimal_ids':np.flatnonzero(masks[best]).tolist(),'MinRisk_ids':sorted(sm.tolist()),'proxy_ids':proxy.tolist(),
                     'optimal_count':int(masks[best].sum()),'MinRisk_count':len(sm),'proxy_count':len(proxy),'optimal_cost':float(spent[best]),'MinRisk_cost':float(c[sm].sum()),'proxy_cost':float(c[proxy].sum())}
                rows.append(row)
                subsets.extend({'direction':name,'strength':strength,'cost_condition':association,'mask':int(i),'gain':float(values[i]),'cost':float(spent[i]),'feasible':bool(feasible[i])} for i in range(2**n))
    # Independent analytic checks, including both counterexamples from the plan.
    h=np.eye(2);gg=np.diag([1.,0.]);uu=np.array([[1.,2.]]);v=np.array([1.,-2.])
    before=gain(h,gg,uu);after=gain(h+np.outer(v,v),gg,uu)
    assert abs(before-1/6)<1e-12 and abs(after-1/2)<1e-12
    relative=[gain(np.diag([1+b,1]),np.eye(2),np.array([[0.,1.]]))-gain(np.diag([1+b,1]),np.eye(2),np.array([[1.,0.]])) for b in [0,1]]
    assert abs(relative[0])<1e-12 and abs(relative[1]-1/3)<1e-12
    zero_cost_bound=fraction_bound(np.eye(1),np.eye(1),np.array([[1.],[0.]]),np.array([0.,1.]),0.)
    assert zero_cost_bound>=.5 and gain(hd,g,u[-1:])==0 and max_null_error<1e-12
    curve=[]
    for name,d in cfg['directions'].items():
        d=np.array(d)
        for t in np.linspace(0,20,201):curve.append({'direction':name,'strength':t,'gain':gain(hd+t*np.outer(d,d),g,u[0:1])})
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'condition_results.csv',index=False)
    pd.DataFrame(subsets).to_csv(a.out/'all_subset_values.csv.gz',index=False,compression='gzip')
    pd.DataFrame(fixed).to_csv(a.out/'fixed_set_values.csv',index=False)
    pd.DataFrame(curve).to_csv(a.out/'fixed_set_curves.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,q in pd.DataFrame(curve).groupby('direction'):axes[0].plot(q.strength,q.gain,label=name)
    axes[0].set(xlabel='Ordinary information strength',ylabel='Fixed set Bayes risk reduction');axes[0].legend()
    for name,q in frame.groupby('cost_condition'):axes[1].plot(np.arange(len(q)),q.headroom,'o-',label=name)
    axes[1].set(xlabel='Geometry / strength condition',ylabel='Exact advantage over MinRisk');axes[1].legend(fontsize=8)
    axes[2].scatter(frame.headroom,frame.headroom_upper_bound);mx=max(frame.headroom_upper_bound.max(),.01);axes[2].plot([0,mx],[0,mx],'--',color='gray')
    axes[2].set(xlabel='Exact advantage over MinRisk',ylabel='Certified upper bound');fig.tight_layout();fig.savefig(a.out/'exact_geometry.png',dpi=180);plt.close(fig)
    checks={'passed':True,'conditions':len(rows),'enumerated_subsets':len(subsets),'subsets_per_condition':2**n,'max_null_error':max_null_error,'bound_violation':max_bound_error,
            'counterexample_fixed':[before,after],'counterexample_relative':relative,'zero_cost_handled':True,'zero_information_gain':0.,'seconds':time.perf_counter()-start,
            'certified_exclusions':int(frame.excludes_delta.sum()),'threshold':cfg['delta'],'scope':'known Gaussian model; no classification or novel-theorem claim'}
    write(a.out/'verification.json',checks);print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
