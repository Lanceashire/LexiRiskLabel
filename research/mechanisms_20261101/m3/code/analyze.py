"""World-paired M3 contrasts and outcome-blind pilot variance planning."""
from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
from scipy.stats import t, chi2, norm
from study import COSTS, sha, write

def world_frames(run,plan,M=None):
    paths={int(p.parent.name[1:]):p for p in Path(run).glob('worker_*/w[0-9]*/outcomes.csv') if p.parent.name[1:].isdigit()}
    assert set(paths)==set(plan['world_ids']),('Missing/extra worlds',len(paths),len(plan['world_ids']))
    frames=[];resources=[]
    for w in plan['world_ids']:
        p=paths[w];complete=json.loads((p.parent/'complete.json').read_text())
        assert complete['plan_sha256']==sha(plan['_path'])
        f=pd.read_csv(p)
        if M is not None:f=f[f.replicate<M]
        f['J_to_MinRisk']=np.nan
        for learner in ['LR','XGB']:
            ff=f[f.learner==learner];rep=ff.pivot(index='replicate',columns=['cost_condition','method'],values='risk')
            def risk(c,m):return rep[c,m]
            vals={}
            for c in COSTS:
                vals[f'J/{c}']=risk(c,'MinRisk')-risk(c,'Pending-aware')
                vals[f'conditioning/{c}']=risk(c,'D-only')-risk(c,'Pending-aware')
                vals[f'conditioning_batch_interaction/{c}']=(risk(c,'D-only')-risk(c,'Pending-aware'))-(risk(c,'Static-D')-risk(c,'Static-Pending'))
                for name,method in [('fixed_count','FixedK-Pending'),('gain_budget','GainBudget-Pending'),('same_count','MinRisk-k'),('spendcap','MinRisk-spendcap'),('matched','CostCountMatched')]:
                    vals[f'{name}/{c}']=risk(c,method)-risk(c,'Pending-aware')
            vals['price']=vals['J/information_cheap']-vals['J/information_expensive']
            vals['price_conditioning']=vals['conditioning/information_cheap']-vals['conditioning/information_expensive']
            for name,value in vals.items():
                assert value.notna().all() or name.startswith('matched/'),(w,name)
                v=value.to_numpy(float);finite=np.isfinite(v)
                frames.append({'world_id':w,'learner':learner,'contrast':name,'value':float(np.mean(v[finite])) if finite.any() else np.nan,
                               'label_replicates':len(v),'estimable':bool(finite.all()),'within_world_label_variance':float(np.var(v[finite],ddof=1)) if finite.sum()>1 else np.nan})
        resource=f[f.replicate==f.replicate.min()].copy();resources.append(resource)
    return pd.DataFrame(frames),pd.concat(resources,ignore_index=True)

def plan_from_pilot(run,path,out):
    plan=json.loads(Path(path).read_text());plan['_path']=str(path);assert plan['role']=='variance_pilot'
    # First 4 of 8 label draws have exactly the confirmatory MC sample size.
    f,_=world_frames(run,plan,M=plan['planning']['confirmation_label_M'])
    q=f[f.contrast=='price'];n=plan['planning']['pilot_worlds'];assert n==len(plan['world_ids'])==32
    estimates=[];required=[]
    for learner in ['LR','XGB']:
        x=q[q.learner==learner].value.to_numpy();assert len(x)==n
        sd=float(np.std(x,ddof=1));upper=math.sqrt((n-1)*sd*sd/chi2.ppf(.05,n-1))
        critical=norm.ppf(1-plan['primary_alpha']/(2*plan['primary_family']))
        raw=math.ceil((critical*upper/plan['planning']['target_halfwidth'])**2)
        required.append(raw)
        estimates.append({'learner':learner,'n_pilot_worlds':n,'planning_label_M':4,'world_SD':sd,'gaussian_upper95_SD':upper,'required_n_approx':raw})
    raw=max(required);chosen=min(plan['planning']['max_confirmation_worlds'],max(plan['planning']['min_confirmation_worlds'],int(math.ceil(raw/256))*256))
    result={'planning_complete':True,'outcome_means_not_reported_or_used':True,'variance_estimates':estimates,'max_raw_required_n':raw,'chosen_confirmation_n':chosen,
            'resource_cap_reached':raw>plan['planning']['max_confirmation_worlds'],'target_halfwidth':plan['planning']['target_halfwidth'],
            'qualification':'Gaussian chi-square SD upper bound and t/normal planning are approximations, not finite-sample guarantees. At the fixed resource cap retain unresolved conclusions.',
            'pilot_not_pooled_into_confirmation':True,'pilot_plan_sha256':sha(path),'pilot_worlds_sha256':__import__('hashlib').sha256(json.dumps(plan['world_ids']).encode()).hexdigest()}
    write(Path(out)/'variance_planning.json',result);print(json.dumps(result,indent=2),flush=True);return result

def analyze(run,path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);plan=json.loads(Path(path).read_text());plan['_path']=str(path)
    assert plan['role']=='confirmation';f,res=world_frames(run,plan)
    f.to_csv(out/'world_contrasts.csv',index=False);res.to_csv(out/'world_resources.csv',index=False)
    primary=f[f.contrast=='price'];secondary=f[f.contrast!='price']
    assert len(primary[['learner','contrast']].drop_duplicates())==plan['primary_family']==2
    assert len(secondary[['learner','contrast']].drop_duplicates())==plan['secondary_family']==50
    summaries=[]
    for role,frame,family,alpha in [('primary',primary,plan['primary_family'],plan['primary_alpha']),('secondary',secondary,plan['secondary_family'],plan['secondary_alpha'])]:
        for (learner,name),q in frame.groupby(['learner','contrast'],sort=True):
            x=q.loc[q.estimable,'value'].dropna().to_numpy();n=len(x)
            if n>1:
                mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n));radius=float(t.ppf(1-alpha/(2*family),n-1)*se);low=mean-radius;high=mean+radius
                delta=plan['delta'];verdict='unresolved_at_registered_endpoint'
                if low>delta:verdict='meaningful_positive'
                elif high < -delta:verdict='meaningful_negative'
                elif low>0:verdict='positive_direction_only'
                elif high<0:verdict='negative_direction_only'
                elif low>-delta and high<delta:verdict='within_registered_equivalence_band'
                elif high<delta:verdict='meaningful_positive_excluded'
                elif low>-delta:verdict='meaningful_negative_excluded'
            else:mean=se=low=high=np.nan;verdict='not_estimable'
            summaries.append({'role':role,'learner':learner,'contrast':name,'n_worlds':n,'registered_worlds':len(plan['world_ids']),
                              'mean':mean,'SE':se,'low':low,'high':high,'family':family,'family_alpha':alpha,'delta':plan['delta'],'verdict':verdict,
                              'estimand_scope':'matched-state subset' if n<len(plan['world_ids']) else 'all registered worlds'})
    s=pd.DataFrame(summaries);s.to_csv(out/'summary.csv',index=False)
    risk_paths=sorted(p for p in Path(run).glob('worker_*/w[0-9]*/outcomes.csv') if p.parent.name[1:].isdigit())
    all_out=pd.concat([pd.read_csv(p) for p in risk_paths],ignore_index=True)
    # Descriptive metrics include all methods, with no new inferential promotion.
    means=all_out.groupby(['world_id','learner','cost_condition','method'],dropna=False).agg(
        risk=('risk','mean'),brier=('brier','mean'),realized_selected_loss=('realized_selected_loss','mean'),count=('count','first'),cost=('cost','first'),
        proxy_pending=('proxy_pending','first'),true_expected_selected_loss=('true_expected_selected_loss','first'),composition_premium=('composition_premium','first'),m_star=('m_star','first'),Kstar=('Kstar','first')).reset_index()
    means.to_csv(out/'world_method_means.csv',index=False)
    means.groupby(['learner','cost_condition','method']).mean(numeric_only=True).drop(columns='world_id').to_csv(out/'method_descriptive_means.csv')
    write(out/'analysis_complete.json',{'completed':True,'worlds':len(plan['world_ids']),'primary_contrasts':2,'secondary_contrasts':50,'single_fixed_endpoint':True,'plan_sha256':sha(path),
                                      'pilot_pooled':False,'inference':'world-paired approximate t simultaneous intervals','nominal_total_alpha':plan['primary_alpha']+plan['secondary_alpha']})
    print(s[s.role=='primary'].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','confirmation']);p.add_argument('--run',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();(plan_from_pilot if a.mode=='pilot' else analyze)(a.run,a.plan,a.out)
