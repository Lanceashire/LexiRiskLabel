"""Frozen M5 conditional test-group inference, evaluated once after all locks.

These intervals condition on all fitted procedures. They do not quantify
training-population uncertainty, and replay seeds are not independent units.
"""
from common import *
from data import load

def endpoint_status(lo,hi,delta):
    if lo>delta:return 'meaningful_positive'
    if lo>=-delta and hi<=delta:return 'within_practical_band'
    if hi<0:return 'negative_direction'
    if hi<delta:return 'positive_threshold_excluded'
    if lo>0:return 'positive_direction_only'
    return 'unresolved'

def analyze_dataset(name,plan):
    raw,y,exposure,group,receipt=load(name)
    cfg=json.loads((HERE/'pre_label_scope.json').read_text());seeds=cfg['seeds'];root=HERE/'runs'
    ids=None;delta=[];main={};one_step={};diagnostic={};descriptive=[]
    for seed in seeds:
        run=root/f'{name}_{seed}';assert json.loads((run/'complete.json').read_text())['complete']
        split=np.load(run/'split.npz');current=split['test']
        if ids is None:ids=current
        assert np.array_equal(ids,current),'T must be sealed and shared across every replay'
        delta.append(json.loads((run/'identity.json').read_text())['delta'])
        d=pd.read_csv(run/'common_state_no_T_outcomes.csv')
        for learner in ['LR','XGB']:
            losses={}
            for method in METHODS:
                z=np.load(run/learner/method/'predictions.npz');assert np.array_equal(z['test_ids'],ids)
                loss=(z['prediction']-y[ids])**2;assert loss.shape==(10,len(ids))
                losses[method]=loss.mean(axis=0)
                descriptive.append({'dataset':name,'seed':seed,'learner':learner,'method':method,'ten_round_Brier':float(loss.mean())})
            for comparator in ['MinRisk',METHODS[-1]]:
                main.setdefault((learner,comparator),[]).append(losses[comparator]-losses['Pending-aware'])
            for r in [2,5,8]:
                for treatment in ['full','empty','high','low','random']:
                    p=run/'common_state'/learner
                    a=np.load(p/f'r{r}_{treatment}_MinRisk.npz')['prediction'];b=np.load(p/f'r{r}_{treatment}_Pending-aware.npz')['prediction']
                    gain=(a-y[ids])**2-(b-y[ids])**2
                    one_step.setdefault((learner,treatment),[]).append(gain)
                    if treatment=='full':
                        row=d[(d.learner==learner)&(d['round']==r)&(d.treatment==treatment)&(d.method=='Pending-aware')].iloc[0]
                        diagnostic.setdefault(learner,[]).append((gain,float(row.predicted_gain),float(row.constant_prediction),int(row.diagnostic_outside_M4_range)))
    delta=float(np.mean(delta));entries=[];linear_vectors=[];nonlinear=[]
    for learner in ['LR','XGB']:
        for comparator in ['MinRisk',METHODS[-1]]:
            v=np.mean(main[(learner,comparator)],axis=0)
            entries.append({'dataset':name,'learner':learner,'endpoint':'ten_round_PA_gain_vs_'+comparator,'delta':delta,'kind':'algorithm_gain'})
            linear_vectors.append(v)
        for a,b,title in [('high','low','O_direction_high_minus_low'),('full','empty','O_quantity_full_minus_empty')]:
            v=np.mean(one_step[(learner,a)],axis=0)-np.mean(one_step[(learner,b)],axis=0)
            entries.append({'dataset':name,'learner':learner,'endpoint':title,'delta':delta,'kind':'mechanism_interaction'})
            linear_vectors.append(v)
        gains=np.array([a for a,b,c,d in diagnostic[learner]]);pred=np.array([b for a,b,c,d in diagnostic[learner]]);constant=np.array([c for a,b,c,d in diagnostic[learner]])
        nonlinear.append((gains,pred,constant))
    # Eight linear and two diagnostic endpoints per task. Each test group is
    # resampled once with shared weights for every learner, seed and treatment.
    linear=np.array(linear_vectors);groups=group[ids];unique,inverse=np.unique(groups,return_inverse=True)
    size=np.bincount(inverse);agg=np.zeros((len(unique),len(linear)))
    for j,v in enumerate(linear):np.add.at(agg[:,j],inverse,v)
    diag_agg=[]
    for gains,pred,constant in nonlinear:
        a=np.zeros((len(unique),len(gains)))
        for j,v in enumerate(gains):np.add.at(a[:,j],inverse,v)
        diag_agg.append((a,pred,constant))
    point=list(linear.mean(axis=1))
    for learner,(gains,pred,constant) in zip(['LR','XGB'],nonlinear):
        actual=gains.mean(axis=1);skill=float(np.mean(np.abs(actual-constant)-np.abs(actual-pred)))
        entries.append({'dataset':name,'learner':learner,'endpoint':'M4_diagnostic_MAE_skill_vs_M4_constant','delta':0.,'kind':'explanation_prediction',
            'states':len(actual),'out_of_M4_feature_range_fraction':float(np.mean([d for a,b,c,d in diagnostic[learner]])),
            'MAE_model':float(np.mean(np.abs(actual-pred))),'MAE_constant':float(np.mean(np.abs(actual-constant))),
            'sign_agreement_fraction':float(np.mean((actual>0)==(pred>0))),'mean_predicted_gain':float(pred.mean()),'mean_actual_gain':float(actual.mean())})
        point.append(skill)
    boot=np.empty((plan['bootstrap_repetitions'],len(entries)));rng=np.random.default_rng(plan['bootstrap_seed']+(0 if name=='LendingClub2017' else 1))
    for k in range(0,len(boot),128):
        n=min(128,len(boot)-k);weights=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=n)
        den=weights@size;boot[k:k+n,:len(linear)]=(weights@agg)/den[:,None]
        for j,(a,pred,constant) in enumerate(diag_agg):
            value=(weights@a)/den[:,None];boot[k:k+n,len(linear)+j]=np.mean(np.abs(value-constant)-np.abs(value-pred),axis=1)
    tail=plan['family_alpha']/(2*plan['primary_family_size']);low,high=np.quantile(boot,[tail,1-tail],axis=0)
    for j,row in enumerate(entries):
        row.update({'estimate':point[j],'low':float(low[j]),'high':float(high[j]),'test_rows':len(ids),'test_groups':len(unique),'replay_seeds_descriptive_count':len(seeds),
            'interval_scope':'conditional test-group percentile bootstrap; approximate Bonferroni simultaneous family',
            'bootstrap_centered_p':float((1+np.sum(np.abs(boot[:,j]-point[j])>=abs(point[j])))/(len(boot)+1)),
            'positive_direction':bool(low[j]>0),'negative_direction':bool(high[j]<0),
            'within_delta':bool(low[j]>=-row['delta'] and high[j]<=row['delta']),
            'status':endpoint_status(float(low[j]),float(high[j]),row['delta'])})
    supplementary=[]
    for (learner,treatment),values in one_step.items():
        supplementary.append({'dataset':name,'learner':learner,'treatment':treatment,'mean_PA_gain_vs_MinRisk':float(np.mean(values)),
            'inference':'descriptive; matched random half is a quantity control, not a separate significance search'})
    pd.DataFrame(descriptive).to_csv(HERE/'analysis'/f'{name}_seed_method_Brier.csv',index=False)
    pd.DataFrame(supplementary).to_csv(HERE/'analysis'/f'{name}_ordinary_controls.csv',index=False)
    np.savez_compressed(HERE/'analysis'/f'{name}_bootstrap.npz',point=point,low=low,high=high,bootstrap=boot,test_ids=ids)
    return entries

def main():
    plan=json.loads((HERE/'analysis_plan.json').read_text());reg=json.loads((HERE/'registration.json').read_text())
    assert sha(HERE/'analysis_plan.json')==reg['source_hashes']['analysis_plan.json']
    (HERE/'analysis').mkdir(exist_ok=True);all_entries=[]
    for name in ['LendingClub2017','SouthGermanCredit']:all_entries+=analyze_dataset(name,plan)
    assert len(all_entries)==plan['primary_family_size']==20
    order=sorted(range(len(all_entries)),key=lambda j:all_entries[j]['bootstrap_centered_p']);adjusted=0.
    for rank,j in enumerate(order):
        adjusted=max(adjusted,min(1.,(len(all_entries)-rank)*all_entries[j]['bootstrap_centered_p']))
        all_entries[j]['Holm_adjusted_bootstrap_p']=adjusted
    pd.DataFrame(all_entries).to_csv(HERE/'analysis/primary_endpoints.csv',index=False)
    write(HERE/'analysis/results.json',{'registration_sha256':sha(HERE/'registration.json'),'analysis_plan':plan,'endpoints':all_entries,
        'full_training_population_inference':False,'real_reject_counterfactual_identified':False,'strict_as_of_online_learning':False})
    print(pd.DataFrame(all_entries)[['dataset','learner','endpoint','estimate','low','high','status']].to_string(index=False),flush=True)
if __name__=='__main__':main()
