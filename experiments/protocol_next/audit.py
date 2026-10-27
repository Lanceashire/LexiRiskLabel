"""E0: small, risk-oriented implementation audit for the new diagnostic runtime."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from runtime import *
from gate_protocol import FutureProtocol
import run_mechanisms as previous_mechanisms


def run_audit(config,identity,output):
    out=Path(output);private=out/'private/audit';private.mkdir(parents=True,exist_ok=True)
    context=build_context('hmeq',2201,2,config,private);a=context.acquisition
    utility,_=bootstrap_ig(a,config)
    selectors=config['selectors'];reference={m:select_e1(a,m,config,utility=utility).selected_ids for m in selectors}
    _,y,_,_,_=legacy.load('hmeq')
    perturbations={}
    for kind,ids in [('candidate',context.feedback.c_ids),('ordinary',context.feedback.o_ids),('outer_test',context.evaluation._ids)]:
        changed=y.copy();changed[ids]=1-changed[ids]
        alternative=build_context('hmeq',2201,2,config,private,label_override=changed)
        us,_=bootstrap_ig(alternative.acquisition,config)
        assert np.array_equal(utility,us)
        assert np.array_equal(a.theta,alternative.acquisition.theta)
        for method in selectors:
            result=select_e1(alternative.acquisition,method,config,utility=us)
            assert np.array_equal(reference[method],result.selected_ids),(kind,method)
        perturbations[kind]={'passed':True,'selectors':len(selectors),'committee_recomputed':True}
    changed_identities={}
    for component in ['configuration','code','datasets']:
        other=deepcopy(identity)
        if component=='configuration':other[component]['lr_ridge']=10.
        else:other[component][next(iter(other[component]))]='changed-for-audit'
        try:verify_identity(identity,other)
        except RuntimeError:changed_identities[component]=True
        else:raise AssertionError('identity change silently accepted')
    # A lock token must exist before O/S labels can be released.
    try:context.feedback.reveal(np.empty(0,int),'not_locked')
    except RuntimeError:locked_before_release=True
    else:raise AssertionError('feedback released before lock')
    selected=reference['MinRisk'];token=context.feedback.lock(selected,'MinRisk',a.seed)
    ids,xx,yy=context.feedback.reveal(selected,token)
    assert len(ids)==len(np.unique(ids))
    try:context.feedback.lock(np.r_[selected,selected[:1]],'duplicated',a.seed)
    except AssertionError:duplicate_rejected=True
    else:raise AssertionError('duplicate acquisition accepted')
    cache=FitCache(config,identity['identity_hash'],private)
    empty=np.empty(0,int);key=context.feedback.lock(empty,'OrdinaryOnly',a.seed)
    r1=cache.fit(context,empty,'LR',key);r2=cache.fit(context,empty,'LR',key)
    assert not r1['cache_hit'] and r2['cache_hit'] and r1['fit_key']==r2['fit_key'] and cache.fits==1
    future=FutureProtocol(**config['future']);one=replace(future,planning_horizon=1);no_release=replace(future,actual_future_release=False)
    assert one.actual_future_release and one.environment_parameters()==future.environment_parameters()
    assert no_release.planning_horizon==2 and no_release.planner_parameters()==future.planner_parameters()
    prob=np.linspace(.05,.95,future.future_batch_size);prices=np.full(future.future_batch_size,.2)
    approved=future.approvals(prob,prices);assert len(approved)<=future.capacity and prices[approved].sum()<=future.risk_cap+1e-10
    assert len(future.release(approved,np.zeros(future.future_batch_size,int))[0])==len(approved)
    assert len(no_release.release(approved,np.zeros(future.future_batch_size,int))[0])==0
    # Record inaccessible CORR support from the frozen negative-result block.
    support=[];private_support=[]
    for dataset in config['datasets']:
        for seed in config['seeds']:
            directory=ROOT/'results/research_v2/final_real'/f'{dataset}_{seed}_correction_RB-CORR'
            for r in range(1,11):
                log=json.loads((directory/f'round_{r:02d}_selection.json').read_text())
                pi=np.asarray(log['inclusion_probabilities']);candidate=np.asarray(log['candidate_ids'])
                unreachable=candidate[pi==0]
                support.append({'dataset':dataset,'seed':seed,'round':r,'candidate_n':len(candidate),'pi_zero_count':len(unreachable),
                                'positive_support_fraction':float((pi>0).mean()) if len(pi) else None})
                private_support.append({**support[-1],'pi_zero_ids':unreachable})
    write_jsonl(out/'corr_support_audit.jsonl',support);write_jsonl(private/'corr_support_ids.jsonl',private_support)
    linear=previous_mechanisms.correctness()
    # Parameter changes with identical actions are one intervention, not extra evidence.
    other=replace(a,exchanges=0);s1=frozen.select('Vopt',a);s2=frozen.select('Vopt',other)
    assert np.array_equal(s1.selected_ids,s2.selected_ids)
    same_key=context.feedback.lock(s1.selected_ids,'Vopt',a.seed)==context.feedback.lock(s2.selected_ids,'Vopt-exchange-parameter-zero',a.seed)
    assert same_key
    result={'status':'PASS_FOR_E1','hidden_label_perturbation':perturbations,'identity_changes_rejected':changed_identities,
            'lock_before_release':locked_before_release,'duplicate_collection_rejected':duplicate_rejected,
            'identical_training_cached':True,'same_action_deduplicated':True,'linear_and_exhaustive_checks':linear,
            'future_parameters':config['future'],'one_step_and_actual_future_release_separated':True,
            'future_capacity_and_risk_enforced':True,'corr_pi_zero_explicitly_recorded':True,'corr_unbiased_claim_domain':'positive-probability support only',
            'legacy_findings':[{'issue':'BRIDGE pair search and exchange confounded','status':'frozen result retained; four-cell diagnostic required before group-search attribution'},
                               {'issue':'GATE planning 64/32/8 vs actual environment 160/80/20','status':'explicit common FutureProtocol implemented; legacy result not reinterpreted as corrected E4'},
                               {'issue':'legacy no-release changed planner only','status':'actual environment release is an independent parameter; scale-up remains disabled'}],
            'scope':'implementation audit; mathematical checks do not establish learning improvement'}
    write_json(out/'audit_report.json',result);return result
