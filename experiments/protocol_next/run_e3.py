"""Small CORR diagnostic: fixed selection/design/g/learner; only target changes.

Current hidden labels reach training only through lock/reveal for the sampled S.
All-candidate residuals are accessed later in a separate, explicitly offline audit.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from runtime import *
from run_e1b import validation_blocks


def finite_design(s):
    selected=frozen.corr_select(s,'optimized');design=selected.sampling_design
    atoms=[np.asarray(x,int) for x in design['sets']];q=np.asarray(design['weights'],float)
    incidence=np.zeros((len(q),len(s.cost)))
    for j,ids in enumerate(atoms):
        assert len(ids)==len(np.unique(ids)) and s.cost[ids].sum()<=s.budget+1e-7
        incidence[j,ids]=1
    pi=q@incidence;pij=incidence.T@(q[:,None]*incidence)
    assert np.isclose(q.sum(),1) and np.all(q>=0)
    assert np.allclose(pi,selected.inclusion_probabilities,rtol=0,atol=1e-12)
    assert np.allclose(pij.diagonal(),pi,rtol=0,atol=1e-12)
    assert np.array_equal(pi>0,s.cost<=s.budget+1e-10)
    return atoms,q,incidence,pi,pij


def crossfit_residuals(s,folds):
    prediction=np.zeros(len(s.visible_y));splitter=StratifiedKFold(folds,shuffle=True,random_state=s.seed+917)
    for train,validation in splitter.split(s.visible_x,s.visible_y):
        theta=frozen.fit_logistic(s.visible_x[train],s.visible_y[train],s.ridge)
        prediction[validation]=expit(s.visible_x[validation]@theta)
    residual=s.visible_y-prediction
    return {'folds':folds,'visible_n':len(residual),'oof_brier':float(np.mean(residual**2)),
            'oof_residual_mean':float(residual.mean()),'oof_abs_residual_q95':float(np.quantile(abs(residual),.95)),
            'scope':'historically observed D only; cannot validate residuals in unreachable population'},prediction


def offline_variance(s,atoms,q,incidence,pi,pij,g,y,alphas):
    eligible=np.flatnonzero(pi>0);p=pi[eligible];gg=g[eligible];yy=y[eligible];a=incidence[:,eligible]
    kernel=pij[np.ix_(eligible,eligible)]/(p[:,None]*p[None,:])-1
    r=(1-2*gg)*(yy-gg);factor=r/len(eligible)
    full_var=float(factor@kernel@factor);diag_var=float(np.sum(factor**2*np.diag(kernel)))
    records=[];max_error=0.
    for alpha in alphas:
        targets=gg[None,:]+alpha*a/p[None,:]*(yy-gg)[None,:]
        expectation=q@targets;mean_error=(1-alpha)*(gg-yy)
        target_var=q@((targets-expectation)**2)
        theory_var=alpha**2*(1/p-1)*(yy-gg)**2
        score=np.mean(gg[None,:]**2+(1-2*gg)[None,:]*targets,axis=1)
        mean=float(q@score);var=float(q@((score-mean)**2));theory=alpha**2*full_var
        truth=float(np.mean((gg-yy)**2))
        bias=float((1-alpha)*np.mean((1-2*gg)*(gg-yy)))
        error=max(float(np.max(abs(expectation-yy-mean_error))),float(np.max(abs(target_var-theory_var))),abs(var-theory),abs(mean-truth-bias))
        max_error=max(max_error,error)
        assert np.allclose(expectation-yy,mean_error,rtol=1e-9,atol=1e-10)
        assert np.allclose(target_var,theory_var,rtol=1e-9,atol=1e-10)
        assert np.isclose(var,theory,rtol=1e-9,atol=1e-10)
        assert abs(mean-truth-bias)<1e-10
        records.append({'alpha':alpha,'mean_target_bias_squared':float(np.mean(mean_error**2)),
          'mean_target_variance':float(np.mean(theory_var)),'mean_target_mse':float(np.mean(mean_error**2+theory_var)),
          'fixed_g_brier_design_variance':var,'full_pi_ij_variance':theory,
          'diagonal_only_proxy':alpha**2*diag_var,'fixed_g_brier_estimator_bias':bias,
          'scope':'offline hidden-residual audit on positive-pi current candidates; not model generalization variance'})
    return records,max_error


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base
    out=base/'e3';out.mkdir(exist_ok=False);private=base/'private/e3';private.mkdir(parents=True,exist_ok=True)
    identity=json.loads((base/'source_identity.json').read_text());config=identity['configuration'];verify_identity(identity,run_identity(config))
    stage=json.loads((HERE/'e3_config.json').read_text());params={**config['xgb'],'objective':stage['objective'],'reg_lambda':1.0}
    code_hash=digest({x:sha(HERE/x) for x in ['run_e3.py','e3_config.json','runtime.py','run_e1b.py']})
    write_json(out/'activation.json',{'configuration':stage,'parent_identity':identity['identity_hash'],'source_hash':code_hash,
       'source_files':{x:sha(HERE/x) for x in ['run_e3.py','e3_config.json','runtime.py','run_e1b.py']},'regressor_parameters':params,
       'scope':'all six round-5 anchors; three predeclared draws; no alpha tuning; no multi-round correction expansion'})
    outcomes=[];resources=[];diagnostics=[];states=[];audit=[];oracle_jobs=[];cache={};fits=0;hits=0;crossfit_fits=0
    max_variance_error=0.
    for dataset in config['datasets']:
        for seed in config['seeds']:
            c=build_context(dataset,seed,stage['anchor_round'],config,private);s=c.acquisition
            atoms,q,incidence,pi,pij=finite_design(s);eligible=np.flatnonzero(pi>0);g=expit(s.candidate_x@s.theta)
            rngseed=stage['draw_seed_offset']+s.seed
            components=np.random.default_rng(rngseed).choice(len(q),size=stage['acquisition_repeats'],p=q)
            actions=[]
            for draw,component in enumerate(components):
                ids=atoms[int(component)];token=c.feedback.lock(ids,f'E3_draw{draw}',rngseed)
                actions.append({'draw':draw,'component':int(component),'selected':ids,'token':token})
            design_hash=digest({'q':q,'incidence':digest(incidence),'pi':pi,'pij':digest(pij)})
            write_json(private/'designs'/f'{c.state_id}.json',{'atoms':atoms,'q':q,'pi':pi,'pij':pij,'g':g,
                        'C_ids':c.feedback.c_ids,'actions':actions,'design_hash':design_hash,'all_locked_before_reveal':True})
            residual_info,oof=crossfit_residuals(s,stage['historical_residual_crossfit_folds']);crossfit_fits+=residual_info['folds']
            np.savez_compressed(private/f'{c.state_id}_visible_oof.npz',ids=c.feedback.d_ids,y=s.visible_y,prediction=oof)
            vs,ve,xs,xe,ys,ye=validation_blocks(c,config)
            np.savez_compressed(private/f'{c.state_id}_validation_ids.npz',search_ids=vs,eval_ids=ve)
            state={**c.manifest,'reachable_n':len(eligible),'unreachable_n':len(pi)-len(eligible),
                'reachable_candidate_fraction':float(len(eligible)/len(pi)),'current_candidate_group':'initial-price rejected group',
                'target_coverage_scope':'positive-pi fraction of current rejected candidates, not all future or outer-test people',
                'mixture_components':len(q),'unique_drawn_sets':len({tuple(sorted(a['selected'])) for a in actions}),
                'draws':len(actions),'design_hash':design_hash,'g_hash':digest(g),'pi_hash':digest(pi),
                'pi_min_positive':float(pi[eligible].min()),'inverse_pi_max':float((1/pi[eligible]).max()),
                'inverse_pi_q50':float(np.quantile(1/pi[eligible],.5)),'inverse_pi_q95':float(np.quantile(1/pi[eligible],.95)),
                'V_eval_n':len(ve),'V_eval_hash':digest(ve),'V_search_not_used':True,**residual_info}
            states.append(state)
            for action in actions:
                selected=action['selected'];ids,xx,yy=c.feedback.reveal(selected,action['token'])
                visible_y={int(i):float(v) for i,v in zip(ids,yy)}
                base_ids=np.r_[c.feedback.d_ids,c.feedback.o_ids]
                expanded_ids=np.r_[base_ids,c.feedback.c_ids[eligible]]
                expanded_x=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[eligible]])
                by=np.asarray([visible_y[int(i)] for i in base_ids])
                selected_y=np.asarray([visible_y[int(i)] for i in c.feedback.c_ids[selected]])
                selected_hash=digest(np.sort(c.feedback.c_ids[selected]));w=1/pi[selected]
                ess=float(w.sum()**2/(w@w)) if len(w) else 0.
                total_n=len(base_ids);total_ess=float((total_n+w.sum())**2/(total_n+w@w))
                resource={'state_id':c.state_id,'dataset':dataset,'seed':seed,'draw':action['draw'],
                    'selected_hash':selected_hash,'count':len(selected),'cost':float(s.cost[selected].sum()),'budget':s.budget,
                    'round_correction_ESS':ess,'single_fork_cumulative_correction_ESS':ess,
                    'historical_multi_round_correction_ESS':None,'observed_plus_current_ipw_ESS':total_ess,
                    'ESS_scope':'unit weights on frozen D/O plus current selected inverse-pi weights; no historical randomized correction trajectory',
                    'selected_weight_max':float(w.max()) if len(w) else 0.,'selected_weight_q95':float(np.quantile(w,.95)) if len(w) else 0.,
                    'historical_oof_variance_proxy':float(state['oof_brier']*np.sum(1/pi[eligible]-1))}
                resources.append(resource)
                for branch,alpha in stage['branches'].items():
                    start=time.perf_counter()
                    if branch=='Observed':train_ids,train_x,train_y=ids,xx,yy.astype(float);pseudo=None
                    else:
                        pseudo=g.copy();pseudo[selected]+=alpha*(selected_y-g[selected])/pi[selected]
                        train_ids,train_x,train_y=expanded_ids,expanded_x,np.r_[by,pseudo[eligible]]
                    order=np.argsort(train_ids,kind='stable');train_ids=train_ids[order];train_x=train_x[order];train_y=train_y[order]
                    assert len(train_ids)==len(np.unique(train_ids)) and np.isfinite(train_y).all()
                    key=digest({'dataset':dataset,'ids':digest(train_ids),'x':digest(train_x),'y':digest(train_y),
                                'params':params,'fit_seed':seed*1000+c.round,'code':code_hash})
                    reused=key in cache
                    if reused:result=cache[key];hits+=1
                    else:
                        model=XGBRegressor(**params,random_state=seed*1000+c.round)
                        model.fit(train_x[:,1:].astype(np.float32),train_y)
                        actual=json.loads(model.get_booster().save_config())['learner'];backend=actual['generic_param']['device']
                        assert backend.startswith('cuda'),backend
                        rawv=model.predict(xe[:,1:].astype(np.float32));rawt=model.predict(c.evaluation._x[:,1:].astype(np.float32))
                        pv=np.clip(rawv,0,1);pt=np.clip(rawt,0,1)
                        result={'V_eval_brier':float(np.mean((pv-ye)**2)),**c.evaluation.evaluate(pt),'actual_backend':backend,
                           'prediction_clip_fraction_V_eval':float(np.mean((rawv<0)|(rawv>1))),
                           'prediction_clip_fraction_T':float(np.mean((rawt<0)|(rawt>1)))}
                        cache[key]=result;fits+=1
                        dest=private/'predictions'/f'{key}.npz';dest.parent.mkdir(exist_ok=True)
                        np.savez_compressed(dest,eval_ids=ve,eval_y=ye,eval_prediction=pv,eval_raw_prediction=rawv,
                          test_ids=c.evaluation._ids,test_y=c.evaluation._y,test_prediction=pt,test_raw_prediction=rawt,
                          train_ids=train_ids,train_targets=train_y,selected_ids=c.feedback.c_ids[selected])
                    gradient=2*(expit(train_x@s.theta)-train_y)
                    diag={'state_id':c.state_id,'dataset':dataset,'seed':seed,'draw':action['draw'],'branch':branch,'alpha':alpha,
                          'train_target_min':float(train_y.min()),'train_target_max':float(train_y.max()),
                          'train_target_outside_fraction':float(np.mean((train_y<0)|(train_y>1))),
                          'candidate_pseudo_outside_fraction':None if pseudo is None else float(np.mean((pseudo[eligible]<0)|(pseudo[eligible]>1))),
                          'gradient_min_at_frozen_g':float(gradient.min()),'gradient_max_at_frozen_g':float(gradient.max()),
                          'gradient_abs_q99_at_frozen_g':float(np.quantile(abs(gradient),.99)),
                          'gradient_scope':'derivative of squared loss at the frozen D-only g, not a traced boosting gradient'}
                    diagnostics.append(diag)
                    outcomes.append({'state_id':c.state_id,'dataset':dataset,'seed':seed,'round':c.round,'draw':action['draw'],
                       'branch':branch,'alpha':alpha,'selected_hash':selected_hash,'design_hash':design_hash,'pi_hash':digest(pi),'g_hash':digest(g),
                       'learner':'XGBRegressor','train_n':len(train_y),'training_ids_hash':digest(train_ids),'training_targets_hash':digest(train_y),
                       'fit_key':key,'cache_hit':reused,'elapsed_seconds':time.perf_counter()-start,**result})
            oracle_jobs.append((c,s,atoms,q,incidence,pi,pij,g))
            print('E3',c.state_id,'reachable',len(eligible),'/',len(pi),'unique draws',state['unique_drawn_sets'],'new fits',fits,'reuse',hits,flush=True)
        # No diagnostic outcome changes any later state, atom, draw or alpha.
    # All branch fitting is finished before hidden current-population residual audit.
    for c,s,atoms,q,incidence,pi,pij,g in oracle_jobs:
        rows,error=offline_variance(s,atoms,q,incidence,pi,pij,g,c.feedback._y[c.feedback.c_ids],[0.,.25,.5,1.])
        max_variance_error=max(max_variance_error,error)
        audit.extend({'state_id':c.state_id,'dataset':c.dataset,'seed':c.seed,**r} for r in rows)
    write_jsonl(out/'state_manifest.jsonl',states);write_jsonl(out/'resources.jsonl',resources)
    write_jsonl(out/'target_diagnostics.jsonl',diagnostics);write_jsonl(out/'branch_outcomes.jsonl',outcomes)
    write_jsonl(out/'offline_design_variance.jsonl',audit)
    checks=[]
    for row in outcomes:
        p=np.load(private/'predictions'/f"{row['fit_key']}.npz")
        assert abs(float(np.mean((p['eval_prediction']-p['eval_y'])**2))-row['V_eval_brier'])<1e-12
        assert abs(float(np.mean((p['test_prediction']-p['test_y'])**2))-row['brier'])<1e-12
        assert row['actual_backend'].startswith('cuda')
        assert digest(p['train_targets'])==row['training_targets_hash']
    for sid in {r['state_id'] for r in outcomes}:
        group=[r for r in outcomes if r['state_id']==sid]
        assert len({r['design_hash'] for r in group})==len({r['pi_hash'] for r in group})==len({r['g_hash'] for r in group})==1
        for draw in range(stage['acquisition_repeats']):
            gr=[r for r in group if r['draw']==draw]
            assert len(gr)==5 and len({r['selected_hash'] for r in gr})==1
            assert len({r['training_ids_hash'] for r in gr if r['branch']!='Observed'})==1
    assert all(r['cost']<=r['budget']+1e-7 for r in resources)
    write_json(out/'verification.json',{'passed':True,'max_exact_design_formula_error':max_variance_error,
       'checks':['frozen parent identity','pi and all pi_ij from full finite mixture','all atoms risk feasible',
       'all draws locked before labels','same selection/pi/g/learner across target branches','no candidate labels outside S used in training',
       'crossfit historical residuals on visible D','V_eval and outer T disjoint from training',
       'actual CUDA regressor backend','saved prediction Brier recomputation','exact target bias/variance and full design Brier variance'],
       'historical_multi_round_CORR_replayed':False})
    completion={'completed':True,'states':len(states),'acquisition_draws':len(resources),'nominal_fit_slots':len(outcomes),
       'actual_unique_regressor_fits':fits,'fit_cache_hits':hits,'historical_crossfit_LR_fits':crossfit_fits,
       'independent_trajectory_clusters':6,'development_seed_values':2,'draws_per_state':stage['acquisition_repeats'],
       'unique_sampled_sets':sum(s['unique_drawn_sets'] for s in states),'confirmation':False,'scale_up':False}
    write_json(out/'completion.json',completion);print(json.dumps(completion),flush=True)


if __name__=='__main__':main()
