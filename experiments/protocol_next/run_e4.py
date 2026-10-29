"""E4 minimal paired GATE mechanism; actual release is an environment control.

Three underlying synthetic states, a shared finite family, 16 selection worlds,
independent 8-world model/true checks. Matched LR is the mechanism endpoint;
actual GPU XGBoost is a secondary transfer check on identical chosen actions.
"""
from __future__ import annotations
import argparse,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from runtime import *
from gate_protocol import FutureProtocol
import run_mechanisms as previous


def proposals(s):
    raw=[('OrdinaryOnly',np.empty(0,int)),('MinRisk',frozen.select('MinRisk',s).selected_ids),
         ('Vopt',frozen.select('Vopt',s).selected_ids),('RB-BRIDGE',frozen.select('RB-BRIDGE',s).selected_ids),
         ('one_fixed_random_feasible',frozen.random_fill(s.cost,s.budget,np.random.default_rng(s.seed+391)))]
    out=[];mapping={}
    for source,ids in raw:
        ids=np.sort(np.asarray(ids,int));key=tuple(ids.tolist())
        if key in mapping:out[mapping[key]]['sources'].append(source)
        else:mapping[key]=len(out);out.append({'ids':ids,'sources':[source]})
    for p in out:assert s.cost[p['ids']].sum()<=s.budget+1e-7
    return out


def world_grid(s,actions,future,prices,uniforms,theta_worlds,truth,oracle,protocol):
    n=len(uniforms);m=len(actions);scores=np.zeros((n,m,3));ref=s.reference_x
    for w in range(n):
        theta=theta_worlds[w]
        probability=lambda x:truth(x) if oracle else expit(x@theta)
        po,pc,pf=probability(s.ordinary_x),probability(s.candidate_x),probability(future)
        yo=(uniforms[w,:len(po)]<po).astype(int)
        yc=(uniforms[w,len(po):len(po)+len(pc)]<pc).astype(int)
        yf=(uniforms[w,len(po)+len(pc):]<pf).astype(int)
        target=probability(ref)
        base=frozen.fit_logistic(np.vstack([s.visible_x,s.ordinary_x]),np.r_[s.visible_y,yo],s.ridge)
        fixed=protocol.approvals(expit(future@base),prices)
        for j,p in enumerate(actions):
            ids=p['ids'];xx=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]]);yy=np.r_[s.visible_y,yo,yc[ids]]
            direct=frozen.fit_logistic(xx,yy,s.ridge)
            dynamic=protocol.approvals(expit(future@direct),prices)
            scores[w,j,0]=np.mean((expit(ref@direct)-target)**2)
            for col,approved in [(1,dynamic),(2,fixed)]:
                post=frozen.fit_logistic(np.vstack([xx,future[approved]]),np.r_[yy,yf[approved]],s.ridge)
                scores[w,j,col]=np.mean((expit(ref@post)-target)**2)
    return scores


def actual_fit(cache,s,ids,x,y,learner,config,private,test,yt,pt,future):
    order=np.argsort(ids,kind='stable');ids=ids[order];x=x[order];y=y[order]
    key=digest({'seed':s.seed,'ids':ids,'x':digest(x),'y':y,'learner':learner,'params':config['xgb'] if learner=='XGB' else s.ridge})
    if key in cache:return cache[key],True
    if learner=='LR':
        model=frozen.fit_logistic(x,y,s.ridge);backend='cpu';predict=lambda z:expit(z@model)
    else:
        model=XGBClassifier(**config['xgb'],random_state=s.seed+7000,objective='binary:logistic',eval_metric='logloss')
        model.fit(x[:,1:].astype(np.float32),y);backend=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        assert backend.startswith('cuda');predict=lambda z:model.predict_proba(z[:,1:].astype(np.float32))[:,1]
    prediction=predict(test);fp=predict(future)
    result={'fit_key':key,'backend':backend,'prediction':prediction,'future_probability':fp,
            'brier':float(np.mean((prediction-yt)**2)),'excess_brier':float(np.mean((prediction-pt)**2)),
            'train_n':len(y),'train_hash':digest(ids)};cache[key]=result
    path=private/'actual_predictions'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,train_ids=ids,train_y=y,test_prediction=prediction,test_y=yt,test_true_probability=pt,future_prediction=fp)
    return result,False


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base
    out=base/'e4';out.mkdir(exist_ok=False);private=base/'private/e4';private.mkdir(parents=True,exist_ok=True)
    identity=json.loads((base/'source_identity.json').read_text());config=identity['configuration'];verify_identity(identity,run_identity(config))
    stage=json.loads((HERE/'e4_config.json').read_text());protocol=FutureProtocol(future_batch_size=stage['future_batch_size'],capacity=stage['capacity'],
        risk_cap=stage['risk_cap'],threshold=stage['threshold'],selection_worlds=stage['selection_worlds'],verification_worlds=stage['model_verification_worlds'])
    source={x:sha(HERE/x) for x in ['run_e4.py','e4_config.json','runtime.py','gate_protocol.py']}
    write_json(out/'activation.json',{'configuration':stage,'source':source,'source_hash':digest(source),'parent_identity':identity['identity_hash'],
        'generator_source':sha(ROOT/'experiments/research_v2/run_mechanisms.py'),'planning_and_environment_common_parameters':protocol.environment_parameters()})
    records=[];ranks=[];world_pairs=[];states=[];cache={};hits=0;logical_locks=[];internal_refits=0
    for seed in stage['seeds']:
        s,yo,yc,test,yt,pt,truth=previous.generate('C1',seed);actions=proposals(s)
        rng=np.random.default_rng(seed+41700)
        history=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x]);future=history[rng.integers(0,len(history),protocol.future_batch_size)]
        prices=expit(future@s.theta);width=len(yo)+len(yc)+len(future)
        # Disjoint RNG seeds register independent selection/model-check/true-check worlds.
        us=np.random.default_rng(seed+51000).random((stage['selection_worlds'],width))
        uv=np.random.default_rng(seed+61000).random((stage['model_verification_worlds'],width))
        ut=np.random.default_rng(seed+71000).random((stage['true_generator_verification_worlds'],width))
        h,_,_=frozen.geometry(s,False);cov=frozen.inverse(h)
        ts=np.random.default_rng(seed+81000).multivariate_normal(s.theta,cov,len(us))
        tv=np.random.default_rng(seed+91000).multivariate_normal(s.theta,cov,len(uv))
        true_grid=world_grid(s,actions,future,prices,ut,np.tile(s.theta,(len(ut),1)),truth,True,protocol)
        internal_refits+=len(ut)*(1+3*len(actions))
        plans={};scorefiles={}
        for planner in stage['planners']:
            oracle=planner=='oracle_true_generator'
            selection=world_grid(s,actions,future,prices,us,ts,truth,oracle,protocol)
            validation=true_grid if oracle else world_grid(s,actions,future,prices,uv,tv,truth,False,protocol)
            internal_refits+=len(us)*(1+3*len(actions))+(0 if oracle else len(uv)*(1+3*len(actions)))
            mean=selection.mean(axis=0);vm=validation.mean(axis=0);tm=true_grid.mean(axis=0)
            indices={'H1_release_dynamic':int(np.argmin(mean[:,0])),'H2_release_dynamic':int(np.argmin(mean[:,1])),
                     'H2_release_fixed':int(np.argmin(mean[:,2]))}
            indices['H2_no_release_dynamic']=indices['H2_release_dynamic'];indices['H2_fixed_matched_action']=indices['H2_release_dynamic']
            plans[planner]=indices
            for condition,col in [('H1_release_dynamic',0),('H2_release_dynamic',1),('H2_release_fixed',2)]:
                chosen=indices[condition];rho=spearmanr(vm[:,col],tm[:,col]).statistic if len(actions)>1 else None
                ranks.append({'seed':seed,'planner':planner,'condition':condition,'chosen_proposal':chosen,'candidate_family_n':len(actions),
                    'selection_risk':float(mean[chosen,col]),'model_verification_risk':float(vm[chosen,col]),
                    'model_verification_family_regret':float(vm[chosen,col]-vm[:,col].min()),
                    'independent_world_ranking_disagrees':int(chosen!=np.argmin(vm[:,col])),
                    'true_verification_risk':float(tm[chosen,col]),'true_verification_family_regret':float(tm[chosen,col]-tm[:,col].min()),
                    'model_true_verification_rank_correlation':None if rho is None or not np.isfinite(rho) else float(rho),
                    'scope':'finite-world/family diagnosis only, not an upper bound; no reselection on verification'})
            # Evaluate chosen H1 and H2 actions under the SAME true H2 release environment.
            one,two=indices['H1_release_dynamic'],indices['H2_release_dynamic']
            world_pairs.append({'seed':seed,'planner':planner,'horizon_gain_true_verification':float(tm[one,1]-tm[two,1]),
                'different_horizon_actions':bool(one!=two),'H1_true_H2_risk':float(tm[one,1]),'H2_true_H2_risk':float(tm[two,1]),
                'finite_family_room_after_H1':float(tm[one,1]-tm[:,1].min()),'verification_worlds':len(ut)})
            scorefiles[planner]={'selection':selection,'model_validation':validation}
        # Lock all candidate actions/planner choices before revealing actual O/C/future labels.
        lock={'seed':seed,'candidate_family':[{'ids':p['ids'],'sources':p['sources']} for p in actions],'plans':plans,
              'future_feature_hash':digest(future),'actual_hidden_labels_in_planner_API':False,'all_choices_locked_before_actual_release':True}
        write_json(private/'selection_locks'/f'seed{seed}.json',lock);logical_locks.append(lock)
        yf=np.random.default_rng(seed+101000).binomial(1,truth(future))
        np.savez_compressed(private/f'seed{seed}_worlds.npz',selection_uniforms=us,model_verification_uniforms=uv,true_verification_uniforms=ut,
             selection_theta=ts,model_verification_theta=tv,true_grid=true_grid,
             **{planner+'_'+kind:value for planner,values in scorefiles.items() for kind,value in values.items()})
        write_json(private/f'seed{seed}_actual_environment.json',{'future_x':future,'future_y':yf,'future_price':prices,'ordinary_y':yo,'candidate_y':yc})
        nd,no,nc=len(s.visible_y),len(yo),len(yc);dids=np.arange(nd);oids=np.arange(nd,nd+no);cids=np.arange(nd+no,nd+no+nc);fids=np.arange(nd+no+nc,nd+no+nc+len(future))
        state={'seed':seed,'underlying_environment':'C1; C2 is the paired fixed-approval control','candidate_family_n':len(actions),
               'candidate_sources':[p['sources'] for p in actions],'planning_size':len(future),'evaluation_size':len(future),
               'future_capacity':protocol.capacity,'future_risk_cap':protocol.risk_cap,'future_threshold':protocol.threshold,
               'selection_worlds':len(us),'model_verification_worlds':len(uv),'true_verification_worlds':len(ut)};states.append(state)
        for learner in ['LR','XGB']:
            base_ids=np.r_[dids,oids];base_x=np.vstack([s.visible_x,s.ordinary_x]);base_y=np.r_[s.visible_y,yo]
            ordinary,reused=actual_fit(cache,s,base_ids,base_x,base_y,learner,config,private,test,yt,pt,future);hits+=reused
            fixed=protocol.approvals(ordinary['future_probability'],prices)
            for planner,plan in plans.items():
                for condition,j in plan.items():
                    selected=actions[j]['ids'];ids=np.r_[base_ids,cids[selected]];xx=np.vstack([base_x,s.candidate_x[selected]]);yy=np.r_[base_y,yc[selected]]
                    direct,reused=actual_fit(cache,s,ids,xx,yy,learner,config,private,test,yt,pt,future);hits+=reused
                    is_fixed='fixed' in condition;release=condition!='H2_no_release_dynamic'
                    dynamic=protocol.approvals(direct['future_probability'],prices)
                    approved=fixed if is_fixed else dynamic
                    actual_protocol=replace(protocol,actual_future_release=release,future_approval='fixed' if is_fixed else 'dynamic',
                                            planning_horizon=1 if condition.startswith('H1_') else 2)
                    released_ids,released_y=actual_protocol.release(approved,yf)
                    if not release:
                        assert len(released_ids)==0;post=direct
                    else:
                        post,reused=actual_fit(cache,s,np.r_[ids,fids[released_ids]],np.vstack([xx,future[released_ids]]),np.r_[yy,released_y],
                            learner,config,private,test,yt,pt,future);hits+=reused
                    assert len(approved)<=protocol.capacity and prices[approved].sum()<=protocol.risk_cap+1e-10
                    current_ordinary=float(expit(s.ordinary_x@s.theta).sum());next_ordinary=float(prices[approved].sum());explore=float(s.cost[selected].sum())
                    records.append({'seed':seed,'planner':planner,'learner':learner,'condition':condition,'proposal_id':j,'selected_hash':digest(selected),
                        'selected_count':len(selected),'current_exploration_predicted_risk':explore,'current_exploration_budget':s.budget,
                        'current_ordinary_predicted_risk':current_ordinary,'future_ordinary_predicted_risk':next_ordinary,
                        'total_predicted_risk':current_ordinary+next_ordinary+explore,
                        'total_realized_loss':float(yo.sum()+yc[selected].sum()+yf[approved].sum()),
                        'total_true_expected_loss':float(truth(s.ordinary_x).sum()+truth(s.candidate_x)[selected].sum()+truth(future)[approved].sum()),
                        'future_approval_count':len(approved),'future_approval_hash':digest(np.sort(approved)),
                        'future_released_count':len(released_ids),'actual_future_release':release,
                        'future_set_change_vs_fixed':len(set(approved)^set(fixed)),
                        'brier':post['brier'],'excess_brier':post['excess_brier'],'direct_excess_brier':direct['excess_brier'],
                        'actual_backend':post['backend'],'direct_fit_key':direct['fit_key'],'post_fit_key':post['fit_key'],'post_train_n':post['train_n']})
        print('E4 seed',seed,'unique proposals',len(actions),'actual model fits',len(cache),'internal LR world refits',internal_refits,flush=True)
    # Invariants target the previously confounded mechanisms, not a benchmark grid.
    import pandas as pd
    frame=pd.DataFrame(records)
    for (seed,planner,learner),group in frame.groupby(['seed','planner','learner']):
        by=group.set_index('condition');h1=by.loc['H1_release_dynamic'];h2=by.loc['H2_release_dynamic'];nr=by.loc['H2_no_release_dynamic'];fx=by.loc['H2_fixed_matched_action']
        assert h1.actual_future_release and h2.actual_future_release
        assert h2.selected_hash==nr.selected_hash==fx.selected_hash
        assert nr.future_released_count==0 and nr.post_fit_key==nr.direct_fit_key
        assert h2.future_approval_hash==nr.future_approval_hash and h2.total_realized_loss==nr.total_realized_loss
        assert group[group.condition.str.contains('fixed')].future_approval_hash.nunique()==1
    for (seed,planner,condition),group in frame.groupby(['seed','planner','condition']):assert group.selected_hash.nunique()==1
    for key,result in cache.items():
        p=np.load(private/'actual_predictions'/f'{key}.npz')
        assert abs(np.mean((p['test_prediction']-p['test_y'])**2)-result['brier'])<1e-12
        assert abs(np.mean((p['test_prediction']-p['test_true_probability'])**2)-result['excess_brier'])<1e-12
    write_jsonl(out/'state_manifest.jsonl',states);write_jsonl(out/'actual_outcomes.jsonl',records)
    write_jsonl(out/'ranking_diagnostics.jsonl',ranks);write_jsonl(out/'independent_true_world_pairs.jsonl',world_pairs)
    write_json(out/'verification.json',{'passed':True,'checks':['same 160/80/20/0.5 planning/environment parameters',
       'shared deduplicated candidate family and common random uniforms','disjoint selection/model-verification/true-verification RNG streams',
       'no actual O/C/future label input in planner API','choices locked before actual future generation/release',
       'H1 and H2 actual environments both release future labels','no-release actually excludes future rows and preserves direct predictions',
       'fixed future approvals identical across actions within learner','matched-S feedback control','same actions for LR/GPU XGB',
       'all current/future risk budgets enforced','total ordinary plus exploration risk/loss reported','saved prediction errors recomputed'],
       'oracle_scope':'known synthetic label generator only, not available for real credit decisions'})
    completion={'completed':True,'independent_underlying_states':len(states),'development_seeds':len(stage['seeds']),
      'actual_policy_learner_slots':len(records),'actual_unique_LR_fits':sum(r['backend']=='cpu' for r in cache.values()),
      'actual_unique_GPU_XGB_fits':sum(r['backend'].startswith('cuda') for r in cache.values()),'actual_model_cache_hits':hits,
      'internal_planning_LR_refits':internal_refits,'internal_world_fits_are_not_independent_benchmarks':True,
      'confirmation':False,'real_data_expansion':False}
    write_json(out/'completion.json',completion);print(json.dumps(completion),flush=True)


if __name__=='__main__':main()
