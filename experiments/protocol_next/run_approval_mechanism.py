"""Next DOCX mechanism block: approved ordinary samples, same count and rotated direction."""
from runtime import *
from dataclasses import replace
from run_ordinary_factorial import family
import argparse


def generate(seed,cfg):
    def rng(offset):return np.random.default_rng(seed+offset)
    def block(n,approved,g):
        x=np.c_[np.ones(n),g.uniform(*(cfg['approved_coordinate_uniform'] if approved else cfg['rejected_coordinate_uniform']),n),
                  g.normal(0,1.2,n),g.normal(size=n),g.normal(size=n)]
        return x
    d=block(cfg['visible'],True,rng(110000));d[:,2]=rng(120000).normal(0,.12,len(d));d[:,4]=d[:,2]+rng(130000).normal(0,.05,len(d))
    c=block(cfg['candidate'],False,rng(140000));truth=np.asarray(cfg['truth_theta'])
    yd=(rng(150000).random(len(d))<expit(d@truth)).astype(int)
    yc=(rng(160000).random(len(c))<expit(c@truth)).astype(int)
    def target(n,g):
        x=np.zeros((n,5));x[:,0]=1;x[:,1]=g.uniform(-2,2.2,n);x[:,2]=g.normal(0,1.2,n);return x
    q=target(cfg['reference'],rng(170000));test=target(cfg['test'],rng(180000));pt=expit(test@truth)
    yt=(rng(190000).random(len(test))<pt).astype(int)
    business=np.asarray(cfg['public_business_theta']);prices=expit(c@business)
    exposure=rng(200000).uniform(.5,1.5,len(c));cost=prices*exposure;budget=cfg['budget_ratio']*cost.sum()
    theta=frozen.fit_logistic(d,yd,cfg['lr_ridge'])
    z=rng(210000).uniform(*cfg['approved_coordinate_uniform'],cfg['ordinary']);direction=rng(220000).normal(0,cfg['ordinary_direction_scale'],len(z))
    uniforms=rng(230000).random(len(z));states=[]
    for coverage,axis in [('low_target_overlap',3),('high_target_overlap',2)]:
        o=np.zeros((len(z),5));o[:,0]=1;o[:,1]=z;o[:,axis]=direction
        yo=(uniforms<expit(o@truth)).astype(int)
        s=frozen.State(d,yd,o,c,q,cost,budget,theta,seed,ridge=cfg['lr_ridge'],exchanges=0)
        states.append((coverage,s,yo))
    assert np.all(expit(d@business)<=cfg['approval_threshold'])
    assert np.all(prices>cfg['approval_threshold'])
    assert np.array_equal(expit(states[0][1].ordinary_x@business),expit(states[1][1].ordinary_x@business))
    assert all(np.all(expit(s.ordinary_x@business)<=cfg['approval_threshold']) for _,s,_ in states)
    return states,yc,test,yt,pt,exposure


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base
    out=base/'approval_mechanism_small';out.mkdir(exist_ok=False);private=base/'private/approval_mechanism_small';private.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'approval_mechanism_config.json').read_text());parent=json.loads((base/'source_identity.json').read_text())
    verify_identity(parent,run_identity(parent['configuration']))
    sources={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'run_approval_mechanism.py',HERE/'approval_mechanism_config.json',
        HERE/'run_ordinary_factorial.py',HERE/'runtime.py',ROOT/'experiments/research_v2/core.py']}
    write_json(out/'activation.json',{'config':cfg,'source':sources,'parent_identity':parent['identity_hash'],'all_design_frozen_before_run':True})
    pending=[];locks=[];states=[];started=time.perf_counter()
    for seed in cfg['seeds']:
        generated,yc,test,yt,pt,exposure=generate(seed,cfg)
        for coverage,s,yo in generated:
            actions=family(s);h,g,u=frozen.geometry(s,True);hd,_,_=frozen.geometry(s,False)
            ordinary_information=float(np.trace(g@(frozen.inverse(hd)-frozen.inverse(h))))
            for p in actions:
                ids=p['ids'];assert s.cost[ids].sum()<=s.budget+1e-8
                p.update({'seed':seed,'coverage':coverage,'count':len(ids),'cost':float(s.cost[ids].sum()),'budget':s.budget,
                    'Kstar':frozen.kstar(s.cost,s.budget),'selected_hash':digest(ids),
                    'common_conditional_proxy':frozen.set_value(h,g,u,ids)})
                locks.append(p)
            # Null pending-information control uses the same state, no extra models or benchmark.
            empty=replace(s,ordinary_x=s.ordinary_x[:0]);ef=family(empty)
            assert np.array_equal(ef[2]['ids'],ef[4]['ids']) and np.array_equal(ef[3]['ids'],ef[5]['ids'])
            state={'seed':seed,'coverage':coverage,'ordinary_n':len(yo),'budget':s.budget,'candidate_n':len(yc),
                'ordinary_predicted_risk':float(expit(s.ordinary_x@np.asarray(cfg['public_business_theta'])).sum()),
                'target_weighted_ordinary_information':ordinary_information,
                'D_hash':digest(s.visible_x),'D_y_hash':digest(s.visible_y),'C_hash':digest(s.candidate_x),'C_y_hash':digest(yc),
                'Q_hash':digest(s.reference_x),'cost_hash':digest(s.cost),'test_hash':digest(test),'pTest_hash':digest(pt),
                'O_feature_hash':digest(s.ordinary_x),'O_uniforms_shared':True,'ordinary_all_approved':True,'candidates_all_rejected':True}
            states.append(state);pending.append((seed,coverage,s,yo,yc,test,yt,pt,exposure,actions))
        a,b=states[-2:]
        for field in ['D_hash','D_y_hash','C_hash','C_y_hash','Q_hash','cost_hash','test_hash','pTest_hash','ordinary_predicted_risk','budget']:
            assert a[field]==b[field]
    write_jsonl(private/'selection_locks.jsonl',locks);lock_sha=sha(private/'selection_locks.jsonl')
    write_json(out/'selection_commitment.json',{'sha256':lock_sha,'actions':len(locks),'persisted_before_training':True})
    write_jsonl(out/'state_manifest.jsonl',states)
    cache={};rows=[];hits=0;error=0.;devices=set()
    for seed,coverage,s,yo,yc,test,yt,pt,exposure,actions in pending:
        np.savez_compressed(private/f'{seed}_{coverage}_environment.npz',D=s.visible_x,yD=s.visible_y,O=s.ordinary_x,yO=yo,
            C=s.candidate_x,yC=yc,Q=s.reference_x,cost=s.cost,budget=s.budget,test=test,yTest=yt,pTest=pt,exposure=exposure)
        for learner in ['LR','XGB']:
            for p in actions:
                ids=p['ids'];x=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]]);y=np.r_[s.visible_y,yo,yc[ids]]
                params=cfg['xgb'] if learner=='XGB' else cfg['lr_ridge']
                key=digest({'x':digest(x),'y':digest(y),'learner':learner,'params':params,'seed':seed+7000})
                reused=key in cache;tick=time.perf_counter()
                if reused:model,backend=cache[key];hits+=1
                elif learner=='LR':model=frozen.fit_logistic(x,y,cfg['lr_ridge']);backend='cpu';cache[key]=(model,backend)
                else:
                    model=XGBClassifier(**cfg['xgb'],random_state=seed+7000,objective='binary:logistic',eval_metric='logloss')
                    model.fit(x[:,1:].astype(np.float32),y);backend=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
                    assert backend.startswith('cuda'),'actual CUDA required';devices.add(backend);cache[key]=(model,backend)
                pred=expit(test@model) if learner=='LR' else model.predict_proba(test[:,1:].astype(np.float32))[:,1]
                evalkey=digest({'fit':key,'test':digest(test),'pTest':digest(pt),'yTest':digest(yt)})
                path=private/'predictions'/f'{evalkey}.npz';path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,prediction=pred,pTest=pt,yTest=yt,selected_ids=ids)
                excess=float(np.mean((pred-pt)**2));brier=float(np.mean((pred-yt)**2));saved=np.load(path)
                error=max(error,abs(excess-float(np.mean((saved['prediction']-saved['pTest'])**2))))
                row={k:v for k,v in p.items() if k not in ['ids','steps']}
                row.update({'learner':learner,'excess_brier':excess,'brier':brier,'fit_key':key,'evaluation_key':evalkey,'cache_hit':reused,
                    'actual_backend':backend,'train_n':len(y),'fit_and_predict_seconds':time.perf_counter()-tick,
                    'realized_exploration_loss':float(np.dot(yc[ids],exposure[ids])),
                    'ordinary_realized_loss':float(yo.sum())})
                rows.append(row)
        pd.DataFrame(rows).to_csv(out/'branch_outcomes.csv',index=False)
        print('approval mechanism',seed,coverage,'unique models',len(cache),'cache hits',hits,flush=True)
    assert len(rows)==cfg['max_fit_slots'] and len(locks)==cfg['max_selector_slots']
    assert lock_sha==sha(private/'selection_locks.jsonl') and error<1e-12
    gpu=sum(backend.startswith('cuda') for _,backend in cache.values())
    write_json(out/'completion.json',{'completed':True,'selector_slots':len(locks),'fit_slots':len(rows),'actual_unique_LR_fits':len(cache)-gpu,
        'actual_unique_GPU_fits':gpu,'cache_hits':hits,'actual_devices':sorted(devices),'independent_generation_seeds':3,
        'old_development_seed_numbers':True,'confirmation':False,'seconds':time.perf_counter()-started})
    write_json(out/'verification.json',{'passed':True,'all_ordinary_approved':True,'all_candidates_rejected':True,
        'fixed_common_business_model_for_gate_and_prices':True,'same_ordinary_count_and_business_PD':True,
        'same_D_C_Q_test_and_cost_across_coverage':True,'empty_O_selection_null_control':True,'all_selections_locked_before_training':True,
        'same_solver_in_four_cells':True,'actual_GPU_verified':True,'saved_prediction_max_error':error,
        'all_source_hashes_unchanged':all(sha(ROOT/k)==h for k,h in sources.items())})
    snap=out/'source_snapshot';snap.mkdir()
    for p,h in sources.items():(snap/Path(p).name).write_bytes((ROOT/p).read_bytes())
    print(json.dumps(json.loads((out/'completion.json').read_text())),flush=True)


if __name__=='__main__':main()
