"""Audit all retained round records and export MD section-11 tables, without refits."""
from runtime import *
import run_formal_md as engine
import argparse


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);a=ap.parse_args();base=a.base
    out=base/'formal_md';identity=json.loads((out/'frozen_identity.json').read_text());private=base/'private/formal_md/cases'
    state_rows=[];selections=[];branches=[];max_risk_error=0.;max_loss_error=0.;case_count=0
    resources={'retained_real_fit_calls':0,'retained_fantasy_fit_calls':0,'retained_committee_fit_calls':0,
        'D0_tuning_fit_calls':0,'initial_teacher_fit_calls':0,'retained_neighbor_LR_refits':0}
    for folder in sorted(private.iterdir()):
        if not (folder/'completion.json').exists():continue
        completed=json.loads((folder/'completion.json').read_text());assert completed['identity_hash']==identity['identity_hash'];case_count+=1
        manifest=json.loads((folder/'state_manifest.json').read_text());split=np.load(folder/'split.npz');tune=json.loads((folder/'initial_tuning.json').read_text())
        raw,y,exposure,_,_,groups,months=engine.dataset(manifest['dataset'])
        d0=split['D0'];test=split['outer_test'];reference=split['reference']
        assert not np.intersect1d(groups[split['outer_train']],groups[test]).size
        if months is not None:assert months[split['outer_train']].max()<months[test].min() and months[reference].max()<=manifest['time0_initial_feature_pool_max_month']
        prehash=digest({'data':manifest['source_hash'],'D0_ordered':digest(d0),'source':identity['sources']['experiments/research_v2/run_real.py'],'versions':identity['versions']})
        pricehash=digest({'preprocessor':prehash,'D0_ordered':digest(d0),'parameters':tune['xgb'],'fit_seed':manifest['seed']})
        ordinary=pd.read_csv(folder/(folder.name+'_rho0.1_OrdinaryOnly')/'trajectory.csv').set_index('round')
        resources['D0_tuning_fit_calls']+=tune['actual_tuning_GPU_fits'];resources['initial_teacher_fit_calls']+=1
        for ep in sorted(folder.iterdir()):
            if not ep.is_dir() or not (ep/'completion.json').exists():continue
            saved=json.loads((ep/'completion.json').read_text())['summary'];trajectory=pd.read_csv(ep/'trajectory.csv').set_index('round')
            mr=pd.read_csv(folder/(folder.name+f"_rho{saved['rho']:g}_MinRisk")/'trajectory.csv').set_index('round')
            visible=np.sort(d0);cum_cost=cum_loss=cum_oloss=0.
            for rid in range(1,11):
                p=ep/f'round_{rid:02d}_selection.json';lock=json.loads(p.read_text());r=trajectory.loc[rid]
                d,o,c,s,q=[np.asarray(lock[k],int) for k in ['D_ids','O_ids','C_ids','selected_ids','reference_ids']]
                costs=np.asarray(lock['costs'],float)
                assert np.array_equal(d,visible) and np.array_equal(q,reference)
                assert len(np.unique(s))==len(s) and np.isin(s,c).all()
                for left,right in [(d,o),(d,c),(o,c),(q,d),(q,o),(q,c),(test,np.r_[d,o,c,q])]:assert not np.intersect1d(left,right).size
                local=np.flatnonzero(np.isin(c,s));spent=float(costs[local].sum());actual_loss=float(exposure[s]@y[s]);oloss=float(exposure[o]@y[o])
                assert spent<=lock['B']+1e-7 and r['count']==len(s)
                max_risk_error=max(max_risk_error,abs(spent-r.risk));cum_cost+=spent;cum_loss+=actual_loss;cum_oloss+=oloss
                max_loss_error=max(max_loss_error,abs(cum_cost-r.cumulative_exploration_cost),abs(cum_loss-r.cumulative_exploration_loss),abs(cum_oloss-r.ordinary_loss))
                after=np.sort(np.r_[d,o,s]);assert len(np.unique(after))==len(after);visible=after
                state=digest({'D':digest(d),'O':digest(o),'C':digest(c),'Q':digest(q),'costs':digest(costs),'B':lock['B'],'preprocessor':prehash})
                s_hash=digest(np.sort(s));context={'state_id':state,'episode_id':saved['episode_id'],'dataset':manifest['dataset'],'seed':manifest['seed'],'round':rid,
                    'outer_fold':manifest['outer_fold'],'outer_repeat':manifest['outer_repeat'],'replay_stream':manifest['replay_stream'],'selector':saved['method'],'rho':saved['rho']}
                state_rows.append({**context,'input_hash':manifest['source_hash'],'protocol':'MD_full_process_outer_evaluation','anchor_policy':'policy-specific history',
                    'D_hash':digest(d),'O_hash':digest(o),'C_hash':digest(c),'reference_hash':digest(q),'preprocessor_hash':prehash,'price_hash':pricehash,
                    'cost_hash':digest(costs),'B':lock['B'],'label_release_state':'selection persisted before actual O/S reveal',
                    'code_hash':identity['identity_hash'],'config_hash':digest(identity['config'])})
                k=frozen.kstar(costs,lock['B']);minimum=float(np.sort(costs)[:len(s)].sum())
                selections.append({**context,'selected_hash':s_hash,'recorded_selected_hash':lock['selected_hash'],'count':len(s),'cost':spent,'realized_loss':actual_loss,
                    'Kstar':k,'slack':lock['B']-float(np.sort(costs)[:k].sum()),'composition_premium':spent-minimum,
                    'time':lock['selection_seconds'],'representation_hash':r.representation_hash,'proxy_value':None,
                    'proxy_status':'same-state mechanism values are in E1/E2 logs; not recorded by frozen full-trajectory runner'})
                branches.append({**context,'selected_hash':s_hash,'learner':'XGBClassifier','learner_config':tune['xgb'],'fit_seed':manifest['seed']+rid,
                    'Brier_base':None,'Brier_selected':float(r.Brier),'gain_vs_ordinary':None,'gain_vs_minrisk':None,
                    'Brier_ordinary_policy':float(ordinary.loc[rid,'Brier']),'gain_vs_ordinary_policy':float(ordinary.loc[rid,'Brier']-r.Brier),
                    'gain_vs_minrisk_policy':float(mr.loc[rid,'Brier']-r.Brier),'comparison_scope':'policy-round comparisons; prior D differs, not same-state marginal gain',
                    'target_Brier':None if pd.isna(r.target_Brier) else float(r.target_Brier),'rest_Brier':None if pd.isna(r.rest_Brier) else float(r.rest_Brier),
                    'actual_backend':r.actual_backend,'cached_real_model':bool(r.cached_real_model),'fit_time':None,
                    'fit_time_status':'per-fit duration not instrumented in frozen runner; selection time retained'})
                resources['retained_real_fit_calls']+=int(not r.cached_real_model)
                diag=lock.get('selection_diagnostics',{});resources['retained_fantasy_fit_calls']+=diag.get('actual_fantasy_GPU_fits',0)
                resources['retained_neighbor_LR_refits']+=diag.get('internal_LR_refits',0)
                if saved['method']=='LRBE-IG' and len(c):
                    rng=np.random.default_rng(manifest['seed']+rid*1000)
                    for _ in range(identity['config']['IG_bootstrap_members']):
                        sampled=rng.integers(0,len(d),len(d))
                        resources['retained_committee_fit_calls']+=int(len(np.unique(y[d[sampled]]))>=2)
    assert case_count==60 and len(state_rows)==7200 and max_risk_error<1e-6 and max_loss_error<1e-6
    write_jsonl(out/'state_manifest.jsonl',state_rows);write_jsonl(out/'selection_summary.jsonl',selections);write_jsonl(out/'branch_outcomes.jsonl',branches)
    audit={'passed':True,'cases':60,'episodes':720,'rounds':7200,'all_D_histories_reconstructed':True,'all_group_time_reference_test_boundaries_checked':True,
        'all_unique_sampling_and_budget_checks':True,'max_risk_roundtrip_error':max_risk_error,'max_realized_ledger_error':max_loss_error,
        'resources_retained_segments':resources,'physical_retry_fit_count_exact':False,
        'resource_limit':'retained accepted round segments only; interrupted overwritten partial fits not recoverable and not counted as independent evidence',
        'price_preprocessor_hash_scope':'deterministic source/data/initial-order/parameter identities; actual per-round candidate costs independently hashed',
        'same_state_baseline_not_fabricated':True,'per_fit_durations_not_fabricated':True,'exporter_sha256':sha(__file__)}
    write_json(out/'round_record_audit.json',audit);print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
