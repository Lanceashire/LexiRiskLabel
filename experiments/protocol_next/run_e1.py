"""E1: frozen MinRisk anchors, shared selections, LR/XGB forks, cached fits."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from runtime import *
from audit import run_audit


def selection_summary(context,method,draw,selection):
    s=context.acquisition;selected=np.sort(np.asarray(selection.selected_ids,int));global_ids=context.feedback.c_ids[selected]
    mr=np.argsort(s.cost,kind='stable')[:frozen.kstar(s.cost,s.budget)]
    h,g,u=frozen.geometry(s);physical=frozen.finish(s,selected).diagnostics if method!='FullFeedback' else {}
    union=set(selected)|set(mr);selected_hash=digest(np.sort(global_ids));steps=selection.diagnostics.get('steps',[])
    return {'state_id':context.state_id,'dataset':context.dataset,'seed':context.seed,'round':context.round,'selector':method,'random_draw':draw if method=='Random-feasible' else None,
            'selector_seed':selection.diagnostics.get('selector_seed',s.seed),'representation_hash':context.manifest['representation_hash'],
            'selected_hash':selected_hash,'intervention_id':digest({'state_id':context.state_id,'selected_hash':selected_hash}),
            'count':len(selected),'cost':float(s.cost[selected].sum()),'realized_loss':float(context.feedback._y[global_ids]@np.asarray(legacy.load(context.dataset)[2])[global_ids]),
            'Kstar':len(mr),'slack':s.budget-float(s.cost[mr].sum()),'max_swap_count':physical.get('max_swap_count'),
            'jaccard_vs_minrisk':len(set(selected)&set(mr))/len(union) if union else 1.,'inserted_vs_minrisk':len(set(selected)-set(mr)),
            'proxy_value':frozen.set_value(h,g,u,selected),'positive_pair_steps':selection.diagnostics.get('positive_pair_steps',0),
            'pair_gain':float(sum(step['gain'] for step in steps if len(step['ids'])==2)),
            'selection_seconds':selection.diagnostics.get('selection_seconds',0.),'budget_feasible':float(s.cost[selected].sum())<=s.budget+1e-7,
            'diagnostic_only':method=='FullFeedback','ordinary_n':context.manifest['ordinary_n'],'candidate_n':len(s.cost),
            'initial_visible_n':context.manifest['initial_visible_n'],'current_visible_n':len(s.visible_y),
            'composition_premium':float(s.cost[selected].sum()-np.sort(s.cost)[:len(selected)].sum())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path,default=HERE/'protocol.json');ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
    config=json.loads(args.config.read_text());identity=run_identity(config);out=args.output
    if out.exists():
        stored=out/'source_identity.json'
        if not stored.exists():raise RuntimeError('existing output lacks identity')
        verify_identity(json.loads(stored.read_text()),identity)
        if (out/'completion.json').exists():print('VERIFIED_COMPLETE',flush=True);return
        raise RuntimeError('incomplete output retained; choose a new output directory')
    out.mkdir(parents=True);write_json(out/'source_identity.json',identity)
    audit=run_audit(config,identity,out);print('E0 PASS: hidden labels, cache, identity, release and mathematical checks',flush=True)
    if args.audit_only:return
    private=out/'private';cache=FitCache(config,identity['identity_hash'],private)
    states=[];selections=[];branches=[];started=time.perf_counter();ig_seconds=0.
    for dataset in config['datasets']:
        for seed in config['seeds']:
            for round_id in config['anchor_rounds']:
                c=build_context(dataset,seed,round_id,config,private);c.manifest['code_hash']=digest(identity['code']);states.append(c.manifest)
                utility,seconds=bootstrap_ig(c.acquisition,config);ig_seconds+=seconds
                requests=[]
                for method in config['selectors']:
                    for draw in range(config['random_draws'] if method=='Random-feasible' else 1):
                        selection=select_e1(c.acquisition,method,config,draw,utility)
                        key=c.feedback.lock(selection.selected_ids,method,selection.diagnostics.get('selector_seed',c.acquisition.seed))
                        summary=selection_summary(c,method,draw,selection);summary['ig_scoring_seconds']=seconds if method=='LRBE-IG' else 0.
                        selections.append(summary);requests.append((method,draw,selection,key,summary))
                baselines={}
                for method,draw,selection,key,summary in requests:
                    for learner in config['learners']:
                        fitted=cache.fit(c,selection.selected_ids,learner,key)
                        record={k:v for k,v in summary.items() if k in ['state_id','dataset','seed','round','selector','random_draw','selected_hash','representation_hash','intervention_id','diagnostic_only']}
                        record.update({'learner':learner,'learner_config':{'ridge':config['lr_ridge']} if learner=='LR' else config['xgb'],
                                       'fit_seed':seed*1000+round_id,'actual_backend':fitted['actual_backend'],'updater':fitted['updater'],
                                       'fit_time':fitted['charged_fit_seconds'],'original_fit_time':fitted['fit_seconds'],'cache_hit':fitted['cache_hit'],
                                       'fit_key':fitted['fit_key'],'train_n':fitted['train_n'],'train_hash':fitted['train_hash'],**fitted['metrics']})
                        baselines[(method,draw,learner)]=fitted['metrics'];branches.append(record)
                for record in branches[-len(requests)*len(config['learners']):]:
                    learner=record['learner'];base=baselines[('OrdinaryOnly',0,learner)];mr=baselines[('MinRisk',0,learner)]
                    record['Brier_base']=base['brier'];record['Brier_selected']=record['brier'];record['gain_vs_ordinary']=base['brier']-record['brier'];record['gain_vs_minrisk']=mr['brier']-record['brier']
                    for region in ['target','rest']:
                        k=region+'_brier';record['gain_'+region+'_vs_ordinary']=base[k]-record[k] if base[k] is not None and record[k] is not None else None
                        record['gain_'+region+'_vs_minrisk']=mr[k]-record[k] if mr[k] is not None and record[k] is not None else None
                write_jsonl(out/'state_manifest.jsonl',states);write_jsonl(out/'selection_summary.jsonl',selections);write_jsonl(out/'branch_outcomes.jsonl',branches)
                write_json(private/'states'/f'{c.state_id}.json',{'D_ids':c.feedback.d_ids,'O_ids':c.feedback.o_ids,'C_ids':c.feedback.c_ids,'reference_ids':json.loads((ROOT/'results/research_v2/final_real'/f'{dataset}_{seed}_fixed_MinRisk'/f'round_{round_id:02d}_selection.json').read_text())['reference_ids'],
                                                               'candidate_costs':c.acquisition.cost,'IG_scores':utility,'B':c.acquisition.budget})
                print(c.state_id,'states',len(states),'slots',len(branches),'unique_fits',cache.fits,'cache_hits',cache.hits,flush=True)
    assert len(states)==18 and len(branches)==432
    complete={'completed':True,'states':len(states),'nominal_fit_slots':len(branches),'actual_unique_fits':cache.fits,'fit_cache_hits':cache.hits,
              'unique_interventions':len(set(r['intervention_id'] for r in selections)),'selected_method_slots':len(selections),
              'independent_trajectory_clusters':6,'independent_seed_values':2,'random_draws':config['random_draws'],
              'ig_committee_fits_separate':len(states)*config['committee_members'],'ig_seconds':ig_seconds,'elapsed_seconds':time.perf_counter()-started}
    write_json(out/'completion.json',complete)
    manifest=[]
    for p in sorted(out.rglob('*')):
        if p.is_file():manifest.append({'path':p.relative_to(out).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size,'private':'private' in p.relative_to(out).parts})
    write_json(out/'manifest.json',manifest);print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
