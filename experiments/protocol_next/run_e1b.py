"""Finite feasible-set offline reference, selected on V_search, evaluated on V_eval.

This uses extra research validation labels. It is NOT a deployable competitor,
not a global optimum, and not an upper bound. Outer T is not scored here.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json,time
import numpy as np
from runtime import *


def validation_blocks(context,config):
    raw,y,_,_,_=legacy.load(context.dataset);_,_,development,_=legacy.split(raw,context.seed,context.dataset)
    original=json.loads((ROOT/'results/research_v2/final_real'/f'{context.dataset}_{context.seed}_fixed_MinRisk'/f'round_{context.round:02d}_selection.json').read_text())
    qid=np.asarray(original['reference_ids'],int);remaining=np.setdiff1d(development,qid)
    rng=np.random.default_rng(config['e1b']['validation_seed']+context.seed);remaining=rng.permutation(remaining);middle=len(remaining)//2
    search,ev=remaining[:middle],remaining[middle:]
    assert not np.intersect1d(search,ev).size
    assert not np.intersect1d(np.r_[search,ev],np.r_[qid,context.feedback.d_ids,context.feedback.o_ids,context.feedback.c_ids,context.evaluation._ids]).size
    pre=legacy.preprocessor(raw);pre.fit(raw.iloc[context.initial_ids])
    def transform(ids):return np.c_[np.ones(len(ids)),np.asarray(pre.transform(raw.iloc[ids]),float)]
    return search,ev,transform(search),transform(ev),y[search],y[ev]


def propose(context,config):
    s=context.acquisition;state_record=json.loads((context.feedback.private_directory/'states'/f'{context.state_id}.json').read_text())
    if not np.array_equal(state_record['D_ids'],context.feedback.d_ids) or not np.array_equal(state_record['candidate_costs'],s.cost):raise RuntimeError('cached IG/state identity mismatch')
    utility=np.asarray(state_record['IG_scores']);proposals=[];mapping={}
    def add(ids,source):
        ids=np.sort(np.asarray(ids,int));signature=tuple(map(int,ids))
        if s.cost[ids].sum()>s.budget+1e-7:return
        if signature in mapping:proposals[mapping[signature]]['sources'].append(source);return
        mapping[signature]=len(proposals);proposals.append({'selected':ids,'sources':[source]})
    for method in ['OrdinaryOnly','MinRisk','LRBE-IG','Vopt','RB-BRIDGE','Vopt-Kstar']:
        add(select_e1(s,method,config,utility=utility).selected_ids,method)
    rng=np.random.default_rng(config['e1b']['validation_seed']+context.seed*100+context.round)
    mr=np.argsort(s.cost,kind='stable')[:frozen.kstar(s.cost,s.budget)];outside=np.setdiff1d(np.arange(len(s.cost)),mr)
    # Fixed 1% of B cost matching; failures never widen the tolerance.
    attempts=0
    while len(proposals)<min(10,config['e1b']['feasible_sets']) and attempts<500 and len(mr) and len(outside):
        attempts+=1;i=int(rng.choice(mr));j=int(rng.choice(outside));new=np.r_[mr[mr!=i],j]
        if abs(s.cost[new].sum()-s.cost[mr].sum())<=.01*s.budget+1e-10:add(new,'risk_matched_1pct_B_exchange')
    random_attempts=0
    while len(proposals)<config['e1b']['feasible_sets'] and random_attempts<2000:
        random_attempts+=1;add(frozen.random_fill(s.cost,s.budget,rng),'random_feasible')
    return proposals[:config['e1b']['feasible_sets']],{'matched_exchange_attempts':attempts,'random_generation_attempts':random_attempts,
                                                     'risk_matching_tolerance':.01*s.budget,'requested_unique_sets':config['e1b']['feasible_sets'],
                                                     'actual_unique_sets':len(proposals[:config['e1b']['feasible_sets']])}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base;out=base/'e1b';out.mkdir(exist_ok=False)
    identity=json.loads((base/'source_identity.json').read_text());config=identity['configuration'];private=base/'private'
    write_json(out/'activation.json',{'basis':'E1 full-feedback and feasible-method effects vary by learner/state; E2b/E2c did not identify a stable repair, so residual feasible-set value remains uncertain.',
                                     'scope':'all six pre-fixed round-5 anchors, sixteen distinct feasible proposals each, no outer T ranking',
                                     'validation_source':'unused development labels after removing Q_ref; independent random V_search and V_eval',
                                     'extra_validation_labels':True,'deployable_competitor':False,'not_an_upper_bound':True,
                                     'proposal_cost_matching_tolerance':'.01*B, never relaxed','source_hash':sha(__file__),'parent_identity':identity['identity_hash']})
    summaries=[];outcomes=[];choices=[];fit_count=0
    for dataset in config['datasets']:
        for seed in config['seeds']:
            c=build_context(dataset,seed,config['e1b']['anchor_round'],config,private)
            proposals,generation=propose(c,config)
            # Every candidate set is fixed and logged before any current O/C labels are released.
            for j,p in enumerate(proposals):
                p['token']=c.feedback.lock(p['selected'],f'E1b_set{j}',c.acquisition.seed)
                p['hash']=digest(np.sort(c.feedback.c_ids[p['selected']]))
            write_json(private/'e1b_sets'/f'{c.state_id}.json',{'proposals':proposals,'generation':generation,'all_locked_before_reveal':True})
            search_ids,eval_ids,search_x,eval_x,search_y,eval_y=validation_blocks(c,config)
            write_json(private/'e1b_validation'/f'{c.state_id}.json',{'V_search_ids':search_ids,'V_eval_ids':eval_ids})
            summary={'state_id':c.state_id,'dataset':dataset,'seed':seed,'round':c.round,'V_search_n':len(search_ids),'V_eval_n':len(eval_ids),
                     'V_search_hash':digest(search_ids),'V_eval_hash':digest(eval_ids),'reference_hash':c.manifest['reference_hash'],
                     'extra_validation_labels':True,'diagnostic_only':True,**generation};summaries.append(summary)
            local=[]
            for j,p in enumerate(proposals):
                ids,xx,yy=c.feedback.reveal(p['selected'],p['token'])
                for learner in config['learners']:
                    start=time.perf_counter();fit_seed=seed*1000+c.round
                    if learner=='LR':
                        model=frozen.fit_logistic(xx,yy,config['lr_ridge']);backend='cpu'
                        ps=expit(search_x@model);pe=expit(eval_x@model)
                    else:
                        model=XGBClassifier(**config['xgb'],random_state=fit_seed,objective='binary:logistic',eval_metric='logloss')
                        model.fit(xx[:,1:].astype(np.float32),yy);backend=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
                        if not backend.startswith('cuda'):raise RuntimeError('E1b actual GPU backend mismatch')
                        ps=model.predict_proba(search_x[:,1:].astype(np.float32))[:,1];pe=model.predict_proba(eval_x[:,1:].astype(np.float32))[:,1]
                    fit_count+=1
                    dest=private/'e1b_predictions'/f'{c.state_id}_set{j}_{learner}.npz';dest.parent.mkdir(exist_ok=True)
                    np.savez_compressed(dest,search_ids=search_ids,search_y=search_y,search_prediction=ps,eval_ids=eval_ids,eval_y=eval_y,eval_prediction=pe)
                    local.append({'state_id':c.state_id,'dataset':dataset,'seed':seed,'round':c.round,'set_id':j,'learner':learner,'selected_hash':p['hash'],
                                  'sources':p['sources'],'count':len(p['selected']),'cost':float(c.acquisition.cost[p['selected']].sum()),'budget':c.acquisition.budget,
                                  'V_search_brier':float(np.mean((ps-search_y)**2)),'V_eval_brier':float(np.mean((pe-eval_y)**2)),
                                  'actual_backend':backend,'fit_seconds':time.perf_counter()-start,'extra_validation_labels':True,'diagnostic_only':True})
            for learner in config['learners']:
                group=[r for r in local if r['learner']==learner]
                chosen=min(group,key=lambda r:(r['V_search_brier'],r['set_id']))
                mr=next(r for r in group if 'MinRisk' in r['sources']);ordinary=next(r for r in group if 'OrdinaryOnly' in r['sources'])
                choices.append({'state_id':c.state_id,'dataset':dataset,'seed':seed,'round':c.round,'learner':learner,
                                'selected_on_V_search_only':True,'chosen_set_id':chosen['set_id'],'chosen_selected_hash':chosen['selected_hash'],
                                'chosen_V_search_brier':chosen['V_search_brier'],'chosen_V_eval_brier':chosen['V_eval_brier'],
                                'MinRisk_V_eval_brier':mr['V_eval_brier'],'OrdinaryOnly_V_eval_brier':ordinary['V_eval_brier'],
                                'gain_V_eval_vs_minrisk':mr['V_eval_brier']-chosen['V_eval_brier'],
                                'gain_V_eval_vs_ordinary':ordinary['V_eval_brier']-chosen['V_eval_brier'],
                                'diagnostic_only':True,'uses_extra_validation_labels':True})
            outcomes.extend(local);print('E1b',c.state_id,'distinct sets',len(proposals),'fits',fit_count,flush=True)
    write_jsonl(out/'state_manifest.jsonl',summaries);write_jsonl(out/'all_set_outcomes.jsonl',outcomes);write_jsonl(out/'validation_selected_reference.jsonl',choices)
    write_json(out/'completion.json',{'completed':True,'states':len(summaries),'actual_unique_sets':sum(x['actual_unique_sets'] for x in summaries),
                                       'actual_fits':fit_count,'validation_selected_references':len(choices),'outer_T_used_for_selection_or_scoring':False,
                                       'note':'New validation predictions require fitting these fixed training sets; E1 persisted T predictions, not fitted models or V predictions. Refit slots do not add independent data clusters.'})


if __name__=='__main__':main()
