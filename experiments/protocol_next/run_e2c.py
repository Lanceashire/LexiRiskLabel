"""Conditional E2c: frozen D-only tree leaves and separate Q_ref sensitivity."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import argparse,json,time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from runtime import *
from run_e1 import selection_summary
from run_e2b import load_e1_cache


def leaf_context(raw_context,config,leaf_config,private):
    c=raw_context;s=c.acquisition;params={**config['xgb'],'n_estimators':leaf_config['leaf_trees'],'max_depth':leaf_config['leaf_depth']}
    model=XGBClassifier(**params,random_state=c.seed*1000+c.round+710000,objective='binary:logistic',eval_metric='logloss')
    model.fit(s.visible_x[:,1:].astype(np.float32),s.visible_y)
    conf=json.loads(model.get_booster().save_config())['learner'];device=conf['generic_param']['device']
    if not device.startswith('cuda'):raise RuntimeError('leaf embedding GPU backend mismatch')
    def apply(x):return model.apply(x[:,1:].astype(np.float32)).astype(int)
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False);encoder.fit(apply(s.visible_x))
    def transform(x):return np.c_[np.ones(len(x)),encoder.transform(apply(x))]
    dx,ox,cx,qx=[transform(x) for x in [s.visible_x,s.ordinary_x,s.candidate_x,s.reference_x]]
    theta=frozen.fit_logistic(dx,s.visible_y,leaf_config['head_ridge'])
    state=replace(s,visible_x=dx,ordinary_x=ox,candidate_x=cx,reference_x=qx,theta=theta,ridge=leaf_config['head_ridge'])
    metadata={**c.manifest,'representation_hash':digest({'kind':'D_only_frozen_tree_leaf','D':digest(dx),'O':digest(ox),'C':digest(cx),'Q_ref':digest(qx),'params':params})}
    f=c.feedback;feedback=FeedbackService(f.d_ids,f.o_ids,f.c_ids,dx,ox,cx,f._y,private,c.state_id)
    e=c.evaluation;evaluation=EvaluationService(e._ids,transform(e._x),e._y,e._target,private)
    result=Context(c.state_id,c.dataset,c.seed,c.round,state,feedback,evaluation,metadata,c.initial_ids)
    path=private/'leaf_models'/f'{c.state_id}.json';path.parent.mkdir(exist_ok=True);model.save_model(path)
    leaf_evidence={'state_id':c.state_id,'training_ids_hash':digest(f.d_ids),'training_labels_hash':digest(s.visible_y),
                   'trained_on_D_only':True,'tree_count':leaf_config['leaf_trees'],'max_depth':leaf_config['leaf_depth'],
                   'leaf_feature_count':dx.shape[1]-1,'actual_backend':device,'model_sha256':sha(path),
                   'representation_hash':metadata['representation_hash']}
    return result,leaf_evidence


def reference512(raw_context,leaf_config):
    c=raw_context;raw,y,_,_,_=legacy.load(c.dataset);_,_,development,_=legacy.split(raw,c.seed,c.dataset)
    qid=development[:leaf_config['reference_sensitivity_n']]
    if np.intersect1d(qid,np.r_[c.feedback.d_ids,c.feedback.o_ids,c.feedback.c_ids,c.evaluation._ids]).size:raise AssertionError('Q/T or Q/training overlap')
    pre=legacy.preprocessor(raw);pre.fit(raw.iloc[c.initial_ids]);qx=np.c_[np.ones(len(qid)),np.asarray(pre.transform(raw.iloc[qid]),float)]
    if not np.array_equal(qx[:len(c.acquisition.reference_x)],c.acquisition.reference_x):raise AssertionError('reference not nested')
    state=replace(c.acquisition,reference_x=qx)
    metadata={**c.manifest,'representation_hash':digest({'parent_representation':c.manifest['representation_hash'],'reference_x':digest(qx)}),'reference_hash':digest(qid)}
    return replace(c,acquisition=state,manifest=metadata)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base;out=base/'e2c';out.mkdir(exist_ok=False)
    identity=json.loads((base/'source_identity.json').read_text());config=identity['configuration'];lc=json.loads((HERE/'e2c_config.json').read_text());private=base/'private'
    write_json(out/'activation.json',{'reason':'E1 paired selections produce different LR versus XGB effects; E2b finds no pair-construction improvement over singleton construction.',
                                     'anchors':'all six dataset/seed combinations at pre-fixed round 5','config':lc,'config_hash':digest(lc),
                                     'parent_identity':identity['identity_hash'],'source_hash':sha(__file__),
                                     'separation':'raw128 versus leaf128 changes representation; raw128 versus raw512 changes only reference size. No new solver or parameter tuning.'})
    cache=FitCache(config,identity['identity_hash'],private);load_e1_cache(cache,base)
    selections=[];branches=[];leaves=[]
    for dataset in config['datasets']:
        for seed in config['seeds']:
            raw=build_context(dataset,seed,lc['anchor_round'],config,private)
            leaf,evidence=leaf_context(raw,config,lc,private);leaves.append(evidence)
            large=reference512(raw,lc)
            for condition,selected_context in [('raw_ref128',raw),('leaf_ref128',leaf),('raw_ref512',large)]:
                requests=[]
                for method in lc['selectors']:
                    s=selected_context.acquisition;selection=frozen.select(method,s)
                    # Commit the intervention before even computing label-based resource summaries.
                    selected_context.feedback.lock(selection.selected_ids,condition+'_'+method,s.seed)
                    selected_hash=digest(np.sort(selected_context.feedback.c_ids[selection.selected_ids]))
                    summary=selection_summary(selected_context,method,0,selection);summary.update({'condition':condition,'reference_n':len(s.reference_x)})
                    selections.append(summary);requests.append((method,selection,selected_hash))
                local=[];metrics={}
                for method,selection,selected_hash in requests:
                    for learner in config['learners']:
                        # Full XGBoost ALWAYS regrows trees on original raw features.
                        # Reference changes are not part of the learner's training data.
                        training_context=leaf if learner=='LR' and condition=='leaf_ref128' else raw
                        token=training_context.feedback.lock(selection.selected_ids,condition+'_'+method,training_context.acquisition.seed)
                        fitted=cache.fit(training_context,selection.selected_ids,learner,token);metrics[(method,learner)]=fitted['metrics']
                        local.append({'state_id':raw.state_id,'dataset':dataset,'seed':seed,'round':raw.round,'condition':condition,'selector':method,'learner':learner,
                                      'selected_hash':selected_hash,'selection_representation_hash':selected_context.manifest['representation_hash'],
                                      'training_representation_hash':training_context.manifest['representation_hash'],
                                      'brier':fitted['metrics']['brier'],'target_brier':fitted['metrics']['target_brier'],'rest_brier':fitted['metrics']['rest_brier'],
                                      'fit_key':fitted['fit_key'],'cache_hit':fitted['cache_hit'],'actual_backend':fitted['actual_backend']})
                for row in local:
                    learner=row['learner'];row['gain_vs_ordinary']=metrics[('OrdinaryOnly',learner)]['brier']-row['brier'];row['gain_vs_minrisk']=metrics[('MinRisk',learner)]['brier']-row['brier']
                branches.extend(local)
            print('E2c',dataset,seed,'new fits',cache.fits,'reused',cache.hits,flush=True)
    write_jsonl(out/'selection_summary.jsonl',selections);write_jsonl(out/'branch_outcomes.jsonl',branches);write_jsonl(out/'leaf_evidence.jsonl',leaves)
    write_json(out/'completion.json',{'completed':True,'states':6,'conditions':len(lc['conditions']),'method_slots':len(selections),'nominal_fit_slots':len(branches),
                                       'actual_new_learner_fits':cache.fits,'reused_learner_fits':cache.hits,'separate_D_only_leaf_model_fits':len(leaves),
                                       'full_XGB_always_uses_original_features':True,'hyperparameter_search':False})


if __name__=='__main__':main()
