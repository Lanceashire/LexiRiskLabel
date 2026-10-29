"""One small learner-matched candidate, frozen before new fantasy outcomes.

Four-or-fewer fixed batches, 2 selection + 2 independent check worlds per state.
Actual validation/test data are read only after all six choices are locked.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from runtime import *


def propose(s):
    result=[];seen={}
    for method in ['OrdinaryOnly','MinRisk','Vopt','RB-BRIDGE']:
        ids=np.sort(frozen.select(method,s).selected_ids);key=tuple(ids.tolist())
        if key in seen:result[seen[key]]['sources'].append(method)
        else:seen[key]=len(result);result.append({'ids':ids,'sources':[method]})
    return result


def table(frame,headers):
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:+.8f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);args=ap.parse_args();base=args.base
    out=base/'fantasy_small';out.mkdir(exist_ok=False);private=base/'private/fantasy_small';private.mkdir(parents=True,exist_ok=True)
    parent=json.loads((base/'source_identity.json').read_text());config=parent['configuration'];verify_identity(parent,run_identity(config))
    stage=json.loads((HERE/'fantasy_config.json').read_text())
    write_json(out/'activation.json',{'configuration':stage,'source_hash':sha(__file__),'config_hash':sha(HERE/'fantasy_config.json'),
       'parent_identity':parent['identity_hash'],'activation_basis':'E1 model-dependent gains and unstable frozen-leaf transfer leave true-learner linearization mismatch unresolved; this is the single small candidate allowed by section 12',
       'actual_outcomes_blinded_until_all_choices_locked':True})
    pending=[];scores=[];selection=[];models={};hits=0;start=time.perf_counter()
    for dataset in config['datasets']:
        for seed in config['seeds']:
            c=build_context(dataset,seed,stage['anchor_round'],config,private);s=c.acquisition;family=propose(s)
            h,_,_=frozen.geometry(s,False);cov=frozen.inverse(h)
            selections=np.random.default_rng(s.seed+132000).multivariate_normal(s.theta,cov,stage['selection_worlds'])
            checks=np.random.default_rng(s.seed+142000).multivariate_normal(s.theta,cov,stage['verification_worlds'])
            theta=np.vstack([selections,checks]);nsel=len(selections)
            uniforms=np.vstack([np.random.default_rng(s.seed+152000).random((len(selections),len(s.ordinary_x)+len(s.candidate_x))),
                                np.random.default_rng(s.seed+162000).random((len(checks),len(s.ordinary_x)+len(s.candidate_x)))])
            values=np.zeros((len(theta),len(family)));fit_keys=[]
            for w,t in enumerate(theta):
                prob=expit(np.vstack([s.ordinary_x,s.candidate_x])@t)
                fantasy_y=(uniforms[w]<prob).astype(int);target=expit(s.reference_x@t)
                for j,proposal in enumerate(family):
                    ids=proposal['ids'];xx=np.vstack([s.visible_x,s.ordinary_x,s.candidate_x[ids]])
                    yy=np.r_[s.visible_y,fantasy_y[:len(s.ordinary_x)],fantasy_y[len(s.ordinary_x)+ids]]
                    global_ids=np.r_[c.feedback.d_ids,c.feedback.o_ids,c.feedback.c_ids[ids]];order=np.argsort(global_ids,kind='stable')
                    xx=xx[order];yy=yy[order];global_ids=global_ids[order]
                    key=digest({'dataset':dataset,'x':digest(xx),'y':yy,'ids':global_ids,'fit_seed':seed*1000+c.round,'params':config['xgb']})
                    if key in models:model=models[key];hits+=1
                    else:
                        model=XGBClassifier(**config['xgb'],random_state=seed*1000+c.round,objective='binary:logistic',eval_metric='logloss')
                        model.fit(xx[:,1:].astype(np.float32),yy)
                        backend=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
                        if not backend.startswith('cuda'):raise RuntimeError('fantasy XGB GPU backend required')
                        models[key]=model
                    prediction=model.predict_proba(s.reference_x[:,1:].astype(np.float32))[:,1]
                    values[w,j]=float(np.mean((prediction-target)**2));fit_keys.append(key)
                    scores.append({'state_id':c.state_id,'dataset':dataset,'seed':seed,'world':w,'world_role':'selection' if w<nsel else 'verification',
                        'proposal_id':j,'score':values[w,j],'fit_key':key,'actual_backend':'cuda:0'})
            mean=values[:nsel].mean(axis=0);verify=values[nsel:].mean(axis=0);chosen=int(np.argmin(mean))
            # Tokens for all proposals do not reveal labels; lock both family and choice.
            for p in family:
                p['token']=c.feedback.lock(p['ids'],'fantasy_candidate',s.seed)
                p['selected_hash']=digest(np.sort(c.feedback.c_ids[p['ids']]))
                assert s.cost[p['ids']].sum()<=s.budget+1e-7
            lock={'state_id':c.state_id,'chosen_proposal':chosen,'selection_scores':mean,'independent_check_scores':verify,
                  'family':family,'actual_outcomes_read_before_choice':False}
            write_json(private/'selection_locks'/f'{c.state_id}.json',lock)
            np.savez_compressed(private/f'{c.state_id}_worlds.npz',theta=theta,uniforms=uniforms,score=values)
            row={'state_id':c.state_id,'dataset':dataset,'seed':seed,'candidate_family_n':len(family),'chosen_proposal':chosen,
                 'chosen_sources':family[chosen]['sources'],'chosen_hash':family[chosen]['selected_hash'],
                 'chosen_count':len(family[chosen]['ids']),'chosen_cost':float(s.cost[family[chosen]['ids']].sum()),'budget':s.budget,
                 'same_model_check_regret':float(verify[chosen]-verify.min()),'same_model_choice_replicated':bool(chosen==np.argmin(verify)),
                 'score_is_not_actual_outcome_guarantee':True};selection.append(row)
            pending.append((c,family,chosen,mean,verify))
            print('Fantasy planning locked',c.state_id,'batches',len(family),'GPU fits',len(models),flush=True)
    # Only now may evaluation read previously observed research V/T outcomes.
    e1b=pd.read_json(base/'e1b/all_set_outcomes.jsonl',lines=True)
    e1=pd.read_json(base/'branch_outcomes.jsonl',lines=True)
    actual=[];decisions=[];max_prediction_error=0.
    for c,family,chosen,mean,check in pending:
        group=[]
        for j,p in enumerate(family):
            ref=e1b[(e1b.state_id==c.state_id)&(e1b.learner=='XGB')&(e1b.selected_hash==p['selected_hash'])]
            assert len(ref)==1,'missing/ambiguous frozen candidate-set validation';ref=ref.iloc[0]
            assert str(ref.actual_backend).startswith('cuda')
            archive=np.load(base/'private/e1b_predictions'/f'{c.state_id}_set{int(ref.set_id)}_XGB.npz')
            vb=float(np.mean((archive['eval_prediction']-archive['eval_y'])**2));max_prediction_error=max(max_prediction_error,abs(vb-ref.V_eval_brier))
            old=e1[(e1.state_id==c.state_id)&(e1.learner=='XGB')&(e1.selected_hash==p['selected_hash'])]
            assert len(old)>0 and old.brier.max()-old.brier.min()<1e-12
            row={'state_id':c.state_id,'dataset':c.dataset,'seed':c.seed,'proposal_id':j,'sources':p['sources'],
                 'selected_hash':p['selected_hash'],'count':len(p['ids']),'cost':float(c.acquisition.cost[p['ids']].sum()),
                 'selection_score':float(mean[j]),'independent_check_score':float(check[j]),'V_eval_brier':vb,'T_brier':float(old.brier.iloc[0]),
                 'actual_predictions_reused':True,'chosen_by_fantasy_only':j==chosen}
            actual.append(row);group.append(row)
        mr=next(x for x in group if 'MinRisk' in x['sources']);ordinary=next(x for x in group if 'OrdinaryOnly' in x['sources']);pick=group[chosen]
        rho=spearmanr(mean,[x['V_eval_brier'] for x in group]).statistic if len(group)>1 else np.nan
        decisions.append({'state_id':c.state_id,'dataset':c.dataset,'seed':c.seed,'chosen_sources':pick['sources'],
            'same_as_minrisk':pick['selected_hash']==mr['selected_hash'],'chosen_count':pick['count'],'minrisk_count':mr['count'],
            'chosen_cost':pick['cost'],'minrisk_cost':mr['cost'],'V_eval_extra_gain_vs_minrisk':mr['V_eval_brier']-pick['V_eval_brier'],
            'V_eval_gain_vs_ordinary':ordinary['V_eval_brier']-pick['V_eval_brier'],'T_extra_gain_vs_minrisk':mr['T_brier']-pick['T_brier'],
            'fantasy_actual_V_rank_correlation':None if not np.isfinite(rho) else float(rho),'new_actual_fits':0})
    d=pd.DataFrame(decisions);summary=d.groupby('dataset',as_index=False).agg(
        V_eval_extra_gain_vs_minrisk=('V_eval_extra_gain_vs_minrisk','mean'),V_eval_gain_vs_ordinary=('V_eval_gain_vs_ordinary','mean'),
        T_extra_gain_vs_minrisk=('T_extra_gain_vs_minrisk','mean'),positive_seed_count=('V_eval_extra_gain_vs_minrisk',lambda v:int((v>1e-12).sum())),
        same_as_minrisk_count=('same_as_minrisk','sum'))
    gate=bool((summary.V_eval_extra_gain_vs_minrisk>1e-12).all() and (d.V_eval_extra_gain_vs_minrisk>1e-12).sum()>=4)
    assert max_prediction_error<1e-12
    write_jsonl(out/'fantasy_world_scores.jsonl',scores);write_jsonl(out/'locked_selections.jsonl',selection)
    write_jsonl(out/'actual_candidate_outcomes.jsonl',actual);write_jsonl(out/'paired_decisions.jsonl',decisions)
    summary.to_csv(out/'dataset_summary.csv',index=False)
    write_json(out/'verification.json',{'passed':True,'max_reused_prediction_score_error':max_prediction_error,
        'checks':['parent source/data identity','D-only fantasy model','common O/C Bernoulli uniforms across batches',
        'same actual XGB retraining program','CUDA backend checked','all six choices locked before validation files read',
        'independent check worlds not used for choice','all proposed sets within same budgets',
        'V_eval predictions matched by exact selected hash and recomputed','outer T secondary only','no fresh-validation claim'],
        'independent_seed_clusters':6,'development_seed_values_per_task':2})
    comp={'completed':True,'states':6,'nominal_fantasy_fit_slots':len(scores),'actual_unique_GPU_fantasy_fits':len(models),'fantasy_fit_cache_hits':hits,
          'new_actual_outcome_fits':0,'frozen_actual_candidate_scores_reused':len(actual),'selection_worlds':2,'verification_worlds':2,
          'elapsed_seconds':time.perf_counter()-start,'confirmation':False};write_json(out/'completion.json',comp)
    write_json(out/'decision.json',{'predeclared_gate_passed':gate,'candidate_frozen_for_confirmation':False,
        'action':'keep only one development candidate for review; no confirmation yet' if gate else 'stop this fantasy candidate; do not expand worlds/candidate batches after seeing outcomes',
        'V_eval_positive_seed_states':int((d.V_eval_extra_gain_vs_minrisk>1e-12).sum()),'new_dataset_started':False,'confirmation_started':False})
    text=f'''# 真实学习器匹配：小批次 XGBoost 幻想重训诊断

本轮执行原协议第 12 节第一优先项：在少量事先固定批次上用实际学习器幻想重训，减少线性代理与树模型重训程序不匹配的问题。它是最多一个新候选的开发诊断，不是新一轮大规模基准或确认性结果。

固定六个 round-5 共同状态。候选族仅 OrdinaryOnly、MinRisk、Vopt、RB-BRIDGE，重复选集去重。每状态仅 2 个选择世界和 2 个独立核验世界；两个 RNG 流独立，全部候选共用每世界的 O/C 幻想标签。幻想标签生成模型只用可见 D 的逻辑回归及 Laplace 不确定性，不接触真实 O/C、V、T 标签。每个候选在 D 真标签 + 幻想 O/S 上按原 100 棵树 GPU XGBoost 真实训练，在 Q_ref 上估计期望 Brier 排名。

全部六个状态的选择锁定后，才读取已有 E1b 的 V_eval 预测与 E1 的 T 结果。按全局选集哈希匹配、重新计分，不重新训练真实结果，也不按 V/T 挑选候选。V_eval 是先前已用于开发的块，不能称为新盲测。独立幻想核验世界只审计排序，不重选动作。

## 真实验证表现

增益为 MinRisk Brier − 所选批次 Brier，正数为改善；先在每数据集两个原有种子内呈现配对均值，随机幻想世界不增加数据独立性。

{table(summary,['数据集','V_eval 相对 MinRisk','V_eval 相对 OrdinaryOnly','外部 T 相对 MinRisk','V_eval 改善种子数/2','与 MinRisk 同选集数/2'])}

逐状态所选来源、实际采集人数和风险，以及幻想分数与真实结果的排序关系保存在 paired_decisions.jsonl 和 actual_candidate_outcomes.jsonl。同预算下选集数量、实际风险可能不同，不能把预测排名当成同风险改善保证。

## 预设决定

实验前规定：三个数据集的 V_eval 均值都优于 MinRisk，且六个种子状态至少四个改善，才保留该候选供审阅；即便通过也不直接启动正式确认。当前通过条件：{gate}。不得看完结果再增世界、换候选族或增加随机种子来追逐正结果。

有效幻想拟合槽 {len(scores)}，实际不同 GPU 拟合 {len(models)}，缓存复用 {hits}；真实结果新增拟合为 0，复用 {len(actual)} 个已有候选集评价。逐预测重算误差最大 {max_prediction_error:.3g}，信息边界与实际 CUDA 后端核验通过。

这个方法只减少部分学习器线性化误差，不能修复错误的世界模型。2+2 个世界是小型开发预算，不足以消除 Monte Carlo 排序噪声，其标准误不是实际泛化收益保证。原始幻想世界和模型训练键本地保留；只整理必要 Pro 审阅证据，不恢复明细上传。
'''
    (out/'XGBoost_幻想重训小实验报告.md').write_text(text)
    print(json.dumps(comp),flush=True);print(summary.to_string(index=False),flush=True);print('GATE',gate,flush=True)


if __name__=='__main__':main()
