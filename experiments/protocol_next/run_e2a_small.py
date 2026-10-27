"""Registered small E2a: exact conjugate target plus unchanged-set LR transfer.

Selectors receive only H/G/U/cost/B. All actions are locked before synthetic
hidden labels are generated. No selected seeds, no GPU transfer or tuning grid.
"""
from __future__ import annotations
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from runtime import *


def exhaustive(h,g,u,cost,budget):
    sets=[];values=[]
    for mask in range(1<<len(cost)):
        ids=np.flatnonzero((mask>>np.arange(len(cost)))&1)
        if cost[ids].sum()<=budget+1e-10:
            sets.append(ids);values.append(frozen.set_value(h,g,u,ids))
    best=min(range(len(sets)),key=lambda j:(-values[j],float(cost[sets[j]].sum()),tuple(sets[j])))
    return sets[best],float(values[best]),len(sets)


def greedy(h,g,u,cost,budget,paired):
    cov=frozen.inverse(h);free=np.ones(len(cost),bool);selected=[];remaining=budget;pair_steps=0
    pairs=np.asarray(list(itertools.combinations(range(len(cost)),2)),int)
    while free.any():
        gains=frozen.marginal(cov,g,u);valid=free&(cost<=remaining+1e-10)
        if not valid.any():break
        score=np.where(valid,gains/cost,-np.inf);i=int(np.argmax(score));chosen=np.array([i]);best=float(score[i])
        if paired:
            pp=pairs[free[pairs].all(axis=1)&(cost[pairs].sum(axis=1)<=remaining+1e-10)]
            pg=frozen.pair_gains(cov,g,u,pp)
            if len(pp):
                j=int(np.argmax(pg/cost[pp].sum(axis=1)))
                if pg[j]/cost[pp[j]].sum()>best+1e-14:chosen=pp[j];pair_steps+=1
        for i in chosen:selected.append(int(i));free[i]=False;remaining-=cost[i];cov=frozen.update(cov,u[i])
    ids=np.asarray(selected,int);single=frozen.marginal(frozen.inverse(h),g,u);valid=np.flatnonzero(cost<=budget+1e-10)
    if len(valid):
        best=int(valid[np.argmax(single[valid])])
        if single[best]>frozen.set_value(h,g,u,ids)+1e-14:ids=np.asarray([best])
    return ids,pair_steps


def table(df,headers):
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:+.8f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);a=ap.parse_args();base=a.base
    out=base/'e2a_small';out.mkdir(exist_ok=False);private=base/'private/e2a_small';private.mkdir(parents=True,exist_ok=True)
    config=json.loads((HERE/'e2a_small_config.json').read_text());identity=json.loads((base/'source_identity.json').read_text())
    verify_identity(identity,run_identity(identity['configuration']))
    write_json(out/'activation.json',{'configuration':config,'source_hash':sha(__file__),'config_hash':sha(HERE/'e2a_small_config.json'),
       'parent_identity':identity['identity_hash'],'activation_reason':'complete missing E2a exact optimization positive control after E2b/E2c/E3/E4; no expansion of stopped real-data routes',
       'registered_grid_note':'3 existing seeds for the small pilot; the suggested full 10-seed/300-endpoint grid is not claimed completed'})
    rows=[];controls=[];cache={};hits=0;shared={};max_inverse_error=0.;max_variance_identity_error=0.
    for seed in config['seeds']:
        rng=np.random.default_rng(seed+120000);L=config['L'];truth=rng.normal(size=2)
        extra=np.c_[rng.uniform(*config['other_target_coordinate_uniform'],config['other_candidates']),
                    rng.uniform(*config['other_nuisance_magnitude_uniform'],config['other_candidates'])*rng.choice([-1,1],config['other_candidates'])]
        semantic=np.vstack([config['core_pair'],config['bait'],extra]);perm=rng.permutation(len(semantic));u=semantic[perm]
        cc=np.asarray(config['cost_multiset'],float)
        expensive=cc.copy()
        for i,j in config['expensive_case_swaps_semantic_indices']:expensive[[i,j]]=expensive[[j,i]]
        assert np.array_equal(np.sort(cc),np.sort(expensive))
        shared[seed]={'truth':truth,'u':u,'perm':perm,'costs':{'core_pair_cheap':cc[perm],'core_pair_expensive':expensive[perm]}}
        for b in config['ordinary_b']:
            h=np.diag([1.,1.+b]);g=np.diag([1.,0.]);cov=frozen.inverse(h)
            signed=frozen.group_gain(cov,g,np.asarray(config['core_pair']))-float(frozen.marginal(cov,g,np.asarray(config['core_pair'])).sum())
            analytic=2/3-2/(2+L*L/(1+b))
            assert abs(signed-analytic)<1e-12
            if b<L*L-1:assert signed>1e-12
            elif b==L*L-1:assert abs(signed)<1e-12
            else:assert signed<-1e-12
            for price_case,cost in shared[seed]['costs'].items():
                budget=config['budget'];opt,value,nfeasible=exhaustive(h,g,u,cost,budget)
                proposals={};steps={}
                k=frozen.kstar(cost,budget);proposals['MinRisk']=np.argsort(cost,kind='stable')[:k]
                for method,paired in [('Vopt-single-noexchange',False),('BRIDGE-allpairs-noexchange',True)]:
                    proposals[method],steps[method]=greedy(h,g,u,cost,budget,paired)
                proposals['Random-feasible']=frozen.random_fill(cost,budget,np.random.default_rng(seed+121000))
                proposals['Exhaustive-Gaussian-opt']=opt
                state_id=f'seed{seed}_b{int(b)}_{price_case}'
                write_json(private/'selection_locks'/f'{state_id}.json',{'state_id':state_id,'candidate_semantic_ids':perm,
                    'selections':{m:ids for m,ids in proposals.items()},'all_locked_before_hidden_label_generation':True,
                    'selection_inputs':'H/G/U/cost/B only','candidate_x':u,'cost':cost,'budget':budget})
                controls.append({'state_id':state_id,'seed':seed,'ordinary_b':b,'price_case':price_case,
                   'core_pair_gamma':signed,'core_pair_gamma_analytic':analytic,'core_pair_feasible':bool(cost[np.flatnonzero(np.isin(perm,[0,1]))].sum()<=budget+1e-10),
                   'feasible_subset_count':nfeasible,'exact_best_gaussian_variance_decrease':value})
    # Every cell/selector is frozen before any candidate/ordinary/test outcomes.
    for seed in config['seeds']:
        state=shared[seed];truth,u,perm=state['truth'],state['u'],state['perm'];z=norm.ppf((np.arange(1500)+.5)/1500)
        test=np.c_[z,np.zeros(len(z))];pt=expit(test@truth);rng=np.random.default_rng(seed+122000)
        candidate_uniform=rng.random(len(u));yc=(candidate_uniform<expit(u@truth)).astype(int)
        candidate_gaussian_y=u@truth+rng.normal(size=len(u));ordinary_uniform=rng.random(4);ordinary_noise=rng.normal(size=4)
        yt=(rng.random(len(z))<pt).astype(int);d=np.zeros((2,2));yd=np.array([0,1])
        np.savez_compressed(private/f'seed{seed}_environment.npz',truth=truth,candidate_x=u,candidate_semantic_ids=perm,
             candidate_y=yc,candidate_gaussian_y=candidate_gaussian_y,ordinary_uniform=ordinary_uniform,
             ordinary_gaussian_noise=ordinary_noise,test_x=test,test_y=yt,test_true_probability=pt)
        for b in config['ordinary_b']:
            ox=np.c_[np.zeros(4),np.array([1,-1,1,-1])*np.sqrt(b/4)]
            h=np.eye(2)+ox.T@ox;g=np.diag([1.,0.]);yo=(ordinary_uniform<expit(ox@truth)).astype(int);ogy=ox@truth+ordinary_noise
            assert np.allclose(h,np.diag([1,1+b]))
            base_x=np.vstack([d,ox]);base_y=np.r_[yd,yo];base_theta=frozen.fit_logistic(base_x,base_y,1.)
            pbase=expit(test@base_theta);baseline=float(np.mean((pbase-pt)**2));base_precision=np.eye(2)+ox.T@ox
            for price_case,cost in state['costs'].items():
                state_id=f'seed{seed}_b{int(b)}_{price_case}';locked=json.loads((private/'selection_locks'/f'{state_id}.json').read_text())
                optimum=next(c['exact_best_gaussian_variance_decrease'] for c in controls if c['state_id']==state_id)
                for method,raw_ids in locked['selections'].items():
                    ids=np.asarray(raw_ids,int);semantic_ids=np.sort(perm[ids]);gauss_cov=frozen.inverse(h+u[ids].T@u[ids])
                    expected=float(np.trace(g@gauss_cov));reduction=float(np.trace(g@frozen.inverse(h)))-expected
                    helper=frozen.set_value(h,g,u,ids);max_variance_identity_error=max(max_variance_identity_error,abs(reduction-helper))
                    cov=frozen.inverse(h)
                    for i in ids:cov=frozen.update(cov,u[i])
                    max_inverse_error=max(max_inverse_error,float(np.max(abs(cov-gauss_cov))))
                    gaussian_mean=gauss_cov@(ox.T@ogy+u[ids].T@candidate_gaussian_y[ids]);real_gauss=float((gaussian_mean[0]-truth[0])**2)
                    # Canonical ordering permits exact reuse of identical training actions.
                    ordered=np.asarray(sorted(ids,key=lambda i:int(perm[i])),int);xx=np.vstack([base_x,u[ordered]]);yy=np.r_[base_y,yc[ordered]]
                    key=digest({'x':digest(xx),'y':yy,'ridge':1.})
                    if key in cache:theta=cache[key];hits+=1
                    else:
                        theta=frozen.fit_logistic(xx,yy,1.);cache[key]=theta
                    # Reuse coefficients, never a different seed's evaluation metrics.
                    p=expit(test@theta)
                    entry={'lr_excess_brier':float(np.mean((p-pt)**2)),'lr_brier':float(np.mean((p-yt)**2))}
                    evaluation_key=digest({'fit_key':key,'test_probability':digest(pt),'test_y':digest(yt)})
                    target=private/'predictions'/f'{evaluation_key}.npz';target.parent.mkdir(exist_ok=True)
                    if not target.exists():
                        np.savez_compressed(target,prediction=p,y=yt,true_probability=pt,train_x=xx,train_y=yy,selected_semantic_ids=semantic_ids)
                    assert len(ids)==len(np.unique(ids)) and cost[ids].sum()<=config['budget']+1e-10
                    assert reduction<=optimum+1e-12
                    rows.append({'state_id':state_id,'seed':seed,'ordinary_b':b,'price_case':price_case,'method':method,
                       'selected_semantic_ids':semantic_ids.tolist(),'count':len(ids),'cost':float(cost[ids].sum()),'budget':config['budget'],
                       'selected_hash':digest(semantic_ids),'exact_gaussian_variance_decrease':reduction,'exact_gaussian_posterior_risk':expected,
                       'gap_to_exhaustive':max(0.,optimum-reduction),'realized_gaussian_target_MSE':real_gauss,
                       'lr_baseline_excess_brier':baseline,'lr_gain_vs_ordinary':baseline-entry['lr_excess_brier'],
                       'lr_fit_key':key,'lr_evaluation_key':evaluation_key,'actual_backend':'cpu','selected_from_hidden_labels':False,**entry})
        print('E2a small seed',seed,'endpoints',len(rows),'unique LR fits',len(cache),flush=True)
    frame=pd.DataFrame(rows);reference=frame[frame.method=='Vopt-single-noexchange'][['state_id','exact_gaussian_variance_decrease','lr_excess_brier']].rename(
       columns={'exact_gaussian_variance_decrease':'singleton_gaussian_gain','lr_excess_brier':'singleton_lr_error'})
    paired=frame.merge(reference,on='state_id',validate='many_to_one');paired['extra_gaussian_gain_vs_singleton']=paired.exact_gaussian_variance_decrease-paired.singleton_gaussian_gain
    paired['lr_extra_gain_vs_singleton']=paired.singleton_lr_error-paired.lr_excess_brier
    write_jsonl(out/'branch_outcomes.jsonl',paired.to_dict(orient='records'));write_jsonl(out/'mechanism_controls.jsonl',controls)
    summary=paired.groupby(['ordinary_b','price_case','method'],as_index=False).agg(
       exact_gaussian_gain=('exact_gaussian_variance_decrease','mean'),gap_to_exhaustive=('gap_to_exhaustive','mean'),
       extra_gaussian_gain_vs_singleton=('extra_gaussian_gain_vs_singleton','mean'),lr_gain_vs_ordinary=('lr_gain_vs_ordinary','mean'),
       lr_extra_gain_vs_singleton=('lr_extra_gain_vs_singleton','mean'),LR_positive_seed_count=('lr_extra_gain_vs_singleton',lambda v:int((v>1e-12).sum())),
       LR_positive_vs_ordinary_count=('lr_gain_vs_ordinary',lambda v:int((v>1e-12).sum())))
    summary.to_csv(out/'descriptive_summary.csv',index=False)
    bridge=summary[summary.method=='BRIDGE-allpairs-noexchange']
    controls_frame=pd.DataFrame(controls);controls_frame.to_csv(out/'mechanism_controls.csv',index=False)
    max_prediction_error=0.
    for row in rows:
        p=np.load(private/'predictions'/f"{row['lr_evaluation_key']}.npz")
        max_prediction_error=max(max_prediction_error,abs(float(np.mean((p['prediction']-p['true_probability'])**2))-row['lr_excess_brier']),
             abs(float(np.mean((p['prediction']-p['y'])**2))-row['lr_brier']))
    assert max_prediction_error<1e-12
    verify={'passed':True,'endpoint_slots':len(rows),'independent_seed_clusters':3,'actual_unique_LR_fits':len(cache),'LR_fit_cache_hits':hits,
       'new_GPU_fits':0,'max_gaussian_variance_identity_error':max_variance_identity_error,'max_sequential_inverse_error':max_inverse_error,
       'max_saved_prediction_metric_error':max_prediction_error,'cached_coefficients_rescored_in_each_seed':True,
       'checks':['all 4096 subsets enumerated per cell','exact target variance, not a surrogate under Gaussian conjugacy',
       'analytic core-pair Gamma threshold b=L^2-1','same cost multiset under price permutations','all choices locked before outcomes',
       'same unchanged selected S evaluated with Bernoulli-logistic targets','identical LR training cached','all unfavorable seeds/conditions retained'],
       'full_300_endpoint_grid_run':False,'confirmation':False}
    assert max_inverse_error<1e-10 and max_variance_identity_error<1e-12
    write_json(out/'verification.json',verify);write_json(out/'completion.json',{'completed':True,**verify})
    write_json(out/'decision.json',{'positive_control_completed':True,'full_10_seed_grid_completed':False,
       'real_credit_candidate_promoted':False,'formal_confirmation_started':False,
       'interpretation':'Exact Gaussian optimization validity and nonlinear LR transfer are separate endpoints; no transfer claim follows from a proxy-positive control.'})
    text=f'''# E2a 小型正控制：精确优化与逻辑回归迁移

本轮补齐原计划尚未执行的精确优化正控制。固定 3 个既有种子、3 档普通信息、2 种成本安排、5 种选择器，共 90 个方法端点，但独立生成种子只有 3 个。原协议建议的 10 种子/300 端点完整网格未运行，本轮是小型开发试验。

这是人为构造的二维机制对照，不是信用真实数据。Gaussian 先验精度为 I、观测噪声方差为 1，目标方向 G=diag(1,0)。四条普通观测只有第二方向信息，总精度为 H=diag(1,1+b)。核心候选为 (1,10)、(1,-10)，诱饵为 (0.1,0)，另有九个按预注册种子生成的弱目标/强干扰候选。b=0/99/198，核心对的互补临界值恰为 L²−1=99。

成本多重集始终相同：六个成本 1、六个成本 2，总预算 2。仅交换预设候选的成本位置，使核心对从可同时采集变为不可同时采集，不按结果改成本。候选物理顺序由固定种子置换，防止 MinRisk 因编号默认选中核心对。

比较 MinRisk、单点 Vopt、全配对 BRIDGE 构造、单次固定随机可行采集及穷举最优。两种贪婪都禁用交换，以隔离成对构造。每个池枚举全部 4096 个子集；普通基准解析计算。Gaussian 指标是精确后验目标方差下降，即该共轭模型下的条件 Bayes 预测风险下降，不是把一般代理改善冒称真实预测改善。另记录单个真实参数/噪声实现的 Gaussian 预测误差，二者不能混称。

随后把完全相同的选集交给 ridge=1 的逻辑回归，标签按同一真实参数的 Bernoulli-logistic 模型生成。零特征的两条可见记录标为 0/1，只为实现常数似然的无信息起点且保证分类器支持两类，不携带参数信息。所有普通/候选标签只在全部选集锁定后生成；成本与选集不使用标签。

## BRIDGE 相对单点构造的结果

Gaussian 增益正数表示精确风险下降更多；LR 增益正数表示同选集迁移后超额 Brier 更低。

{table(bridge[['ordinary_b','price_case','extra_gaussian_gain_vs_singleton','gap_to_exhaustive','lr_extra_gain_vs_singleton','LR_positive_seed_count','LR_positive_vs_ordinary_count']],['普通精度 b','成本安排','额外 Gaussian 增益','距穷举最优差距','LR 额外增益','LR 优于单点种子数/3','LR 优于普通基线种子数/3'])}

完整五种选择器与每个种子的结果均保存在本轮输出，不删不利条件。核心 Gamma 应在 b=0 为正、b=99 为零、b=198 为负，且价格昂贵时核心对不再预算可行；原始 Gamma 与可行性在 mechanism_controls.csv。

## 能支持什么

精确优化成功，只能支持“在已知条件下成对构造能利用该精确目标的互补性”；不能据此推广到真实数据、任意逻辑回归或重新生长的 XGBoost。LR 迁移必须单独看上表，不以 Gaussian 正控制代替它。

同一模型的普通信息精度 b 不等于逻辑回归的实际 Fisher 信息：非线性链接和饱和会改变信息结构，正是迁移诊断要保留的差异。只有三个开发种子，不做确认性显著性或算法优越性宣称。

最终有效运行有 {len(cache)} 次不同 CPU LR 拟合、缓存复用 {hits} 次；仅复用模型系数，每个种子都用其自己的测试真概率与标签重新计分。实现核验时的未交付试跑保留于 work/next_protocol/retained，不当作独立证据。Gaussian 部分均为二维解析计算/小池穷举，无 GPU 新训练。最大精确方差恒等误差 {max_variance_identity_error:.3g}，顺序逆更新与直接逆的最大误差 {max_inverse_error:.3g}。预算、成本多重集、Gamma 阈值与隐藏标签边界检查通过。

本轮不恢复已停止的 CORR/GATE，不启动大规模确认，不因某个正控制成功就宣布真实数据算法改进。下一步依据全部正负证据整理论文主张与必要 Pro 审阅材料。
'''
    (out/'E2a_小型正控制与LR迁移报告.md').write_text(text)
    print(json.dumps(verify),flush=True);print(bridge.to_string(index=False),flush=True)


if __name__=='__main__':main()
