from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

RES=Path(__file__).resolve().parents[1]/'results_phase1'
out=pd.read_csv(RES/'dual_evaluation_all_rounds.csv')
orig=pd.concat([pd.read_csv(RES/'uci_all_rounds.csv').assign(dataset='uci'),pd.read_csv(RES/'hmeq_all_rounds.csv').assign(dataset='hmeq')],ignore_index=True)
ext=out[out.evaluation_scope=='external_fixed'].copy(); inte=out[out.evaluation_scope=='internal_visible'].copy()
metrics=['accuracy','precision','recall','specificity','balanced_accuracy','gmean','roc_auc','pr_auc','brier','ece','calibration_slope','calibration_intercept']
par=[]
for _,r in ext.iterrows():
    q=orig[(orig.dataset==r.dataset)&(orig.seed==r.seed)&(orig['round']==r['round'])&(orig.treatment==r.treatment)].iloc[0]
    for m in metrics:
        if np.isfinite(r[m]) and np.isfinite(q[m]): par.append(abs(float(r[m])-float(q[m])))
pd.DataFrame([{'max_abs_diff':max(par),'n_compared':len(par),'passed':max(par)<=1e-8}]).to_csv(RES/'DUAL_EVAL_PARITY.csv',index=False)
gap=[]
for _,r in inte.iterrows():
    q=ext[(ext.dataset==r.dataset)&(ext.seed==r.seed)&(ext['round']==r['round'])&(ext.treatment==r.treatment)].iloc[0]
    row={'dataset':r.dataset,'seed':r.seed,'round':r['round'],'treatment':r.treatment}
    for m in ['accuracy','recall','specificity','balanced_accuracy','roc_auc','pr_auc','brier','ece']:
        row[m+'_gap_internal_minus_external']=float(r[m]-q[m])
    gap.append(row)
pd.DataFrame(gap).to_csv(RES/'dual_evaluation_gap.csv',index=False)
stats=[]
for name in ['uci','hmeq']:
    for m in ['recall','balanced_accuracy','roc_auc','pr_auc']:
        vals=[]
        for seed in range(20):
            ni=inte[(inte.dataset==name)&(inte.seed==seed)&(inte.treatment=='NoExplore')&(inte['round']==10)][m].iloc[0]
            oi=inte[(inte.dataset==name)&(inte.seed==seed)&(inte.treatment=='Oracle')&(inte['round']==10)][m].iloc[0]
            ne=ext[(ext.dataset==name)&(ext.seed==seed)&(ext.treatment=='NoExplore')&(ext['round']==10)][m].iloc[0]
            oe=ext[(ext.dataset==name)&(ext.seed==seed)&(ext.treatment=='Oracle')&(ext['round']==10)][m].iloc[0]
            vals.append((ni-oi)-(ne-oe))
        vals=np.asarray(vals,float); p=float(wilcoxon(vals).pvalue) if np.any(vals!=0) else 1.0
        rng=np.random.RandomState(880000+len(stats)); boots=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(10000)])
        stats.append({'dataset':name,'round':10,'metric':m,'mean_distortion_gap':float(vals.mean()),'median_distortion_gap':float(np.median(vals)),'bootstrap_ci_low':float(np.quantile(boots,.025)),'bootstrap_ci_high':float(np.quantile(boots,.975)),'wilcoxon_p':p})
st=pd.DataFrame(stats); st['holm_p']=st.groupby('dataset')['wilcoxon_p'].transform(lambda s:multipletests(s,method='holm')[1]); st.to_csv(RES/'dual_evaluation_statistics.csv',index=False)
print('postprocess complete',pd.read_csv(RES/'DUAL_EVAL_PARITY.csv').to_dict('records'))

