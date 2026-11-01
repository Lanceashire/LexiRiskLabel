"""Ex ante explanation model. Fit on M4 development only; never selects data."""
from common import *
from sklearn.linear_model import Ridge
M4=HERE.parent/'m4_study/family_confirmation'
def main():
    summary=pd.read_csv(M4/'world_summary.csv');rows=[]
    for w in sorted(summary.world_id.unique()):
        p=M4/f'w{w}';z=np.load(p/'inputs.npz');locks=json.loads((p/'selection_locks.json').read_text())
        for direction in ['target','unlock']:
            condition='linear_'+direction
            s=core.State(z['linear_D'],z['linear_yD'],z[condition+'_O'],z['linear_C'],z['linear_Q'],z['linear_cost'],float(z['linear_B']),z['linear_theta'],int(w),ridge=1.,exchanges=0)
            sets={a['method']:np.asarray(a['selected_indices'],int) for a in locks if a['condition']==condition}
            features,scale=indicators(s,sets)
            for learner in ['LR','XGB']:
                r=summary[(summary.world_id==w)&(summary.condition==condition)&(summary.learner==learner)].iloc[0]
                rows.append({'world':int(w),'condition':condition,'learner':learner,'scale':scale,'gain':r.Pending_aware_gain,
                             **dict(zip(FEATURES,features))})
    df=pd.DataFrame(rows);models={}
    for learner in ['LR','XGB']:
        d=df[df.learner==learner];x=d[FEATURES].to_numpy();y=d.gain.to_numpy()/d.scale.to_numpy();mean=x.mean(axis=0);sd=x.std(axis=0);sd=np.where(sd>1e-12,sd,1.)
        fit=Ridge(alpha=1.,fit_intercept=True).fit((x-mean)/sd,y)
        models[learner]={'feature_mean':mean,'feature_sd':sd,'coef':fit.coef_,'intercept':fit.intercept_,
            'constant_baseline':float(y.mean()),'training_feature_min':x.min(axis=0),'training_feature_max':x.max(axis=0),
            'training_worlds':512,'training_states':1024,'training_only_MAE':float(np.mean(np.abs(fit.predict((x-mean)/sd)-y))),
            'training_performance_is_not_validation':True}
    df.to_csv(HERE/'diagnostic_development.csv',index=False)
    write(HERE/'diagnostic_frozen.json',{'purpose':'Explain and predict external consistency, never new acquisition policy','features':FEATURES,'normalizer':'trace(G inverse(H_D))',
        'ridge_alpha':1.,'new_task_labels_used':False,'M4_summary_sha256':sha(M4/'world_summary.csv'),'source_sha256':sha(__file__),'models':models})
    print('FROZEN',len(df),'old-state learner rows; new labels used: false',flush=True)
def predict(s,learner,sets=None):
    cfg=json.loads((HERE/'diagnostic_frozen.json').read_text());m=cfg['models'][learner];x,scale=indicators(s,sets)
    mean=np.asarray(m['feature_mean']);sd=np.asarray(m['feature_sd']);coef=np.asarray(m['coef'])
    pred=float(((x-mean)/sd)@coef+m['intercept'])*scale
    out=(x<np.asarray(m['training_feature_min']))|(x>np.asarray(m['training_feature_max']))
    return {'predicted_gain':pred,'constant_prediction':m['constant_baseline']*scale,'features':x,'scale':scale,'out_of_M4_range':out}
if __name__=='__main__':main()
