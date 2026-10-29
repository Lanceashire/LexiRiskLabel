"""One registered-state metamorphic check of the formal six-selector boundary."""
from runtime import *
import run_formal_md as engine
import argparse


def choose(method,s,ids,params,fit_seed,cfg):
    d,o,c=ids
    if method=='ParBaLS':return engine.select_neighbor(s,{'universes':cfg['neighbor_universes'],'posterior_draws':cfg['neighbor_posterior_draws']})[0],0
    if method=='FantasyCandidate':
        selected,diag=engine.fantasy_select(s,d,o,c,params,fit_seed,cfg);return selected,diag['actual_fantasy_GPU_fits']
    if method=='LRBE-IG':
        rng=np.random.default_rng(s.seed);predictions=[];fits=0
        for member in range(cfg['IG_bootstrap_members']):
            sampled=rng.integers(0,len(d),len(d));y=s.visible_y[sampled]
            if len(np.unique(y))<2:predictions.append(np.full(len(c),y.mean()));continue
            m=engine.gpu_fit(s.visible_x[sampled,1:],y,params,s.seed+member);fits+=1
            predictions.append(m.predict_proba(s.candidate_x[:,1:].astype(np.float32))[:,1])
        p=np.asarray(predictions);utility=np.maximum(frozen.entropy(p.mean(axis=0))-frozen.entropy(p).mean(axis=0),0)
        return frozen.select(method,s,utility).selected_ids,fits
    return frozen.select(method,s).selected_ids,0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);a=ap.parse_args();base=a.base;out=base/'formal_md'
    identity=json.loads((out/'frozen_identity.json').read_text());cfg=identity['config'];folder=base/'private/formal_md/cases/hmeq_repeat1_fold1_stream1'
    manifest=json.loads((folder/'state_manifest.json').read_text());split=np.load(folder/'split.npz');tune=json.loads((folder/'initial_tuning.json').read_text())
    lock=json.loads((folder/(folder.name+'_rho0.1_OrdinaryOnly')/'round_01_selection.json').read_text())
    raw,y,_,_,_,_,_=engine.dataset('hmeq');pre=legacy.preprocessor(raw);pre.fit(raw.iloc[split['D0']])
    d,o,c,q=[np.asarray(lock[k],int) for k in ['D_ids','O_ids','C_ids','reference_ids']]
    hidden=np.r_[o,c,split['outer_test']];assert not np.intersect1d(hidden,d).size
    altered=y.copy();altered[hidden]=1-altered[hidden]
    def state(labels):
        def transform(ids):return np.c_[np.ones(len(ids)),np.asarray(pre.transform(raw.iloc[ids]),float)]
        dx,ox,cx,qx=[transform(i) for i in [d,o,c,q]];theta=frozen.fit_logistic(dx,labels[d],tune['ridge'])
        return frozen.State(dx,labels[d],ox,cx,qx,np.asarray(lock['costs']),lock['B'],theta,manifest['seed']+1000,ridge=tune['ridge'])
    original,changed=state(y),state(altered)
    for name in ['visible_x','visible_y','ordinary_x','candidate_x','reference_x','theta','cost']:
        assert np.array_equal(getattr(original,name),getattr(changed,name))
    results=[];gpu=0
    for method in cfg['main_methods']:
        left,n1=choose(method,original,(d,o,c),tune['xgb'],manifest['seed']+1,cfg)
        right,n2=choose(method,changed,(d,o,c),tune['xgb'],manifest['seed']+1,cfg);gpu+=n1+n2
        assert np.array_equal(np.sort(left),np.sort(right)),method+' depends on hidden labels or nondeterministic selection'
        results.append({'selector':method,'same_selection':True,'selected_hash':digest(np.sort(c[left])),'committee_and_fantasy_recomputed':True})
    assert all(sha(ROOT/k)==h for k,h in identity['sources'].items())
    write_json(out/'formal_selector_visibility_audit.json',{'passed':True,'state':'hmeq_repeat1_fold1_stream1_round1',
        'mutation':'simultaneously flip O/C and outer T labels, retain D labels/features/prices',
        'selectors':results,'actual_audit_GPU_fits':gpu,'outcome_trajectories_modified':False,
        'audit_fits_are_not_new_experiments':True,'source_identity_unchanged':True,'auditor_sha256':sha(__file__)})
    print(json.dumps({'passed':True,'selectors':len(results),'actual_audit_GPU_fits':gpu,'existing_predictions_unchanged':True}))


if __name__=='__main__':main()
