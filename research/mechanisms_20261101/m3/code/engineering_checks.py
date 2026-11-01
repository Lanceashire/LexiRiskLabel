"""Checks of the estimand boundary and unchanged generator, before new outcomes."""
import json,sys
from pathlib import Path
from dataclasses import replace
import numpy as np
from study import *

def main():
    legacy=Path('LOCAL_WORKSPACE/2026-10-31/zhe/work/LexiRiskLabel/experiments/mechanism_v2')
    sys.path.insert(0,str(legacy));from run_paired_mechanism_states import generate,action_family
    old=json.loads((legacy/'mechanism_protocol.json').read_text())['pilot'];cfg=dict(old,ordinary_n=4000)
    opts={'time_limit':.5,'mip_rel_gap':.1};reports=[]
    for w in [19900001,19900002]:
        e=environment(w,cfg);prior,T,yT,pT,exp=generate(w,old)
        ss={x[3]:x[4] for x in prior if x[1]=='target' and x[2]==4000}
        for c in COSTS:
            s=state(e,e['costs'][c],w,cfg);p=ss[c]
            for key,a,b in [('D',s.visible_x,p.visible_x),('yD',s.visible_y,p.visible_y),('O',s.ordinary_x,p.ordinary_x),('C',s.candidate_x,p.candidate_x),('Q',s.reference_x,p.reference_x),('cost',s.cost,p.cost),('T',e['T'],T),('pT',e['pT'],pT),('theta',s.theta,p.theta)]:
                assert np.array_equal(a,b),(w,c,key)
            assert s.budget==p.budget
            acts,status=acquire(s,e['candidate_ids'],opts);oldacts=action_family(p,e['candidate_ids'])
            for m in oldacts:assert np.array_equal(acts[m],oldacts[m]),m
            null,_=acquire(replace(s,ordinary_x=s.ordinary_x[:0]),e['candidate_ids'],opts)
            assert np.array_equal(null['D-only'],null['Pending-aware']) and np.array_equal(null['Static-D'],null['Static-Pending'])
            zero,_=acquire(replace(s,budget=0),e['candidate_ids'],opts)
            assert all(v is None or len(v)==0 for v in zero.values())
            permutation=np.random.default_rng(w).permutation(len(s.cost));pp=replace(s,candidate_x=s.candidate_x[permutation],cost=s.cost[permutation])
            pa,_=acquire(pp,e['candidate_ids'][permutation],opts)
            for m in METHODS:
                if m=='CostCountMatched':continue # time-limited solver incumbent need not be unique
                assert np.array_equal(np.sort(e['candidate_ids'][acts[m]]),np.sort(e['candidate_ids'][permutation][pa[m]])),m
            # Hidden O/C/Y and test arrays are not State fields or selector args.
            assert not any('test' in k or k in ['yO','yC','truth','pT','T'] for k in vars(s))
            assert len(acts['FixedK-Pending'])==core.kstar(s.cost,s.budget)
            reports.append({'world':w,'cost':c,'generator_exact_legacy_parity':True,'original_four_selectors_identical':True,'emptyO_null':True,'zero_budget':True,'stableID_invariance_nonmatching_methods':True,'cost_matching_valid':status['valid_incumbent']})
    write(HERE/'engineering_checks.json',{'passed':True,'checks':reports,'future_labels_not_an_acquisition_argument':True,'confirmatory_worlds_used':0,'model_outcome_training_run':False})
    print('engineering boundary and legacy parity checks passed',flush=True)
if __name__=='__main__':main()
