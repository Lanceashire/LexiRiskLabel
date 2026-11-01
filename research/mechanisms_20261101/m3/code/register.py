"""Separate registrations; no confirmatory outcomes used to choose sample size."""
from pathlib import Path
import copy,json,sys
from study import HERE,sha,write

def main():
    mode=sys.argv[1];original=Path('LOCAL_WORKSPACE/2026-10-31/zhe/work/LexiRiskLabel/experiments/mechanism_v2')
    if mode=='engineering':
        acceptance=Path('LOCAL_WORKSPACE/2026-10-31/ji-x/work/M2_mechanism_bridge/M2_FINAL_ACCEPTANCE.json')
        ev=json.loads(acceptance.read_text());assert ev['M2_complete'] and all(ev['items'].values())
        policy={'version':'M3_authorized_20261101_v1','order':['M0','M1','M2','M3','M4','M5'],'current_stage':'M3',
                'historical_status':{'M0':'complete; prior interface and evidence audits','M1':'complete; exact bounded study','M2':'complete; eight original terminal verdicts, bounded reversal studies with unresolved strong claims retained'},
                'M3_training_authorized':True,'authorization':'那就继续做m3把，然后把m3做完来再进行一次推断把材料综合提交到github上面去',
                'parent_original_governance_sha256':sha(original/'stage_governance.json'),'original_M2_acceptance_sha256':sha(acceptance),
                'unresolved_not_promoted':['XGB unique directional root','single histogram effect','universal estimator explanation'],'later_stages_authorized':False}
        write(HERE/'stage_governance_v1.json',policy)
    cfg=copy.deepcopy(json.loads((original/'mechanism_protocol.json').read_text())['pilot']);cfg['ordinary_n']=4000;cfg['xgb']['max_bin']=256
    assert json.loads((HERE/'world_id_collision_audit.json').read_text())['passed']
    source_files=['study.py','analyze.py','driver.py','engineering_checks.py','register.py','deps/core.py','deps/exact.py','deps/stage_gate.py','stage_governance_v1.json','world_id_collision_audit.json']
    base={'version':'M3_cost_pairing_v1_20261101','registration_frozen':True,'config':cfg,
          'cost_conditions':['neutral','information_cheap','information_expensive'],'delta':.0001,'primary_family':2,'primary_alpha':.04,'secondary_family':50,'secondary_alpha':.01,
          'primary_estimand':'price = [R(MinRisk)-R(Pending-aware)]_information_cheap - [R(MinRisk)-R(Pending-aware)]_information_expensive',
          'primary_direction':'two-sided','label_rng_tag':94001,'matching_solver':{'time_limit':.5,'mip_rel_gap':.1},
          'same_world_common_labels_and_features_across_costs_and_methods':True,'same_cost_multiset_and_absolute_budget':True,
          'effective_EAD':'cost/0.75, exposure multiset preserved while individual exposure pairing changes',
          'budget_interpretation':'sum business-model PD times effective EAD, not a realized loss guarantee',
          'comparators':'D-only/PA are classic target-weighted information design instances; Static versions isolate batch updating; GainBudget-Pending is a raw marginal gain under feasibility comparator, not a reproduction of ConBatch-BAL or ParBaLS',
          'claim_scope':'cost-pairing mechanism in original M3 anchor, no new SOTA/ParBaLS superiority or full overall-plan completion claim',
          'stop_rule':'all fixed registered worlds once; no interim significance testing, no outcome-based extension or selected costs/learners',
          'terminal_rule':'at registered N report support, reverse, effect excluded, equivalence or unresolved. Completed bounded study never means all mechanisms positive or identified.',
          'planning':{'pilot_worlds':32,'confirmation_label_M':4,'min_confirmation_worlds':256,'max_confirmation_worlds':4096,'target_halfwidth':.0001,'rounding_grid':256,'rule':'max across learners of [z_(1-.04/4) * upper95_SD / .0001]^2; chi-square Gaussian SD bound; cap fixed before pilot; use first4 of8 label reps to match confirmation variance, never use effect mean'},
          'parent_protocol_sha256':sha(original/'mechanism_protocol.json'),'source_hashes':{p:sha(HERE/p) for p in source_files}}
    if mode=='engineering':base.update(role='engineering',label_M=1,world_ids=[19900001,19900002])
    elif mode=='pilot':
        assert json.loads((HERE/'engineering_checks.json').read_text())['passed']
        assert json.loads((HERE/'engineering_run/completion.json').read_text())['completed']
        base.update(role='variance_pilot',label_M=8,world_ids=list(range(20000001,20000033)))
    elif mode=='confirmation':
        p=json.loads((HERE/'pilot_analysis/variance_planning.json').read_text());n=p['chosen_confirmation_n']
        base.update(role='confirmation',label_M=4,world_ids=list(range(21000001,21000001+n)),confirmation_n=n,pilot_planning_sha256=sha(HERE/'pilot_analysis/variance_planning.json'),pilot_not_pooled=True)
    else:raise ValueError(mode)
    path=HERE/f'{mode}_registration.json';assert not path.exists(),('registration already exists',path)
    write(path,base);print(path,sha(path),flush=True)
if __name__=='__main__':main()
