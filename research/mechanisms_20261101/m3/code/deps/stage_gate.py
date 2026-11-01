"""Mandatory phase guard for newly executed mechanism research."""
from pathlib import Path
import json

GOVERNANCE=Path(__file__).resolve().parent/'stage_governance.json'

def require_stage(stage,governance=None):
    policy=json.loads(GOVERNANCE.read_text()) if governance is None else governance
    if stage!=policy['current_stage']:
        raise RuntimeError(f"Stage gate blocks {stage}: current incomplete stage is {policy['current_stage']}")
    for earlier in policy['order'][:policy['order'].index(stage)]:
        if not policy['historical_status'][earlier].startswith('complete;'):
            raise RuntimeError(f'Stage gate blocks {stage}: {earlier} is not complete')
    return policy

def validate_acceptance(stage,evidence,policy=None):
    policy=require_stage(stage,policy)
    missing=[k for k in policy[f'{stage}_acceptance_items'] if evidence.get('items',{}).get(k) is not True]
    unresolved=[r for r in evidence.get('mandatory_mechanisms',[]) if r.get('verdict') in [None,'unresolved','pending']]
    if missing or unresolved or not evidence.get('mandatory_mechanisms'):
        raise RuntimeError(f'Stage remains incomplete: {len(missing)} missing items, {len(unresolved)} unresolved mechanisms')
    return True
