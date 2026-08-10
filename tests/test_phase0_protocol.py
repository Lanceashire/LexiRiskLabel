import itertools, inspect
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
import phase0_engine as e

def test_policy_identity():
    p=np.array([.1,.4,.8]); c=np.array([1.,2.,3.])
    assert not np.allclose(e.policy_score("RiskTilted-Uncertainty",p,c),e.policy_score("Uncertainty",p,c))
    assert np.allclose(e.policy_score("MostRisky",p,c),p)

def test_batch_partition():
    r=np.arange(37); b=np.array_split(r,10)
    assert len(b)==10 and all(len(x)>0 for x in b)
    assert sorted(np.concatenate(b).tolist())==r.tolist()
    assert len(set(np.concatenate(b).tolist()))==len(r)

def test_optimizer_against_bruteforce():
    cost=np.array([1.,2.,3.,4.,5.]); score=np.array([.1,.9,.3,.8,.4]); budget=7.
    chosen,k=e.optimize(cost,score,budget)
    bestk=max(sum(mask) for mask in itertools.product([0,1],repeat=len(cost)) if sum(m*c for m,c in zip(mask,cost))<=budget+1e-9)
    assert k==bestk and len(chosen)==k and cost[chosen].sum()<=budget+1e-9

def test_kstar_consistency():
    cost=np.array([1.,2.,3.,4.]); budget=5.
    ks=[e.optimize(cost,e.policy_score(p,np.ones(4),cost),budget)[1] for p in ["CountOnly","CountOnly-MinRisk","LeastRisky","MostRisky","Uncertainty","RiskTilted-Uncertainty"]]
    assert len(set(ks))==1

def test_risk_budget():
    cost=np.array([.5,1.,1.5,2.]); chosen,k=e.optimize(cost,np.ones(4),2.5)
    assert cost[chosen].sum()<=2.5+1e-9

def test_hidden_label_api():
    assert "y" not in list(inspect.signature(e.optimize).parameters)
    assert "y" not in list(inspect.signature(e.policy_score).parameters)

def test_holm_correction():
    raw=np.array([.01,.04,.2,.5]); adj=multipletests(raw,method="holm")[1]
    assert np.all(adj>=raw) and np.all(adj<=1)

def test_reproducibility_batches():
    a=np.random.RandomState(0).permutation(np.arange(100)); b=np.random.RandomState(0).permutation(np.arange(100)); assert np.array_equal(a,b)

