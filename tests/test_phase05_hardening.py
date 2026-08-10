import inspect
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import phase0_engine as e
ROOT=Path(__file__).resolve().parents[1]


class IdentityPre:
    def transform(self, frame):
        return frame.to_numpy(float)


class MockModel:
    def predict_proba(self, x):
        p = 1.0 / (1.0 + np.exp(-x[:, 0]))
        return np.column_stack([1.0 - p, p])


def frozen_reference_score(raw, visible, rejected, reference, pre, model, gamma=1.0, delta=0.5):
    Zv=pre.transform(raw.iloc[visible]); Zr=pre.transform(raw.iloc[rejected]); Zf=pre.transform(raw.iloc[reference])
    pv=np.clip(model.predict_proba(Zv)[:,1],1e-9,1-1e-9)
    pr=np.clip(model.predict_proba(Zr)[:,1],1e-9,1-1e-9)
    pf=np.clip(model.predict_proba(Zf)[:,1],1e-9,1-1e-9)
    V=np.column_stack([np.ones(len(Zv)),Zv]); R=np.column_stack([np.ones(len(Zr)),Zr]); F=np.column_stack([np.ones(len(Zf)),Zf])
    wv=np.clip(pv*(1-pv),1e-6,None)
    wi=np.clip(pr*(1-pr)*(pr**delta),1e-10,None)
    Vw=V*np.sqrt(wv)[:,None]
    H=np.einsum("ni,nj->ij",Vw,Vw)
    ridge=max(float(np.trace(H)/H.shape[0])*1e-6,1e-10)
    Hinv=e._ridge_inverse(H+ridge*np.eye(H.shape[0]))
    G=np.einsum("n,ni,nj->ij",pf**gamma,F,F)/max(len(F),1)
    HG=np.einsum("ij,jk,kl->il",Hinv,G,Hinv)
    lev=np.einsum("ij,jk,ik->i",R,Hinv,R)
    num=np.einsum("ij,jk,ik->i",R,HG,R)
    return np.maximum(wi*num/(1+wi*lev),0.0)


def test_fave_api_has_no_hidden_labels():
    assert "y" not in inspect.signature(e.fave_value).parameters
    assert "y" not in inspect.signature(e.select_round).parameters


def test_frozen_fave_numerical_parity():
    raw = pd.DataFrame({"x": [-2., -1., 0., 1., 2., 3., -3., 0.5],
                        "z": np.arange(8, dtype=float)})
    visible=np.array([0,1,2,3]); rejected=np.array([4,5]); reference=np.array([4,5,6,7])
    expected=frozen_reference_score(raw,visible,rejected,reference,IdentityPre(),MockModel())
    actual=e.fave_value(raw,visible,rejected,reference,IdentityPre(),MockModel())
    assert np.allclose(actual,expected,rtol=1e-12,atol=1e-12)
    audit=pd.read_csv(ROOT/"results_phase0/fave_score_audit.csv")
    assert set(audit.loc[audit.item=="reference_zip","status"])=={"PASS"}
    assert set(audit.loc[audit.item=="gamma","value"])=={"1.0"}
    assert set(audit.loc[audit.item=="delta","value"])=={"0.5"}


def test_fave_score_is_finite_nonnegative():
    raw = pd.DataFrame({"x": [-2., -1., 0., 1., 2., 3., -3., 0.5], "z": np.arange(8, dtype=float)})
    values = e.fave_value(raw, np.array([0,1,2,3]), np.array([4,5]), np.array([4,5,6,7]), IdentityPre(), MockModel())
    assert values.shape == (2,)
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_hidden_label_shuffle_invariance():
    old_device=e.DEVICE
    e.DEVICE="cpu"
    try:
        raw=pd.DataFrame({"x":np.linspace(-2,2,16),"z":np.sin(np.linspace(-2,2,16))})
        visible=np.arange(8); batch=np.arange(8,16)
        y1=np.array([0,1,0,1,0,0,1,0, 1,0,1,0,1,0,0,1])
        y2=y1.copy(); y2[8:]=y2[8:][::-1]
        pre1,m1,p1=e.fit_model(raw,y1,visible,101)
        pre2,m2,p2=e.fit_model(raw,y2,visible,101)
        assert np.allclose(p1,p2,rtol=0,atol=0)
        cand=batch[p1[batch]>=e.TAU]
        if len(cand):
            costs=p1[cand]
            score1=e.fave_value(raw,visible,cand,batch,pre1,m1)
            score2=e.fave_value(raw,visible,cand,batch,pre2,m2)
            a,k1=e.select_round("FAVE-v2.1",p1[cand],costs,0.5*costs.sum(),score1)
            b,k2=e.select_round("FAVE-v2.1",p2[cand],costs,0.5*costs.sum(),score2)
            assert k1==k2 and np.array_equal(a,b)
    finally:
        e.DEVICE=old_device


def test_fave_is_distinct_from_risk_tilted_uncertainty():
    p=np.array([0.2,0.5,0.8]); c=np.array([1.,2.,3.])
    assert not np.allclose(e.policy_score("RiskTilted-Uncertainty",p,c), e.policy_score("Uncertainty",p,c))
    assert "FAVE-v2.1" in e.POLICIES



