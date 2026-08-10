import inspect
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import phase0_engine as e


class IdentityPre:
    def transform(self, frame):
        return frame.to_numpy(float)


class MockModel:
    def predict_proba(self, x):
        p = 1.0 / (1.0 + np.exp(-x[:, 0]))
        return np.column_stack([1.0 - p, p])


def test_fave_api_has_no_hidden_labels():
    assert "y" not in inspect.signature(e.fave_value).parameters


def test_fave_reference_score_is_finite_nonnegative():
    raw = pd.DataFrame({"x": [-2., -1., 0., 1., 2., 3., -3., 0.5], "z": np.arange(8, dtype=float)})
    values = e.fave_value(
        raw,
        visible=np.array([0, 1, 2, 3]),
        rejected=np.array([4, 5]),
        reference=np.array([4, 5, 6, 7]),
        pre=IdentityPre(),
        model=MockModel(),
    )
    assert values.shape == (2,)
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_fave_is_distinct_from_risk_tilted_uncertainty():
    p = np.array([0.2, 0.5, 0.8])
    c = np.array([1., 2., 3.])
    assert not np.allclose(
        e.policy_score("RiskTilted-Uncertainty", p, c),
        e.policy_score("Uncertainty", p, c),
    )
    assert "FAVE-v2.1" in e.POLICIES
