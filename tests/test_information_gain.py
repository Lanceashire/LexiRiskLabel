from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ig_runner", ROOT / "experiments/information_gain/run_information_gain.py")
IG = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(IG)

def test_entropy_and_information_gain_are_nonnegative():
    np.testing.assert_allclose(IG.entropy_binary(np.array([.5])), np.log(2.0))
    score = IG.entropy_binary(np.array([.5])) - IG.entropy_binary(np.array([[.1], [.9]])).mean(axis=0)
    assert score[0] > 0

def test_seed_rule_is_deterministic_and_member_specific():
    assert IG.committee_seed(1001, 2, 3) == IG.committee_seed(1001, 2, 3)
    assert IG.committee_seed(1001, 2, 3) != IG.committee_seed(1001, 2, 4)

def test_registered_inputs_and_exposure():
    missing = [
        str(IG.BASE.DATASETS[dataset]["path"])
        for dataset in IG.CONFIG["datasets"]
        if not IG.BASE.DATASETS[dataset]["path"].is_file()
    ]
    if missing:
        pytest.skip(
            "Registered raw datasets are intentionally not distributed; "
            "materialize them before running the data-level IG input audit."
        )
    for dataset in IG.CONFIG["datasets"]:
        assert IG.validate_registered_input(dataset)["validated"]

def test_ppdai_is_separate_reconstruction():
    assert IG.REGISTRY["ppdai"]["evidence_role"] == "reconstructed_source_exploratory"

def test_candidate_label_shuffle_invariance():
    raw = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = np.array([0,1,0,1,0,1,1,1,0,0,1,0]); visible = np.arange(6); candidate = np.arange(6,12)
    pre = IG.p1.make_pre(raw); pre.fit(raw.iloc[visible]); pre._ecomic_x_all = np.asarray(pre.transform(raw))
    first, _ = IG.information_gain_score(raw, y, visible, candidate, pre, seed=7, round_id=1, members=3)
    shuffled = y.copy(); shuffled[candidate] = shuffled[candidate][::-1]
    second, _ = IG.information_gain_score(raw, shuffled, visible, candidate, pre, seed=7, round_id=1, members=3)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)

def test_jaccard():
    assert IG.jaccard(np.array([], int), np.array([], int)) == 1.0
    assert IG.jaccard(np.array([1,2]), np.array([2,3])) == pytest.approx(1/3)
