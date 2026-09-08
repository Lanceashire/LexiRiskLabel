from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_SPEC = importlib.util.spec_from_file_location("mechanism_runner", ROOT / "experiments/mechanism_conditions/run_mechanism_conditions.py")
RUN = importlib.util.module_from_spec(RUN_SPEC)
assert RUN_SPEC.loader is not None
RUN_SPEC.loader.exec_module(RUN)
AN_SPEC = importlib.util.spec_from_file_location("mechanism_analysis", ROOT / "experiments/mechanism_conditions/analyze_mechanism_conditions.py")
AN = importlib.util.module_from_spec(AN_SPEC)
assert AN_SPEC.loader is not None
AN_SPEC.loader.exec_module(AN)


def test_task_generation_is_deterministic_and_disjoint():
    first, second = RUN.generate_task(2999), RUN.generate_task(2999)
    np.testing.assert_array_equal(first["y"], second["y"])
    np.testing.assert_allclose(first["raw"], second["raw"])
    parts = [set(first[name].tolist()) for name in ["visible", "batch", "test0", "test1"]]
    assert all(parts[i].isdisjoint(parts[j]) for i in range(len(parts)) for j in range(i + 1, len(parts)))
    assert first["raw"].iloc[first["visible"]].region.mean() < first["raw"].iloc[first["pool"]].region.mean()


def test_exposure_manipulation_preserves_total_predicted_risk():
    pd_values = np.linspace(0.5, 0.95, 200)
    exposure = RUN.exposure_vectors(pd_values, 2999)
    assert np.isfinite(exposure["high"]).all() and (exposure["high"] > 0).all()
    np.testing.assert_allclose(np.dot(pd_values, exposure["low"]), np.dot(pd_values, exposure["high"]), rtol=1e-12)
    assert exposure["high"].std() / exposure["high"].mean() > 0.5


def test_weighted_metrics_are_endpoint_only():
    y0, p0 = np.array([0, 1]), np.array([0.1, 0.8])
    y1, p1 = np.array([0, 1]), np.array([0.4, 0.6])
    low = RUN.weighted_metrics(y0, p0, y1, p1, 0.1)
    high = RUN.weighted_metrics(y0, p0, y1, p1, 0.5)
    assert low["brier"] < high["brier"]
    assert low["brier_region0"] == high["brier_region0"]
    assert low["brier_region1"] == high["brier_region1"]


def test_budget_identity():
    budget, mk, ms, selected = 10.0, 8.0, 5.0, 9.0
    slack, released, premium, unused = budget - mk, mk - ms, selected - ms, budget - selected
    assert abs((slack + released) - (premium + unused)) < 1e-12


def test_bootstrap_is_deterministic():
    values = np.array([1.0, 2.0, 3.0])
    assert AN.paired_bootstrap(values, draws=1000) == AN.paired_bootstrap(values, draws=1000)
