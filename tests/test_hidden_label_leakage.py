import inspect
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import phase0_engine as e

def test_policy_api_has_no_hidden_labels():
    assert "y" not in inspect.signature(e.optimize).parameters
    assert "y" not in inspect.signature(e.policy_score).parameters

def test_oracle_metrics_are_separate():
    assert "selected" in inspect.signature(e.eval_metrics).parameters or "visible" in inspect.signature(e.eval_metrics).parameters
