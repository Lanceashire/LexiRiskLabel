from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ig_analysis", ROOT / "experiments/information_gain/analyze_information_gain.py")
A = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(A)

def test_paired_bootstrap_deterministic():
    values = np.array([1.,2.,3.]); first = A.paired_bootstrap(values, draws=1000)
    assert first == A.paired_bootstrap(values, draws=1000) and first[0] == 2.0 and first[1] > 0

def test_primary_contrasts_first():
    assert A.CONTRASTS[0][:2] == ("LRBE-IG", "QualityFirst-IG")
    assert A.CONTRASTS[1][:2] == ("LRBE-IG", "CountOnly-MinRisk")
