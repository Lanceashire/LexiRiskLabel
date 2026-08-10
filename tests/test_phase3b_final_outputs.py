from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results_phase3"

def test_lendingclub_formal_coverage_and_gpu():
    files = list((R / "raw" / "lendingclub").glob("*.csv"))
    assert len(files) == 580
    for f in files:
        x = pd.read_csv(f)
        assert len(x) == 11
        assert set(x["compute_backend"].dropna().unique()) == {"gpu"}
        assert bool(x["hidden_label_isolation"].all())
        assert bool(x["risk_ok"].all())

def test_phase3b_statistics_and_profiles():
    s = pd.read_csv(R / "PHASE3B_PRIMARY_STATS_ALL.csv")
    assert len(s) == 72
    assert set(s.dataset) == {"ppdai", "lendingclub"}
    p = pd.read_csv(R / "LENDINGCLUB_COMPUTE_PROFILE.csv")
    assert len(p) == 580
    assert (p.total_time >= 0).all()

def test_parity_artifacts_pass():
    for name in ["SOLVER_PARITY.csv", "FAVE_ACCELERATION_PARITY.csv", "LENDINGCLUB_DATASET_PARITY.csv"]:
        x = pd.read_csv(R / name)
        assert bool(x["pass"].all())