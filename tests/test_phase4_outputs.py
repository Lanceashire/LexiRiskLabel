from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_phase4"
ANALYSIS = OUT / "analysis"


def test_required_phase4_outputs_exist():
    required_tables = [
        "MECHANISM_DATASET_PROFILE.csv", "QUANTITY_PERFORMANCE_ANALYSIS.csv",
        "SELECTED_SET_COMPOSITION.csv", "SELECTED_SET_OVERLAP.csv",
        "THRESHOLD_CROSSING_ANALYSIS.csv", "FEEDBACK_PERFORMANCE_SENSITIVITY.csv",
        "RISK_HETEROGENEITY_ANALYSIS.csv", "POLICY_COMPLEXITY_TRADEOFF.csv",
    ]
    required_reports = [
        "PHASE4_PROTOCOL.md", "PHASE4_MECHANISM_REPORT.md",
        "PHASE4_BOUNDARY_CONDITIONS.md", "PHASE4_COUNTEREVIDENCE.md",
        "PHASE4_MECHANISM_FINDINGS.csv", "PHASE4_FINAL_SYNTHESIS.md",
        "PHASE4_INTEGRITY_AUDIT.md",
    ]
    assert all((ANALYSIS / name).is_file() for name in required_tables)
    assert all((OUT / name).is_file() for name in required_reports)
    assert len(list((OUT / "figures").glob("M*.png"))) == 11


def test_complete_cell_structure_and_dataset_separation():
    quantity = pd.read_csv(ANALYSIS / "QUANTITY_PERFORMANCE_ANALYSIS.csv")
    assert len(quantity) == 4 * 20 * 4 * 8
    assert set(quantity.dataset) == {"uci", "hmeq", "ppdai", "lendingclub"}
    correlations = pd.read_csv(ANALYSIS / "QUANTITY_PERFORMANCE_CORRELATIONS.csv")
    assert correlations.dataset.nunique() == 4
    assert "pooled" not in correlations.dataset.astype(str).str.lower().tolist()


def test_unretained_applicant_diagnostics_are_not_fabricated():
    overlap = pd.read_csv(ANALYSIS / "SELECTED_SET_OVERLAP.csv")
    assert len(overlap) == 4 * 20 * 4 * 3
    assert overlap.jaccard.isna().all()
    assert overlap.status.eq("NOT_ESTIMABLE_SELECTED_IDS_NOT_RETAINED").all()
    crossing = pd.read_csv(ANALYSIS / "THRESHOLD_CROSSING_ANALYSIS.csv")
    assert crossing.exact_default_upcrossings.isna().all()
    assert crossing.exact_default_downcrossings.isna().all()
    replay = pd.read_csv(ANALYSIS / "DIAGNOSTIC_REPLAY_AUDIT.csv")
    assert (~replay.replay_attempted).all()
    assert (~replay.replay_result_used).all()


def test_corrected_coverage_and_policy_scope():
    quantity = pd.read_csv(ANALYSIS / "QUANTITY_PERFORMANCE_ANALYSIS.csv")
    assert quantity.exploration_coverage_ratio.notna().all()
    assert quantity.exploration_coverage_ratio.between(0, 1).all()
    complexity = pd.read_csv(ANALYSIS / "POLICY_COMPLEXITY_TRADEOFF.csv")
    assert set(complexity.policy) == {
        "CountOnly-MinRisk", "RiskTilted-Uncertainty", "FAVE-v2.1"
    }


def test_frozen_input_hashes_match_pre_post():
    hashes = pd.read_csv(OUT / "PHASE4_FROZEN_INPUT_HASHES.csv")
    pre = hashes[hashes.stage.eq("PRE_PHASE4")].drop(columns="stage").sort_values("path").reset_index(drop=True)
    post = hashes[hashes.stage.eq("POST_PHASE4")].drop(columns="stage").sort_values("path").reset_index(drop=True)
    pd.testing.assert_frame_equal(pre, post)

