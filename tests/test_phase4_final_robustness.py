from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_phase4"
ANALYSIS = OUT / "analysis"


def test_final_sensitivity_design_and_cluster_bootstrap():
    data = pd.read_csv(ANALYSIS / "FEEDBACK_PERFORMANCE_SENSITIVITY_FINAL.csv")
    assert len(data) == 4 * (4 + 1 + 1) * 2
    assert set(data.analysis_type) == {
        "WITHIN_BUDGET", "BUDGET_CENTERED", "ACROSS_BUDGET_DESCRIPTIVE"
    }
    assert data.n_seeds.eq(20).all()
    assert data.bootstrap_iterations.eq(10_000).all()
    within = data[data.analysis_type.eq("WITHIN_BUDGET")]
    assert within.groupby(["dataset", "metric_pair"]).size().eq(4).all()
    pooled = data[data.analysis_type.eq("ACROSS_BUDGET_DESCRIPTIVE")]
    assert pooled.interpretation.str.contains("BUDGET_CONFOUNDED_DESCRIPTIVE").all()


def test_quantity_analysis_handles_constant_input_honestly():
    data = pd.read_csv(ANALYSIS / "QUANTITY_PERFORMANCE_ANALYSIS_FINAL.csv")
    assert len(data) == 48
    constant = data[data.estimability_status.eq("CONSTANT_INPUT_NOT_ESTIMABLE")]
    assert len(constant) == 2
    assert constant.dataset.eq("ppdai").all()
    assert constant.rho.astype(str).eq("0.025").all()
    assert constant.spearman_rho.isna().all()
    assert constant.cluster_bootstrap_ci_low.isna().all()
    assert constant.cluster_bootstrap_ci_high.isna().all()


def test_matched_quantity_table_is_complete():
    matched = pd.read_csv(ANALYSIS / "QUANTITY_FIRST_MATCHED_COMPARISON_FINAL.csv")
    assert len(matched) == 4 * 20 * 4 * 3
    assert set(matched.policy) == {
        "CountOnly-MinRisk", "RiskTilted-Uncertainty", "FAVE-v2.1"
    }
    minrisk = matched[matched.policy.eq("CountOnly-MinRisk")]
    assert minrisk.expected_risk_excess_vs_minrisk.abs().lt(1e-10).all()


def test_m7_m8_final_terminology_and_files():
    assert (OUT / "figures" / "M7_mean_pd_true_defaults.png").is_file()
    assert (OUT / "figures" / "M8_net_threshold_status_change.png").is_file()
    assert not (OUT / "figures" / "M7_default_test_score_near_threshold.png").exists()
    assert not (OUT / "figures" / "M8_true_default_threshold_crossings.png").exists()
    report = (OUT / "PHASE4_FINAL_ROBUSTNESS_REPORT.md").read_text(encoding="utf-8")
    assert "NET THRESHOLD-STATUS CHANGE" in report
    assert "BUDGET_CONFOUNDED_DESCRIPTIVE" in report


def test_final_status_and_frozen_hashes():
    status = pd.read_json(OUT / "PHASE4_FINAL_STATUS.json", typ="series")
    assert status["phase4_status"] == "PASS_WITH_MECHANISTIC_LIMITATIONS"
    assert bool(status["ready_for_paper"])
    hashes = pd.read_csv(OUT / "PHASE4_FINAL_PATCH_FROZEN_INPUT_HASHES.csv")
    pre = hashes[hashes.stage.eq("PRE_FINAL_PATCH")].drop(columns="stage").sort_values("path").reset_index(drop=True)
    post = hashes[hashes.stage.eq("POST_FINAL_PATCH")].drop(columns="stage").sort_values("path").reset_index(drop=True)
    pd.testing.assert_frame_equal(pre, post)

