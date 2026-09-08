"""Summarize the fresh risk-bounded exploration baselines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "ccfb_extension"
RAW = BASE / "raw"
OUT = BASE / "supplemental_analysis"


def final_rows(root: Path, methods: set[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*.csv")):
        d = pd.read_csv(path)
        if d.empty or "method" not in d.columns:
            continue
        method = str(d["method"].iloc[0])
        if method not in methods:
            continue
        rows.append(d.loc[d["round"].idxmax()].copy())
    return pd.DataFrame(rows)


def bootstrap_ci(x: np.ndarray, seed: int = 20260902, n: int = 10000) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    supplemental = final_rows(RAW / "supplemental_exploration", {"RiskBounded-Random", "RiskBounded-Epsilon"})
    if len(supplemental) != 80:
        raise RuntimeError(f"Expected 80 supplemental final rows, found {len(supplemental)}")
    count = final_rows(RAW / "kstar_ablation", {"CountOnly-MinRisk", "LRBE-FAVE"})
    common_cols = ["dataset", "seed", "rho", "method", "cumulative_feedback_count",
                   "cumulative_predicted_risk", "cumulative_realized_loss", "recall",
                   "brier", "fd_ratio", "risk_budget", "budget_violation",
                   "split_test_hash", "split_d0_hash", "hidden_label_isolation",
                   "outer_test_used_only_for_metrics", "gpu_enabled"]
    all_rows = pd.concat([supplemental, count], ignore_index=True)
    all_rows = all_rows[[c for c in common_cols if c in all_rows.columns]].copy()
    all_rows.to_csv(OUT / "supplemental_common_cell_final_rows.csv", index=False)
    summary = all_rows.groupby(["dataset", "method"], as_index=False).agg(
        n_cells=("seed", "size"), n_seeds=("seed", "nunique"),
        feedback_count_mean=("cumulative_feedback_count", "mean"),
        feedback_count_std=("cumulative_feedback_count", "std"),
        predicted_risk_mean=("cumulative_predicted_risk", "mean"),
        realized_loss_mean=("cumulative_realized_loss", "mean"),
        recall_mean=("recall", "mean"), brier_mean=("brier", "mean"),
        fd_ratio_mean=("fd_ratio", "mean"), realized_budget_violation_rate=("budget_violation", "mean"),
        all_hidden_label_isolation=("hidden_label_isolation", "all"),
        all_outer_test_metrics_only=("outer_test_used_only_for_metrics", "all"),
        all_gpu=("gpu_enabled", "all"),
    )
    summary.to_csv(OUT / "supplemental_method_summary.csv", index=False)
    key = ["dataset", "seed", "rho"]
    wide = all_rows.pivot_table(index=key, columns="method", values=["cumulative_feedback_count", "cumulative_predicted_risk", "cumulative_realized_loss", "recall", "brier", "fd_ratio"])
    comparisons = []
    pairs = [("RiskBounded-Random", "CountOnly-MinRisk"), ("RiskBounded-Epsilon", "CountOnly-MinRisk"), ("RiskBounded-Random", "RiskBounded-Epsilon")]
    for dataset in sorted(all_rows.dataset.unique()):
        w = wide.loc[dataset] if dataset in wide.index.get_level_values(0) else pd.DataFrame()
        for a, b in pairs:
            if w.empty or a not in w.columns.get_level_values(1) or b not in w.columns.get_level_values(1):
                continue
            for metric in ["cumulative_feedback_count", "cumulative_predicted_risk", "cumulative_realized_loss", "recall", "brier", "fd_ratio"]:
                x = w[(metric, a)].to_numpy(float); y = w[(metric, b)].to_numpy(float); d = x - y
                lo, hi = bootstrap_ci(d, seed=20260902 + len(comparisons))
                try:
                    p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
                except ValueError:
                    p = 1.0
                comparisons.append({"dataset": dataset, "method_a": a, "method_b": b,
                                    "metric": metric, "n_paired_cells": len(d),
                                    "mean_difference": float(d.mean()), "std_difference": float(d.std(ddof=1)) if len(d)>1 else 0.0,
                                    "bootstrap_95ci_low": lo, "bootstrap_95ci_high": hi,
                                    "paired_t_pvalue": float(ttest_rel(x, y).pvalue) if len(d)>1 else np.nan,
                                    "wilcoxon_pvalue": p})
    pd.DataFrame(comparisons).to_csv(OUT / "supplemental_paired_comparisons.csv", index=False)
    audit = {
        "supplemental_files": int(len(supplemental)), "expected_files": 80,
        "datasets": sorted(supplemental.dataset.unique().tolist()),
        "methods": sorted(supplemental.method.unique().tolist()),
        "seed_count_by_dataset": supplemental.groupby("dataset").seed.nunique().to_dict(),
        "same_split_hash_per_dataset_seed": bool(supplemental.groupby(["dataset", "seed"])["split_test_hash"].nunique().max() == 1),
        "same_d0_hash_per_dataset_seed": bool(supplemental.groupby(["dataset", "seed"])["split_d0_hash"].nunique().max() == 1),
        "hidden_label_isolation_all": bool(supplemental.hidden_label_isolation.all()),
        "outer_test_metrics_only_all": bool(supplemental.outer_test_used_only_for_metrics.all()),
        "gpu_enabled_all": bool(supplemental.gpu_enabled.all()),
        "realized_violation_rate": float(supplemental.budget_violation.mean()),
    }
    (OUT / "SUPPLEMENTAL_BASELINE_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report = ["# Supplemental exploration baseline report", "", "This addendum uses 80 fresh final-round trajectories (4 datasets × 10 seeds × 2 risk-bounded randomized methods). Raw rows are immutable after writing.", "", "## Integrity", "", f"- Files: {audit['supplemental_files']}/{audit['expected_files']}", f"- Datasets: {', '.join(audit['datasets'])}", f"- Same split hashes per dataset-seed: {audit['same_split_hash_per_dataset_seed']}", f"- Same D0 hashes per dataset-seed: {audit['same_d0_hash_per_dataset_seed']}", f"- Hidden-label isolation: {audit['hidden_label_isolation_all']}", f"- Outer-test metrics-only use: {audit['outer_test_metrics_only_all']}", f"- GPU backend recorded for every row: {audit['gpu_enabled_all']}", "", "## Interpretation", "", "RiskBounded-Random greedily samples a deterministic pseudo-random order subject to the predicted-risk budget. RiskBounded-Epsilon retains the exact feasible K* count while adding a preregistered 10% random perturbation to the min-risk utility. These are external-style controls, not claims of reproducing any one prior paper's full online algorithm.", "", "The paired comparisons and confidence intervals are in `supplemental_paired_comparisons.csv`; aggregate values are in `supplemental_method_summary.csv`."]
    (OUT / "SUPPLEMENTAL_BASELINE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
