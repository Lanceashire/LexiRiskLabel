"""Analysis-only addendum for the reviewer-requested identification checks.

This script never refits a model and never changes frozen or fresh raw runs.
It uses final-round rows from the existing CCFB fresh-seed trajectories to
produce matched-count/matched-risk contrasts and a budget-sweep Pareto table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "ccfb_extension" / "supplemental_analysis"
RAW = ROOT / "results" / "ccfb_extension" / "raw"


def load_final(experiment: str, methods: list[str]) -> pd.DataFrame:
    rows = []
    base = RAW / experiment
    for dataset_dir in sorted(base.glob("*")):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for path in sorted(dataset_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            if frame.empty or "method" not in frame:
                continue
            method = str(frame["method"].iloc[0])
            if method not in methods:
                continue
            last = frame.loc[frame["round"].idxmax()].copy()
            rows.append(last)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def matched_count_risk() -> pd.DataFrame:
    methods = ["CountOnly-MinRisk", "LRBE-FAVE"]
    # K* ablation is the cleanest same-budget comparison, with one row per
    # dataset x seed x rho (rho is fixed at 0.1 in this experiment).
    d = load_final("kstar_ablation", methods)
    if d.empty:
        return d
    cols = ["dataset", "seed", "rho", "method", "cumulative_feedback_count",
            "cumulative_predicted_risk", "recall", "brier", "fd_ratio",
            "split_test_hash", "split_d0_hash", "hidden_label_isolation",
            "outer_test_used_only_for_metrics", "gpu_enabled"]
    d = d[[c for c in cols if c in d.columns]].copy()
    key = ["dataset", "seed", "rho"]
    rows = []
    for (dataset, seed, rho), g in d.groupby(key, sort=True):
        if set(g["method"]) != set(methods):
            continue
        for a, b in [("LRBE-FAVE", "CountOnly-MinRisk")]:
            x = g.set_index("method")
            rows.append({
                "dataset": dataset, "seed": int(seed), "rho": float(rho),
                "method_a": a, "method_b": b,
                "count_a": float(x.loc[a, "cumulative_feedback_count"]),
                "count_b": float(x.loc[b, "cumulative_feedback_count"]),
                "count_difference": float(x.loc[a, "cumulative_feedback_count"] - x.loc[b, "cumulative_feedback_count"]),
                "predicted_risk_a": float(x.loc[a, "cumulative_predicted_risk"]),
                "predicted_risk_b": float(x.loc[b, "cumulative_predicted_risk"]),
                "predicted_risk_difference": float(x.loc[a, "cumulative_predicted_risk"] - x.loc[b, "cumulative_predicted_risk"]),
                "recall_difference": float(x.loc[a, "recall"] - x.loc[b, "recall"]),
                "brier_difference": float(x.loc[a, "brier"] - x.loc[b, "brier"]),
                "fd_ratio_difference": float(x.loc[a, "fd_ratio"] - x.loc[b, "fd_ratio"]),
                "same_split_test_hash": bool(x["split_test_hash"].nunique() == 1),
                "same_split_d0_hash": bool(x["split_d0_hash"].nunique() == 1),
                "hidden_label_isolation": bool(x["hidden_label_isolation"].all()),
                "outer_test_metrics_only": bool(x["outer_test_used_only_for_metrics"].all()),
                "all_gpu": bool(x["gpu_enabled"].all()),
            })
    return pd.DataFrame(rows)


def pareto_table() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "results" / "ccfb_extension" / "04_budget_sweep_summary.csv")
    rows = []
    for dataset, g in p.groupby("dataset", sort=True):
        g = g.copy().reset_index(drop=True)
        # Lower expected risk and higher recall are preferred. A point is
        # efficient if no point dominates it, with one strict improvement.
        for i, r in g.iterrows():
            risk = float(r["predicted_risk_mean"])
            rec = float(r["recall_mean"])
            dominates = ((g["predicted_risk_mean"] <= risk + 1e-12) &
                         (g["recall_mean"] >= rec - 1e-12) &
                         ((g["predicted_risk_mean"] < risk - 1e-12) |
                          (g["recall_mean"] > rec + 1e-12)))
            rows.append({**r.to_dict(), "pareto_recall_risk": not bool(dominates.any())})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matched = matched_count_risk()
    matched.to_csv(OUT / "matched_count_risk_cellwise.csv", index=False)
    if not matched.empty:
        summary = matched.groupby(["dataset", "method_a", "method_b"], as_index=False).agg(
            n_cells=("seed", "size"),
            mean_count_difference=("count_difference", "mean"),
            mean_predicted_risk_difference=("predicted_risk_difference", "mean"),
            mean_recall_difference=("recall_difference", "mean"),
            mean_brier_difference=("brier_difference", "mean"),
            mean_fd_ratio_difference=("fd_ratio_difference", "mean"),
            same_split_test_hash=("same_split_test_hash", "all"),
            same_split_d0_hash=("same_split_d0_hash", "all"),
            hidden_label_isolation=("hidden_label_isolation", "all"),
            outer_test_metrics_only=("outer_test_metrics_only", "all"),
            all_gpu=("all_gpu", "all"),
        )
        summary.to_csv(OUT / "matched_count_risk_summary.csv", index=False)
    pareto = pareto_table()
    pareto.to_csv(OUT / "budget_sweep_pareto.csv", index=False)
    report = {
        "analysis": "reviewer_requested_identification_checks",
        "raw_results_unchanged": True,
        "matched_count_risk_cells": int(len(matched)),
        "pareto_points": int(len(pareto)),
        "pareto_efficient_points": int(pareto["pareto_recall_risk"].sum()),
        "datasets": sorted(pareto["dataset"].unique().tolist()),
        "methods": sorted(matched["method_a"].unique().tolist()) if not matched.empty else [],
        "interpretation": "Matched cells isolate composition/risk differences after holding split, seed, budget and (for LRBE policies) K* fixed; Pareto table spans all existing fresh budget-sweep points.",
    }
    (OUT / "SUPPLEMENTAL_ANALYSIS_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
