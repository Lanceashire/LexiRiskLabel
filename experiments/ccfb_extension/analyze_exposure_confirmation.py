"""Validate hardened reruns and quantify endogenous trajectory divergence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/ccfb_extension/raw"
OUT = ROOT / "results/ccfb_extension/dasfaa_revision/exposure_confirmation"
DATASETS = ("uci", "hmeq", "ppdai", "lendingclub")
METHODS = ("CountOnly-MinRisk", "LRBE-FAVE")
SAFE = {m: m.replace("-", "_") for m in METHODS}
KEYS = ["dataset", "method", "seed", "rho", "round"]
NUMERIC_FIELDS = [
    "accuracy", "precision", "recall", "specificity", "balanced_accuracy",
    "gmean", "roc_auc", "pr_auc", "brier", "ece", "calibration_slope",
    "calibration_intercept", "feedback_count", "k_star", "risk_budget",
    "predicted_exploration_risk", "realized_exploration_loss",
    "cumulative_feedback_count", "cumulative_predicted_risk", "cumulative_realized_loss",
]
EXACT_FIELDS = ["split_test_hash", "split_d0_hash"]


def compare_frames(new: pd.DataFrame, old: pd.DataFrame, atol: float = 1e-12) -> dict:
    """Fail closed on schema, row identity, numeric and missingness changes."""
    required = KEYS + NUMERIC_FIELDS + EXACT_FIELDS
    for name, frame in [("new", new), ("old", old)]:
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{name}: missing required parity fields: {missing}")
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f"{name}: null or duplicate row keys")
    a, b = new.set_index(KEYS).sort_index(), old.set_index(KEYS).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Parity row keys differ")
    maximum, numeric_equal = 0.0, True
    for field in NUMERIC_FIELDS:
        x = pd.to_numeric(a[field], errors="raise").to_numpy(float)
        y = pd.to_numeric(b[field], errors="raise").to_numpy(float)
        if np.isinf(x).any() or np.isinf(y).any():
            raise ValueError(f"Nonfinite infinity in {field}")
        same_nan = np.array_equal(np.isnan(x), np.isnan(y))
        finite = np.isfinite(x) & np.isfinite(y)
        delta = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
        maximum = max(maximum, delta)
        numeric_equal &= same_nan and delta <= atol
    exact_equal = all(a[c].equals(b[c]) and not a[c].isna().any() for c in EXACT_FIELDS)
    return {"max_abs_numeric_difference": maximum, "exact_fields_equal": bool(exact_equal),
            "numeric_fields_checked": len(NUMERIC_FIELDS), "row_keys_equal": True,
            "parity_passed": bool(numeric_equal and exact_equal)}


def load(experiment: str, dataset: str, method: str) -> pd.DataFrame:
    path = RAW / experiment / dataset / f"seed1001_rho0.100_{SAFE[method]}.csv"
    return pd.read_csv(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parity_rows: list[dict[str, object]] = []
    divergence_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        hardened = {m: load("exposure_hardened_confirmation", dataset, m) for m in METHODS}
        for method, new in hardened.items():
            old = load("budget_sweep", dataset, method)
            comparison = compare_frames(new, old)
            parity_rows.append({
                "dataset": dataset,
                "method": method,
                "rows": len(new),
                **comparison,
                "dataset_hash_recorded": bool(new["dataset_input_sha256"].nunique() == 1),
                "exposure_validation_all_passed": bool(new["exposure_validation_passed"].all()),
                "gpu_all_rounds": bool(new["gpu_enabled"].all()),
            })

        left = hardened[METHODS[0]].query("round > 0").set_index("round")
        right = hardened[METHODS[1]].query("round > 0").set_index("round")
        for round_id in left.index.intersection(right.index):
            a, b = left.loc[round_id], right.loc[round_id]
            divergence_rows.append({
                "dataset": dataset,
                "seed": 1001,
                "rho": 0.1,
                "round": int(round_id),
                "incoming_batch_shared": a["incoming_batch_hash"] == b["incoming_batch_hash"],
                "candidate_pool_shared": a["candidate_pool_hash"] == b["candidate_pool_hash"],
                "numeric_budget_shared": bool(np.isclose(a["risk_budget"], b["risk_budget"], rtol=0, atol=1e-9)),
                "countonly_budget": float(a["risk_budget"]),
                "fave_budget": float(b["risk_budget"]),
            })

    parity = pd.DataFrame(parity_rows)
    divergence = pd.DataFrame(divergence_rows)
    parity.to_csv(OUT / "HARDENED_RERUN_PARITY.csv", index=False)
    divergence.to_csv(OUT / "TRAJECTORY_DIVERGENCE_AUDIT.csv", index=False)
    summary = {
        "hardened_runs": int(len(parity)),
        "all_hardened_runs_match_registered_pre_hardening_results": bool(parity["parity_passed"].all()),
        "all_hardened_rounds_used_gpu": bool(parity["gpu_all_rounds"].all()),
        "trajectory_cells": int(len(divergence)),
        "incoming_batch_shared_cells": int(divergence["incoming_batch_shared"].sum()),
        "candidate_pool_shared_cells": int(divergence["candidate_pool_shared"].sum()),
        "candidate_pool_diverged_cells": int((~divergence["candidate_pool_shared"]).sum()),
        "numeric_budget_shared_cells": int(divergence["numeric_budget_shared"].sum()),
        "numeric_budget_diverged_cells": int((~divergence["numeric_budget_shared"]).sum()),
    }
    (OUT / "CONFIRMATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_hardened_runs_match_registered_pre_hardening_results"]:
        raise SystemExit("Hardened rerun parity failed")
    if summary["incoming_batch_shared_cells"] != summary["trajectory_cells"]:
        raise SystemExit("Incoming batch parity failed")


if __name__ == "__main__":
    main()
