"""Fail-closed paired analysis for the independent IG evidence block."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
POLICIES = CONFIG["policies"]
CONTRASTS = [
    ("LRBE-IG", "QualityFirst-IG", "primary_quantity_constraint"),
    ("LRBE-IG", "CountOnly-MinRisk", "primary_ig_increment"),
    ("LRBE-IG", "LRBE-FAVE", "secondary_ig_vs_fave"),
    ("QualityFirst-IG", "QualityFirst-FAVE", "secondary_qualityfirst_ig_vs_fave"),
    ("LRBE-FAVE", "CountOnly-MinRisk", "secondary_fave_increment")]
METRICS = ["cumulative_feedback_count", "cumulative_predicted_risk",
    "cumulative_realized_loss", "recall", "brier", "predicted_budget_violation_any",
    "realized_budget_violation_any", "runtime_seconds_elapsed"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate(root):
    run_config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    files = sorted((root / "trajectories").rglob("*.csv"))
    expected = len(run_config["datasets"]) * len(run_config["seeds"]) * len(run_config["budgets"]) * len(POLICIES)
    if len(files) != expected:
        raise ValueError(f"incomplete trajectories: {len(files)} != {expected}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    keys = ["dataset", "seed", "rho", "method", "round"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate trajectory rows")
    counts = frame.groupby(keys[:-1]).size()
    if not counts.eq(CONFIG["rounds"] + 1).all():
        raise ValueError("incomplete ten-round episodes")
    if set(frame.method) != set(POLICIES):
        raise ValueError("policy set mismatch")
    for flag in ["exposure_validation_passed", "hidden_label_isolation",
                 "outer_test_used_only_for_metrics", "gpu_enabled"]:
        if not frame[flag].astype(bool).all():
            raise ValueError(f"integrity failure: {flag}")
    for field in ["split_test_hash", "split_d0_hash", "stream_hash", "batch_stream_hash"]:
        if not frame.groupby(["dataset", "seed", "rho"])[field].nunique().eq(1).all():
            raise ValueError(f"exogenous mismatch: {field}")
    if frame.predicted_budget_violation.astype(bool).any():
        raise ValueError("predicted budget feasibility failure")
    return frame, run_config


def paired_bootstrap(values, draws=10000, seed=20260907):
    values = np.asarray(values, float)
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    means = values[rng.randint(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def analyze(root, output):
    frame, run_config = load_and_validate(root)
    final = frame.loc[frame["round"].eq(CONFIG["rounds"])].copy()
    summaries = []
    for keys, group in final.groupby(["dataset", "evidence_role", "rho", "method"]):
        dataset, role, rho, method = keys
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise").astype(float)
            summaries.append({"dataset": dataset, "evidence_role": role, "rho": rho,
                "method": method, "metric": metric, "n_seeds": len(values),
                "mean": float(values.mean()), "sd": float(values.std(ddof=1))})
    paired = []
    for (dataset, role, rho), group in final.groupby(["dataset", "evidence_role", "rho"]):
        indexed, seeds = group.set_index(["seed", "method"]), sorted(group.seed.unique())
        for left, right, contrast_role in CONTRASTS:
            for metric in METRICS:
                diffs = np.array([float(indexed.loc[(s, left), metric]) -
                                  float(indexed.loc[(s, right), metric]) for s in seeds])
                mean, low, high = paired_bootstrap(diffs)
                paired.append({"dataset": dataset, "evidence_role": role, "rho": rho,
                    "contrast_role": contrast_role, "left": left, "right": right,
                    "metric": metric, "n_seeds": len(seeds), "mean_difference": mean,
                    "ci_low": low, "ci_high": high,
                    "interval_excludes_zero": bool(low > 0 or high < 0)})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / "method_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    audit_files = sorted((root / "state_audit").rglob("*.csv")) if (root / "state_audit").exists() else []
    if audit_files:
        audit = pd.concat([pd.read_csv(path) for path in audit_files], ignore_index=True)
        audit.groupby(["dataset", "evidence_role", "rho", "method"]).agg(
            cells=("round", "size"), mean_count_sacrifice=("count_sacrifice", "mean"),
            mean_composition_premium=("composition_premium", "mean"),
            mean_jaccard_vs_countonly=("selection_jaccard_vs_countonly", "mean"),
            mean_premium_over_slack=("premium_over_slack", "mean")).reset_index().to_csv(
                output / "state_audit_summary.csv", index=False)
    integrity = {"trajectory_files": len(list((root / "trajectories").rglob("*.csv"))),
        "episodes": int(len(final)), "datasets": run_config["datasets"], "seeds": run_config["seeds"],
        "budgets": run_config["budgets"], "policies": POLICIES,
        "committee_members": run_config["members"], "bootstrap_draws": 10000,
        "canonical_gpu_all_rows": bool(frame.gpu_enabled.astype(bool).all()),
        "predicted_budget_feasible_all_rows": bool(~frame.predicted_budget_violation.astype(bool).any()),
        "hidden_label_isolation_all_rows": bool(frame.hidden_label_isolation.astype(bool).all()),
        "ppdai_kept_separate": bool(final.loc[final.dataset.eq("ppdai"), "evidence_role"].eq("reconstructed_source_exploratory").all())}
    # Keep manifest-tracked JSON byte-identical across operating systems.
    (output / "integrity_summary.json").write_bytes(
        (json.dumps(integrity, indent=2) + "\n").encode("utf-8")
    )
    files = [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(output.glob("*")) if p.is_file() and p.name != "manifest.csv"]
    pd.DataFrame(files).to_csv(output / "manifest.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(Path(args.root), Path(args.output))
