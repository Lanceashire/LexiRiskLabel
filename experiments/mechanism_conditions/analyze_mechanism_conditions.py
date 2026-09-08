"""Fail-closed analysis for the controlled mechanism-conditions experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
POLICIES = CONFIG["policies"]
CONTRASTS = [
    ("LRBE-IG", "QualityFirst-IG", "primary_quantity_constraint"),
    ("LRBE-IG", "CountOnly-MinRisk", "primary_ig_increment"),
    ("LRBE-IG", "MaxCount-Random", "secondary_ig_vs_random"),
]
METRICS = ["selected_count", "count_sacrifice", "selected_predicted_risk",
    "realized_loss", "composition_premium", "unused_budget", "ig_score_sum",
    "selected_region1_share", "selection_jaccard_vs_minrisk", "brier", "recall"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(values, draws=None, seed=None):
    values = np.asarray(values, float)
    draws = int(draws or CONFIG["bootstrap_iterations"])
    seed = int(seed or CONFIG["bootstrap_seed"])
    rng = np.random.RandomState(seed)
    means = values[rng.randint(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_and_validate(root):
    run_config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    files = sorted((root / "seeds").glob("seed*.csv"))
    if len(files) != len(run_config["seeds"]):
        raise ValueError(f"incomplete seed files: {len(files)} != {len(run_config['seeds'])}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    expected_rows = len(run_config["seeds"]) * len(CONFIG["exposure_conditions"]) * len(POLICIES) * len(CONFIG["target_region_weights"])
    if len(frame) != expected_rows:
        raise ValueError(f"incomplete rows: {len(frame)} != {expected_rows}")
    keys = ["seed", "exposure_condition", "method", "target_region1_weight"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate experimental cells")
    if set(frame.method) != set(POLICIES):
        raise ValueError("policy mismatch")
    for flag in ["gpu_enabled", "predicted_budget_feasible", "hidden_label_isolation",
                 "test_label_isolation", "preprocessing_visible_only",
                 "selection_independent_of_target_weight"]:
        if not frame[flag].astype(bool).all():
            raise ValueError(f"integrity failure: {flag}")
    for field in ["visible_hash", "batch_hash", "accepted_hash", "candidate_hash",
                  "candidate_pd_hash", "test0_hash", "test1_hash"]:
        if not frame.groupby("seed")[field].nunique().eq(1).all():
            raise ValueError(f"common-state mismatch: {field}")
    if not frame.groupby(["seed", "exposure_condition"])["cost_hash"].nunique().eq(1).all():
        raise ValueError("cost vector mismatch within a common-state cell")
    selection_fields = ["selected_ids", "selected_count", "selected_predicted_risk"]
    for field in selection_fields:
        if not frame.groupby(["seed", "exposure_condition", "method"])[field].nunique().eq(1).all():
            raise ValueError(f"target weights changed selection: {field}")
    if np.abs(frame["budget_identity_residual"].astype(float)).max() > 1e-6:
        raise ValueError("budget decomposition identity failed")
    if not frame.loc[frame.method.isin(["CountOnly-MinRisk", "LRBE-IG", "MaxCount-Random"]), "count_sacrifice"].eq(0).all():
        raise ValueError("a K*-preserving policy sacrificed count")
    return frame, run_config


def interval_row(values, **metadata):
    mean, low, high = paired_bootstrap(values)
    return {**metadata, "n_seeds": len(values), "mean_difference": mean,
            "ci_low": low, "ci_high": high,
            "interval_excludes_zero": bool(low > 0 or high < 0)}


def analyze(root, output):
    frame, run_config = load_and_validate(root)
    output.mkdir(parents=True, exist_ok=True)
    summaries = frame.groupby(["exposure_condition", "target_region1_weight", "method"])[METRICS].agg(["mean", "std"]).reset_index()
    summaries.columns = ["_".join(str(x) for x in c if x != "").rstrip("_") if isinstance(c, tuple) else c for c in summaries.columns]
    summaries.to_csv(output / "method_summary.csv", index=False)

    paired = []
    for (exposure, weight), group in frame.groupby(["exposure_condition", "target_region1_weight"]):
        indexed = group.set_index(["seed", "method"])
        seeds = sorted(group.seed.unique())
        for left, right, role in CONTRASTS:
            for metric in METRICS:
                differences = np.asarray([float(indexed.loc[(seed, left), metric]) - float(indexed.loc[(seed, right), metric]) for seed in seeds])
                paired.append(interval_row(differences, exposure_condition=exposure,
                    target_region1_weight=weight, contrast_role=role, left=left,
                    right=right, metric=metric))
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(output / "paired_comparisons.csv", index=False)

    unique_selection = frame.loc[frame.target_region1_weight.eq(float(CONFIG["target_region_weights"][0]))].copy()
    pivot_h1 = unique_selection.loc[unique_selection.method.eq("QualityFirst-IG")].pivot(index="seed", columns="exposure_condition", values="count_sacrifice")
    interactions = [interval_row((pivot_h1["high"] - pivot_h1["low"]).to_numpy(),
        interaction="H1_high_minus_low_exposure_QF_count_sacrifice")]
    ig_brier = frame.loc[frame.method.isin(["LRBE-IG", "CountOnly-MinRisk"])].pivot(
        index=["seed", "exposure_condition", "target_region1_weight"], columns="method", values="brier")
    ig_brier["contrast"] = ig_brier["LRBE-IG"] - ig_brier["CountOnly-MinRisk"]
    h2_values = {}
    low_weight, high_weight = map(float, CONFIG["target_region_weights"])
    for exposure in CONFIG["exposure_conditions"]:
        table = ig_brier.xs(exposure, level="exposure_condition")["contrast"].unstack("target_region1_weight")
        change = table[high_weight] - table[low_weight]
        h2_values[exposure] = change
        interactions.append(interval_row(change.to_numpy(),
            interaction=f"H2_weight_{high_weight}_minus_{low_weight}_IG_Brier_contrast_{exposure}_exposure"))
    h3 = h2_values["high"] - h2_values["low"]
    interactions.append(interval_row(h3.to_numpy(), interaction="H3_exposure_difference_in_H2"))
    interaction_frame = pd.DataFrame(interactions)
    interaction_frame.to_csv(output / "interaction_tests.csv", index=False)

    make_figure(frame, paired_frame, output)
    integrity = {"experiment_id": CONFIG["experiment_id"], "seed_files": len(list((root / "seeds").glob("seed*.csv"))),
        "seeds": run_config["seeds"], "rows": len(frame), "training_branches": len(run_config["seeds"]) * 2 * 4,
        "committee_members": run_config["committee_members"], "bootstrap_iterations": CONFIG["bootstrap_iterations"],
        "gpu_all_rows": bool(frame.gpu_enabled.astype(bool).all()),
        "common_state_checks_passed": True, "target_weight_selection_invariance_passed": True,
        "predicted_budget_feasible_all_rows": bool(frame.predicted_budget_feasible.astype(bool).all()),
        "hidden_and_test_label_isolation_all_rows": bool((frame.hidden_label_isolation.astype(bool) & frame.test_label_isolation.astype(bool)).all()),
        "maximum_abs_budget_identity_residual": float(np.abs(frame.budget_identity_residual).max())}
    (output / "integrity_summary.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    write_report(paired_frame, interaction_frame, integrity, output)
    manifest = [{"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(output.glob("*")) if path.is_file() and path.name != "manifest.csv"]
    pd.DataFrame(manifest).to_csv(output / "manifest.csv", index=False)


def make_figure(frame, paired, output):
    unique = frame.loc[frame.target_region1_weight.eq(float(CONFIG["target_region_weights"][0]))]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    qf = unique.loc[unique.method.eq("QualityFirst-IG")]
    qf.groupby("exposure_condition").count_sacrifice.mean().reindex(CONFIG["exposure_conditions"]).plot.bar(ax=axes[0], color=["#4C78A8", "#E45756"])
    axes[0].set(title="QualityFirst-IG count sacrifice", xlabel="Exposure condition", ylabel="Mean $K^*-|S|$")
    for policy, color in [("LRBE-IG", "#4C78A8"), ("QualityFirst-IG", "#E45756")]:
        values = unique.loc[unique.method.eq(policy)].groupby("exposure_condition").composition_premium.mean().reindex(CONFIG["exposure_conditions"])
        axes[1].plot(CONFIG["exposure_conditions"], values, marker="o", label=policy, color=color)
    axes[1].set(title="Composition premium", xlabel="Exposure condition", ylabel="Predicted-risk units")
    axes[1].legend(frameon=False, fontsize=8)
    subset = paired.loc[(paired.contrast_role == "primary_ig_increment") & (paired.metric == "brier")].copy()
    exposure_order = {name: index for index, name in enumerate(CONFIG["exposure_conditions"])}
    subset["_exposure_order"] = subset.exposure_condition.map(exposure_order)
    subset = subset.sort_values(["_exposure_order", "target_region1_weight"])
    positions = np.arange(len(subset))
    axes[2].errorbar(positions, subset.mean_difference, yerr=[subset.mean_difference - subset.ci_low, subset.ci_high - subset.mean_difference], fmt="o", color="#2F4B7C", capsize=3)
    axes[2].axhline(0, color="black", linewidth=.8)
    axes[2].set_xticks(positions, [f"{e}\nw={w:g}" for e, w in zip(subset.exposure_condition, subset.target_region1_weight)])
    axes[2].set(title="LRBE-IG minus MinRisk", xlabel="Exposure and target weight", ylabel="Brier difference (lower favors IG)")
    fig.tight_layout()
    fig.savefig(output / "mechanism_conditions.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "mechanism_conditions.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(paired, interactions, integrity, output):
    key = paired.loc[((paired.contrast_role == "primary_quantity_constraint") & (paired.metric == "selected_count")) |
                     ((paired.contrast_role == "primary_ig_increment") & paired.metric.isin(["brier", "selected_predicted_risk"]))]
    lines = ["# Controlled mechanism-conditions results", "", "This report is generated from the complete validated evidence block.", "",
        f"- Seeds: {len(integrity['seeds'])}", f"- Retraining branches: {integrity['training_branches']}",
        f"- GPU validation: {integrity['gpu_all_rows']}", f"- Bootstrap draws: {integrity['bootstrap_iterations']}", "", "## Primary paired contrasts", "",
        "| Exposure | Target region-1 weight | Contrast | Metric | Mean difference | 95% interval |",
        "|---|---:|---|---|---:|---:|"]
    for row in key.itertuples():
        lines.append(f"| {row.exposure_condition} | {row.target_region1_weight:.1f} | {row.left} - {row.right} | {row.metric} | {row.mean_difference:.6g} | [{row.ci_low:.6g}, {row.ci_high:.6g}] |")
    lines += ["", "## Preregistered interaction tests", "", "| Test | Mean | 95% interval |", "|---|---:|---:|"]
    for row in interactions.itertuples():
        lines.append(f"| {row.interaction} | {row.mean_difference:.6g} | [{row.ci_low:.6g}, {row.ci_high:.6g}] |")
    lines += ["", "Intervals containing zero are treated as inconclusive. Same-state one-step results do not identify field causal effects or long-run dynamic policy value.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(Path(args.root), Path(args.output))
