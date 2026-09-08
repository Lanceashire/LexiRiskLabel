"""Independent bootstrap information-gain evidence block.

The five policies share exogenous splits and batches, then follow their own
feedback histories. Endogenous candidate pools and budgets may diverge after
round one. Frozen Phase 1--4 artifacts are never modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import phase1_runner as p1
import phase2_runner as p2

BASE_PATH = ROOT / "experiments" / "ccfb_extension" / "run_ccfb_extension.py"
SPEC = importlib.util.spec_from_file_location("ccfb_extension_for_ig", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
REGISTRY = json.loads((HERE / "input_registry.json").read_text(encoding="utf-8"))
POLICIES = CONFIG["policies"]


def digest_indices(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def validate_registered_input(dataset):
    cfg, reg = BASE.DATASETS[dataset], REGISTRY[dataset]
    path = cfg["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_hash = BASE._sha(path)
    if observed_hash != reg["sha256"]:
        raise ValueError(f"{dataset} SHA-256 mismatch: {observed_hash} != {reg['sha256']}")
    df, target, y, exposure, _ = BASE.dataset_frame(dataset)
    if len(df) != int(reg["rows"]):
        raise ValueError(f"{dataset} row mismatch: {len(df)} != {reg['rows']}")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{dataset} target is not binary")
    if not np.isfinite(exposure).all() or not (exposure > 0).all():
        raise ValueError(f"{dataset} exposure is not finite and strictly positive")
    return {"dataset": dataset, "path": path.relative_to(ROOT).as_posix(), "rows": len(df),
            "sha256": observed_hash, "target": target, "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()), "exposure_min": float(exposure.min()),
            "exposure_max": float(exposure.max()), "parity": reg["parity"],
            "evidence_role": reg["evidence_role"], "validated": True}


def entropy_binary(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def committee_seed(seed, round_id, member):
    return int(seed * 100000 + round_id * 1000 + member)


def information_gain_score(raw, y, visible, candidate, pre, *, seed, round_id, members, workers=None):
    """Empirical-committee MI; reads labels only at visible indices."""
    candidate, visible = np.asarray(candidate, int), np.asarray(visible, int)
    if not len(candidate):
        return np.empty(0), 0.0
    if members < 2:
        raise ValueError("committee members must be at least two")
    cached = getattr(pre, "_ecomic_x_all", None)
    if cached is None:
        x_visible = np.asarray(pre.transform(raw.iloc[visible]))
        x_candidate = np.asarray(pre.transform(raw.iloc[candidate]))
    else:
        cached = np.asarray(cached)
        x_visible, x_candidate = cached[visible], cached[candidate]
    visible_y = np.asarray(y[visible], dtype=np.int32)
    started = time.perf_counter()
    def fit_member(member):
        model_seed = committee_seed(seed, round_id, member)
        rng = np.random.RandomState(model_seed)
        sample = rng.randint(0, len(visible), size=len(visible))
        sampled_y = visible_y[sample]
        if len(np.unique(sampled_y)) < 2:
            return np.full(len(candidate), float(sampled_y[0]))
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
            objective="binary:logistic", eval_metric="logloss", random_state=model_seed,
            n_jobs=1, tree_method="hist", device="cpu")
        model.fit(x_visible[sample], sampled_y)
        return model.predict_proba(x_candidate)[:, 1]
    worker_count = min(members, int(workers or CONFIG.get("committee_workers", 1)))
    if worker_count == 1:
        predictions = np.vstack([fit_member(member) for member in range(members)])
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="ig-member") as pool:
            predictions = np.vstack(list(pool.map(fit_member, range(members))))
    score = entropy_binary(predictions.mean(axis=0)) - entropy_binary(predictions).mean(axis=0)
    return np.maximum(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0.0), float(time.perf_counter() - started)


def selection_for_policy(policy, p, cost, candidate, budget, raw, y, visible, batch,
                         pre, model, *, seed, round_id, members, ig_cache=None):
    if policy in {"LRBE-IG", "QualityFirst-IG"}:
        key = (digest_indices(visible), digest_indices(candidate), seed, round_id, members)
        if ig_cache is not None and key in ig_cache:
            utility, committee_seconds = ig_cache[key]
        else:
            utility, committee_seconds = information_gain_score(
                raw, y, visible, candidate, pre, seed=seed, round_id=round_id, members=members)
            if ig_cache is not None:
                ig_cache[key] = (utility, committee_seconds)
        if policy == "LRBE-IG":
            local, kstar = p2.lrbe_select(cost, utility, budget)
            return local, kstar, utility, "lrbe_bootstrap_ig", committee_seconds
        local = BASE.max_quality_select(cost, utility, budget)
        kstar = p2.lrbe_select(cost, -cost, budget)[1]
        return local, kstar, utility, "quality_first_bootstrap_ig", committee_seconds
    local, kstar, utility, structure = BASE.select(
        policy, p, cost, candidate, budget, raw, visible, batch, pre, model, None)
    if kstar < 0:
        kstar = p2.lrbe_select(cost, -cost, budget)[1]
    return local, kstar, utility, structure, 0.0


def base_metadata(dataset, seed, rho, policy, members, fixed, dataset_hash, evidence_role):
    test, visible, stream = fixed["test"], fixed["visible"], fixed["stream"]
    batches = [np.asarray(x, int) for x in fixed["batches"]]
    return {"experiment": CONFIG["extension_id"], "dataset": dataset,
        "evidence_role": evidence_role, "seed": seed, "rho": rho, "method": policy,
        "committee_members": members, "committee_backend": "cpu" if "IG" in policy else "not_used",
        "dataset_input_sha256": dataset_hash, "dataset_parity": REGISTRY[dataset]["parity"],
        "exposure_validation_passed": True, "split_test_hash": digest_indices(test),
        "split_d0_hash": digest_indices(visible), "stream_hash": digest_indices(stream),
        "batch_stream_hash": digest_indices(np.concatenate(batches)),
        "gpu_enabled": bool(p2.USE_GPU and p2.GPU_AVAILABLE), "preprocessing_visible_only": True,
        "committee_visible_labels_only": True, "outer_test_used_only_for_metrics": True,
        "hidden_label_isolation": True}


def run_episode(dataset, seed, rho, policy, members):
    df, target, y, exposure, raw = BASE.dataset_frame(dataset)
    fixed = p1.fixed_split(df, target, seed)
    test = np.asarray(fixed["test"], int)
    batches = [np.asarray(x, int) for x in fixed["batches"]]
    visible = np.asarray(fixed["visible"], int).copy()
    pop_rate = float(y[np.asarray(fixed["stream"], int)].mean())
    common = base_metadata(dataset, seed, rho, policy, members, fixed,
                           REGISTRY[dataset]["sha256"], REGISTRY[dataset]["evidence_role"])
    rows, cumulative_count = [], 0
    cumulative_predicted = cumulative_realized = committee_total = 0.0
    predicted_any = realized_any = False
    started = time.perf_counter()
    pre, model, pred, backend = p2.fit_predict_gpu(raw, y, visible, seed * 10000)
    metric = p1.metrics(y[test], pred[test], y[visible], pop_rate, np.empty(0, int),
        np.empty(0), np.empty(0, int), np.empty(0, int), np.empty(0, int))
    metric.update(common)
    metric.update({"round": 0, "selection_structure": "reference", "committee_runtime_seconds": 0.0,
        "n_explored_round": 0, "feedback_count": 0, "k_star": 0, "risk_budget": 0.0,
        "predicted_exploration_risk": 0.0, "realized_exploration_loss": 0.0,
        "predicted_budget_violation": False, "realized_budget_violation": False,
        "predicted_budget_violation_any": False, "realized_budget_violation_any": False,
        "cumulative_feedback_count": 0, "cumulative_predicted_risk": 0.0,
        "cumulative_realized_loss": 0.0, "incoming_batch_hash": "ROUND_ZERO",
        "candidate_pool_hash": "ROUND_ZERO", "visible_before_hash": digest_indices(visible),
        "selected_ids": "[]", "compute_backend": backend,
        "runtime_seconds_elapsed": float(time.perf_counter() - started),
        "committee_runtime_seconds_cumulative": 0.0})
    rows.append(metric)
    for round_id, batch in enumerate(batches, 1):
        visible_before = visible.copy()
        pre, model, pred, backend_before = p2.fit_predict_gpu(raw, y, visible_before, seed * 10000 + round_id)
        p_batch = pred[batch]
        accepted, candidate = batch[p_batch < p1.TAU], batch[p_batch >= p1.TAU]
        pc = p_batch[p_batch >= p1.TAU]
        cost, budget = pc * exposure[candidate], float(rho * np.sum(pc * exposure[candidate]))
        local, kstar, utility, structure, committee_seconds = selection_for_policy(
            policy, pc, cost, candidate, budget, raw, y, visible_before, batch, pre, model,
            seed=seed, round_id=round_id, members=members)
        committee_total += committee_seconds
        selected = candidate[local]
        predicted_risk = float(cost[local].sum()) if len(local) else 0.0
        realized_loss = float(exposure[selected][y[selected] == 1].sum()) if len(selected) else 0.0
        predicted_violation = predicted_risk > budget + 1e-7
        realized_violation = realized_loss > budget + 1e-7
        predicted_any, realized_any = predicted_any or predicted_violation, realized_any or realized_violation
        gained = np.concatenate([accepted, selected]).astype(int)
        visible = np.concatenate([visible_before, gained]).astype(int)
        cumulative_count += len(selected); cumulative_predicted += predicted_risk; cumulative_realized += realized_loss
        _, _, pred_after, backend_after = p2.fit_predict_gpu(raw, y, visible, seed * 10000 + 500 + round_id)
        metric = p1.metrics(y[test], pred_after[test], y[visible], pop_rate, y[batch], p_batch,
                            gained, candidate, accepted)
        metric.update(common)
        metric.update({"round": round_id, "selection_structure": structure,
            "committee_runtime_seconds": committee_seconds, "n_explored_round": len(selected),
            "feedback_count": len(selected), "k_star": kstar, "risk_budget": budget,
            "predicted_exploration_risk": predicted_risk, "realized_exploration_loss": realized_loss,
            "predicted_budget_violation": predicted_violation, "realized_budget_violation": realized_violation,
            "predicted_budget_violation_any": predicted_any, "realized_budget_violation_any": realized_any,
            "cumulative_feedback_count": cumulative_count, "cumulative_predicted_risk": cumulative_predicted,
            "cumulative_realized_loss": cumulative_realized, "incoming_batch_hash": digest_indices(batch),
            "candidate_pool_hash": digest_indices(candidate), "visible_before_hash": digest_indices(visible_before),
            "selected_ids": json.dumps([int(v) for v in selected]),
            "compute_backend": f"{backend_before}+{backend_after}",
            "selected_default_rate_oracle": float(y[selected].mean()) if len(selected) else np.nan,
            "selected_mean_pd": float(pc[local].mean()) if len(local) else np.nan,
            "selected_mean_exposure": float(exposure[selected].mean()) if len(selected) else np.nan,
            "runtime_seconds_elapsed": float(time.perf_counter() - started),
            "committee_runtime_seconds_cumulative": committee_total})
        rows.append(metric)
    return pd.DataFrame(rows)


def jaccard(a, b):
    left, right = set(map(int, a)), set(map(int, b))
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def run_state_audit(dataset, seed, rho, members):
    """Immediate selection audit on identical CountOnly-anchored states."""
    df, target, y, exposure, raw = BASE.dataset_frame(dataset)
    fixed = p1.fixed_split(df, target, seed)
    visible = np.asarray(fixed["visible"], int).copy()
    rows = []
    for round_id, batch_raw in enumerate(fixed["batches"], 1):
        batch, visible_before = np.asarray(batch_raw, int), visible.copy()
        pre, model, pred, backend = p2.fit_predict_gpu(raw, y, visible_before, seed * 10000 + round_id)
        p_batch = pred[batch]
        accepted, candidate = batch[p_batch < p1.TAU], batch[p_batch >= p1.TAU]
        pc = p_batch[p_batch >= p1.TAU]
        cost = pc * exposure[candidate]
        budget = float(rho * cost.sum())
        ordered_cost = np.sort(cost)
        cumulative = np.cumsum(ordered_cost)
        kstar = int(np.searchsorted(cumulative, budget + 1e-10, side="right")) if len(cost) else 0
        min_kstar = float(cumulative[kstar - 1]) if kstar else 0.0
        slack = float(budget - min_kstar)
        cache, details = {}, {}
        for policy in POLICIES:
            local, observed_kstar, utility, structure, committee_seconds = selection_for_policy(
                policy, pc, cost, candidate, budget, raw, y, visible_before, batch, pre, model,
                seed=seed, round_id=round_id, members=members, ig_cache=cache)
            selected = candidate[local]
            selected_risk = float(cost[local].sum()) if len(local) else 0.0
            minimum_same_count = float(ordered_cost[:len(local)].sum()) if len(local) else 0.0
            details[policy] = (selected, structure, committee_seconds, selected_risk,
                               minimum_same_count, selected_risk - minimum_same_count)
        anchor = details["CountOnly-MinRisk"][0]
        for policy, detail in details.items():
            selected, structure, committee_seconds, selected_risk, minimum_same_count, premium = detail
            rows.append({"experiment": CONFIG["extension_id"], "dataset": dataset,
                "evidence_role": REGISTRY[dataset]["evidence_role"], "seed": seed, "rho": rho,
                "round": round_id, "anchor": "CountOnly-MinRisk", "method": policy,
                "n_visible": len(visible_before), "n_candidate": len(candidate), "k_star": kstar,
                "selected_count": len(selected), "count_sacrifice": int(kstar - len(selected)),
                "risk_budget": budget, "selected_predicted_risk": selected_risk,
                "minimum_risk_at_selected_count": minimum_same_count,
                "composition_premium": premium, "maximum_count_slack": slack,
                "premium_over_slack": float(premium / slack) if slack > 1e-12 else np.nan,
                "selection_jaccard_vs_countonly": jaccard(selected, anchor),
                "selected_ids": json.dumps([int(v) for v in selected]),
                "visible_hash": digest_indices(visible_before), "candidate_hash": digest_indices(candidate),
                "cost_hash": hashlib.sha256(np.asarray(cost, np.float64).tobytes()).hexdigest(),
                "selection_structure": structure, "committee_members": members,
                "committee_backend": "cpu" if "IG" in policy else "not_used",
                "committee_runtime_seconds": committee_seconds, "canonical_backend": backend,
                "gpu_enabled": bool(p2.USE_GPU and p2.GPU_AVAILABLE),
                "common_state_diagnostic_only": True})
        visible = np.concatenate([visible_before, accepted, anchor]).astype(int)
    return pd.DataFrame(rows)


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def run(args):
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"output already exists; use a new path or --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or CONFIG["datasets"]
    seeds = args.seeds or CONFIG["seeds"]
    budgets = args.budgets or CONFIG["extension_budgets"]
    atomic_csv(pd.DataFrame([validate_registered_input(x) for x in datasets]), output / "input_audit.csv")
    run_config = {"extension_id": CONFIG["extension_id"], "datasets": datasets,
        "seeds": seeds, "budgets": budgets, "policies": POLICIES, "members": args.members,
        "state_audit": args.state_audit, "canonical_gpu_requested": bool(p2.USE_GPU),
        "canonical_gpu_available": bool(p2.GPU_AVAILABLE), "committee_backend": "cpu",
        "committee_seed_rule": CONFIG["committee_seed_rule"]}
    # Write a canonical LF-terminated byte stream so the recorded SHA-256 stays
    # stable when a Windows checkout applies Git's text-file normalization.
    (output / "run_config.json").write_bytes(
        (json.dumps(run_config, indent=2) + "\n").encode("utf-8")
    )
    if not p2.USE_GPU or not p2.GPU_AVAILABLE:
        raise RuntimeError("canonical GPU backend is required for this requested run")
    total = len(datasets) * len(seeds) * len(budgets) * len(POLICIES)
    completed = 0
    for dataset in datasets:
        for seed in seeds:
            for rho in budgets:
                for policy in POLICIES:
                    safe = policy.replace("-", "_")
                    dest = output / "trajectories" / dataset / f"seed{seed}_rho{rho:.3f}_{safe}.csv"
                    if not dest.exists():
                        atomic_csv(run_episode(dataset, seed, rho, policy, args.members), dest)
                    elif not args.resume:
                        raise FileExistsError(dest)
                    completed += 1
                    print(f"PROGRESS {completed}/{total} dataset={dataset} seed={seed} rho={rho:.3f} policy={policy}", flush=True)
                if args.state_audit:
                    dest = output / "state_audit" / dataset / f"seed{seed}_rho{rho:.3f}.csv"
                    if not dest.exists():
                        atomic_csv(run_state_audit(dataset, seed, rho, args.members), dest)
    manifest = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.csv":
            manifest.append({"path": path.relative_to(output).as_posix(),
                             "sha256": BASE._sha(path), "bytes": path.stat().st_size})
    atomic_csv(pd.DataFrame(manifest), output / "manifest.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=CONFIG["datasets"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--budgets", nargs="+", type=float)
    parser.add_argument("--members", type=int, default=CONFIG["committee_members"])
    parser.add_argument("--state-audit", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
