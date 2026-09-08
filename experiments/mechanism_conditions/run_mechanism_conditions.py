"""Preregistered common-state mechanism experiment for acquisition conditions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import phase1_runner as p1
import phase2_runner as p2

IG_PATH = ROOT / "experiments" / "information_gain" / "run_information_gain.py"
IG_SPEC = importlib.util.spec_from_file_location("ig_for_mechanism_conditions", IG_PATH)
IG = importlib.util.module_from_spec(IG_SPEC)
assert IG_SPEC.loader is not None
IG_SPEC.loader.exec_module(IG)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
POLICIES = CONFIG["policies"]


def sigmoid(value):
    value = np.clip(np.asarray(value, float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def digest(values, dtype=np.int64):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def generate_block(rng, n, fixed_region=None):
    x = rng.normal(size=(n, int(CONFIG["dgp"]["features"])))
    if fixed_region is None:
        region = rng.binomial(1, float(CONFIG["dgp"]["region_probability"]), size=n)
    else:
        region = np.full(n, int(fixed_region), dtype=int)
    logit = -1.0 + 1.3 * x[:, 0] - 0.8 * x[:, 1] + region * (1.5 * x[:, 2] + 0.7 * x[:, 3])
    probability = sigmoid(logit)
    y = rng.binomial(1, probability).astype(int)
    raw = pd.DataFrame(x, columns=[f"x{j + 1}" for j in range(x.shape[1])])
    raw["region"] = region
    return raw, y, probability


def generate_task(seed):
    rng = np.random.RandomState(int(seed))
    blocks = [
        generate_block(rng, int(CONFIG["n_pool"])),
        generate_block(rng, int(CONFIG["n_batch"])),
        generate_block(rng, int(CONFIG["n_test_per_region"]), fixed_region=0),
        generate_block(rng, int(CONFIG["n_test_per_region"]), fixed_region=1),
    ]
    raw = pd.concat([b[0] for b in blocks], ignore_index=True)
    y = np.concatenate([b[1] for b in blocks]).astype(int)
    true_probability = np.concatenate([b[2] for b in blocks])
    n_pool, n_batch, n_test = int(CONFIG["n_pool"]), int(CONFIG["n_batch"]), int(CONFIG["n_test_per_region"])
    pool = np.arange(n_pool, dtype=int)
    batch = np.arange(n_pool, n_pool + n_batch, dtype=int)
    test0 = np.arange(n_pool + n_batch, n_pool + n_batch + n_test, dtype=int)
    test1 = np.arange(n_pool + n_batch + n_test, n_pool + n_batch + 2 * n_test, dtype=int)
    selection_score = -1.0 + 1.3 * raw.loc[pool, "x1"].to_numpy() - 0.8 * raw.loc[pool, "x2"].to_numpy() + 1.5 * raw.loc[pool, "region"].to_numpy()
    visible = pool[np.argsort(selection_score, kind="mergesort")[:int(CONFIG["visible_count"])]].astype(int)
    return {"raw": raw, "y": y, "true_probability": true_probability, "pool": pool,
            "visible": visible, "batch": batch, "test0": test0, "test1": test1}


def exposure_vectors(candidate_pd, seed):
    candidate_pd = np.asarray(candidate_pd, float)
    low = np.ones(len(candidate_pd), dtype=float)
    rng = np.random.RandomState(int(seed) * 100000 + 771)
    high = rng.lognormal(mean=0.0, sigma=float(CONFIG["high_exposure_log_sigma"]), size=len(candidate_pd))
    denominator = float(np.dot(candidate_pd, high))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("invalid high-exposure normalization")
    high *= float(candidate_pd.sum()) / denominator
    if not np.isfinite(high).all() or not (high > 0).all():
        raise ValueError("exposure must be finite and strictly positive")
    return {"low": low, "high": high}


def weighted_metrics(y0, p0, y1, p1_values, region_weight):
    region_weight = float(region_weight)
    brier0, brier1 = float(np.mean((y0 - p0) ** 2)), float(np.mean((y1 - p1_values) ** 2))
    tp = (1.0 - region_weight) * float(np.mean(y0 * (p0 >= CONFIG["threshold"]))) + region_weight * float(np.mean(y1 * (p1_values >= CONFIG["threshold"])))
    positives = (1.0 - region_weight) * float(np.mean(y0)) + region_weight * float(np.mean(y1))
    recall = tp / positives if positives > 0 else np.nan
    return {"brier": (1.0 - region_weight) * brier0 + region_weight * brier1,
            "recall": recall, "brier_region0": brier0, "brier_region1": brier1}


def jaccard(left, right):
    a, b = set(map(int, left)), set(map(int, right))
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def select_policy(policy, cost, ig_score, candidate, budget, seed):
    ordered = np.argsort(cost, kind="mergesort")
    kstar = int(np.searchsorted(np.cumsum(cost[ordered]), budget + 1e-10, side="right")) if len(cost) else 0
    if policy == "CountOnly-MinRisk":
        local, observed = p2.lrbe_select(cost, -cost, budget)
    elif policy == "LRBE-IG":
        local, observed = p2.lrbe_select(cost, ig_score, budget)
    elif policy == "QualityFirst-IG":
        local, observed = IG.BASE.max_quality_select(cost, ig_score, budget), kstar
    elif policy == "MaxCount-Random":
        random_utility = p2.deterministic_hash_scores(candidate, seed=int(seed), round_id=8801)
        local, observed = p2.lrbe_select(cost, random_utility, budget)
    else:
        raise ValueError(policy)
    if observed != kstar:
        raise AssertionError(f"K* mismatch for {policy}: {observed} != {kstar}")
    return np.asarray(local, int), kstar


def branch_training_seed(seed):
    """Use common training randomness for every branch of a synthetic task."""
    return int(seed) * 10000 + 100


def acquisition_isolation_audit(raw, y, visible, batch, candidate, prediction,
                                ig_score, pre, seed, members):
    """Verify that hidden and test labels cannot affect pre-acquisition state."""
    hidden_y = np.asarray(y, int).copy()
    hidden_mask = np.ones(len(hidden_y), dtype=bool)
    hidden_mask[np.asarray(visible, int)] = False
    hidden_y[hidden_mask] = 1 - hidden_y[hidden_mask]
    _, _, perturbed_prediction, backend = p2.fit_predict_gpu(
        raw, hidden_y, visible, int(seed) * 10000 + 1)
    perturbed_candidate = batch[perturbed_prediction[batch] >= float(CONFIG["threshold"])]
    perturbed_ig, _ = IG.information_gain_score(
        raw, hidden_y, visible, perturbed_candidate, pre, seed=int(seed), round_id=1,
        members=int(members), workers=int(CONFIG["committee_workers"]))
    return bool(
        backend == "gpu"
        and np.array_equal(candidate, perturbed_candidate)
        and np.array_equal(prediction, perturbed_prediction)
        and np.array_equal(ig_score, perturbed_ig)
    )


def test_label_isolation_audit(raw, y, visible_after, test0, test1, seed,
                               reference_prediction):
    """Verify that test-label perturbations cannot change the fitted predictor."""
    perturbed_y = np.asarray(y, int).copy()
    test = np.concatenate([test0, test1]).astype(int)
    perturbed_y[test] = 1 - perturbed_y[test]
    _, _, perturbed_prediction, backend = p2.fit_predict_gpu(
        raw, perturbed_y, visible_after, branch_training_seed(seed))
    return bool(backend == "gpu" and np.array_equal(reference_prediction, perturbed_prediction))


def run_seed(seed, members):
    task = generate_task(seed)
    raw, y = task["raw"], task["y"]
    visible, batch = task["visible"], task["batch"]
    started = time.perf_counter()
    pre, model, prediction, initial_backend = p2.fit_predict_gpu(raw, y, visible, int(seed) * 10000 + 1)
    if initial_backend != "gpu":
        raise RuntimeError(f"canonical GPU required, observed {initial_backend}")
    batch_pd = prediction[batch]
    accepted = batch[batch_pd < float(CONFIG["threshold"])]
    candidate = batch[batch_pd >= float(CONFIG["threshold"])]
    candidate_pd = prediction[candidate]
    if not len(candidate):
        raise RuntimeError(f"seed {seed} produced an empty candidate set")
    ig_score, committee_seconds = IG.information_gain_score(
        raw, y, visible, candidate, pre, seed=int(seed), round_id=1,
        members=int(members), workers=int(CONFIG["committee_workers"]))
    hidden_label_isolation = acquisition_isolation_audit(
        raw, y, visible, batch, candidate, prediction, ig_score, pre, seed, members)
    if not hidden_label_isolation:
        raise AssertionError("hidden-label acquisition invariance failed")
    exposures = exposure_vectors(candidate_pd, seed)
    total_risk = {name: float(np.dot(candidate_pd, value)) for name, value in exposures.items()}
    if not np.isclose(total_risk["low"], total_risk["high"], rtol=1e-10, atol=1e-10):
        raise AssertionError("exposure conditions do not share total predicted risk")
    common_hashes = {"visible_hash": digest(visible), "batch_hash": digest(batch),
        "accepted_hash": digest(accepted), "candidate_hash": digest(candidate),
        "candidate_pd_hash": digest(candidate_pd, np.float64),
        "test0_hash": digest(task["test0"]), "test1_hash": digest(task["test1"])}
    rows = []
    test_label_isolation = None
    for exposure_name, exposure in exposures.items():
        cost = candidate_pd * exposure
        budget = float(CONFIG["risk_budget_ratio"] * cost.sum())
        cost_hash = digest(cost, np.float64)
        order = np.argsort(cost, kind="mergesort")
        cumsum = np.cumsum(cost[order])
        selections = {}
        kstars = {}
        for policy in POLICIES:
            local, kstar = select_policy(policy, cost, ig_score, candidate, budget, seed)
            selections[policy], kstars[policy] = local, kstar
        anchor = candidate[selections["CountOnly-MinRisk"]]
        for policy in POLICIES:
            branch_started = time.perf_counter()
            local, kstar = selections[policy], kstars[policy]
            selected = candidate[local]
            visible_after = np.concatenate([visible, accepted, selected]).astype(int)
            _, _, after_prediction, after_backend = p2.fit_predict_gpu(
                raw, y, visible_after, branch_training_seed(seed))
            if after_backend != "gpu":
                raise RuntimeError(f"canonical GPU required, observed {after_backend}")
            if test_label_isolation is None:
                test_label_isolation = test_label_isolation_audit(
                    raw, y, visible_after, task["test0"], task["test1"], seed,
                    after_prediction)
                if not test_label_isolation:
                    raise AssertionError("test-label prediction invariance failed")
            selected_count = len(local)
            selected_risk = float(cost[local].sum()) if selected_count else 0.0
            realized_loss = float(exposure[local][y[selected] == 1].sum()) if selected_count else 0.0
            min_selected_count = float(cumsum[selected_count - 1]) if selected_count else 0.0
            min_kstar = float(cumsum[kstar - 1]) if kstar else 0.0
            slack = budget - min_kstar
            released = min_kstar - min_selected_count
            premium = selected_risk - min_selected_count
            unused = budget - selected_risk
            identity_residual = (slack + released) - (premium + unused)
            base = {"experiment": CONFIG["experiment_id"], "seed": int(seed),
                "exposure_condition": exposure_name, "method": policy,
                "committee_members": int(members), "committee_backend": "cpu",
                "committee_runtime_seconds": committee_seconds, "canonical_backend": f"{initial_backend}+{after_backend}",
                "gpu_enabled": True, "n_visible": len(visible), "visible_region1_share": float(raw.iloc[visible]["region"].mean()),
                "n_batch": len(batch), "n_accepted": len(accepted), "n_candidate": len(candidate),
                "k_star": kstar, "selected_count": selected_count, "count_sacrifice": kstar - selected_count,
                "risk_budget": budget, "selected_predicted_risk": selected_risk,
                "realized_loss": realized_loss, "minimum_risk_at_selected_count": min_selected_count,
                "minimum_risk_at_kstar": min_kstar, "maximum_count_slack": slack,
                "released_minimum_risk": released, "composition_premium": premium,
                "unused_budget": unused, "budget_identity_residual": identity_residual,
                "premium_over_available_risk": premium / (slack + released) if slack + released > 1e-12 else np.nan,
                "ig_score_sum": float(ig_score[local].sum()) if selected_count else 0.0,
                "selected_region1_share": float(raw.iloc[selected]["region"].mean()) if selected_count else np.nan,
                "selection_jaccard_vs_minrisk": jaccard(selected, anchor),
                "selected_ids": json.dumps([int(v) for v in selected]),
                "exposure_mean": float(exposure.mean()), "exposure_cv": float(exposure.std() / exposure.mean()),
                "predicted_budget_feasible": bool(selected_risk <= budget + 1e-7),
                "hidden_label_isolation": hidden_label_isolation,
                "test_label_isolation": test_label_isolation,
                "preprocessing_visible_only": True, "selection_independent_of_target_weight": True,
                "branch_training_seed": branch_training_seed(seed),
                "branch_runtime_seconds": float(time.perf_counter() - branch_started),
                "seed_runtime_seconds_elapsed": float(time.perf_counter() - started),
                "cost_hash": cost_hash, **common_hashes}
            p0, p1_values = after_prediction[task["test0"]], after_prediction[task["test1"]]
            y0, y1 = y[task["test0"]], y[task["test1"]]
            for region_weight in CONFIG["target_region_weights"]:
                row = dict(base)
                row["target_region1_weight"] = float(region_weight)
                row.update(weighted_metrics(y0, p0, y1, p1_values, region_weight))
                rows.append(row)
    return pd.DataFrame(rows)


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume or a new path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds or CONFIG["seeds"]
    run_config = dict(CONFIG)
    run_config.update({"seeds": seeds, "committee_members": int(args.members),
        "canonical_gpu_requested": bool(p2.USE_GPU), "canonical_gpu_available": bool(p2.GPU_AVAILABLE)})
    (output / "run_config.json").write_bytes(
        (json.dumps(run_config, indent=2) + "\n").encode("utf-8"))
    if not p2.USE_GPU or not p2.GPU_AVAILABLE:
        raise RuntimeError("canonical GPU backend is required")
    for completed, seed in enumerate(seeds, 1):
        destination = output / "seeds" / f"seed{int(seed)}.csv"
        if not destination.exists():
            atomic_csv(run_seed(int(seed), int(args.members)), destination)
        print(f"PROGRESS {completed}/{len(seeds)} seed={seed}", flush=True)
    manifest = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.csv":
            manifest.append({"path": path.relative_to(output).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    atomic_csv(pd.DataFrame(manifest), output / "manifest.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--members", type=int, default=CONFIG["committee_members"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
