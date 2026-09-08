"""Fresh-seed CCFB extension runner.

This module is intentionally separate from frozen Phase 1--4 code.  It uses
the exact frozen split and fit functions, but records all new evidence under
results/ccfb_extension.  The outer test set is used only by ``p1.metrics``
after an acquisition decision is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import phase1_runner as p1  # frozen import
import phase2_runner as p2  # frozen import; retains GPU implementation

CONFIG_PATH = Path(__file__).with_name("config.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / "results" / "ccfb_extension"
RAW = OUT / "raw"
LAMBDA_RAW = OUT / "lambda_sensitivity"

DATASETS = {
    "uci": {
        "path": ROOT / "data" / "raw" / "uci" / "uci_default_credit_card.csv",
        "target": "default", "exposure": None, "risk_definition": "unit_exposure; EL=PD_hat",
        "evidence_role": "development_benchmark",
    },
    "hmeq": {
        "path": ROOT / "data" / "raw" / "hmeq" / "hmeq.csv",
        "target": "BAD", "exposure": "LOAN", "risk_definition": "proxy; EL=PD_hat*LOAN; LGD=1",
        "evidence_role": "development_monetary_validation",
    },
    "ppdai": {
        "path": ROOT / "data" / "processed" / "ppdai" / "ppdai.csv",
        "target": "label", "exposure": "exposure", "risk_definition": "proxy; EL=PD_hat*exposure; LGD=1",
        "evidence_role": "exploratory_external",
    },
    "lendingclub": {
        "path": ROOT / "data" / "processed" / "lendingclub" / "lendingclub.csv",
        "target": "target", "exposure": "exposure", "risk_definition": "proxy; EL=PD_hat*exposure; LGD=1",
        "evidence_role": "primary_external_validation",
    },
}

ABLATION_METHODS = [
    "CountOnly-MinRisk", "QualityFirst-Uncertainty", "QualityFirst-FAVE",
    "LRBE-Uncertainty", "LRBE-FAVE", "Joint-Best-Val-Uncertainty",
]
BUDGET_METHODS = ["CountOnly-MinRisk", "QualityFirst-Uncertainty", "QualityFirst-FAVE", "LRBE-FAVE"]
REFERENCE_METHODS = ["NoExplore", "Oracle"]
SUPPLEMENTAL_METHODS = ["RiskBounded-Random", "RiskBounded-Epsilon"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_exposure(values: pd.Series, *, dataset: str, column: str) -> np.ndarray:
    """Return strictly positive finite exposure values or fail closed."""
    numeric = pd.to_numeric(values, errors="coerce")
    raw_missing = values.isna()
    non_numeric = values.notna() & numeric.isna()
    array = numeric.to_numpy(dtype=float)
    nonfinite = ~np.isfinite(array)
    nonpositive = np.isfinite(array) & (array <= 0.0)
    if raw_missing.any() or non_numeric.any() or nonfinite.any() or nonpositive.any():
        raise ValueError(
            f"Invalid exposure in {dataset}.{column}: "
            f"missing={int(raw_missing.sum())}, "
            f"non_numeric={int(non_numeric.sum())}, "
            f"nonfinite={int(nonfinite.sum())}, "
            f"nonpositive={int(nonpositive.sum())}. "
            "Exposure must be finite and strictly positive."
        )
    return array


def dataset_frame(dataset: str) -> tuple[pd.DataFrame, str, np.ndarray, np.ndarray, pd.DataFrame]:
    cfg = DATASETS[dataset]
    df = pd.read_csv(cfg["path"], low_memory=False)
    target = cfg["target"]
    y = df[target].astype(int).to_numpy()
    excluded = {target, "ID"}
    if cfg["exposure"]:
        excluded.add(cfg["exposure"])
    raw = df[[c for c in df.columns if c not in excluded]].copy()
    if cfg["exposure"] is None:
        exposure = np.ones(len(df), dtype=float)
    else:
        exposure = validate_exposure(
            df[cfg["exposure"]], dataset=dataset, column=cfg["exposure"]
        )
    return df, target, y, exposure, raw


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return np.nan_to_num((values - lo) / (hi - lo), nan=0.0, posinf=0.0, neginf=0.0)


def quality_values(kind: str, p: np.ndarray, cost: np.ndarray, candidate: np.ndarray,
                   raw: pd.DataFrame, visible: np.ndarray, batch: np.ndarray, pre, model) -> np.ndarray:
    if kind == "Uncertainty":
        return np.asarray(p, dtype=float) * (1.0 - np.asarray(p, dtype=float))
    if kind == "FAVE":
        return p2.fave_value_gpu(raw, visible, candidate, batch, pre, model)
    raise ValueError(f"Unknown quality kind: {kind}")


def max_quality_select(cost: np.ndarray, utility: np.ndarray, budget: float) -> np.ndarray:
    """Quality-first knapsack: no K* cardinality constraint."""
    cost = np.asarray(cost, dtype=float)
    utility = np.maximum(np.nan_to_num(utility, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if not len(cost) or budget <= 0:
        return np.empty(0, dtype=int)
    feasible = cost <= budget + 1e-12
    if not feasible.any():
        return np.empty(0, dtype=int)
    idx = np.flatnonzero(feasible)
    c, u = cost[idx], utility[idx]
    con = LinearConstraint(c.reshape(1, -1), -np.inf, budget)
    result = milp(c=-u, integrality=np.ones(len(c), dtype=int),
                  bounds=Bounds(np.zeros(len(c)), np.ones(len(c))), constraints=[con])
    if not result.success:
        raise RuntimeError(f"Quality-first optimizer failed: {result.message}")
    return idx[np.flatnonzero(np.rint(result.x) > 0.5)].astype(int)


def joint_select(cost: np.ndarray, quality: np.ndarray, budget: float, lam: float) -> np.ndarray:
    """Normalized Q - lambda R joint objective, constrained only by budget."""
    score = normalize(quality) - float(lam) * normalize(cost)
    # A zero/negative objective should not force a selection; this is the
    # defining contrast with LRBE's fixed K* stage.
    positive = score > 0.0
    if not positive.any() or budget <= 0:
        return np.empty(0, dtype=int)
    idx = np.flatnonzero(positive & (cost <= budget + 1e-12))
    if not len(idx):
        return np.empty(0, dtype=int)
    c, s = cost[idx], score[idx]
    con = LinearConstraint(c.reshape(1, -1), -np.inf, budget)
    result = milp(c=-s, integrality=np.ones(len(c), dtype=int),
                  bounds=Bounds(np.zeros(len(c)), np.ones(len(c))), constraints=[con])
    if not result.success:
        raise RuntimeError(f"Joint optimizer failed: {result.message}")
    return idx[np.flatnonzero(np.rint(result.x) > 0.5)].astype(int)


def select(method: str, p: np.ndarray, cost: np.ndarray, candidate: np.ndarray, budget: float,
           raw: pd.DataFrame, visible: np.ndarray, batch: np.ndarray, pre, model,
           joint_lambda: float | None) -> tuple[np.ndarray, int, np.ndarray, str]:
    if method == "NoExplore":
        return np.empty(0, dtype=int), 0, np.empty(0), "none"
    if method == "Oracle":
        return np.arange(len(candidate), dtype=int), len(candidate), np.ones(len(candidate)), "oracle_all"
    if method == "CountOnly-MinRisk":
        utility = -cost
        local, k = p2.lrbe_select(cost, utility, budget)
        return local, k, utility, "lrbe_minrisk"
    if method == "RiskBounded-Random":
        # External-style randomized exploration within the current policy's
        # candidate pool and predicted-risk budget. A deterministic hash gives a reproducible
        # pseudo-random order without consuming labels or touching the test set.
        scores = p2.deterministic_hash_scores(candidate, seed=int(visible[0]) if len(visible) else 0,
                                               round_id=int(batch[0]) if len(batch) else 0)
        order = np.argsort(scores, kind="mergesort")
        chosen = []
        remaining = float(budget)
        for j in order:
            if cost[j] <= remaining + 1e-12:
                chosen.append(int(j)); remaining -= float(cost[j])
        local = np.asarray(chosen, dtype=int)
        return local, len(local), scores, "risk_bounded_random_greedy"
    if method == "RiskBounded-Epsilon":
        # Epsilon exploration: preserve the exact feasible K* cardinality,
        # then perturb the min-risk objective with a fixed epsilon random score.
        random_score = p2.deterministic_hash_scores(candidate, seed=int(visible[0]) if len(visible) else 0,
                                                     round_id=int(batch[0]) + 7919 if len(batch) else 7919)
        c_norm = normalize(cost)
        utility = -(1.0 - 0.10) * c_norm + 0.10 * random_score
        local, k = p2.lrbe_select(cost, utility, budget)
        return local, k, utility, "risk_bounded_epsilon"
    if method.startswith("LRBE-"):
        kind = method.removeprefix("LRBE-")
        utility = quality_values(kind, p, cost, candidate, raw, visible, batch, pre, model)
        local, k = p2.lrbe_select(cost, utility, budget)
        return local, k, utility, f"lrbe_{kind.lower()}"
    if method.startswith("QualityFirst-"):
        kind = method.removeprefix("QualityFirst-")
        utility = quality_values(kind, p, cost, candidate, raw, visible, batch, pre, model)
        return max_quality_select(cost, utility, budget), -1, utility, f"quality_first_{kind.lower()}"
    if method == "Joint-Best-Val-Uncertainty":
        if joint_lambda is None:
            raise ValueError("Joint method requires a validation-selected lambda")
        utility = quality_values("Uncertainty", p, cost, candidate, raw, visible, batch, pre, model)
        return joint_select(cost, utility, budget, joint_lambda), -1, utility, "joint_normalized"
    raise ValueError(f"Unknown method: {method}")


def choose_joint_lambda(raw: pd.DataFrame, y: np.ndarray, exposure: np.ndarray, d0: np.ndarray,
                        seed: int, rho: float, dataset: str) -> tuple[float, pd.DataFrame]:
    """Inner-only pseudo-acquisition validation for Joint-Best-Val.

    D0 is split deterministically into model training, a pseudo candidate pool,
    and a held-out validation set.  Labels from the pseudo pool are revealed
    only after selection; evaluation is on the disjoint held-out validation
    set.  No stream or outer-test label is inspected here.
    """
    labels = y[d0]
    try:
        split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=seed + 17001)
        tr_local, hold_local = next(split1.split(d0, labels))
        train_idx, hold_idx = d0[tr_local], d0[hold_local]
        split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 17002)
        pool_local, eval_local = next(split2.split(hold_idx, y[hold_idx]))
        pool_idx, eval_idx = hold_idx[pool_local], hold_idx[eval_local]
    except ValueError:
        # This can occur for extremely small/minority-stratified D0.  The
        # explicit fallback is deterministic and logged; it never touches test.
        perm = np.random.RandomState(seed + 17001).permutation(d0)
        ntr, npool = max(2, int(.6 * len(d0))), max(1, int(.2 * len(d0)))
        train_idx, pool_idx, eval_idx = perm[:ntr], perm[ntr:ntr + npool], perm[ntr + npool:]
    pre, model, pred, backend = p2.fit_predict_gpu(raw, y, train_idx, seed * 1000 + 71)
    p_pool = pred[pool_idx]
    accepted = pool_idx[p_pool < p1.TAU]
    candidate = pool_idx[p_pool >= p1.TAU]
    pc = p_pool[p_pool >= p1.TAU]
    cost = pc * exposure[candidate]
    budget = float(rho * cost.sum())
    quality = pc * (1 - pc)
    rows = []
    for lam in CONFIG["joint_lambda_grid"]:
        local = joint_select(cost, quality, budget, lam)
        selected = candidate[local]
        fit_idx = np.concatenate([train_idx, accepted, selected]).astype(int)
        pre2, model2, pred2, backend2 = p2.fit_predict_gpu(raw, y, fit_idx, seed * 1000 + 72 + int(100 * lam))
        p_eval = pred2[eval_idx]
        rows.append({
            "dataset": dataset, "seed": seed, "rho": rho, "lambda": float(lam),
            "validation_pr_auc": float(average_precision_score(y[eval_idx], p_eval)) if len(np.unique(y[eval_idx])) > 1 else np.nan,
            "validation_recall": float(recall_score(y[eval_idx], p_eval >= p1.TAU, zero_division=0)),
            "validation_selected_count": int(len(selected)),
            "validation_predicted_risk": float(cost[local].sum()) if len(local) else 0.0,
            "validation_budget": budget, "validation_backend": f"{backend}+{backend2}",
            "selection_source": "D0_inner_train_pseudopool_validation_only",
        })
    frame = pd.DataFrame(rows)
    ordered = frame.sort_values(["validation_pr_auc", "validation_recall", "lambda"], ascending=[False, False, True], na_position="last")
    return float(ordered.iloc[0]["lambda"]), frame


def run_episode(dataset: str, seed: int, rho: float, method: str, experiment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, target, y, exposure, raw = dataset_frame(dataset)
    fixed = p1.fixed_split(df, target, seed)
    test, batches, visible = fixed["test"], fixed["batches"], fixed["visible"].copy()
    stream = fixed["stream"]
    d0_hash = hashlib.sha256(np.asarray(visible, dtype=np.int64).tobytes()).hexdigest()
    test_hash = hashlib.sha256(np.asarray(test, dtype=np.int64).tobytes()).hexdigest()
    dataset_hash = _sha(DATASETS[dataset]["path"])
    pop_rate = float(y[stream].mean())
    lambda_rows = pd.DataFrame()
    joint_lambda = None
    if method == "Joint-Best-Val-Uncertainty":
        joint_lambda, lambda_rows = choose_joint_lambda(raw, y, exposure, visible.copy(), seed, rho, dataset)
        lambda_rows.insert(0, "experiment", experiment)
        lambda_rows.insert(1, "chosen_lambda", joint_lambda)
    rows: list[dict] = []
    cumulative_pred, cumulative_real, cumulative_count = 0.0, 0.0, 0
    t0 = time.perf_counter()
    for round_id in range(0, len(batches) + 1):
        if round_id == 0:
            pre, model, pred, backend = p2.fit_predict_gpu(raw, y, visible, seed * 10000)
            # The frozen helper expects explicit empty batch vectors at round
            # zero (its defaults predate the phase-0 record convention).
            metric = p1.metrics(
                y[test], pred[test], y[visible], pop_rate,
                np.empty(0, dtype=int), np.empty(0, dtype=float),
                np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int),
            )
            metric.update({
                "experiment": experiment, "dataset": dataset, "evidence_role": DATASETS[dataset]["evidence_role"],
                "seed": seed, "rho": rho, "budget_multiplier": rho / CONFIG["base_rho"], "round": 0,
                "method": method, "selection_structure": "reference", "joint_lambda": joint_lambda,
                "n_explored_round": 0, "feedback_count": 0, "k_star": 0, "risk_budget": 0.0,
                "predicted_exploration_risk": 0.0, "realized_exploration_loss": 0.0,
                "risk_ratio": np.nan, "budget_violation": False, "budget_utilization": np.nan,
                "cumulative_feedback_count": 0, "cumulative_predicted_risk": 0.0,
                "cumulative_realized_loss": 0.0, "visible_default_rate": float(y[visible].mean()),
                "fd_ratio": float(y[visible].mean() / pop_rate) if pop_rate else np.nan,
                "split_test_hash": test_hash, "split_d0_hash": d0_hash, "risk_definition": DATASETS[dataset]["risk_definition"],
                "dataset_input_sha256": dataset_hash, "exposure_validation_passed": True,
                "incoming_batch_hash": "ROUND_ZERO", "candidate_pool_hash": "ROUND_ZERO",
                "compute_backend": backend, "gpu_enabled": bool(p2.USE_GPU and p2.GPU_AVAILABLE),
                "outer_test_used_only_for_metrics": True, "hidden_label_isolation": True,
            })
            rows.append(metric)
            continue
        batch = np.asarray(batches[round_id - 1], dtype=int)
        visible_before = visible.copy()
        pre, model, pred, backend_before = p2.fit_predict_gpu(raw, y, visible_before, seed * 10000 + round_id)
        p_batch = pred[batch]
        accepted = batch[p_batch < p1.TAU]
        candidate = batch[p_batch >= p1.TAU]
        incoming_batch_hash = hashlib.sha256(np.asarray(batch, dtype=np.int64).tobytes()).hexdigest()
        candidate_pool_hash = hashlib.sha256(np.asarray(candidate, dtype=np.int64).tobytes()).hexdigest()
        pc = p_batch[p_batch >= p1.TAU]
        cost = pc * exposure[candidate]
        budget = float(rho * cost.sum())
        local, kstar, utility, structure = select(method, pc, cost, candidate, budget, raw, visible_before, batch, pre, model, joint_lambda)
        selected = candidate[local]
        pred_risk = float(cost[local].sum()) if len(local) else 0.0
        real_loss = float(exposure[selected][y[selected] == 1].sum()) if len(selected) else 0.0
        gained = np.concatenate([accepted, selected]).astype(int)
        visible = np.concatenate([visible_before, gained]).astype(int)
        cumulative_pred += pred_risk
        cumulative_real += real_loss
        cumulative_count += len(selected)
        pre_after, model_after, pred_after, backend_after = p2.fit_predict_gpu(raw, y, visible, seed * 10000 + 500 + round_id)
        metric = p1.metrics(y[test], pred_after[test], y[visible], pop_rate, y[batch], p_batch, gained, candidate, accepted)
        risk_ratio = real_loss / (pred_risk + 1e-12) if pred_risk > 0 else np.nan
        metric.update({
            "experiment": experiment, "dataset": dataset, "evidence_role": DATASETS[dataset]["evidence_role"],
            "seed": seed, "rho": rho, "budget_multiplier": rho / CONFIG["base_rho"], "round": round_id,
            "method": method, "selection_structure": structure, "joint_lambda": joint_lambda,
            "n_explored_round": int(len(selected)), "feedback_count": int(len(selected)), "k_star": int(kstar),
            "risk_budget": budget, "predicted_exploration_risk": pred_risk,
            "realized_exploration_loss": real_loss, "risk_ratio": risk_ratio,
            "budget_violation": bool(real_loss > budget + 1e-9),
            "budget_utilization": float(pred_risk / budget) if budget > 0 else np.nan,
            "cumulative_feedback_count": int(cumulative_count), "cumulative_predicted_risk": float(cumulative_pred),
            "cumulative_realized_loss": float(cumulative_real), "visible_default_rate": float(y[visible].mean()),
            "fd_ratio": float(y[visible].mean() / pop_rate) if pop_rate else np.nan,
            "split_test_hash": test_hash, "split_d0_hash": d0_hash, "risk_definition": DATASETS[dataset]["risk_definition"],
            "dataset_input_sha256": dataset_hash, "exposure_validation_passed": True,
            "incoming_batch_hash": incoming_batch_hash, "candidate_pool_hash": candidate_pool_hash,
            "compute_backend": f"{backend_before}+{backend_after}", "gpu_enabled": bool(p2.USE_GPU and p2.GPU_AVAILABLE),
            "outer_test_used_only_for_metrics": True, "hidden_label_isolation": True,
            "selected_default_rate_oracle": float(y[selected].mean()) if len(selected) else np.nan,
            "selected_mean_pd": float(pc[local].mean()) if len(local) else np.nan,
            "selected_mean_exposure": float(exposure[selected].mean()) if len(selected) else np.nan,
            "runtime_seconds_elapsed": float(time.perf_counter() - t0),
        })
        rows.append(metric)
    return pd.DataFrame(rows), lambda_rows


def write_run(frame: pd.DataFrame, experiment: str, dataset: str, seed: int, rho: float, method: str) -> Path:
    dest = RAW / experiment / dataset
    dest.mkdir(parents=True, exist_ok=True)
    safe = method.replace("-", "_")
    path = dest / f"seed{seed}_rho{rho:.3f}_{safe}.csv"
    frame.to_csv(path, index=False)
    return path


def one(experiment: str, dataset: str, seed: int, rho: float, method: str, force: bool = False) -> Path:
    safe = method.replace("-", "_")
    path = RAW / experiment / dataset / f"seed{seed}_rho{rho:.3f}_{safe}.csv"
    if path.exists() and not force:
        return path
    frame, lambda_frame = run_episode(dataset, seed, rho, method, experiment)
    result = write_run(frame, experiment, dataset, seed, rho, method)
    if len(lambda_frame):
        LAMBDA_RAW.mkdir(parents=True, exist_ok=True)
        lambda_frame.to_csv(LAMBDA_RAW / f"{experiment}_{dataset}_seed{seed}_rho{rho:.3f}.csv", index=False)
    return result


def run_reference(dataset: str, seed: int, force: bool) -> None:
    for method in REFERENCE_METHODS:
        one("reference", dataset, seed, CONFIG["base_rho"], method, force)


def run_smoke(force: bool) -> None:
    # Covers all selection structures, but only one small monetary dataset and
    # one fresh seed.  It is a correctness test, not formal evidence.
    for method in ABLATION_METHODS:
        one("smoke", "hmeq", CONFIG["fresh_seeds"][0], CONFIG["base_rho"], method, force)
    for method in REFERENCE_METHODS:
        one("smoke", "hmeq", CONFIG["fresh_seeds"][0], CONFIG["base_rho"], method, force)


def run_full(force: bool) -> None:
    for dataset in CONFIG["datasets"]:
        for seed in CONFIG["fresh_seeds"]:
            run_reference(dataset, seed, force)
            for method in ABLATION_METHODS:
                one("kstar_ablation", dataset, seed, CONFIG["base_rho"], method, force)
            for multiplier in CONFIG["budget_multipliers"]:
                rho = float(CONFIG["base_rho"] * multiplier)
                for method in BUDGET_METHODS:
                    one("budget_sweep", dataset, seed, rho, method, force)
            print(f"CCFB extension completed {dataset} seed={seed}", flush=True)


def run_supplemental(force: bool) -> None:
    """Fresh external-style exploration baselines at the preregistered rho."""
    for dataset in CONFIG["datasets"]:
        for seed in CONFIG["fresh_seeds"]:
            for method in SUPPLEMENTAL_METHODS:
                one("supplemental_exploration", dataset, seed, CONFIG["base_rho"], method, force)
            print(f"Supplemental exploration completed {dataset} seed={seed}", flush=True)


def write_run_manifest() -> None:
    files = []
    for path in sorted(RAW.rglob("*.csv")) if RAW.exists() else []:
        files.append({"path": str(path.relative_to(OUT)).replace("\\", "/"), "sha256": _sha(path), "bytes": path.stat().st_size})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(files).to_csv(OUT / "RAW_RUN_MANIFEST.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full", "supplemental", "one"], required=True)
    ap.add_argument("--dataset", choices=sorted(DATASETS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--rho", type=float, default=CONFIG["base_rho"])
    ap.add_argument("--method", choices=ABLATION_METHODS + BUDGET_METHODS + REFERENCE_METHODS + SUPPLEMENTAL_METHODS)
    ap.add_argument("--experiment", default="manual")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        run_smoke(args.force)
    elif args.mode == "full":
        run_full(args.force)
    elif args.mode == "supplemental":
        run_supplemental(args.force)
    else:
        if args.dataset is None or args.seed is None or args.method is None:
            ap.error("--mode one requires --dataset, --seed, and --method")
        one(args.experiment, args.dataset, args.seed, args.rho, args.method, args.force)
    write_run_manifest()


if __name__ == "__main__":
    main()
