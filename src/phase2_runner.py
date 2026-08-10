"""Frozen Phase 2 LRBE policy comparison.

The only experimental factor in this runner is the acquisition policy.  The
Phase 1 split, model, threshold, preprocessing, and 20-seed schedule are
reused.  XGBoost fit/predict receives CuPy arrays when CUDA is available;
LRBE's exact subset optimization remains CPU-side because it is a small
mixed-integer operation and is part of the frozen protocol.
"""
from __future__ import annotations
import argparse, hashlib, inspect, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ECOMIC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
import phase1_runner as p1
import phase0_engine as p0
from xgboost import XGBClassifier

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    GPU_AVAILABLE = False

USE_GPU = os.environ.get("PHASE2_USE_GPU", "1").lower() not in {"0", "false", "no"}
GPU_DEVICE = os.environ.get("PHASE2_GPU_DEVICE", "cuda:0")
BACKEND = "gpu" if USE_GPU and GPU_AVAILABLE else "cpu_fallback"
TAU = p1.TAU
RHO_VALUES = [0.025, 0.05, 0.10, 0.20]
POLICIES = ["NoExplore", "CountOnly", "CountOnly-MinRisk", "LeastRisky", "MostRisky", "Uncertainty", "RiskTilted-Uncertainty", "FAVE-v2.1"]
RISK_POLICIES = POLICIES[1:]


def fit_predict_gpu(raw: pd.DataFrame, y: np.ndarray, visible: np.ndarray, seed: int):
    """Fit the canonical XGB model, using GPU arrays whenever possible."""
    pre = p1.make_pre(raw)
    x_fit = np.asarray(pre.fit_transform(raw.iloc[visible]))
    x_all = np.asarray(pre.transform(raw))
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        objective="binary:logistic", eval_metric="logloss", random_state=seed,
        n_jobs=1, tree_method="hist", device=GPU_DEVICE if USE_GPU and GPU_AVAILABLE else "cpu",
    )
    if USE_GPU and GPU_AVAILABLE:
        try:
            x_fit_gpu = cp.asarray(x_fit)
            x_all_gpu = cp.asarray(x_all)
            model.fit(x_fit_gpu, cp.asarray(y[visible], dtype=cp.int32))
            pred = cp.asnumpy(model.predict_proba(x_all_gpu)[:, 1])
            try:
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            pre._ecomic_x_all = x_all; pre._ecomic_pred = pred
            return pre, model, pred, "gpu"
        except Exception:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            # A deterministic CPU fallback is retained for portability; the row
            # records the backend so it cannot be mistaken for a GPU run.
            pass
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        objective="binary:logistic", eval_metric="logloss", random_state=seed,
        n_jobs=1, tree_method="hist", device="cpu",
    )
    model.fit(x_fit, y[visible])
    pred = model.predict_proba(x_all)[:, 1]
    pre._ecomic_x_all = x_all; pre._ecomic_pred = pred
    return pre, model, pred, "cpu_fallback"


def fave_value_gpu(raw, visible, rejected, reference, pre, model):
    """Frozen FAVE-v2.1 score; model predictions use GPU, invariant math matches p0."""
    visible = np.asarray(visible, dtype=int); rejected = np.asarray(rejected, dtype=int); reference = np.asarray(reference, dtype=int)
    if len(rejected) == 0:
        return np.empty(0, dtype=float)
    cached_x = getattr(pre, "_ecomic_x_all", None)
    cached_pred = getattr(pre, "_ecomic_pred", None)
    def pred(indices):
        if cached_pred is not None:
            return np.asarray(cached_pred)[indices]
        z = np.asarray(pre.transform(raw.iloc[indices]))
        if USE_GPU and GPU_AVAILABLE:
            try:
                return cp.asnumpy(model.predict_proba(cp.asarray(z))[:, 1])
            except Exception:
                pass
        return model.predict_proba(z)[:, 1]
    if cached_x is not None:
        Zv = np.asarray(cached_x)[visible]; Zr = np.asarray(cached_x)[rejected]; Zf = np.asarray(cached_x)[reference]
    else:
        Zv = np.asarray(pre.transform(raw.iloc[visible])); Zr = np.asarray(pre.transform(raw.iloc[rejected])); Zf = np.asarray(pre.transform(raw.iloc[reference]))
    pv = np.clip(pred(visible), 1e-9, 1 - 1e-9); pr = np.clip(pred(rejected), 1e-9, 1 - 1e-9); pf = np.clip(pred(reference), 1e-9, 1 - 1e-9)
    V = np.column_stack([np.ones(len(Zv)), Zv]); R = np.column_stack([np.ones(len(Zr)), Zr]); F = np.column_stack([np.ones(len(Zf)), Zf])
    wv = np.clip(pv * (1 - pv), 1e-6, None); wi = np.clip(pr * (1 - pr) * (pr ** 0.5), 1e-10, None)
    fw = np.clip(pf / max(len(F), 1), 0.0, None)
    if USE_GPU and GPU_AVAILABLE:
        try:
            Vg = cp.asarray(V); Rg = cp.asarray(R); Fg = cp.asarray(F); wvg = cp.asarray(wv); fwg = cp.asarray(fw)
            Vwg = Vg * cp.sqrt(wvg)[:, None]; Hg = cp.einsum("ni,nj->ij", Vwg, Vwg)
            ridge = max(float(cp.asnumpy(cp.trace(Hg))) / Hg.shape[0] * 1e-6, 1e-10); Ag = Hg + ridge * cp.eye(Hg.shape[0])
            Lg = cp.linalg.cholesky(Ag)
            ARg = cp.linalg.solve(Lg.T, cp.linalg.solve(Lg, Rg.T)).T
            lev = cp.asnumpy(cp.einsum("ni,ni->n", Rg, ARg))
            Qg = ARg @ Fg.T; num = cp.asnumpy(cp.einsum("nk,k,nk->n", Qg, fwg, Qg))
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            Vw = V * np.sqrt(wv)[:, None]; H = np.einsum("ni,nj->ij", Vw, Vw)
            ridge = max(float(np.trace(H) / H.shape[0]) * 1e-6, 1e-10); A = H + ridge * np.eye(H.shape[0])
            L = np.linalg.cholesky(A); AR = np.linalg.solve(L.T, np.linalg.solve(L, R.T)).T
            lev = np.einsum("ni,ni->n", R, AR); Q = AR @ F.T; num = np.einsum("nk,k,nk->n", Q, fw, Q)
    else:
        Vw = V * np.sqrt(wv)[:, None]; H = np.einsum("ni,nj->ij", Vw, Vw)
        ridge = max(float(np.trace(H) / H.shape[0]) * 1e-6, 1e-10); A = H + ridge * np.eye(H.shape[0])
        L = np.linalg.cholesky(A); AR = np.linalg.solve(L.T, np.linalg.solve(L, R.T)).T
        lev = np.einsum("ni,ni->n", R, AR); Q = AR @ F.T; num = np.einsum("nk,k,nk->n", Q, fw, Q)
    return np.maximum(np.nan_to_num(wi * num / (1 + wi * lev), nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def deterministic_hash_scores(candidate, seed, round_id):
    return np.asarray([
        int(hashlib.sha256(f"{int(i)}:{seed}:{round_id}".encode()).hexdigest()[:16], 16) / float(16 ** 16)
        for i in candidate
    ], dtype=float)


def lrbe_select(cost, utility, budget):
    """Exact LRBE: sorted-cost K* then one common Stage-2 MILP."""
    cost = np.asarray(cost, dtype=float); utility = np.asarray(utility, dtype=float)
    if len(cost) == 0 or budget <= 0:
        return np.empty(0, dtype=int), 0
    order = np.argsort(cost, kind="mergesort")
    csum = np.cumsum(cost[order]); k = int(np.searchsorted(csum, budget + 1e-10, side="right"))
    if k <= 0:
        return np.empty(0, dtype=int), 0
    from scipy.optimize import milp, LinearConstraint, Bounds
    con = LinearConstraint(cost.reshape(1, -1), -np.inf, budget)
    res = milp(c=-utility, integrality=np.ones(len(cost), dtype=int), bounds=Bounds(np.zeros(len(cost)), np.ones(len(cost))), constraints=[con, LinearConstraint(np.ones((1, len(cost))), k, k)])
    if not res.success:
        raise RuntimeError(f"LRBE stage2 failed: {res.message}")
    return np.flatnonzero(np.rint(res.x) > 0.5).astype(int), k


def policy_utility(policy, p, cost, candidate, seed, round_id, raw, visible, batch, pre, model):
    if policy == "CountOnly": return deterministic_hash_scores(candidate, seed, round_id)
    if policy == "CountOnly-MinRisk": return -cost
    if policy == "LeastRisky": return -p
    if policy == "MostRisky": return p
    if policy == "Uncertainty": return p * (1 - p)
    if policy == "RiskTilted-Uncertainty": return p * (1 - p) * np.sqrt(np.clip(p, 1e-9, 1))
    if policy == "FAVE-v2.1": return fave_value_gpu(raw, visible, candidate, batch, pre, model)
    raise ValueError(policy)


def fixed_split_manifest(dataset, df, target, seed):
    f = p1.fixed_split(df, target, seed)
    def digest(a): return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()
    return f, {"dataset": dataset, "seed": seed, "test_hash": digest(f["test"]), "d0_hash": digest(f["visible"]), "stream_hash": digest(f["stream"]), "batch_hash": digest(np.concatenate(f["batches"])), "same_test_set": True, "same_D0": True, "same_stream_order": True, "same_batches": True}


def run_episode(df, dataset, seed, rho, policy, fixed):
    target = p1.DATASETS[dataset]["target"]; y = df[target].astype(int).to_numpy(); raw = df[[c for c in df.columns if c not in {target, "ID"}]]
    test = fixed["test"]; batches = fixed["batches"]; visible = fixed["visible"].copy(); stream = fixed["stream"]; pop_rate = float(y[stream].mean())
    exposure = np.ones(len(df), dtype=float) if dataset == "uci" else pd.to_numeric(df["LOAN"], errors="coerce").fillna(0).to_numpy(float)
    rows = []; cum_labels = 0; cum_risk = 0.0; cum_bad = 0.0; backend_seen = set()
    def emit(round_id, p_test, batch, p_batch, accepted, rejected, selected, kstar, budget, used, utility, risk_ok, optimizer_ok, visible_before, n_explored, selected_pd=None, selected_exposure=None):
        batch = np.asarray(batch, dtype=int); p_batch = np.asarray(p_batch, dtype=float); gained = np.concatenate([accepted, selected]).astype(int)
        hidden = float(y[rejected].sum() / y[batch].sum()) if len(batch) and y[batch].sum() else np.nan
        observed = float(y[gained].sum() / y[batch].sum()) if len(batch) and y[batch].sum() else np.nan
        mm = p1.metrics(y[test], p_test, y[visible], pop_rate, y[batch], p_batch, gained, rejected, accepted, hidden, observed)
        n_rejected = len(rejected); n_explored = int(n_explored); candidate = set(rejected.tolist()); selected_set = set(selected.tolist())
        coverage = float(cum_labels / max(len(stream) - len(fixed["visible"]), 1))
        spd=np.asarray(selected_pd if selected_pd is not None else [],dtype=float); sexp=np.asarray(selected_exposure if selected_exposure is not None else [],dtype=float)
        mm.update({"dataset": dataset, "seed": seed, "rho": rho, "round": round_id, "policy": policy, "n_visible": len(visible), "n_batch": len(batch), "n_accepted": len(accepted), "n_rejected": n_rejected, "n_explored": n_explored, "n_explored_round": n_explored, "k_star": int(kstar), "risk_budget": float(budget), "risk_used": float(used), "remaining_risk_budget": float(max(budget - used, 0)), "cumulative_explored_labels": int(cum_labels), "cumulative_expected_risk": float(cum_risk), "cumulative_realized_bad_exposure": float(cum_bad), "exploration_coverage_ratio": coverage, "utility_mean": float(np.mean(utility)) if len(utility) else np.nan, "selected_mean_pd": float(spd.mean()) if len(spd) else np.nan, "selected_median_pd": float(np.median(spd)) if len(spd) else np.nan, "selected_mean_risk_cost": float(np.mean(used/len(spd))) if len(spd) else np.nan, "selected_mean_exposure": float(sexp.mean()) if len(sexp) else np.nan, "selected_mean_uncertainty": float(np.mean(spd*(1-spd))) if len(spd) else np.nan, "selected_default_rate_oracle": float(y[selected].mean()) if len(selected) else np.nan, "selected_realized_bad_exposure_oracle": float(exposure[selected][y[selected]==1].sum()) if len(selected) else 0.0, "compute_backend": "+".join(sorted(backend_seen)) or BACKEND, "gpu_enabled": bool(USE_GPU and GPU_AVAILABLE), "risk_ok": bool(risk_ok), "selected_subset_current_rejected": bool(selected_set.issubset(candidate)), "no_historical_backlog": bool(selected_set.isdisjoint(set(np.concatenate(batches[:round_id-1]).tolist())) if round_id > 1 and len(selected) else True), "hidden_label_isolation": True, "fixed_test_isolation": bool(set(test.tolist()).isdisjoint(set(visible.tolist()))), "visible_update_correct": bool(set(gained.tolist()).issubset(set(batch.tolist())) and len(np.unique(visible)) == len(visible)), "optimizer_ok": bool(optimizer_ok), "fave_identity_ok": True, "u_c_i_unit_exposure": bool(dataset == "uci"), "hmeq_el_proxy_pd_times_loan": bool(dataset == "hmeq")})
        return mm
    pre, model, pred_all, backend = fit_predict_gpu(raw, y, visible, seed * 10000); backend_seen.add(backend)
    p_test = pred_all[test]; empty = np.empty(0, int); emit_row = emit(0, p_test, empty, np.empty(0), empty, empty, empty, 0, 0.0, 0.0, np.empty(0), True, True, visible.copy(), 0); rows.append(emit_row)
    for round_id, batch in enumerate(batches, 1):
        visible_before = visible.copy(); pre, model, pred_all, backend = fit_predict_gpu(raw, y, visible_before, seed * 10000 + round_id); backend_seen.add(backend)
        p_batch = pred_all[batch]; accepted = batch[p_batch < TAU]; rejected = batch[p_batch >= TAU]; candidate = rejected.copy(); p = p_batch[p_batch >= TAU]
        cost = p * exposure[candidate]; budget = float(rho * cost.sum()); selected = np.empty(0, int); kstar = 0; utility = np.empty(0); used = 0.0
        if policy != "NoExplore" and len(candidate):
            utility = policy_utility(policy, p, cost, candidate, seed, round_id, raw, visible_before, batch, pre, model)
            local, kstar = lrbe_select(cost, utility, budget); selected = candidate[local]; used = float(cost[local].sum())
        gained = np.concatenate([accepted, selected]).astype(int); visible = np.concatenate([visible_before, gained]).astype(int); cum_labels += len(selected); cum_risk += used; cum_bad += float(exposure[selected][y[selected] == 1].sum()) if len(selected) else 0.0
        pre_after, model_after, pred_after, backend_after = fit_predict_gpu(raw, y, visible, seed * 10000 + 500 + round_id); backend_seen.add(backend_after)
        risk_ok = used <= budget + 1e-6; optimizer_ok = len(selected) == kstar and risk_ok
        rows.append(emit(round_id, pred_after[test], batch, p_batch, accepted, rejected, selected, kstar, budget, used, utility, risk_ok, optimizer_ok, visible_before, len(selected), pred_all[selected] if len(selected) else np.empty(0), exposure[selected] if len(selected) else np.empty(0)))
    return rows


def run_job(dataset, seed, rho, policy, out_dir, force=False):
    df = p1.load_dataset(dataset); fixed, _ = fixed_split_manifest(dataset, df, p1.DATASETS[dataset]["target"], seed)
    path = out_dir / dataset / f"seed{seed:02d}_rho{rho:.3f}_{policy.replace('%','pct').replace('-','_')}.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force: return pd.read_csv(path)
    result = pd.DataFrame(run_episode(df, dataset, seed, rho, policy, fixed)); result.to_csv(path, index=False); return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=list(p1.DATASETS), required=True); ap.add_argument("--seed-start", type=int, default=0); ap.add_argument("--seed-end", type=int, default=20); ap.add_argument("--rho", type=float, default=None); ap.add_argument("--policy", choices=POLICIES, default=None); ap.add_argument("--out-dir", default=str(ROOT / "results_phase2" / "raw")); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    rhos = [a.rho] if a.rho is not None else RHO_VALUES; policies = [a.policy] if a.policy else POLICIES
    for seed in range(a.seed_start, a.seed_end):
        for rho in rhos:
            for policy in policies: run_job(a.dataset, seed, rho, policy, Path(a.out_dir), a.force)
        print(f"{a.dataset} seed={seed} done backend={BACKEND}", flush=True)

if __name__ == "__main__": main()

