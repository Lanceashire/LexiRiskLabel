"""Analysis, figures, and Chinese report for the CCFB fresh-seed extension."""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr, ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "ccfb_extension"
RAW = OUT / "raw"
FIG = OUT / "figures"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(experiment: str) -> pd.DataFrame:
    paths = sorted((RAW / experiment).rglob("*.csv")) if (RAW / experiment).exists() else []
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def terminal(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["round"].eq(10)].copy() if len(frame) else frame.copy()


def bootstrap_ci(values: np.ndarray, stat=np.mean, n: int = 10000, seed: int = 20260810) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=float)
    for i in range(n):
        samples[i] = stat(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def paired_rows(frame: pd.DataFrame, left: str, right: str, metric: str) -> pd.DataFrame:
    a = frame.loc[frame.method.eq(left), ["dataset", "seed", "rho", metric]].rename(columns={metric: "left"})
    b = frame.loc[frame.method.eq(right), ["dataset", "seed", "rho", metric]].rename(columns={metric: "right"})
    x = a.merge(b, on=["dataset", "seed", "rho"], how="inner")
    x["difference"] = x.left - x.right
    return x


def paired_summary(frame: pd.DataFrame, comparisons: list[tuple[str, str]], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for dataset, group in frame.groupby("dataset"):
        for left, right in comparisons:
            for metric in metrics:
                pair = paired_rows(group, left, right, metric)
                d = pair.difference.to_numpy(float)
                d = d[np.isfinite(d)]
                if not len(d):
                    continue
                ci_lo, ci_hi = bootstrap_ci(d)
                sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
                try:
                    p_t = float(ttest_rel(pair.left, pair.right, nan_policy="omit").pvalue)
                except Exception:
                    p_t = np.nan
                try:
                    p_w = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                except Exception:
                    p_w = np.nan
                rows.append({
                    "dataset": dataset, "comparison": f"{left} minus {right}", "metric": metric,
                    "n_paired_seeds": len(d), "mean_difference": float(np.mean(d)),
                    "std_difference": sd, "effect_size_cohen_dz": float(np.mean(d) / sd) if sd and np.isfinite(sd) else np.nan,
                    "bootstrap_95ci_low": ci_lo, "bootstrap_95ci_high": ci_hi,
                    "paired_t_pvalue": p_t, "wilcoxon_pvalue": p_w,
                })
    return pd.DataFrame(rows)


def reference_recovery(frame: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    ref = terminal(references)
    keys = ["dataset", "seed"]
    no = ref.loc[ref.method.eq("NoExplore"), keys + ["recall", "roc_auc", "pr_auc", "fd_ratio"]].rename(
        columns={x: f"{x}_noexplore" for x in ["recall", "roc_auc", "pr_auc", "fd_ratio"]})
    oracle = ref.loc[ref.method.eq("Oracle"), keys + ["recall", "roc_auc", "pr_auc", "fd_ratio"]].rename(
        columns={x: f"{x}_oracle" for x in ["recall", "roc_auc", "pr_auc", "fd_ratio"]})
    out = frame.merge(no, on=keys, how="left").merge(oracle, on=keys, how="left")
    for metric, name in [("recall", "recall_recovery"), ("roc_auc", "roc_auc_recovery"),
                         ("pr_auc", "pr_auc_recovery"), ("fd_ratio", "feedback_recovery_fdratio")]:
        denom = out[f"{metric}_oracle"] - out[f"{metric}_noexplore"]
        out[name] = np.where(np.abs(denom) > 1e-10, (out[metric] - out[f"{metric}_noexplore"]) / denom, np.nan)
    return out


def reliability_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = frame.loc[frame["round"].gt(0) & ~frame.method.eq("Oracle")].copy()
    group_cols = ["experiment", "dataset", "method", "rho"]
    output = []
    for key, g in rows.groupby(group_cols):
        ratio = g.risk_ratio.to_numpy(float)
        ratio = ratio[np.isfinite(ratio)]
        lo, hi = bootstrap_ci(ratio) if len(ratio) else (np.nan, np.nan)
        output.append(dict(zip(group_cols, key)) | {
            "n_dataset_seed_round": len(g), "risk_ratio_mean": float(np.mean(ratio)) if len(ratio) else np.nan,
            "risk_ratio_median": float(np.median(ratio)) if len(ratio) else np.nan,
            "risk_ratio_std": float(np.std(ratio, ddof=1)) if len(ratio) > 1 else np.nan,
            "risk_ratio_bootstrap_95ci_low": lo, "risk_ratio_bootstrap_95ci_high": hi,
            "budget_violation_rate_realized": float(g.budget_violation.astype(bool).mean()),
            "mean_predicted_risk": float(g.predicted_exploration_risk.mean()),
            "mean_realized_loss": float(g.realized_exploration_loss.mean()),
            "mean_brier": float(g.brier.mean()), "mean_ece": float(g.ece.mean()),
        })
    # Calibration curves are reconstructed from the frozen-format JSON field;
    # they are evaluation-only, never used to fit a calibrator.
    bins = []
    for _, row in terminal(rows).iterrows():
        try:
            values = json.loads(row.reliability_bins)
        except Exception:
            continue
        for value in values:
            bins.append({"experiment": row.experiment, "dataset": row.dataset, "method": row.method,
                         "rho": row.rho, "seed": row.seed, "bin": value["bin"],
                         "count": value["count"], "observed_default_rate": value["accuracy"],
                         "mean_predicted_pd": value["confidence"]})
    return pd.DataFrame(output), pd.DataFrame(bins)


def cluster_correlation(group: pd.DataFrame, x: str, y: str, label: str, n_boot: int = 10000) -> dict:
    """Seed-cluster bootstrap using vectorized resampling.

    Each bootstrap draw samples whole seed clusters with replacement.  The
    common balanced design is evaluated in a NumPy batch; an exact SciPy
    fallback is retained for unexpected unbalanced clusters.  Point estimates
    always use SciPy's tie-aware Pearson/Spearman implementations.
    """
    g = group[["seed", x, y]].dropna().copy()
    if len(g) < 4 or g[x].nunique() < 2 or g[y].nunique() < 2:
        return {"analysis_scope": label, "n": len(g), "pearson_r": np.nan, "spearman_rho": np.nan,
                "bootstrap_95ci_low": np.nan, "bootstrap_95ci_high": np.nan, "estimability": "NOT_ESTIMABLE"}
    pear = float(pearsonr(g[x], g[y]).statistic)
    spear = float(spearmanr(g[x], g[y]).statistic)
    clusters = sorted(g.seed.unique())
    parts = [g.loc[g.seed.eq(seed), [x, y]].to_numpy(float) for seed in clusters]
    rng = np.random.default_rng(20260810)
    sizes = {len(part) for part in parts}
    if len(sizes) == 1:
        values = np.stack(parts, axis=0)  # seed x cell x variable
        draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
        xb = values[draws, :, 0].reshape(n_boot, -1)
        yb = values[draws, :, 1].reshape(n_boot, -1)
        # Average-rank handling keeps the vectorized implementation identical
        # to the tie-aware Spearman definition after cluster resampling.
        # A cluster may be drawn repeatedly.  Average ranks are therefore
        # required for duplicated values in a valid Spearman bootstrap.
        rx = rankdata(xb, axis=1, method="average").astype(float)
        ry = rankdata(yb, axis=1, method="average").astype(float)
        rx -= rx.mean(axis=1, keepdims=True); ry -= ry.mean(axis=1, keepdims=True)
        denom = np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
        boots = (rx * ry).sum(axis=1) / np.maximum(denom, 1e-15)
    else:
        # A rare fallback for an incomplete/filtered cell; it preserves the
        # exact tie-aware statistic and avoids pandas concatenation overhead.
        draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
        boots = []
        for draw in draws:
            sample = np.concatenate([parts[i] for i in draw], axis=0)
            if np.unique(sample[:, 0]).size >= 2 and np.unique(sample[:, 1]).size >= 2:
                boots.append(float(spearmanr(sample[:, 0], sample[:, 1]).statistic))
        boots = np.asarray(boots, dtype=float)
    return {"analysis_scope": label, "n": len(g), "pearson_r": pear, "spearman_rho": spear,
            "bootstrap_95ci_low": float(np.quantile(boots, .025)) if len(boots) else np.nan,
            "bootstrap_95ci_high": float(np.quantile(boots, .975)) if len(boots) else np.nan,
            "estimability": "ESTIMABLE_CLUSTER_BOOTSTRAP"}

def decoupling_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset, g in frame.groupby("dataset"):
        for x, y, name in [("feedback_recovery_fdratio", "recall_recovery", "FeedbackRecovery_vs_RecallRecovery"),
                           ("feedback_recovery_fdratio", "roc_auc_recovery", "FeedbackRecovery_vs_AUCRecovery"),
                           ("feedback_recovery_fdratio", "pr_auc_recovery", "FeedbackRecovery_vs_PRAUCRecovery")]:
            rows.append({"dataset": dataset, "rho": "ALL", "metric_pair": name} | cluster_correlation(g, x, y, "WITHIN_DATASET"))
        for rho, h in g.groupby("rho"):
            for x, y, name in [("feedback_recovery_fdratio", "recall_recovery", "FeedbackRecovery_vs_RecallRecovery"),
                               ("feedback_recovery_fdratio", "roc_auc_recovery", "FeedbackRecovery_vs_AUCRecovery"),
                               ("feedback_recovery_fdratio", "pr_auc_recovery", "FeedbackRecovery_vs_PRAUCRecovery")]:
                rows.append({"dataset": dataset, "rho": rho, "metric_pair": name} | cluster_correlation(h, x, y, "WITHIN_BUDGET"))
    # Pairwise ordering within the exact dataset x seed x budget cells.
    concordance = []
    for (dataset, seed, rho), g in frame.groupby(["dataset", "seed", "rho"]):
        for a, b in itertools.combinations(g.to_dict("records"), 2):
            fx = float(a["feedback_recovery_fdratio"] - b["feedback_recovery_fdratio"])
            dy = float(a["recall_recovery"] - b["recall_recovery"])
            if np.isfinite(fx) and np.isfinite(dy) and abs(fx) > 1e-12 and abs(dy) > 1e-12:
                concordance.append({"dataset": dataset, "seed": seed, "rho": rho,
                                    "method_a": a["method"], "method_b": b["method"],
                                    "feedback_difference": fx, "recall_difference": dy,
                                    "ordering_concordant": bool(np.sign(fx) == np.sign(dy))})
    con = pd.DataFrame(concordance)
    if len(con):
        summary = con.groupby("dataset", as_index=False).agg(
            comparable_pairs=("ordering_concordant", "size"),
            concordance_rate=("ordering_concordant", "mean"))
    else:
        summary = pd.DataFrame(columns=["dataset", "comparable_pairs", "concordance_rate"])
    return pd.DataFrame(rows), con.merge(summary, on="dataset", how="left") if len(con) else con


def save_figures(ablation: pd.DataFrame, budget: pd.DataFrame, reliability: pd.DataFrame, feedback: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    # B1: budget versus feedback quantity.
    for value, filename, ylabel in [("cumulative_feedback_count", "fig_budget_vs_feedback_count", "Feedback count"),
                                   ("recall_recovery", "fig_budget_vs_recall", "Recall recovery")]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        for ax, (dataset, g) in zip(axes.flat, budget.groupby("dataset")):
            for method, h in g.groupby("method"):
                s = h.groupby("rho")[value].agg(["mean", "std"]).reset_index()
                ax.plot(s.rho, s["mean"], marker="o", label=method)
                ax.fill_between(s.rho, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0), alpha=.12)
            ax.set_title(dataset); ax.set_xlabel("Risk budget (rho)"); ax.set_ylabel(ylabel)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / f"{filename}.png", dpi=220); fig.savefig(FIG / f"{filename}.pdf"); plt.close(fig)
    # A: ablation at B0.
    s = ablation.groupby(["dataset", "method"], as_index=False).agg(
        feedback_count=("cumulative_feedback_count", "mean"), recall=("recall", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, value, label in zip(axes, ["feedback_count", "recall"], ["Feedback count", "Recall"]):
        pivot = s.pivot(index="method", columns="dataset", values=value)
        pivot.plot.bar(ax=ax); ax.set_ylabel(label); ax.tick_params(axis="x", rotation=38)
    fig.tight_layout(); fig.savefig(FIG / "fig_kstar_ablation.png", dpi=220); fig.savefig(FIG / "fig_kstar_ablation.pdf"); plt.close(fig)
    # C: predicted versus realized risk.
    fig, ax = plt.subplots(figsize=(7, 6))
    for dataset, g in reliability.groupby("dataset"):
        ax.scatter(g.mean_predicted_risk, g.mean_realized_loss, alpha=.65, label=dataset)
    lim = ax.get_xlim(); ax.plot(lim, lim, "k--", lw=1, label="y=x"); ax.set_xlim(lim)
    ax.set_xlabel("Mean predicted exploration risk"); ax.set_ylabel("Mean realized exploration loss"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig_predicted_vs_realized_risk.png", dpi=220); fig.savefig(FIG / "fig_predicted_vs_realized_risk.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    q = reliability.groupby(["dataset", "method"], as_index=False).budget_violation_rate_realized.mean()
    for dataset, g in q.groupby("dataset"):
        ax.bar([f"{dataset}\n{m}" for m in g.method], g.budget_violation_rate_realized, label=dataset)
    ax.set_ylabel("Realized budget violation rate"); ax.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(FIG / "fig_budget_violation.png", dpi=220); fig.savefig(FIG / "fig_budget_violation.pdf"); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (dataset, g) in zip(axes.flat, feedback.groupby("dataset")):
        for method, h in g.groupby("method"):
            ax.scatter(h.feedback_recovery_fdratio, h.recall_recovery, s=20, alpha=.65, label=method)
        ax.set_title(dataset); ax.set_xlabel("Feedback recovery (FDRatio)"); ax.set_ylabel("Recall recovery")
    axes[0, 0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "fig_feedback_vs_recall.png", dpi=220); fig.savefig(FIG / "fig_feedback_vs_recall.pdf"); plt.close(fig)


def build_report(ablation: pd.DataFrame, budget: pd.DataFrame, reliability: pd.DataFrame,
                 stats: pd.DataFrame, paired: pd.DataFrame, frozen_phase4: pd.DataFrame) -> str:
    violation = float(reliability.budget_violation_rate_realized.mean()) if len(reliability) else np.nan
    within = stats.loc[stats.analysis_scope.eq("WITHIN_BUDGET")] if len(stats) else stats
    median_rho = float(within.spearman_rho.median()) if len(within) else np.nan
    k_cmp = paired.loc[paired.comparison.str.contains("LRBE", na=False) & paired.metric.eq("cumulative_feedback_count")] if len(paired) else paired
    k_ok = bool((k_cmp.mean_difference >= -1e-10).all()) if len(k_cmp) else False
    def table(frame: pd.DataFrame, cols: list[str], n=20) -> str:
        if not len(frame): return "（尚无可用行）"
        selected = frame.loc[:, [c for c in cols if c in frame]].head(n)
        header = "| " + " | ".join(selected.columns.astype(str)) + " |"
        divider = "| " + " | ".join(["---"] * len(selected.columns)) + " |"
        body = ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in selected.itertuples(index=False, name=None)]
        return "\n".join([header, divider] + body)
    conclusion = (
        "现有扩展证据支持把 LRBE 作为可检验的词典序设计，并支持将数量优先表述为受数据集、预算和风险代理限制的经验原则；"
        "但不能把预测 ECL 预算表述为对实际损失的确定性硬上界。"
        if np.isfinite(violation) and violation > 0 else
        "在本次样本中，预测预算的实际违反率没有显示为正值；这不构成对未来实际损失的确定性保证。"
    )
    return f"""# CCFB Extension 研究报告

## 1. 本次实验目的

本扩展验证 **Lexicographic Risk-Budget Exploration（LRBE，词典序风险预算探索）** 的 `K*` 分解、风险预算敏感性、**Expected Credit Loss（ECL，预期信用损失）** 预算可靠性，以及 **Feedback Recovery（反馈恢复）** 与 **Decision Recovery（决策恢复）** 的关系。它不替换冻结的 Phase 1--4，而是使用 fresh seeds `1001--1010` 形成独立证据。

## 2. 与原 Phase 1--4 的关系

固定的数据切分、模型、阈值、FAVE 定义、候选池和风险代理均沿用冻结框架。外部测试集从未参与预处理、拟合、lambda 选择或获取。Phase 4 的修正后预算内统计被按原文件引用，不重算、不覆盖。

## 3. 实验设置

- 数据集：UCI、HMEQ、PPDai（探索性外部证据）和 LendingClub（主要外部验证）。
- 风险：UCI 使用单位 exposure；HMEQ 使用 `PD_hat × LOAN`；外部数据使用 `PD_hat × exposure`；固定 `LGD=1`。因此无真实 EAD/LGD 的结果仅解释为模拟风险预算。
- fresh seeds：{CONFIG['fresh_seeds']}。
- 预算：基准 `rho={CONFIG['base_rho']}`，扫描倍数 {CONFIG['budget_multipliers']}。
- 联合目标：在 D0 内部的训练/伪获取/验证划分选择 lambda，完整敏感性写入 `lambda_sensitivity/`；外部测试未被使用。

## 4. K* 词典序分解消融结果

词典序方法先以成本排序求可行的最大数量 `K*`，再在 `Σz=K*` 下优化质量。联合目标和 Quality-first 不含该数量约束。

{table(paired, ['dataset','comparison','metric','n_paired_seeds','mean_difference','effect_size_cohen_dz','bootstrap_95ci_low','bootstrap_95ci_high','wilcoxon_pvalue'])}

对“LRBE 的数量是否不低于竞争结构”的可审计判断：`{k_ok}`。这只是结果描述，不能替代逐项置信区间。

## 5. 风险预算敏感性结果

预算扫描的最终轮均值保存在 `04_budget_sweep_summary.csv`，图 B1/B2 分别为反馈数量和 Recall Recovery。所有方法在每个 dataset × seed × round × budget 中共享同一候选池与预算。若低预算差异不更大或高预算不收敛，报告与图会如实保留该反证。

## 6. 风险预算可靠性结果

预测预算约束只保证 `R_pred ≤ B`，并不保证随机实现损失 `R_real ≤ B`。本次逐轮平均实际违反率为 `{violation:.4f}`。风险比例、Brier Score（布里尔分数）和校准曲线均在 `06_budget_reliability_summary.csv` 与图中报告；没有把测试集用于概率校准。

{table(reliability, ['experiment','dataset','method','rho','risk_ratio_mean','risk_ratio_bootstrap_95ci_low','risk_ratio_bootstrap_95ci_high','budget_violation_rate_realized','mean_brier','mean_ece'])}

## 7. 反馈—决策解耦分析

本扩展在相同 dataset × seed × budget 内计算 FDRatio 的反馈恢复与 Recall/AUC/PR-AUC 恢复，并以 seed 聚类 bootstrap 报告不确定性。预算内 Spearman rho 的中位数为 `{median_rho:.4f}`。

{table(stats, ['dataset','rho','metric_pair','analysis_scope','n','pearson_r','spearman_rho','bootstrap_95ci_low','bootstrap_95ci_high','estimability'])}

Phase 4 的冻结重用统计行数：`{len(frozen_phase4)}`；它们保留在 `08_feedback_decision_statistics.csv` 的 `evidence_source=FROZEN_PHASE4_REUSED` 行中。

## 8. 统计显著性与置信区间

主比较使用相同 fresh seed 的配对差、Cohen's dz、95% bootstrap CI、配对 t 检验和 Wilcoxon 检验。p-value 不作为唯一判断依据。相关分析采用 seed-cluster bootstrap。

## 9. 支持原假设的证据

- 如果 LRBE 在配对比较中保持相同/更高反馈数量，则它直接支持 `K*` 的设计动机；否则该假设被削弱。
- 预算扫描直接检验 Quantity-first 是否只出现在单个预算点。

## 10. 不支持原假设的证据

- 任何 LRBE 相对 Joint/Quality-first 的置信区间跨过零、预算曲线反转，或复杂策略带来稳定下游优势，均作为反证保留在原始 CSV 与图中。
- 实际预算违反率为正时，不能把“预测 hard budget”表述为实现损失的硬上界。

## 11. Counterevidence（反证）

本报告不删除负结果：PPDai 是探索性证据；EAD/LGD 缺失的数据集使用 proxy；有限的 10 fresh seeds 只能提供中等精度；选择性标签模拟不能自动证明线上因果效果。

## 12. 对论文 Contributions（贡献点）的影响

可保留的贡献是：在内生风险资源约束下，将最大安全反馈数量与反馈质量明确分离，并量化反馈代表性与学习价值可不一致。不能宣称 FAVE 是稳定 SOTA，也不能把 Quantity-first 扩张为无条件普遍定律。

## 13. 哪些 claim 可以保留 / 必须削弱

可保留：ECL 风险预算是透明的规划量；`K*` 提供一个公平比较不同质量函数的共同数量层；反馈恢复和决策恢复必须分别检验。

必须削弱：预测 ECL 不是实际损失的确定上界；外部证据有 PPDai 的较低置信等级；FAVE 的价值是机制探针而非稳定性能优势。

## 14. Reviewer Defense（审稿人防御）

**Q1：这不就是风险和信息价值加权吗？** 不是。`01/02` 直接比较了无固定数量的标准化 Joint Objective 与先求 `K*` 的 LRBE；差异在反馈数量、风险利用和下游指标上被逐项记录。

**Q2：Quantity-first 只在某一个预算有效吗？** `03/04` 在 0.25× 到 4× 基准预算上提供同一 split、同一 seed 的完整曲线，而非单一预算点叙事。

**Q3：预测 PD 下 hard budget 真的安全吗？** 只能说其在预测空间受约束。`05/06` 同时报告 `R_pred`、`R_real`、RiskRatio 和实际违反率；报告不把二者混同。

**Q4：反馈恢复为何不必带来更好性能？** `07/08` 给出预算内相关及策略排序一致率；反馈代表性价值与决策学习价值是独立的经验问题。

**Q5：FAVE 没有赢，论文还有什么贡献？** FAVE 是冻结的复杂信息价值基线与机制探针。论文主张是风险预算下的数量/质量分解及其边界，而不是 FAVE 的 SOTA 声明。

## 15. 最终 CCF B 投稿实验结论

{conclusion}
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ab_raw, b_raw, refs = load_raw("kstar_ablation"), load_raw("budget_sweep"), load_raw("reference")
    if not len(ab_raw) or not len(b_raw) or not len(refs):
        raise SystemExit("Formal runs are incomplete: require kstar_ablation, budget_sweep, and reference raw outputs.")
    ab = reference_recovery(terminal(ab_raw), refs)
    budget = reference_recovery(terminal(b_raw), refs)
    # Requested 01/03 are raw per-round records; 02/04 are terminal summaries.
    ab_raw.to_csv(OUT / "01_kstar_ablation_raw.csv", index=False)
    ab_summary = ab.groupby(["dataset", "method", "rho"], as_index=False).agg(
        seeds=("seed", "nunique"), feedback_count_mean=("cumulative_feedback_count", "mean"),
        feedback_count_std=("cumulative_feedback_count", "std"), predicted_risk_mean=("cumulative_predicted_risk", "mean"),
        realized_loss_mean=("cumulative_realized_loss", "mean"), recall_mean=("recall", "mean"),
        roc_auc_mean=("roc_auc", "mean"), pr_auc_mean=("pr_auc", "mean"))
    ab_summary.to_csv(OUT / "02_kstar_ablation_summary.csv", index=False)
    b_raw.to_csv(OUT / "03_budget_sweep_raw.csv", index=False)
    b_summary = budget.groupby(["dataset", "method", "rho"], as_index=False).agg(
        seeds=("seed", "nunique"), feedback_count_mean=("cumulative_feedback_count", "mean"),
        feedback_count_std=("cumulative_feedback_count", "std"), budget_utilization_mean=("budget_utilization", "mean"),
        visible_default_rate_mean=("visible_default_rate", "mean"), feedback_recovery_mean=("feedback_recovery_fdratio", "mean"),
        recall_mean=("recall", "mean"), recall_recovery_mean=("recall_recovery", "mean"),
        roc_auc_mean=("roc_auc", "mean"), pr_auc_mean=("pr_auc", "mean"),
        predicted_risk_mean=("cumulative_predicted_risk", "mean"), realized_loss_mean=("cumulative_realized_loss", "mean"))
    b_summary.to_csv(OUT / "04_budget_sweep_summary.csv", index=False)
    rel_source = pd.concat([ab_raw, b_raw], ignore_index=True)
    rel_source.to_csv(OUT / "05_budget_reliability_raw.csv", index=False)
    rel, calibration_bins = reliability_summary(rel_source)
    rel.to_csv(OUT / "06_budget_reliability_summary.csv", index=False)
    calibration_bins.to_csv(OUT / "06b_calibration_curve_bins.csv", index=False)
    feedback = budget.copy()
    feedback.to_csv(OUT / "07_feedback_decision_raw.csv", index=False)
    stats, con = decoupling_statistics(feedback)
    frozen_path = ROOT / "results" / "phase4" / "statistics" / "FEEDBACK_PERFORMANCE_SENSITIVITY_FINAL.csv"
    frozen = pd.read_csv(frozen_path) if frozen_path.exists() else pd.DataFrame()
    if len(frozen):
        frozen = frozen.assign(evidence_source="FROZEN_PHASE4_REUSED", extension_analysis="historical_within_budget_reference")
    stats = stats.assign(evidence_source="FRESH_EXTENSION", extension_analysis="fresh_seed_budget_sweep")
    combined_stats = pd.concat([stats, frozen], ignore_index=True, sort=False)
    combined_stats.to_csv(OUT / "08_feedback_decision_statistics.csv", index=False)
    con.to_csv(OUT / "08b_within_budget_concordance_pairs.csv", index=False)
    comparisons = [("LRBE-Uncertainty", "QualityFirst-Uncertainty"), ("LRBE-FAVE", "QualityFirst-FAVE"),
                   ("LRBE-FAVE", "Joint-Best-Val-Uncertainty"), ("LRBE-FAVE", "CountOnly-MinRisk")]
    paired = paired_summary(ab, comparisons, ["cumulative_feedback_count", "cumulative_predicted_risk", "cumulative_realized_loss", "recall", "roc_auc", "pr_auc"])
    paired.to_csv(OUT / "02b_kstar_paired_comparisons.csv", index=False)
    save_figures(ab, budget, rel, feedback)
    (OUT / "FIGURE_NOTES_CN.md").write_text(
        "# 图中文说明\n\n所有图的坐标轴和图例使用英文以便论文排版；图 B1 展示预算—反馈数量，图 B2 展示预算—Recall Recovery，其他图分别展示 K* 消融、预测/实现风险、违反率和反馈—Recall 关系。阴影（如有）为 seed 间标准差，非置信区间。\n",
        encoding="utf-8")
    report = build_report(ab, budget, rel, stats, paired, frozen)
    (OUT / "CCFB_extension_report.md").write_text(report, encoding="utf-8")
    manifest_rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "CCFB_EXTENSION_MANIFEST.csv":
            manifest_rows.append({"path": str(path.relative_to(OUT)).replace("\\", "/"), "sha256": sha(path), "bytes": path.stat().st_size})
    pd.DataFrame(manifest_rows).to_csv(OUT / "CCFB_EXTENSION_MANIFEST.csv", index=False)


if __name__ == "__main__":
    main()
