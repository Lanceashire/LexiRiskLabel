# LexiRiskLabel

**Risk-Budgeted Feedback Acquisition under Selective Labels**

LexiRiskLabel is a reproducible research package for dynamic credit approval
when outcomes are observed only for approved applicants.  The project asks how
to acquire additional ground-truth feedback while respecting a per-round
credit-risk budget.

Current mechanism research examines when additional risk-budgeted exploration adds value beyond routine feedback. Maximum-count acquisition is treated as a baseline and an ablation, with conditional value and model-specific risk boundaries examined in the [M1–M3 evidence package](research/mechanisms_20261101/README.md).

The historical **Lexicographic Risk-Budget Exploration (LRBE)** baseline
first computes the maximum feasible feedback count under its predicted-risk budget

\[
K_t^*=\max_z\sum_i z_i
\quad\text{s.t.}\quad
\sum_i z_i\widehat{PD}_i EAD_i LGD_i\le B_t,
\]

then fixes `K*` and optimizes feedback quality.  This separates the questions
"how many labels fit the predicted-risk budget?" and "whose labels should be
acquired?" instead of collapsing them into one risk--utility weight.

## Status

### Current experiment consolidation and writing verification — 2026-11-02

See [the current mechanism status](research/CURRENT_MECHANISM_RESEARCH_STATUS.md) and [the E0–E3 review package](research/mechanisms_followup_20261102_v3/README.md). The explicit prospective v2 experiment scope, fixed-action explanation and engineering schema checks are complete. Seven LR primary endpoints yield three practical positives, two within-band results and two unresolved results; not all original precision targets were met. Preparation currently means verifying evidence-based writing, with no full manuscript or submission PDF written. Earlier stage statuses below are historical snapshots.

### M5 registered retrospective study completed 2026-11-02

The [historical M1–M5 completion status](research/M4_M5_CURRENT_STATUS_20261102.md) links the completed [M4 evidence](research/mechanisms_m4_20261101/README.md) and [M5 full results and reproduction archive](research/mechanisms_m5_20261102/README.md). M5 completed all 40 replay tasks, 400 ten-round trajectories and 3,600 common-state branches. All 7,600 prediction branches were reconstructed exactly, and all 7,600 label-release commitments were checked. Twenty primary endpoint estimates, intervals and multiplicity adjustments were independently recomputed.

Completion refers to the registered study and audits. No practical positive advantage of Pending-aware over the two comparators was confirmed: all eight ten-round gain estimates were negative, seven excluded the registered positive threshold, and one remained unresolved. The frozen M4 diagnostic did not establish cross-task prediction; both XGBoost diagnostics were worse than the frozen constant baseline outside the M4 feature range. These are conditional retrospective findings, not real-reject counterfactuals, strict as-of online effects, or training-population coverage.

The [earlier 8/40-task checkpoint](research/mechanisms_m5_checkpoint_20261102/README.md) remains a historical snapshot. Raw credit data, applicant-level logs and predictions remain local. A half-synthetic bridge is a proposed follow-up and has not run.

### M1–M3 mechanism evidence added 2026-11-01

The [integrated report and archives](research/mechanisms_20261101/README.md) consolidate the exact M1 study, bounded M2 reversal investigations, and a newly registered M3 cost-pairing confirmation with 2,560 independent worlds and four label replicates per world. All 52 M3 contrasts, positive/negative/unresolved outcomes, frozen source, and audit records are retained. This release predates the M4 and M5 completion results linked above; its archived stage markers retain that historical meaning. These M-stage labels are separate from the historical Phase 0–4 labels below.

### Small development experiments added 2026-10-26

The independent [research v2 implementation](experiments/research_v2/README.md)
evaluates RB-BRIDGE, RB-CORR, and a small closed-loop RB-GATE prototype.
The [experiment report](results/research_v2/analysis/实验汇总.md) contains all
positive and negative findings, with [paired descriptive contrasts](results/research_v2/analysis/paired_descriptive_contrasts.csv)
and [local artifact verification](results/research_v2/verification.json).
The [publication scope](results/research_v2/PUBLICATION.md) explains which
aggregate artifacts are included; per-applicant logs and prediction arrays
remain local.
The real-data development block comprises 84 ten-round episodes across HMEQ,
UCI, and LendingClub, two new seeds per task, and one budget (`rho=0.1`).
Six small mechanism environments use three development seeds each.

These are development results, not confirmatory or state-of-the-art claims.
RB-BRIDGE did not outperform MinRisk in the two-seed mean on any of the three
real tasks; the tested RB-CORR correction worsened Brier; the small RB-GATE
study did not establish an advantage over myopic acquisition. Newly reconstructed
inputs have distinct byte hashes, and the new protocol/schema is kept separate
from the frozen study. Third-party data and private planning conversations are
not redistributed. Historical evidence below is unchanged.

### Frozen study status

- Frozen Phase 0--4 evidence: complete.
- Fresh-seed CCFB extension: complete.
- Independent information-gain primary block: complete (200 dynamic episodes
  and 40 common-state audits).
- Controlled mechanism-conditions block and seed-controlled confirmation:
  complete (20 seeds, 160 GPU acquisition/retraining branches, 320 endpoint
  rows, label-perturbation audits, and 10,000-draw paired analysis).
- Fresh seeds: `1001`--`1010`, disjoint from frozen seeds `0`--`19`.
- Datasets: UCI Default Credit Card, HMEQ, PPDai, and LendingClub.
- Formal extension outputs: 1,120 run CSVs and 12,320 round-level records.
- Current project and submission checks: 76 passed in the validated environment.

This repository supports paper drafting and private collaboration.  It does
not claim that FAVE is a universally superior policy or that a predicted ECL
budget is a deterministic upper bound on realized loss.

## Main empirical findings

1. **Quantity-first is a bounded empirical principle.** Simple maximum-count,
   low-risk acquisition is frequently risk-efficient, but is not claimed to
   be universally optimal across every dataset and budget.
2. **Complex feedback value is not stable downstream superiority.** FAVE can
   change feedback composition, yet does not deliver stable Recall, ROC-AUC,
   or PR-AUC gains over strong simple baselines.
3. **Feedback recovery and decision recovery are not reliably synchronized.**
   Their relationship varies by dataset and risk budget.
4. **Predicted-budget feasibility is not realized-loss feasibility.** Across
   method--dataset--budget groups, the mean round-level realized budget
   violation rate is approximately `0.4798` (range `0.01`--`0.84`).
5. **Acquisition-score gains need not become prediction gains.** In the
   controlled common-state experiment, LRBE-IG increases its IG objective over
   both MinRisk and a feasible random control, while the primary Brier
   contrasts remain inconclusive. QualityFirst-IG sacrifices feasible labels
   under both exposure conditions.

The full evidence-aligned interpretation is in
[`results/ccfb_extension/CCFB_extension_report.md`](results/ccfb_extension/CCFB_extension_report.md).

## Repository map

```text
configs/                    Frozen configurations and protocols
data/metadata/              Dataset provenance, hashes, target/leakage audits
experiments/                Phase descriptions and the CCFB extension runner
figures/                    Cross-phase paper assets
paper/paper_orchestra/      Canonical anonymous LaTeX manuscript and PDF
plans/                      Preregistration and experimental plans
reports/                    Canonical report pointers
results_phase0/             Frozen compatibility results
results_phase1/             Frozen mechanism results
results_phase2/             Frozen policy-comparison results
results_phase3/             Frozen external-validation results
results_phase4/             Frozen mechanism-analysis results
results/ccfb_extension/     Fresh-seed raw results, statistics, figures, report
results/information_gain_primary_20260907/  Independent IG trajectories
results/mechanism_conditions_primary_20260908/  Controlled mechanism evidence
results/mechanism_conditions_seed_controlled_confirmation_20260908/  Confirmed mechanism evidence
scripts/analysis/           Analysis entry points
scripts/reproduce/          Deliberate reproduction entry points
src/                        Frozen Phase 0--4 implementation
tests/                      Frozen integrity and leakage tests
```

The legacy `results_phase*` names are retained intentionally because the
frozen compatibility tests and historical provenance use them.  New CCFB
extension artifacts live under `results/ccfb_extension/`.

## Quick verification without downloading third-party data

Create the environment, then run:

```bash
python -m pytest -q -p no:cacheprovider tests
python experiments/ccfb_extension/analyze_ccfb_extension.py
```

The first command checks frozen Phase 0--4 invariants.  The second reconstructs
all CCFB extension summaries, paired tests, 10,000-draw seed-cluster bootstrap
statistics, PNG/PDF figures, and the Chinese report from committed formal raw
results.  Neither command retrains a model.

## Full experimental reproduction

Third-party raw data are not committed.  Obtain them under their original
terms, verify their hashes, and materialize the documented runtime layout:

```bash
python scripts/data_preparation/materialize_data_layout.py \
  --uci /path/to/uci_default_credit_card.csv \
  --hmeq /path/to/hmeq.csv \
  --ppdai /path/to/ppdai.csv \
  --lendingclub /path/to/lendingclub.csv
```

Then run the fresh-seed extension:

```bash
python experiments/ccfb_extension/run_ccfb_extension.py --mode smoke
python experiments/ccfb_extension/run_ccfb_extension.py --mode full
python experiments/ccfb_extension/analyze_ccfb_extension.py
```

`--mode full` skips already present run files unless `--force` is passed.  Do
not use `--force` merely to obtain more favorable outcomes.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact environment, GPU/CPU
behavior, data-leakage controls, expected runtimes, and result verification.

## Data availability

The repository includes dataset metadata, provenance, row counts, SHA-256
digests, target mappings, and leakage audits, but not third-party data files.
See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Claims and limitations

- LRBE guarantees feasibility under the **predicted applicant-level ECL
  budget**, not under ex-post realized loss.
- PPDai is lower-confidence exploratory external evidence; LendingClub is the
  primary external validation.
- HMEQ and external exposures are proxies where observed EAD/LGD are absent.
- The experiments are offline simulations, not an online causal policy trial.
- FAVE is a frozen complex information-value baseline and mechanism probe,
  not a claimed state-of-the-art winner.

## License and citation

Project-authored code is released under the repository's [MIT License](LICENSE).
Third-party datasets and source materials are excluded and remain governed by
their own terms.  See [LICENSES_AND_DATA_TERMS.md](LICENSES_AND_DATA_TERMS.md)
before redistributing any data-derived material.

Citation metadata are provided in [CITATION.cff](CITATION.cff).  Update the
authors, paper title, venue, DOI, and release version before a public archival
release.

## 中文说明

本仓库是选择性标签条件下风险预算反馈获取研究的可复现工程包。当前机制研究聚焦历史条件、资源控制与迁移边界，最新实验收束和写法核验见上述当前状态。既有算法及历史阶段保留作为参照；新结果须保留未定项、事前范围修订和有限外推范围。第三方真实原始行、个体标签、个体预测及私有日志不随本次公开材料分发。

