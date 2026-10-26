# LexiRiskLabel

**Risk-Budgeted Feedback Acquisition under Selective Labels**

LexiRiskLabel is a reproducible research package for dynamic credit approval
when outcomes are observed only for approved applicants.  The project asks how
to acquire additional ground-truth feedback while respecting a per-round
credit-risk budget.

The central method is **Lexicographic Risk-Budget Exploration (LRBE)**.  It
first computes the maximum safely acquirable feedback count

\[
K_t^*=\max_z\sum_i z_i
\quad\text{s.t.}\quad
\sum_i z_i\widehat{PD}_i EAD_i LGD_i\le B_t,
\]

then fixes `K*` and optimizes feedback quality.  This separates the questions
"how many labels can be acquired safely?" and "whose labels should be
acquired?" instead of collapsing them into one risk--utility weight.

## Status

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

本仓库是选择性标签条件下风险预算反馈获取研究的可复现工程包。代码、配置、冻结结果、fresh-seed 原始结果、统计与论文图均已纳入；第三方原始数据不随仓库重新分发。论文主张应聚焦于 LRBE 的数量—质量词典序分解、Quantity-first 的有限适用范围、预测风险与实际损失的区别，以及反馈恢复与决策恢复的不稳定对应关系。
