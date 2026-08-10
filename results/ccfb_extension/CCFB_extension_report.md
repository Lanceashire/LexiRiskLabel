# CCFB Extension 研究报告

## 1. 本次实验目的

本扩展验证 **Lexicographic Risk-Budget Exploration（LRBE，词典序风险预算探索）** 的 `K*` 分解、风险预算敏感性、**Expected Credit Loss（ECL，预期信用损失）** 预算可靠性，以及 **Feedback Recovery（反馈恢复）** 与 **Decision Recovery（决策恢复）** 的关系。它不替换冻结的 Phase 1--4，而是使用 fresh seeds `1001--1010` 形成独立证据。

## 2. 与原 Phase 1--4 的关系

固定的数据切分、模型、阈值、FAVE 定义、候选池和风险代理均沿用冻结框架。外部测试集从未参与预处理、拟合、lambda 选择或获取。Phase 4 的修正后预算内统计被按原文件引用，不重算、不覆盖。

## 3. 实验设置

- 数据集：UCI、HMEQ、PPDai（探索性外部证据）和 LendingClub（主要外部验证）。
- 风险：UCI 使用单位 exposure；HMEQ 使用 `PD_hat × LOAN`；外部数据使用 `PD_hat × exposure`；固定 `LGD=1`。因此无真实 EAD/LGD 的结果仅解释为模拟风险预算。
- fresh seeds：[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]。
- 预算：基准 `rho=0.1`，扫描倍数 [0.25, 0.5, 1.0, 2.0, 4.0]。
- 联合目标：在 D0 内部的训练/伪获取/验证划分选择 lambda，完整敏感性写入 `lambda_sensitivity/`；外部测试未被使用。

## 4. K* 词典序分解消融结果

词典序方法先以成本排序求可行的最大数量 `K*`，再在 `Σz=K*` 下优化质量。联合目标和 Quality-first 不含该数量约束。

| dataset | comparison | metric | n_paired_seeds | mean_difference | effect_size_cohen_dz | bootstrap_95ci_low | bootstrap_95ci_high | wilcoxon_pvalue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hmeq | LRBE-Uncertainty minus QualityFirst-Uncertainty | cumulative_feedback_count | 10 | 7.0 | 1.4702940882294102 | 4.6 | 10.1 | 0.001953125 |
| hmeq | LRBE-Uncertainty minus QualityFirst-Uncertainty | cumulative_predicted_risk | 10 | 18.565433621406555 | 0.0010232930958926462 | -10204.32358649373 | 11079.880140632391 | 0.921875 |
| hmeq | LRBE-Uncertainty minus QualityFirst-Uncertainty | cumulative_realized_loss | 10 | -2340.0 | -0.04585904243506823 | -31870.0 | 27920.0 | 0.845703125 |
| hmeq | LRBE-Uncertainty minus QualityFirst-Uncertainty | recall | 10 | -0.002992784540562632 | -0.18450262048653715 | -0.011929139455822907 | 0.007163925518919916 | 0.42578125 |
| hmeq | LRBE-Uncertainty minus QualityFirst-Uncertainty | roc_auc | 10 | 0.0007511454910415716 | 0.4179116449363229 | -0.0003847246703399712 | 0.0017238383836849444 | 0.232421875 |
| hmeq | LRBE-Uncertainty minus QualityFirst-Uncertainty | pr_auc | 10 | 0.0010248727238768463 | 0.5030379722712651 | -0.00017192947538758357 | 0.002223788296230737 | 0.193359375 |
| hmeq | LRBE-FAVE minus QualityFirst-FAVE | cumulative_feedback_count | 10 | 67.1 | 8.871199152470687 | 62.6 | 71.4 | 0.001953125 |
| hmeq | LRBE-FAVE minus QualityFirst-FAVE | cumulative_predicted_risk | 10 | 13265.222989916801 | 0.9310624498616993 | 5406.643859371543 | 22174.131932586424 | 0.013671875 |
| hmeq | LRBE-FAVE minus QualityFirst-FAVE | cumulative_realized_loss | 10 | -39120.0 | -0.5640359247183272 | -78180.25 | 3630.2499999999964 | 0.10546875 |
| hmeq | LRBE-FAVE minus QualityFirst-FAVE | recall | 10 | 0.017455509145381888 | 0.7978077966319409 | 0.003660476761208695 | 0.02947095386179518 | 0.037109375 |
| hmeq | LRBE-FAVE minus QualityFirst-FAVE | roc_auc | 10 | -0.0006453825487568787 | -0.29657158572126613 | -0.0018938518268951212 | 0.0006532667073947011 | 0.375 |
| hmeq | LRBE-FAVE minus QualityFirst-FAVE | pr_auc | 10 | -0.001627205511955887 | -0.32354303456640715 | -0.004515290480490729 | 0.0013910250318644446 | 0.322265625 |
| hmeq | LRBE-FAVE minus Joint-Best-Val-Uncertainty | cumulative_feedback_count | 10 | 25.3 | 0.9978121968972269 | 11.7 | 41.1 | 0.001953125 |
| hmeq | LRBE-FAVE minus Joint-Best-Val-Uncertainty | cumulative_predicted_risk | 10 | 115213.16831946373 | 0.7055318080037026 | 28703.09532785416 | 216323.006216675 | 0.01953125 |
| hmeq | LRBE-FAVE minus Joint-Best-Val-Uncertainty | cumulative_realized_loss | 10 | 125630.0 | 0.8040858355561785 | 41680.0 | 222260.0 | 0.013671875 |
| hmeq | LRBE-FAVE minus Joint-Best-Val-Uncertainty | recall | 10 | -0.0025230943655338844 | -0.1569823694753813 | -0.011795094184788612 | 0.0071412906617159405 | 0.6875 |
| hmeq | LRBE-FAVE minus Joint-Best-Val-Uncertainty | roc_auc | 10 | -0.0009686846032160856 | -0.5329402177437909 | -0.002071008937905328 | 4.6897317495487625e-05 | 0.130859375 |
| hmeq | LRBE-FAVE minus Joint-Best-Val-Uncertainty | pr_auc | 10 | -0.0009077179446998929 | -0.33728219334770587 | -0.002512342594123318 | 0.0006158064487620957 | 0.556640625 |
| hmeq | LRBE-FAVE minus CountOnly-MinRisk | cumulative_feedback_count | 10 | 1.1 | 0.33183011234370885 | -0.9 | 3.0 | 0.3125 |
| hmeq | LRBE-FAVE minus CountOnly-MinRisk | cumulative_predicted_risk | 10 | 32605.03533065319 | 1.8965921852888419 | 23059.75448642671 | 43126.58394683897 | 0.001953125 |

对“LRBE 的数量是否不低于竞争结构”的可审计判断：`False`。这只是结果描述，不能替代逐项置信区间。

## 5. 风险预算敏感性结果

预算扫描的最终轮均值保存在 `04_budget_sweep_summary.csv`，图 B1/B2 分别为反馈数量和 Recall Recovery。所有方法在每个 dataset × seed × round × budget 中共享同一候选池与预算。若低预算差异不更大或高预算不收敛，报告与图会如实保留该反证。

## 6. 风险预算可靠性结果

预测预算约束只保证 `R_pred ≤ B`，并不保证随机实现损失 `R_real ≤ B`。本次逐轮平均实际违反率为 `0.4798`。风险比例、Brier Score（布里尔分数）和校准曲线均在 `06_budget_reliability_summary.csv` 与图中报告；没有把测试集用于概率校准。

| experiment | dataset | method | rho | risk_ratio_mean | risk_ratio_bootstrap_95ci_low | risk_ratio_bootstrap_95ci_high | budget_violation_rate_realized | mean_brier | mean_ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_sweep | hmeq | CountOnly-MinRisk | 0.025 | 1.1586319329798775 | 1.1024031452722842 | 1.213710234092168 | 0.56 | 0.08034618072211737 | 0.025260458047507938 |
| budget_sweep | hmeq | CountOnly-MinRisk | 0.05 | 1.0864044670056747 | 1.0447328853573472 | 1.128304001911277 | 0.48 | 0.08017451032996174 | 0.02440387355891245 |
| budget_sweep | hmeq | CountOnly-MinRisk | 0.1 | 1.086192400150332 | 1.0558613382580162 | 1.1162169132149322 | 0.55 | 0.07996966153383252 | 0.024191703476486734 |
| budget_sweep | hmeq | CountOnly-MinRisk | 0.2 | 1.0504601812798497 | 1.022823842501399 | 1.0783777205353795 | 0.5 | 0.0797068370878696 | 0.02434123929752495 |
| budget_sweep | hmeq | CountOnly-MinRisk | 0.4 | 1.046133680079001 | 1.0227414921476736 | 1.0691392225710283 | 0.56 | 0.07924207501113412 | 0.024140840503118332 |
| budget_sweep | hmeq | LRBE-FAVE | 0.025 | 1.1401019653996554 | 1.079634431187473 | 1.196581558660777 | 0.72 | 0.08040950909256932 | 0.02493425068124224 |
| budget_sweep | hmeq | LRBE-FAVE | 0.05 | 1.1301343128609946 | 1.0887816749030705 | 1.1729970743930094 | 0.72 | 0.08025869734585282 | 0.024072499922682917 |
| budget_sweep | hmeq | LRBE-FAVE | 0.1 | 1.0926964090901086 | 1.0613821774678416 | 1.1242498060840964 | 0.75 | 0.08000592626631255 | 0.024828981020431545 |
| budget_sweep | hmeq | LRBE-FAVE | 0.2 | 1.0756215644321272 | 1.0501804914283905 | 1.100954894123853 | 0.72 | 0.07962864205241199 | 0.024461692600917254 |
| budget_sweep | hmeq | LRBE-FAVE | 0.4 | 1.0766507866871016 | 1.054979489408433 | 1.0977701039331198 | 0.79 | 0.07921385720372195 | 0.023882229030332262 |
| budget_sweep | hmeq | QualityFirst-FAVE | 0.025 | 1.2197172097353015 | 1.161089341337009 | 1.276564507699621 | 0.81 | 0.08053299911320204 | 0.026078726017135057 |
| budget_sweep | hmeq | QualityFirst-FAVE | 0.05 | 1.2024207495951498 | 1.154070152964181 | 1.248323130242599 | 0.81 | 0.08034875869750974 | 0.025608896007823817 |
| budget_sweep | hmeq | QualityFirst-FAVE | 0.1 | 1.1756391827254349 | 1.132928230838884 | 1.2175503149104225 | 0.81 | 0.08032553046941752 | 0.025488533582038276 |
| budget_sweep | hmeq | QualityFirst-FAVE | 0.2 | 1.139322405061975 | 1.11110766432122 | 1.167045349617101 | 0.84 | 0.08005161367356774 | 0.024657769214661878 |
| budget_sweep | hmeq | QualityFirst-FAVE | 0.4 | 1.0939846498581325 | 1.072475894325964 | 1.1149618850426237 | 0.79 | 0.07944940187036988 | 0.024890793869359332 |
| budget_sweep | hmeq | QualityFirst-Uncertainty | 0.025 | 1.1922639228723348 | 1.1278592881084102 | 1.2565487498542214 | 0.69 | 0.08055501759052273 | 0.025382766283848487 |
| budget_sweep | hmeq | QualityFirst-Uncertainty | 0.05 | 1.1220038132078691 | 1.07173417251791 | 1.1709878181280209 | 0.69 | 0.0803311629593372 | 0.02528673633743047 |
| budget_sweep | hmeq | QualityFirst-Uncertainty | 0.1 | 1.0803571941117547 | 1.0339283927105445 | 1.1271174489110682 | 0.61 | 0.08005390442907806 | 0.024893572994976583 |
| budget_sweep | hmeq | QualityFirst-Uncertainty | 0.2 | 1.080071266389317 | 1.0449072948240787 | 1.114706329809326 | 0.69 | 0.0794697099179029 | 0.02491706240704691 |
| budget_sweep | hmeq | QualityFirst-Uncertainty | 0.4 | 1.0618586170126385 | 1.0334988321808443 | 1.089455715825082 | 0.7 | 0.07922823883593078 | 0.02560079137118366 |

## 7. 反馈—决策解耦分析

本扩展在相同 dataset × seed × budget 内计算 FDRatio 的反馈恢复与 Recall/AUC/PR-AUC 恢复，并以 seed 聚类 bootstrap 报告不确定性。预算内 Spearman rho 的中位数为 `0.1239`。

| dataset | rho | metric_pair | analysis_scope | n | pearson_r | spearman_rho | bootstrap_95ci_low | bootstrap_95ci_high | estimability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hmeq | ALL | FeedbackRecovery_vs_RecallRecovery | WITHIN_DATASET | 200 | 0.5534152985568352 | 0.5863135005462851 | 0.5121150886125874 | 0.6640433946134617 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | ALL | FeedbackRecovery_vs_AUCRecovery | WITHIN_DATASET | 200 | -0.12316607813865041 | 0.031236054328398957 | -0.2194626810843523 | 0.28520028417882903 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | ALL | FeedbackRecovery_vs_PRAUCRecovery | WITHIN_DATASET | 200 | -0.0740143870257683 | 0.09997378683484796 | -0.10900570829825895 | 0.3305209090239231 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.025 | FeedbackRecovery_vs_RecallRecovery | WITHIN_BUDGET | 40 | 0.34320443141153106 | 0.46610686768282367 | 0.1697491834797714 | 0.6797031401639737 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.025 | FeedbackRecovery_vs_AUCRecovery | WITHIN_BUDGET | 40 | -0.050708426183160876 | -0.16848030018761728 | -0.40949548497281585 | 0.05439187855547413 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.025 | FeedbackRecovery_vs_PRAUCRecovery | WITHIN_BUDGET | 40 | 0.16556446345366613 | -0.010318949343339589 | -0.41190863185886506 | 0.3264908240864986 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.05 | FeedbackRecovery_vs_RecallRecovery | WITHIN_BUDGET | 40 | 0.5157355297908666 | 0.4260449448767517 | 0.0633707752904733 | 0.7013729092801895 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.05 | FeedbackRecovery_vs_AUCRecovery | WITHIN_BUDGET | 40 | -0.18602412884269695 | -0.05103189493433397 | -0.3571710943803284 | 0.27455387593678815 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.05 | FeedbackRecovery_vs_PRAUCRecovery | WITHIN_BUDGET | 40 | -0.006465043557254994 | 0.06885553470919327 | -0.216734022556391 | 0.3760046233613306 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.1 | FeedbackRecovery_vs_RecallRecovery | WITHIN_BUDGET | 40 | 0.4373233475249882 | 0.32527805443983054 | -0.06330358978811895 | 0.5866978904568354 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.1 | FeedbackRecovery_vs_AUCRecovery | WITHIN_BUDGET | 40 | 0.07506207702677455 | -0.13189493433395877 | -0.4660939497925882 | 0.21602178812922587 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.1 | FeedbackRecovery_vs_PRAUCRecovery | WITHIN_BUDGET | 40 | 0.3222895213022792 | 0.2879924953095685 | -0.0026705683484363214 | 0.5648758446960086 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.2 | FeedbackRecovery_vs_RecallRecovery | WITHIN_BUDGET | 40 | 0.38526242159830165 | 0.34338790976633143 | 0.10020443334016003 | 0.56011113206168 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.2 | FeedbackRecovery_vs_AUCRecovery | WITHIN_BUDGET | 40 | 0.003045768279757345 | 0.19268292682926833 | -0.006593945356899291 | 0.41181060783687706 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.2 | FeedbackRecovery_vs_PRAUCRecovery | WITHIN_BUDGET | 40 | -0.11440274755376328 | 0.13714821763602253 | -0.1719309277424896 | 0.4937970507582372 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.4 | FeedbackRecovery_vs_RecallRecovery | WITHIN_BUDGET | 40 | 0.5975841174730516 | 0.5569371106938342 | 0.36195353422387844 | 0.7087743402521343 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.4 | FeedbackRecovery_vs_AUCRecovery | WITHIN_BUDGET | 40 | 0.002429031057509393 | 0.14352720450281425 | -0.12353828743870238 | 0.36329575643226153 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| hmeq | 0.4 | FeedbackRecovery_vs_PRAUCRecovery | WITHIN_BUDGET | 40 | 0.22202245463077483 | 0.17954971857410884 | -0.14913640183881113 | 0.4544122957295924 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| lendingclub | ALL | FeedbackRecovery_vs_RecallRecovery | WITHIN_DATASET | 200 | 0.07458651890052824 | 0.1567485229854144 | -0.0240971311566092 | 0.41766402496557203 | ESTIMABLE_CLUSTER_BOOTSTRAP |
| lendingclub | ALL | FeedbackRecovery_vs_AUCRecovery | WITHIN_DATASET | 200 | 0.23112284079532341 | 0.2403068477862795 | 0.09545645997538971 | 0.44232077696813255 | ESTIMABLE_CLUSTER_BOOTSTRAP |

Phase 4 的冻结重用统计行数：`48`；它们保留在 `08_feedback_decision_statistics.csv` 的 `evidence_source=FROZEN_PHASE4_REUSED` 行中。

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

现有扩展证据支持把 LRBE 作为可检验的词典序设计，并支持将数量优先表述为受数据集、预算和风险代理限制的经验原则；但不能把预测 ECL 预算表述为对实际损失的确定性硬上界。
