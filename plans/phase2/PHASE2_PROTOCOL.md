你现在作为严格的机器学习科研实验工程师，执行：

============================================================
PHASE 2 — RISK-BUDGETED POLICY COMPARISON
风险预算约束下的反馈获取策略比较
============================================================

这是正式实验阶段。

Phase 0：
已经完成 protocol repair。

Phase 1：
已经完成 selective-label mechanism validation，
并正式冻结。

Phase 2 的唯一核心问题：

============================================================
在动态 selective-label credit decision system 中，
当获取 rejected applicant 的反馈本身需要承担真实信用风险时，

在相同风险预算和统一 LRBE 优化框架下，

“选择哪些 rejected labels”

是否比简单地最大化可获得标签数量更重要？
复杂信息价值策略究竟在什么风险预算条件下才有价值？
============================================================

本阶段不是算法开发。

禁止：

- 修改 FAVE；
- 开发 FAVE-v3/v4/v5；
- 调 gamma；
- 调 delta；
- 根据结果修改 threshold；
- 修改 production model；
- 修改 preprocessing；
- 改 risk budget；
- 增加新的 policy 以追求更好结果；
- 进入 PPDai；
- 进入 LendingClub；
- 做 Agent；
- 做 Review branch。

完成后 STOP。

不得根据 Phase 2 结果继续调算法。


============================================================
0. Phase 2 科学问题
============================================================

正式回答：

RQ-P1
在有限信用风险预算下，
controlled exploration 是否能够比 NoExplore
获得更健康的长期反馈？

RQ-P2
Label Quantity 与 Label Quality 哪个更重要？

RQ-P3
在已经最大化可安全获得标签数量 K* 后，
不同 acquisition utility V_i
是否产生稳定的 downstream 差异？

RQ-P4
简单 RiskTilted-Uncertainty
是否已经能够获得复杂 FAVE-v2.1 的大部分收益？

RQ-P5
复杂 policy 的价值是否主要存在于：

low-risk-budget / scarce-budget regime？

RQ-P6
不同 policy 的优势体现在哪里：

- Recall？
- Balanced Accuracy？
- Calibration？
- Feedback health？
- Risk-learning Pareto efficiency？

而不只是 ROC-AUC。


============================================================
1. Phase 2 必须继承 Phase 1 的发现
============================================================

Phase 1 已经建立：

Finding A：

Selective decision 会隐藏不成比例的大量 default labels。

Finding B：

TDR / FDRatio 是最稳定的 feedback distortion signal。

Finding C：

ROC-AUC 稳定不代表 feedback-learning system 健康。

Finding D：

HMEQ 出现 genuine decision/calibration degradation，
UCI 的 external decision degradation 较弱。

Finding E：

Random exploration 能稳定改善 feedback observability，
但 downstream performance recovery 较弱。

因此 Phase 2 的研究问题不是：

“Exploration 有没有用？”

而是：

============================================================
在相同现实风险成本下，
TARGETED exploration
是否比 untargeted / simple acquisition 更有效？
============================================================


============================================================
2. 数据集
============================================================

Phase 2 只运行：

Dataset 1:
UCI Default of Credit Card Clients

Dataset 2:
HMEQ

不运行：

PPDai
LendingClub

后两者属于 Phase 3 external generalization。


============================================================
3. 必须复用 Phase 1 的数据 split
============================================================

对于每个：

dataset × seed

优先直接复用 Phase 1 frozen：

D_test
D0
applicant permutation
10 applicant batches

不要重新生成不同的数据划分。

如果 Phase 1 已保存：

split manifest / IDs / hashes

直接读取。

否则：

必须用 Phase 1 完全相同代码和 seed
deterministically reconstruct，

并做 parity check。

输出：

PHASE2_SPLIT_PARITY.csv

必须确认：

same_test_set = TRUE
same_D0 = TRUE
same_stream_order = TRUE
same_batches = TRUE

任何一项 FALSE：

停止。

输出：

PHASE2_SPLIT_PARITY_FAILURE


============================================================
4. Fixed external evaluation remains primary
============================================================

Phase 1 已经发现：

internal visible evaluation
可能严重夸大 degradation。

因此 Phase 2 的正式 policy comparison：

必须以：

FIXED EXTERNAL D_test

作为主 evaluator。

禁止使用：

visible-heldout Recall

作为论文主比较。

Internal visible evaluation：

如果保留，
只能作为 secondary diagnostic。

所有：

policy superiority claim

必须基于：

fixed external test。


============================================================
5. Seeds
============================================================

正式实验：

20 seeds

固定：

seed = 0,1,...,19

每个：

dataset
policy
rho

全部使用同一组 seeds。

禁止：

- 删除失败 seed；
- 增加 seed 直到显著；
- 针对某 policy 使用不同 seed。


============================================================
6. Rounds
============================================================

固定：

T = 10

remaining stream：

使用：

np.array_split(..., 10)

严禁重新出现：

fixed batch_size = 400。


============================================================
7. Decision rule
============================================================

继续冻结：

tau = 0.5

每轮：

PD < 0.5
→ Approve

PD >= 0.5
→ Reject

探索候选只能来自：

CURRENT ROUND REJECTED SET

即：

E_t ⊆ R_t

严禁：

historical rejected backlog。


============================================================
8. Production model 完全冻结
============================================================

必须使用 Phase 1 canonical production model。

不得：

重新调 XGBoost；

新增 calibration model；

更换 CatBoost / LightGBM；

针对 HMEQ/UCI 使用不同模型优化。

如果 Phase 1 使用：

XGBoost 100 trees
depth 3
learning_rate .1

则 Phase 2 完全复用。

Phase 2 唯一允许改变的实验因素：

exploration acquisition policy。


============================================================
9. Preprocessing 完全冻结
============================================================

继续 Phase 1：

preprocessor 每轮只 fit：

current visible data。

future batch 和 D_test：

只能 transform。

不得在整个数据集上 fit preprocessing。


============================================================
10. UCI 与 HMEQ 的风险定义不同
============================================================

这是非常重要的地方。

------------------------------------------------------------
UCI
------------------------------------------------------------

UCI 没有严格意义上的真实 EAD。

LIMIT_BAL 是信用额度，
不能直接宣称：

actual exposure-at-default。

因此 Phase 2 正式主实验：

UCI 使用：

UNIT-EXPOSURE RISK

定义：

EL_i = p_i

预算：

B_t =
rho × Σ_(i∈R_t) p_i

必须在论文中明确：

UCI experiments use unit-exposure risk budgets.

不要把它写成 monetary credit loss。


------------------------------------------------------------
HMEQ
------------------------------------------------------------

使用：

EAD_i = LOAN_i

定义：

EL_i = p_i × LOAN_i

默认：

LGD = 1

因此这是：

expected-loss proxy

不是经过真实 LGD 校准的银行 EL。

必须明确：

EL proxy = PD × LOAN。


============================================================
11. 风险预算
============================================================

固定：

rho ∈ {

0.025,
0.05,
0.10,
0.20

}

即：

2.5%
5%
10%
20%

每轮：

B_t =
rho
×
Σ_(i∈R_t) EL_i

注意：

R_t 是：

CURRENT ROUND REJECTED。

不是：

历史 rejected pool。


============================================================
12. 为什么不再跑 50% / 100%
============================================================

Phase 2 重点研究：

scarce-budget regime。

此前开发结果已经显示：

高预算下 policy 很容易 saturation。

因此正式 Phase 2：

禁止临时增加：

50%
75%
100%

来制造曲线。

2.5%：
stress regime

5%：
low budget

10%：
moderate budget

20%：
upper main budget


============================================================
13. 正式算法框架：LRBE
============================================================

所有 risk-budgeted policy
必须经过统一：

LEXICOGRAPHIC RISK-BUDGETED EXPLORATION

LRBE。


============================================================
14. Stage 1 — Quantity
============================================================

对于当前 rejected applicants：

R_t

risk cost：

c_i = EL_i

先计算：

K_t* =
max_z Σ_i z_i

subject to：

Σ_i z_i c_i <= B_t

z_i ∈ {0,1}

由于：

c_i >= 0

K* 可以通过：

ascending costs
+
cumulative sum

精确获得。

必须验证：

K* 为真正 maximum feasible count。


============================================================
15. Stage 2 — Quality
============================================================

固定：

Σ_i z_i = K_t*

然后：

maximize：

Σ_i z_i V_i

subject to：

Σ_i z_i c_i <= B_t

Σ_i z_i = K_t*

z_i ∈ {0,1}

所有 quality policies：

完全使用同一 solver。

只有：

V_i

不同。


============================================================
16. 一个重要公平性说明
============================================================

在 round 1：

如果所有 policy 初始 state 相同，

它们面对相同：

R_t
B_t
K_t*

所以 K* 必须完全一致。

在后续 rounds：

由于不同 policy 获取不同 labels，

模型 state 会自然分叉，

因此：

Rejected_t
Budget_t
K*_t

可能逐渐不同。

这是：

DYNAMIC POLICY EFFECT

不是 bug。

因此不要错误要求：

所有 policy 在 round 10 的 K* 必须相同。

正确 invariant 是：

对于任何给定：

candidate set + costs + budget

所有 policy 的 Stage 1：

K*

必须相同。

必须在 optimizer unit test 中验证。


============================================================
17. 正式 policy suite
============================================================

必须运行：

A. NoExplore

B. CountOnly

C. CountOnly-MinRisk

D. LeastRisky

E. MostRisky

F. Uncertainty

G. RiskTilted-Uncertainty

H. FAVE-v2.1

除此之外：

禁止新增 policy。


============================================================
18. NoExplore
============================================================

不探索任何 rejected applicant。

每 seed 只需要跑一次。

不用针对4个 rho 重复训练。

它作为所有 budgets 的共同 baseline。


============================================================
19. Oracle reference
============================================================

Oracle 不属于 deployable exploration policy。

但允许作为 reference upper-bound。

优先复用 Phase 1：

Oracle external-test trajectory。

如果：

model
split
preprocessing

完全一致，

不要重新浪费计算。

Oracle 只能用于：

gap / recovery reference。

不得参与：

risk-budget policy ranking。


============================================================
20. CountOnly
============================================================

Stage 1：

取得最大 K*。

Stage 2：

不得隐式最小化 risk。

因为那会与：

CountOnly-MinRisk

重合。

使用：

deterministic policy-independent tie-breaking。

推荐：

为：

candidate_id + seed + round

生成固定 hash score：

V_i = deterministic_hash(i,seed,round)

这个 score：

绝对不能使用：

PD
EAD
features
uncertainty
y

然后通过相同 LRBE Stage 2
选择一个 deterministic feasible K* set。

这代表：

quantity-first
+
no meaningful quality preference。


============================================================
21. CountOnly-MinRisk
============================================================

Stage 1：

maximum K*。

Stage 2：

minimize：

Σ_i z_i EL_i

subject to：

Σ z_i = K*

Σ z_i EL_i <= B_t

这是正式强 baseline。


============================================================
22. LeastRisky
============================================================

utility：

V_i = -p_i

使用同一 LRBE Stage 2。


============================================================
23. MostRisky
============================================================

utility：

V_i = p_i


============================================================
24. Uncertainty
============================================================

utility：

V_i =
p_i(1-p_i)


============================================================
25. RiskTilted-Uncertainty
============================================================

必须保持正式名称：

RiskTilted-Uncertainty

utility：

V_i =
p_i(1-p_i)p_i^0.5

也就是：

p(1-p)sqrt(p)

禁止叫：

FAVE。


============================================================
26. FAVE-v2.1
============================================================

必须使用 Phase 0 已恢复并冻结的：

TRUE FAVE-v2.1

固定：

gamma = 1.0

delta = 0.5

必须继续使用此前 verified：

influence /
information-matrix value function。

禁止：

用：

p(1-p)sqrt(p)

替代。

禁止：

调 gamma/delta。

禁止：

修改 reference set。

禁止：

根据 Phase 2 结果调整公式。


============================================================
27. FAVE identity gate
============================================================

在正式跑20 seeds之前：

重新运行：

FAVE frozen-reference parity test。

要求：

gamma = 1 verified

delta = .5 verified

RiskTilted != FAVE

numerical parity within tolerance

如果 FAIL：

停止：

PHASE2_FAVE_IDENTITY_FAILURE


============================================================
28. Hidden-label isolation
============================================================

所有 policy selection 输入：

只能包括：

X
PD
risk cost
EAD if legitimate
uncertainty
FAVE score
candidate IDs
budget

绝不能包括：

true y of rejected applicant。

Realized default / bad exposure：

只能由：

OracleEvaluator

在 selection 完成后计算。


============================================================
29. Phase 2 主要评价维度
============================================================

============================================================
A. Decision performance
============================================================

Recall

Balanced Accuracy

Specificity

G-mean

Precision

Accuracy


============================================================
B. Discrimination
============================================================

ROC-AUC

PR-AUC


============================================================
C. Calibration
============================================================

Brier

ECE

Calibration slope

Calibration intercept


============================================================
D. Feedback health
============================================================

Visible TDR

FDRatio


============================================================
E. Acquisition
============================================================

n_explored_round

cum_labels_acquired

K*

candidate count


============================================================
F. Risk
============================================================

Expected risk：

ER_t =
Σ_(i∈E_t) EL_i

Cumulative Expected Risk：

CER =
Σ_t ER_t

Budget utilization：

ER_t / B_t

Realized bad exposure：

HMEQ：

RB_t =
Σ y_i × LOAN_i

UCI：

如果 unit exposure：

Realized bad count：

Σ y_i

不要把 UCI 的 LIMIT_BAL
称为 realized monetary bad exposure。


============================================================
30. 主 endpoint
============================================================

预先固定三组主 outcome：

PRIMARY DECISION:

Final-round Recall

PRIMARY FEEDBACK:

Final-round FDRatio

PRIMARY CALIBRATION:

Final-round Brier

Balanced Accuracy
ECE
PR-AUC
ROC-AUC

作为 secondary。


============================================================
31. 为什么 Recall 仍然是 primary decision metric
============================================================

信贷 default detection 里：

default = positive class。

Phase 1 HMEQ 已发现：

selective labels
主要表现为：

Recall ↓
Specificity ↑

所以 Phase 2 需要研究：

哪种 acquisition policy
最能够减少 default misses。

但不得只看 Recall。


============================================================
32. 轨迹指标
============================================================

对：

Recall
BA
Brier
ECE
TDR
FDRatio
PR-AUC
AUC

同时报告：

round 0

round 1..10

final

trajectory mean

trajectory area / average

避免只使用 final round。


============================================================
33. 与 NoExplore 的 improvement
============================================================

对于越大越好的 metric：

ΔRecall =
Recall_policy - Recall_NoExplore

ΔBA

ΔFDRatio

对于越小越好的：

BrierImprovement =
Brier_NoExplore - Brier_policy

ECEImprovement =
ECE_NoExplore - ECE_policy


============================================================
34. Risk-learning efficiency
============================================================

计算：

RecallGainPerRisk =

ΔRecall
/
CER

FeedbackGainPerRisk =

ΔFDRatio
/
CER

BrierGainPerRisk =

BrierImprovement
/
CER

注意：

如果：

CER≈0

返回 NaN。

这些是：

descriptive efficiency metrics

不是主要 hypothesis test。


============================================================
35. Phase 2 最重要的图：Pareto
============================================================

必须生成：

Figure P1

x =
Cumulative Expected Risk

y =
Recall gain over NoExplore


Figure P2

x =
Cumulative Expected Risk

y =
FDRatio gain


Figure P3

x =
Cumulative Expected Risk

y =
Brier improvement


HMEQ 再增加：

Figure P4

x =
Cumulative Realized Bad Exposure

y =
Recall gain


必须标记：

Pareto frontier。


============================================================
36. Recall vs Budget
============================================================

生成：

rho:

2.5
5
10
20

x-axis：

rho

y-axis：

final Recall

所有 policy curve。

UCI/HMEQ 分开。


============================================================
37. Feedback Health vs Budget
============================================================

生成：

FDRatio vs rho

TDR vs rho


============================================================
38. Calibration vs Budget
============================================================

生成：

Brier vs rho

ECE vs rho


============================================================
39. Labels vs Budget
============================================================

生成：

cum_labels_acquired

vs

rho

用于分析：

Quantity vs Quality。


============================================================
40. Risk utilization vs Budget
============================================================

必须检查：

risk_used <= budget

并画：

budget utilization

以防某些 policy 看起来风险低，
只是因为没有用完预算。

由于 LRBE 先最大 K*：

重点记录：

labels acquired
risk used
remaining risk budget。


============================================================
41. 正式 primary comparisons
============================================================

预注册主要比较：

C1:

FAVE-v2.1
vs
CountOnly-MinRisk


C2:

FAVE-v2.1
vs
RiskTilted-Uncertainty


C3:

RiskTilted-Uncertainty
vs
CountOnly-MinRisk


C4:

Uncertainty
vs
CountOnly-MinRisk


CountOnly
LeastRisky
MostRisky

保留为结构性 baselines，
但不作为最主要 confirmatory comparisons。


============================================================
42. 每个 seed 必须 paired
============================================================

所有 pairwise tests：

dataset
rho
seed

严格配对。

主统计：

Wilcoxon signed-rank。

必须报告：

mean difference

median difference

bootstrap 95% CI

raw p

Holm-adjusted p

win/tie/loss。


============================================================
43. Holm correction family
============================================================

必须预先固定 multiplicity family。

对于：

Final Recall

每个 dataset：

4 primary comparisons
×
4 budgets

共：

16 confirmatory tests。

对这16个 raw p：

统一：

multipletests(
    p,
    method="holm"
)

这是：

PRIMARY CONFIRMATORY FAMILY。


对于：

FDRatio

单独形成第二个16-test family。


对于：

Brier

单独形成第三个16-test family。


其他：

BA
ECE
PR-AUC
ROC-AUC

作为 secondary/exploratory，
也可分别 Holm，
但不要与 primary Recall family 混在一起。


============================================================
44. Bootstrap
============================================================

paired seed differences：

20 seeds。

Bootstrap：

10000 resamples

固定：

bootstrap_seed。

输出：

95% percentile CI。


============================================================
45. Win / tie / loss
============================================================

对于每：

dataset
rho
comparison
metric

输出：

wins
ties
losses

tie tolerance：

必须预定义。

例如：

1e-10

不要人工判断 tie。


============================================================
46. 预注册 Scarcity Hypothesis
============================================================

Phase 2 必须正式检验：

H-SCARCITY：

Policy sophistication matters primarily
in the low-risk-budget regime.

重点比较：

FAVE vs CountOnly-MinRisk

FAVE vs RiskTilted

定义：

LowBudgetDiff =
mean(
difference at rho=.025,
difference at rho=.05
)

HighBudgetDiff =
mean(
difference at rho=.10,
difference at rho=.20
)

每 seed：

ScarcityContrast =
LowBudgetDiff
-
HighBudgetDiff

对：

Recall
FDRatio
Brier improvement

做 paired Wilcoxon / bootstrap CI。

如果：

ScarcityContrast > 0

且方向稳定，

支持：

complex policy advantage is stronger
under scarce budgets。

如果不成立：

H-SCARCITY NOT SUPPORTED。


============================================================
47. Saturation diagnostics
============================================================

每个 rho 记录：

K* / n_rejected

称：

ExplorationCoverageRatio。

如果：

20% budget

已经探索大部分 rejected applicants，

策略差距自然收敛。

输出：

coverage_vs_budget.csv

用于解释：

budget saturation。


============================================================
48. Label Quantity vs Quality
============================================================

输出：

quantity_quality_summary.csv

字段：

dataset
seed
rho
policy

cum_labels
cum_expected_risk
final_recall
final_fdratio
final_brier

并计算：

corr(labels acquired, recall recovery)

corr(labels acquired, FDRatio recovery)

但：

相关性只能作为机制描述。

不要做因果声称。


============================================================
49. Policy 选择了什么样的人
============================================================

Phase 2 只做基本 acquisition diagnostics。

每 policy 记录 selected applicants：

mean PD

median PD

mean risk cost

mean exposure

mean uncertainty

default rate of selected
（oracle evaluator only）

对于 HMEQ：

mean LOAN

realized default exposure。

不要在本阶段做复杂 feature-space mechanism analysis。

完整解释留给 Phase 4。


============================================================
50. Policy selected label composition
============================================================

Offline evaluator 计算：

ExploredDefaultRate =

# defaults in explored set
/
# explored labels

这不能作为 policy input。

用于判断：

MostRisky / RiskTilted / FAVE

是否真正找回更多 previously hidden defaults。


============================================================
51. Phase 2 integrity audit
============================================================

生成：

PHASE2_INTEGRITY_AUDIT.csv

每：

dataset
seed
rho
round
policy

记录：

n_batch
n_rejected
n_explored

Kstar

risk_budget
risk_used

risk_ok

selected_subset_of_current_rejected

no_historical_backlog

hidden_label_isolation

fixed_test_isolation

visible_update_correct

fave_identity_ok

optimizer_ok


============================================================
52. Optimizer invariant
============================================================

对于 synthetic matched candidate pool：

同一个：

costs
budget

验证：

CountOnly
CountOnly-MinRisk
LeastRisky
MostRisky
Uncertainty
RiskTilted
FAVE

Stage 1 K*

完全相同。

注意：

这只是 optimizer invariant。

不要错误要求不同 dynamic trajectories
后期实际 K* 一样。


============================================================
53. Dry Run Gate
============================================================

正式20 seeds前：

先运行：

dataset = UCI, HMEQ

seed = 0

rho = .025,.05,.10,.20

全部 policies。

只检查：

protocol
risk
FAVE identity
K*
output schema

禁止根据性能修改算法。

若 PASS：

立即进入 full formal run。

若 FAIL：

只能修 BUG。

不能修改方法。


============================================================
54. Formal Run
============================================================

正式：

2 datasets

×

20 seeds

×

4 budgets

×

7 risk-budgeted policies

加：

NoExplore once per seed。

可以并行运行。

确保：

每个 job 有唯一输出文件。

不要多个进程写同一 CSV。


============================================================
55. Checkpoint
============================================================

每：

dataset × seed × rho × policy

单独保存结果。

如果中断：

允许 resume missing jobs。

禁止：

重新覆盖已经成功的结果，
除非 hash/config 不一致。


============================================================
56. Config hash
============================================================

正式 run 前生成：

PHASE2_CONFIG_FROZEN.json

并计算：

SHA256。

所有结果文件记录：

config_hash

git_commit if available

FAVE_reference_hash

dataset_hash。


============================================================
57. UCI 特别说明
============================================================

UCI 已被大量用于 development。

因此：

所有 UCI Phase 2 结果：

只能称：

development/benchmark evidence。

不得称：

untouched external validation。


============================================================
58. HMEQ 特别说明
============================================================

HMEQ 是：

monetary exposure validation。

但是：

LOAN × PD

仍是：

risk proxy。

不得称：

actual expected bank loss

除非有真实：

LGD
EAD realization

数据。


============================================================
59. Phase 2 不能做参数选择
============================================================

即使出现：

FAVE 2.5% 赢
5% 输
10% 赢
20% 输

也不得：

根据 rho 切换 gamma/delta。

即使：

RiskTilted 赢 FAVE

也不得：

修改 FAVE。

Phase 2 是：

EVALUATION

不是：

DEVELOPMENT。


============================================================
60. Phase 2 Outcome Interpretation
============================================================

在 UCI + HMEQ 两个数据集上，
只能给：

PROVISIONAL OUTCOME。

不能因为两个数据集就宣称：

最终算法 superiority。

Phase 3 的：

PPDai + LendingClub

才是真正 external validation。


============================================================
61. Provisional Outcome A
============================================================

如果 Phase 2 发现：

FAVE > CountOnly-MinRisk

且：

FAVE > RiskTilted

尤其：

2.5 / 5%

方向稳定，

则写：

PROVISIONAL SUPPORT FOR INFORMATION-VALUE SELECTION

但：

不得写：

FAVE validated.

Phase 3 继续 frozen validation。


============================================================
62. Provisional Outcome B
============================================================

如果：

RiskTilted ≈ FAVE

或：

RiskTilted > FAVE

写：

SIMPLE RISK-AWARE UNCERTAINTY
CAPTURES MOST OF THE OBSERVED VALUE

不要再开发新 FAVE。

Phase 3 仍保留 frozen FAVE
作为 comparison。


============================================================
63. Provisional Outcome C
============================================================

如果：

CountOnly-MinRisk >= FAVE

在：

UCI + HMEQ

大多数 budgets 稳定出现，

写：

PROVISIONAL SUPPORT FOR
QUANTITY-FIRST / SAFE-LABEL PRINCIPLE

仍然不要调算法。

Phase 3 用 external datasets
确认是否普遍成立。


============================================================
64. 重点：负结果也是结果
============================================================

如果所有 quality policies：

CountOnly-MinRisk
RiskTilted
FAVE

差不多，

不要写：

“实验失败”。

应检查：

- labels acquired 是否类似；
- budget 是否饱和；
- candidate heterogeneity 是否不足；
- downstream metrics 是否对 policy selection 不敏感。

研究问题本来就是：

HOW SMART DOES EXPLORATION NEED TO BE?

答案可能是：

not very smart.


============================================================
65. 结果目录
============================================================

建议：

results_phase2/
│
├── protocol/
│   ├── PHASE2_PROTOCOL.md
│   ├── PHASE2_CONFIG_FROZEN.json
│   ├── config.sha256
│   └── split_parity.csv
│
├── raw/
│   ├── uci/
│   └── hmeq/
│
├── summary/
│   ├── all_rounds.csv
│   ├── final_by_seed.csv
│   ├── final_summary.csv
│   ├── paired_tests.csv
│   ├── paired_tests_holm.csv
│   ├── scarcity_tests.csv
│   ├── acquisition_summary.csv
│   ├── quantity_quality_summary.csv
│   └── pareto_points.csv
│
├── figures/
│
├── PHASE2_RESULTS.md
├── PHASE2_STATISTICAL_REPORT.md
├── PHASE2_PARETO_REPORT.md
├── PHASE2_POLICY_REPORT.md
├── PHASE2_INTEGRITY_AUDIT.md
└── PHASE2_PROVISIONAL_CONCLUSION.md


============================================================
66. 必须生成的核心表
============================================================

TABLE 1

Final metrics：

dataset
rho
policy

Recall
BA
AUC
PR-AUC
Brier
ECE
TDR
FDRatio

mean ± SD。


TABLE 2

Acquisition：

cum labels
cum expected risk
budget utilization
coverage ratio


TABLE 3

Primary paired comparisons：

FAVE vs CountOnly-MinRisk
FAVE vs RiskTilted
RiskTilted vs CountOnly-MinRisk
Uncertainty vs CountOnly-MinRisk


TABLE 4

Scarcity contrast。


TABLE 5

Pareto-efficient policy points。


============================================================
67. PHASE2_RESULTS.md
============================================================

必须按：

# 1. Protocol integrity

# 2. UCI results

# 3. HMEQ results

# 4. Label quantity

# 5. Policy quality

# 6. Risk-budget trade-off

# 7. Low-budget regime

# 8. Calibration

# 9. Feedback health

# 10. Pareto analysis

# 11. Statistical tests

# 12. What FAVE actually contributed

# 13. What simple policies achieved

# 14. Dataset-dependent results

# 15. Negative findings

# 16. Implications for Phase 3


============================================================
68. 最终必须回答的6个问题
============================================================

Q1.

Under the same risk budget,
does exploration improve over NoExplore?

Q2.

Does CountOnly-MinRisk remain a strong baseline?

Q3.

Does RiskTilted capture most of FAVE's value?

Q4.

Does FAVE provide stable incremental value?

Q5.

Are policy differences larger at low budgets?

Q6.

Is policy sophistication more important for:
decision,
calibration,
or feedback health?


============================================================
69. Phase 2 PASS 条件
============================================================

不是：

FAVE winning。

Phase 2 PASS 的条件：

[ ] protocol全部通过

[ ] UCI 20 seeds完整

[ ] HMEQ 20 seeds完整

[ ] 四 budgets完整

[ ] 所有 frozen policies完整

[ ] 无 hidden-label leakage

[ ] risk constraint全部满足

[ ] FAVE identity验证

[ ] 主统计完成

[ ] Pareto完成

[ ] scarcity test完成

[ ] 不进行 post-hoc tuning


只要这些成立：

PHASE2 = PASS

无论哪个算法赢。


============================================================
70. Phase 2 STOP RULE
============================================================

Phase 2 完成以后：

STOP。

不得：

- 修改 FAVE；
- 调 gamma/delta；
- 为 HMEQ 发明新 utility；
- 重新找 rho；
- 重新调 tau；
- 继续增加 seeds；
- 自动跑 PPDai；
- 自动跑 LendingClub。


============================================================
71. 最终 ZIP
============================================================

输出：

risk_budget_exploration_PHASE2.zip

必须包含：

全部 source code

tests

configs

raw results

summary

statistics

figures

reports

hashes

requirements/environment。


============================================================
72. 最终终端只打印
============================================================

============================================================
PHASE 2 STATUS
============================================================

Protocol:
PASS / FAIL

UCI:
20/20 seeds

HMEQ:
20/20 seeds

Budgets:
2.5%, 5%, 10%, 20%

Policies:
NoExplore
CountOnly
CountOnly-MinRisk
LeastRisky
MostRisky
Uncertainty
RiskTilted-Uncertainty
FAVE-v2.1

FAVE identity:
PASS / FAIL

Primary result:

UCI low-budget:
best policy = ...
FAVE vs CountOnly-MinRisk = ...
FAVE vs RiskTilted = ...

HMEQ low-budget:
best policy = ...
FAVE vs CountOnly-MinRisk = ...
FAVE vs RiskTilted = ...

Scarcity hypothesis:
SUPPORTED / NOT SUPPORTED / MIXED

Quantity-vs-quality:
...

Pareto:
...

Provisional outcome:
A / B / C / MIXED

Ready for Phase 3:
YES / NO

Output:
risk_budget_exploration_PHASE2.zip

然后停止。