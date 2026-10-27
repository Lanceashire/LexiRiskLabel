# LexiRiskLabel 下一阶段实验协议
## 状态
- 基于仓库提交：45a2d41d403ce0587d5434f564abb76ac7eaaf48。
- 本文件为建议性研究设计，不是新增实验结果；本文列出的运行数是计划槽位，不是统计独立样本数。
- 已有 84 条真实任务轨迹与机制结果作为开发证据，不合并进后续确认性统计。
- 原始数据、逐人日志、选集与预测数组不在本文件包中；使用本地授权数据与完整档案。
- protocol.json 是任务调度规范，不是现有 runner 可直接读取的配置；新增 runner 尚需实现。

## 1. 唯一主问题
在相同普通反馈、候选集合与数值风险预算下，复杂采集策略究竟在哪些条件下获得超出 MinRisk 的实际预测收益？
依次区分：
1. 当前学习器是否存在额外收益空间；
2. 简单采集是否已获得主要收益；
3. 代理目标能否识别剩余收益；
4. 求解器能否找到代理目标下的好选集；
5. 改变训练方式和未来反馈是否产生额外作用。

## 2. 数据与信息边界
D：当前真实可见标签；O：已承诺普通审批的特征；C：可探索候选特征；
Q_ref：独立且可使用的目标参考特征；V：研究阶段独立验证标签；
T：外层评估标签，仅评估服务可见。
采集器不能接触 O/C 当前隐藏标签、T 特征或标签、真实未来标签。
Q_ref 不得直接复用 T；使用未来协变量时必须单列为转导协议。
先锁定选集与选择日志，再由反馈服务揭示 O 和选中的 C 标签。
风险成本 c、数值预算 B、审批模型、预处理与超参数在同状态分叉中固定。
全反馈与预算内验证挑选参考属于离线诊断，不是可部署竞争算法。

## 3. E0：协议与实现审计（立即执行）
输出：audit_report.json、manifest.json、source_identity.json。
检查：
- 改变隐藏候选标签、普通标签或外层测试标签，不改变采集输出。
- 修改任何配置/代码/数据哈希，不能静默复用旧结果。
- 无重复采集、不可重复统计已获标签；预算与实际后端准确记录。
- 不同参数输出同一选集时，标记为同一次实际干预。
- 将标签释放时点、未来批量、普通容量、风险上限显式参数化。
- GATE 的 one-step planning 与 no-future-label-environment 分开。
- 校正分支记录 π=0 人群，禁止对其宣称无偏恢复。
- 相同重训数据和训练设置可缓存；不把重复拟合当作重复样本。
- 所有线性、矩阵与穷举正确性测试保留；数学恒等式通过不代表学习效益通过。

## 4. E1：同状态收益与学习器分叉（立即执行）
固定 MinRisk 轨迹为主锚点：
HMEQ/UCI/LendingClub × 2201/2202 × round 2/5/9，共18个状态。
方法选集：
OrdinaryOnly、MinRisk、LRBE-IG、Vopt、RB-BRIDGE、Vopt-Kstar、
Random-feasible、FullFeedback。
每个选集分别交给逻辑回归与当前 XGBoost 重训；
不得为不同学习器重新选集，必须共享 select_hash。
基础：18×8×2=288 个拟合槽位。
建议随机方法5次固定抽签，其余7种各1次：18×12×2=432槽位。
只在开发块内运行，旧测试结果作为开发终点，不称新盲测。
未修改算法，LR ridge=1，XGBoost沿用旧参数；调参留到冻结后的内层流程。

单步指标：
gain(S) = Brier(A(D+O)) - Brier(A(D+O+S))。
相对 MinRisk：
extra_gain(S) = Brier(A(D+O+MinRisk)) - Brier(A(D+O+S))。
记录总体、冻结拒绝区域、其余区域；总体为主，区域为机制。
FullFeedback 指 D+O+C，允许违反预算，仅作诊断。它不是理论上界；
全反馈差，不足以排除预算内优质子集更好。
不得把不同锚点的单步收益相加当作完整策略收益。

必须记录：
ordinary_n、candidate_n、initial_visible_n、Kstar、count、cost、realized_loss、
slack、max_swap_count、selected_hash、Jaccard、实际换入人数、
同一F下的proxy_value、positive_pair_steps、pair_gain、
每人预测误差、训练与选择耗时、实际后端、representation_hash。

E1统计：
18状态属于6条轨迹；3个状态不是3个独立种子。
随机抽签是算法随机性重复，不新增数据独立样本。
主要输出每条轨迹均值和完整配对差，而非两种子显著性检验。

## 5. E1b：预算内选集空间诊断（只有E1难以区分时）
挑选轮次事先固定的6个状态，每状态预先生成16个预算可行选集。
生成来源：既有方法、随机可行、风险匹配可行交换；不能看T标签选集合。
所有选集释放各自标签后拟合，在独立 V_search 上选最佳，
只在独立 V_eval 上评估该“有限集合离线参照”。
必须单列它使用额外验证标签，不能放进可部署方法主排名。
它不是全局最优，也不是统计上界。
若无独立验证块，只报告全部选集结果，不把T上的最优值当可泛化结果。
不与正式确认测试混用。

## 6. E2：BRIDGE机制与代理诊断（条件执行）
E2a：优化正控制。
用线性高斯共轭问题实现精确目标方差下降，再用逻辑回归验证迁移。
起始二维例为 H=I，G=diag(1,0)，u+=(1,L)，u-=(1,-L)，u0=(0.1,0)。
普通方向信息增加b后，互补正值条件为 b < L²-1（该二维例限定）。
注册完整生成条件；不能删去不利种子或按结果寻找“成功正控制”。

建议第一网格：
3个普通信息覆盖水平 × 2种成本安排 × 10个生成种子 × 5个选择器
=300方法终点。Ordinary基准可解析计算或共享拟合，不重复计入。
选择器：MinRisk、单点Vopt、BRIDGE、随机可行、穷举代理最优。
小池至多12或16个候选；逐项枚举可行子集。
成本安排保持边际成本多重集不变，通过预设交换改变信息与价格的关联。
单独评估代理最优差距与重训收益；不能用代理改善代替实际收益。

E2b：真实状态三层定位。
1. 全配对或注册的随机子池中，是否存在显著超过数值误差的Γ>0；
2. BRIDGE是否找到有用候选组；全配对/穷举能否改善当前求解结果；
3. 相同F上的改善能否对应真实重训收益。
Γ阈值使用绝对+相对浮点容差，并保存原始Γ，不只存二元标记。
若没有成组空间，停止增加partners和exchange次数。

E2b补充：双人搜索×交换优化必须解耦。
当前core.py仅在pairs=True时执行交换，BRIDGE与Vopt不是纯粹的组搜索对照。
新增四格：单人/不交换、单人/交换、双人/不交换、双人/交换。
四格共享同一几何、预算、候选预筛、允许的交换次数和预注册运行上限。
先比较四格的代理目标，再比较相同重训程序的实际损失。
负Γ不意味着所有组合优化都无用：背包装填和局部交换仍可能改善，
但这种改善不能归为“正互补性”。

E2c：代理-学习器匹配。
只有E1显示模型匹配差异后运行。
表示一：原始标准化特征+逻辑头。
表示二：仅用D训练并冻结的小型树叶嵌入+逻辑头。
表示二先采用20棵深度2树、至多约80个叶特征（具体数由树决定），
所有选择器共享同一表示、G、正则化和预算。
分别评估：冻结特征逻辑头、全量重训XGBoost。
冻结叶子上的Fisher只对应冻结逻辑头，不是重新生长树的精确Hessian。
对比Vopt与BRIDGE均须使用相同表示。
单独做Q_ref规模128/512的敏感性，不与新求解器同时改变。

E2d：数量-构成交叉。
每状态去重 k ∈ {floor(0.6Kstar),floor(0.8Kstar),Kstar}，
必要时保留k=0基准；固定k时比较MinRisk-k、IG-k、Vopt-k。
固定数量不等于固定风险；同时报告实际风险和构成溢价。
仅对可匹配状态追加同数量、风险差不超过预注册容差的随机交换对照。
匹配失败标记不可估计，不偷偷放宽容差。
旧epsilon扫描输出相同集合的点不算多个独立结果。

## 7. E3：CORR（先诊断，暂不恢复主线）
保留旧完整校正为冻结负结果。
第一步固定选集、π、g和学习器，只改变使用标签的方式：
A. 普通已观测标签训练；
B. g插件伪目标（包括相同可达候选）；
C. 完整残差校正；
D. 预设收缩α∈{0.25,0.5}，只作为有偏稳定化诊断。
A不等于α=0，必须分开。
所有分支用同一回归器、同样正则化、相同数据权重约定。
记录伪目标范围、越界比例、目标梯度极值、最小π、权重分位数、
每轮及累计ESS、可达群体与目标人群覆盖。
通过随机采集重复区分选样方差与测试样本波动。

对于固定Yi,gi：
E[tildeYi^(α)]-Yi=(1-α)(gi-Yi)；
Var=α²(1/πi-1)(Yi-gi)²。
这只是设计层面的偏差方差，不是泛化保证。

有限混合集合可计算：
πij = sum_S q(S) 1{i,j∈S}。
固定函数Brier的校正项方差使用全部πij核验，
不将对角方差代理误称完整设计方差。
使用真实隐藏残差的核验仅属于离线诊断。
合成中用真p*替代g只做oracle对照；真实方法只能用历史可见标签。
历史残差估计采用交叉拟合，且必须披露不可达区域无法验证。
通过：校正能在明确机制区改善，且有偏稳定化在独立验证中优于普通训练。
失败：关闭校正主线，保留可行混合概率设计与审计工具。

## 8. E4：GATE（先修协议，暂不大规模）
统一规划与评价的future_batch_size/capacity/risk_cap/threshold；
建议用当前评价设定160/80/20作为小试共同配置，而非业务最优值。
主要消融：
planning_horizon=1/2（实际环境都释放未来标签）；
actual_future_release=True/False（真正改变环境，不只是评分函数）；
future_approval=fixed/dynamic。
只先比较关键配对，不全因子扩张。

候选批次族必须相同。所有候选使用共同模拟随机数。
选择世界与核验世界独立；核验结果不用于反复调候选族。
分开估计：
- 有限世界Monte Carlo排序噪声；
- 规划模型对真实生成机制的失配；
- 候选族是否包含有用前瞻动作。
合成用真生成器的规划器是oracle诊断，不是真实信用可用方法。
真实未来标签从不参与当前规划。
比较总体普通+探索风险；免费反馈不等于零经济风险。
停止：正确生成器下也没有额外前瞻价值，则不扩大时域和算力。

## 9. 交叉验证：验证完整采集流程，不是已采集训练集
先完成诊断和最多一个候选修改，再冻结正式协议。

建议内部评估：
外层5折×2次注册重复；每个外层折2个重放随机流。
每个外层折重新：
划分数据→拟合预处理→建立初始可见集→拟合公共业务模型→
构造Q_ref→内层调参→从头采集与重训10轮→T评估。
同一个人的重复记录必须同组；时间数据采用前向切分，不打乱为主时间结果。

外层训练部分：
预留独立协变量参考R；其标签不进入采集器。
外层T的特征和标签均不用于严格归纳式协议。
剩余部分构成回放初始池与到达流，初始遮蔽规则预先固定。

内层3折：
超参数选择必须排除外层T。
调预测器：仅用允许的初始可见标签，预处理逐折重拟合。
调采集策略：每个内层折要完整重放，V_inner只给评分器；
这属于使用历史全标签研究基准的离线元级调参，不能宣称部署中免费获得拒绝人群标签。
如不具备这种开发标签资源，则采集参数在独立开发任务冻结；
不能借用当前隐藏候选标签替代验证集。
CORR的α也服从相同信息约束。

建议有限网格：
LR正则λ∈{0.1,1,10}；
XGB depth∈{2,3,5}，trees∈{100,300}，其他参数固定；
最多6个候选配置/方法，优化预算和失败记录透明。
为机制隔离，先固定学习器参数；调参版本另列。

最终主比较六种：
OrdinaryOnly、MinRisk、LRBE-IG、匹配Vopt、一个最近邻外部方法、
唯一胜出的新候选。单主预算0.1：
3×5×2×2×6=360条十轮轨迹（不含内层调参）。
预算0.025/0.4仅对MinRisk/最近邻/新候选扩展：
3×5×2×2×3×2=360条，合计720（不含内层调参）。
不得在结果出来后更换主预算。

跨数据集：
在两个任务固定方法配置，第三个任务只按可见D0拟合本地模型；
轮换三次。旧数据全被看过，称内部迁移稳健性，不称首次外部验证。
真正外部：新机构或未参与开发的成熟时间队列，事先固定schema与标签时间。
时间外回放只能在该历史时点使用已经成熟/可获得的结果；
若只有最终贷款状态、没有标签可得时点，不宣称严格在线时序验证。
可报告按放款时间排序、标签即时揭示的回顾性敏感性实验，并明确其假设。
LendingClub历史已放款队列不解决真实拒绝人群的反事实标签缺失。

## 10. 统计与预注册
主终点：十轮平均总体Brier（不含初始轮）；次终点：最终Brier与固定区域Brier。
主比较：唯一新候选对MinRisk和匹配最近邻；其他为次要/机制。
每个任务独立呈现；不同货币单位风险不直接合并。
主效果报告绝对差与相对改善；不依赖可能接近零的Oracle-gap比例。
最小相关效应δ需在确认前锁定；可用旧开发OrdinaryOnly Brier的0.5%
作研究阈值示例，并固定0.1%/1%敏感性，不宣称为真实商业收益门槛。
方向判断：
L95>δ：超过预设有意义改进；
U95<0：劣于对照；
U95<δ：排除该幅度的正收益，不自动等价；
CI完整落在[-δ,δ]：在该标准下支持近似等效；
其他：不确定。
以上仅用于适当的确认推断，不能为两种子开发结果生成强结论。

独立性：
合成生成种子可作主要独立簇；
同轨迹轮次、状态、随机采集重复不是新的总体样本；
重叠训练的交叉验证折不是互相独立的实验。
重复CV报告配对效应与折间/重放稳定性，不把20折当n=20独立种子套普通t。
独立封存测试可按申请人/借款人/月份做配对重采样，说明区间条件于已训练模型；
训练集抽样不确定性需要重新抽样整个训练-采集流程，不能由测试bootstrap替代。
若需多重确认比较，对预定任务×主对照家族使用Holm等调整；
不得混用多种显著性方法择优报告。
外层结果一旦用于修改算法，降为开发证据。

## 11. 输出数据表
state_manifest.jsonl：
dataset,input_hash,protocol,anchor_policy,seed,round,state_id,
D/O/C/reference ids或可审计哈希,preprocessor_hash,price_hash,cost_hash,B,
label_release_state,code_hash,config_hash。
selection_summary.jsonl：
state_id,selector,selector_seed,representation_hash,selected_hash,
count,cost,realized_loss,Kstar,slack,maxswap,proxy_value,pair diagnostics,time。
branch_outcomes.jsonl：
state_id,selected_hash,learner,learner_config,fit_seed,
Brier_base,Brier_selected,gain_vs_ordinary,gain_vs_minrisk,
target/rest metrics,actual_backend,fit_time。
confirm_episodes.jsonl：
outer_fold,outer_repeat,replay_seed,algorithm,budget,
ALC_Brier,final_Brier,total_exploration_cost,total_realized_loss,
ordinary_cost,ordinary_loss,all_round_metrics_hash。

所有选集与预测逐样本日志保留本地；公开汇总不得包含不获授权的第三方原始行。

## 12. 改进路线与门槛
第一优先：实际训练程序匹配的目标收益估计。
先冻结树叶表示作模型匹配诊断；必要时仅对少量预先候选批次做真实学习器幻想重训。
共同幻想O和S，模拟标签只能来自历史可见数据形成的模型。
幻想重训只消除部分线性化误差，仍有世界模型错设；
不能把模拟标准误当真实预测收益的置信保证。
BRIDGE只有确有组合空间时才作为求解器插件，不先假设其是主算法。

第二优先：数量保护，只在牺牲数量解释损失后启用。
扫描实际不同的固定数量或可行数量下限，不把相同选集当新结果。
第三优先：有偏低方差CORR，仅在其独立训练消融通过后恢复。
第四优先：GATE，只在正确信息时序与oracle规划诊断通过后恢复。

最多保留一个正式新候选；不新增CVaR、公平性、生存模型等不同目标。
研究创新必须来自经验证的新机制、估计/决策方法与证据，
不是方法缩写、调参、增加数据集或熟知的矩阵恒等式。

## 13. 当前执行队列
立即：E0 + E1（基础288，推荐432槽位）。
条件：E1b或E2，取决于收益空间/代理/求解器诊断。
暂缓：CORR/GATE规模化及全部720条确认轨迹。
阶段交付必须给出：支持什么、反证什么、仍不确定什么、继续/停止哪一路。
失败与不确定是不同结论；不以“必然需要一个胜出算法”为终点。

## 来源索引
R1：仓库 experiments/research_v2/README.md；上述提交。
R2：仓库 experiments/research_v2/core.py；上述提交。
R3：仓库 experiments/research_v2/run_real.py；上述提交。
R4：仓库 experiments/research_v2/run_mechanisms.py 与 run_focused.py；上述提交。
R5：仓库 results/research_v2/analysis/paired_descriptive_contrasts.csv；
    episode_summary.csv；mechanism_summary.csv；上述提交。
R6：scikit-learn 官方 Nested versus non-nested cross-validation；
    Cross-validation: evaluating estimator performance。
R7：Cawley & Talbot, On Over-fitting in Model Selection and Subsequent
    Selection Bias in Performance Evaluation, JMLR 2010。
R8：Bengio & Grandvalet, No Unbiased Estimator of the Variance of K-Fold
    Cross-Validation, JMLR 2004。
R9：Ash et al., Gone Fishing: Neural Active Learning with Fisher Embeddings,
    NeurIPS 2021。
R10：Huang, Bickford Smith & Rainforth, Loss-Driven Bayesian Active Learning,
     AISTATS 2026, PMLR 300:5140–5148。
R11：Bickford Smith et al., Prediction-Oriented Bayesian Active Learning,
     AISTATS 2023。
R12：Hu & Mussmann, Myopic Bayesian Decision Theory for Batch Active Learning
     with Partial Batch Label Sampling, arXiv:2510.09877。
R13：Murray et al., Addressing Bias in Active Learning with Depth Uncertainty
     Networks... or Not, PMLR 163, 2022（研讨会论文）。
