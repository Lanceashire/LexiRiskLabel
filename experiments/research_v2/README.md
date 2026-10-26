# 2026-10-26 小规模研究实验

本目录实现“论文改进建议”聊天末尾的 RB-BRIDGE、RB-CORR、RB-GATE，并采用用户补充文档中的共同预算、普通反馈基线、独立目标参考、配置身份与预测留存要求。

## 本轮范围

- E0：数学互补例、秩更新、小池穷举、概率校正恒等式、预算与信息接口检查。
- 六类开发机制：A1/A2 未覆盖/已覆盖互补方向，B1/B2 错设/匹配平方损失模型，C1/C2 动态/冻结审批。每类 3 个预定开发种子。
- 三份真实数据：HMEQ、UCI、LendingClub；两个预定开发种子；一个主预算 rho=0.1；每条轨迹十轮。
- 固定审批的分类器比较，与平方损失回归器的采集×校正实验分别分析。
- RB-GATE 按聊天要求先限于小型闭环机制环境，不启动大规模真实动态回放。
- 不运行文档示例中的 1,320 条正式轨迹、14×30 机制确认、PPDai、CVaR、公平性或延迟扩展。

## 方法实现与限定

`core.py` 的选择接口只能接收已经可见的标签、本轮普通与候选特征、独立参考特征、成本和预算。普通/候选隐藏标签不在接口内。每轮在读取新标签前先写出完整选集日志。

BRIDGE：信息白化后按目标/非目标方向寻找最多 8 个伙伴，同时比较单人和双人的精确局部联合增益，最后允许一次可行交换。未按单点分数截断候选。随机伙伴与无普通条件化版本用于机制消融。

CORR：以随机顺序可行分组建立正概率覆盖，在可行集合混合上做 12 步条件梯度，每步调用风险背包；最终混入相同的 0.15 覆盖分布。保存支持集、混合权重、精确边际概率和历史伪目标。不可采候选的概率为零，不声称全域无偏。没有截断逆概率权重。

GATE：8 个当前可行批次（含空探索）使用共同的 16 个 Laplace 模拟世界排序，另外 8 个世界留作模拟核验。未来特征只从已观察特征历史重采样。同一世界共享潜在参数生成当前和未来标签。它目前是逻辑模型规划原型。

BAIT-cost 是固定特征二分类 Fisher 目标的风险成本适配：前向到 2B，按单位成本移除损失后向删减至 B。ParBaLS-cost 依作者代码的 multi-pseudo EPIG/EER 准则实现，4 条幻想、32 个后验样本，普通 pending 标签也被幻想，再按边际 EPIG/成本填充。它们是本项目的预算适配实现，不能当成原作者默认大规模配置的直接复现。

## 数据与模型

第三方数据保存在 `data/research_v2/`，来源、字节哈希、行数和旧登记哈希对照在该目录 `provenance.json`。新下载输入未证明与旧登记文件字节一致；不覆盖旧文件、不混合历史结果。

真实回放按申请人行划分为开发协变量 20%、封存测试 20%、回放 60%。回放初始池占 25%，其中按协变量分数观察一半标签；剩余回放申请分成十轮。开发标签完全不用于训练或调参，目标参考只使用开发协变量。预处理固定拟合在初始可见特征上，所有方法共用该表示。

HMEQ 排除 LOAN 预测特征，UCI 排除 ID，LendingClub 使用 `run_real.py` 中明确列出的 21 个字段，排除 last_fico 等更新结果字段。该紧凑 schema 与旧稿全部字段版本不同，结果独立解释。未做结果驱动调参。

真实分类器为 GPU XGBoost（100 树、深度 3、学习率 0.1），CORR 对照全部使用相同配置的 XGBRegressor 平方损失。概率输出限制在 [0,1]。预测可走 CPU 输入的 DMatrix 路径；训练实际 device 和 updater 逐轮读取 booster 配置并验证为 cuda，而非以请求参数替代后端证据。

## 文件

- `config.json`：预定小规模配置。
- `prepare_data.py`：新输入重建及来源记录。
- `run_mechanisms.py`：E0 与六环境机制实验。
- `run_real.py`：真实固定审批与校正交叉实验。
- `results/research_v2/`：各版本结果。最终采用哪个目录，以顶层实验汇总的明确指针为准；原型结果不混入最终汇总。

运行需要 numpy、scipy、scikit-learn、pandas、xgboost、matplotlib 和读取 UCI Excel 的 xlrd。本轮依赖锁定另附于实验输出。GPU 运行需在具有 NVIDIA 设备访问权限的宿主环境执行。

可使用 `EXPERIMENT_PYTHON=/path/to/python bash experiments/research_v2/reproduce_small.sh /new/output/directory` 重跑相同的小规模范围。脚本拒绝覆盖已有输出。若没有数据，先用同一个 Python 环境运行 `fetch_data.sh /local/data/cache`，再核对新来源哈希。当前机器已取得数据，不需要再次下载。

开发结果只用于检查机制与方法是否值得继续，不能把两个或三个种子的均值当成确认结论，不能声称已获得 CCF B/C 等级或 SOTA。

## 参考

- [BAIT 作者论文](https://proceedings.neurips.cc/paper/2021/file/4afe044911ed2c247005912512ace23b-Paper.pdf)
- [ParBaLS 作者实现及配置](https://github.com/ADDAPT-ML/ParBaLS)
- [SAS 官方 HMEQ](https://github.com/sassoftware/sas-viya-dmml-pipelines/blob/master/data/hmeq.csv)
- [UCI 官方数据](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- [仓库记录的 LendingClub 样本来源](https://bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/)

## GitHub 发布范围

本次 GitHub 提交只包含汇总和逐轮统计，不包含逐样本选样、概率、伪目标及 test_predictions.npz。`verification.json` 描述完整本地运行的核验；使用 `verify_results.py` 前需先按复现脚本本地生成完整结果。详见 [发布范围](../../results/research_v2/PUBLICATION.md)。
