# 原实验计划继续执行记录

用户明确要求继续原实验计划，当前按既定顺序执行比较。2026年10月27日本轮已完成四项：审批约束普通覆盖、真实任务近邻适配、普通反馈数量与成本关联、非线性错设子集。实际新增118次LR、118次CUDA XGB拟合。

总报告：`results/protocol_next_20261027/继续按原计划_本轮四项实验报告.md`。各目录给出逐种子配对、描述汇总、配置与核验。预测与生成数据仍在本地private。

使用原venv，在仓库运行下列runner，各自传 `--base <既有结果根目录>`：

- `run_approval_mechanism.py`，报告用 `analyze_approval_mechanism.py`。
- `run_nearest_neighbor.py`。
- `run_approval_volume_cost.py`，两项报告用 `analyze_plan_continuation.py`。
- `run_nonlinear_mechanism.py`，报告和本轮总账用 `finalize_plan_batch.py`。

输出目录不得复用已完成目录，完整身份不匹配即拒绝。运行需要原授权输入及本地锚点，不声称精简公开材料可无数据开箱重跑。近邻明确为Laplace/成本/pending适配，参数在配置冻结；不是作者原NUTS后验全复现。

后续队列见总报告和plan_execution_state.json：中间覆盖控制、冻结方法完整十轮与近邻比较、按注册协议独立确认。保留阴性结果，不为造出赢家添加新模块。
