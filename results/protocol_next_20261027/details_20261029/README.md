# 实验详细数据补传（2026-10-29）

本目录补充已经完成实验的明细，不新增实验，也不改变原结论。先读[实验清单完成总报告](../md_execution_complete_20261029/实验清单完成总报告.md)。

- [全部768条轨迹](all_768_episodes.csv)：主/补充预算720条、三任务配置迁移36条、2016后续队列12条。
- [全部7680轮记录](all_7680_rounds.csv)：每轮Brier、分组指标、选取数量、预算与风险账本、后端和选集哈希。
- [完整文件索引与SHA-256](DETAILS_INDEX.json)：列出所有独立tar.zst归档、原文件数量和压缩前后大小。每个tar.zst归档内还包含逐文件校验清单。
- `archives/formal_<任务>_fold<折号>.tar.zst`：两次划分重复和两次回放，包含选集、D/O/C/R行号、候选成本、划分、初始调参记录、逐轮日志、测试标签及预测。
- `archives/transfer_*.tar.zst`、`later_2016.tar.zst`：迁移及后续队列的同类明细。
- `archives/earlier_*.tar.zst`：早期诊断的逐集合结果、选集、π/Γ、伪目标、预测和模拟世界参数等已保存证据。共享预测缓存按文件哈希分四组，均可独立解压，无二进制分片。
- `archives/all_stage_tables_configs_and_audits.tar.zst`：各阶段完整表格、配对明细、配置、审计、源码快照和报告。

`experiment_profile`与`record_id`区分不同阶段中重用的case/episode名称，合表时请保留。归档路径相对于`outputs/next_protocol_20261027`，全部tar.zst归档可解压到同一目录，路径不重叠。用tar --zstd -xf 文件.tar.zst解压（需要zstd），预测数组用NumPy读取。

第三方原始信用特征输入、环境和凭据未上传。历史DOCX补充文件夹（ordinary_factorial_small、approval_mechanism_small、nearest_neighbor_small、approval_volume_cost、nonlinear_mechanism_small）仅作为已做工作的历史记录，不计入指定Markdown清单的768条轨迹。所有不利结果完整保留。未采集的同状态边际基线和单次拟合耗时保持缺失；不同策略历史下的差值不能当成同状态边际收益。统计区间仍只描述已训练预测的条件敏感性，2016队列仍受标签可得时点缺失限制。

下载和校验：将本目录整体下载后运行`python verify_details.py`。也可单独下载需要的tar.zst归档及DETAILS_INDEX.json，运行`python verify_details.py --allow-missing`；缺失包会明确列出，不会当成完整交付。

用户已明确授权公开逐样本标签、选集行号与预测等全部上述派生明细。为减少下载体积，公开归档改用跨文件压缩的tar.zst；各成员字节与原ZIP完全一致，已解压逐文件核验。
