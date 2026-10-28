# MoonEP + 910C 验证记录（2026-10-28）

本轮参照仓库外 `MoonEP_910c_archive_20261028_113346/files/docs/moonep_ascend_910c_guide.md` 复测。节点来自 `/home/jianzhnie/llmtuner/llm/hyper-parallel-robin/node_list.txt`；以下均为各节点单机 8 卡测试，**没有验证跨机器 EP**。容器为 `hyper-parallel-env`，CANN 9.1.0、Python 3.12、torch / torch_npu 2.12.0。主机扩展和 a2 内核均已重新构建。

## 修复内容

- 对称堆必须显式初始化；记录并核对进程组、设备、rank 和 world size，防止子组意外使用 WORLD。四进程测试覆盖仅 ranks 2/3 初始化的非零根子组。
- 主机 planner 拒绝负计数、不一致的本地计数／路由直方图及 int32 溢出。
- Golden plan 保留完整去重元数据；不再仅拷贝 dst 后重建不完整 plan。
- all2all combine backward 对 hidden 求和，并单独返回路由权重，移除错误的 hidden × weight。
- grad-reduce 本地梯度采用 `[epn,H,Hp]`；计时统计用 float32，规避 910C 不支持的 float64 MaxDim。
- combine 输出 UB 被 Vector Cast／Scalar SetValue 覆盖时，分别使用 MTE3_V／MTE3_S 依赖。原 MTE3_MTE2 不能保护这两类写入。修改后大规模随机与逐列变化输入校验通过，但仍有下述间歇停滞，不能视为完整稳定性认证。
- combine 每轮计时前重新 dispatch；grad-reduce 每轮恢复非零输入，避免原地归并或清零使后续迭代输入失真。恢复操作在事件计时外完成。
- 数值检查失败通过 collective 通知所有 rank，防止单个 rank 失败后其余 rank 继续计时。UDMA-only 脚本在 910C 明确跳过。

## 计时口径

Golden plan 在计时外生成，`plan_us` 为 n/a，不能解释为零规划开销。shmem 模式逐轮记录 NPU 事件，轮间 host group barrier 在事件区间外；禁用 graph。combine 输入恢复不计入 combine。p50/p90 是各 rank 分位数再聚合，非全部 rank 样本合并分位数。

节点有其他训练任务，HBM 占用约 47–62 GiB/card，性能波动明显。以下耗时仅用于检查执行路径，不提供稳定性能或加速比结论；没有停止其他人的任务。

## 正确性与构建

原始证据目录：`/home/jianzhnie/llmtuner/tmp/moonep_verify_20261028/`。日志名均相对此目录。历史失败日志保留，不能混入通过结果。

| 检查 | 节点 | 结果／证据 |
| --- | --- | --- |
| 主机扩展、a2 内核构建 | .42 等 | `build.log`、`kernel_event_build.log` 完成 |
| planner / heap scope / benchmark contract / 计时恢复回归 | .90 | `regression_latest.log`：10 passed |
| 非零根 EP 子组初始化与 WORLD 重用拒绝 | .29 | `subgroup.log`：4 ranks 各 1 passed |
| dispatch / combine / joint | .29 等 | `dispatch.log`、`combine.log`、`joint.log`：ALL RANKS PASS |
| Buffer 示例 | .29 | `example_final.log`：EXAMPLE PASS |
| prefetch 正确性 | .92 | `prefetch_correctness.log`：各 rank 9 passed，2 deselected |
| grad-reduce 正确性 | .92 | `grad_correctness.log`：各 rank 13 passed，2 deselected |
| 随机生产形状全量 joint | .92 | `joint_random_prod.log`：ALL RANKS PASS；S4096/K16/E896/H7168，全部 4096 行 |
| 逐列变化 payload 全量 dispatch / combine | .29 | `debug_pattern.log`：8 ranks DISPATCH_BAD=0、COMBINE_BAD=0；S8192/K8/E896/H3584 |

prefetch／grad-reduce pytest 中 deselected 的生产大张量和 i64 大张量案例未验证，不以小形状通过代替。随机 joint 的 bf16 归并存在中间舍入，按绝对和相对误差阈值判断；权重精确相等。

## Benchmark 覆盖

| 脚本／配置 | 节点 | 结论与证据 |
| --- | --- | --- |
| bench_host_plan.py，逻辑 R8/E128/K8，S64/1024/4096/8192，warmup20/iters100 | .29 等 | CPU 与传输包含模式各 4 shapes 完成；`host_cpu.csv`、`host_npu.csv` |
| bench_barrier.py，8 ranks | .92 等 | 完成；`barrier.log` |
| bench_prefetch.py，默认 14 cases，warmup20/iters100 | .29 | 非零 payload 数值与计时全部通过；`prefetch_checked.log`；最终逐轮同步版本 `prefetch_latest.log` 也完成全部 14 项 |
| bench_grad_reduce.py，默认 14 cases，warmup20/iters100 | .29 | 非零梯度及 consumed slots 检查全部通过；`grad_reduce_checked.log`；早期计时未逐轮恢复输入；最终恢复非零输入版本 `grad_restored.log` 也完成全部 14 项（warmup3/iters10，.92） |
| bench_dispatch_maxvio.py，E896/K8/H3584，MaxVio .2/1/10/20，warmup20/iters100 | .90 | 四组计时完成；`dispatch_maxvio.log/csv`；没有单独逐组 full payload 检查 |
| bench_grad_reduce_maxvio.py，E896/K8/H3584/Hp256，MaxVio .2/1/10/20 | .90 | 四组早期计时完成；最终逐轮非零恢复版本也全部完成（.92，warmup3/iters10），见 `grad_maxvio_restored.log/csv` |
| bench_vs.py，E896/K8/H3584/Hp256，MaxVio .2/1/10/20，warmup20/iters100 | .90 | 最终计时恢复路径完成；`vs_final.log/csv`；数值由同实现 all2all 对比和 joint 另行验证 |
| bench_vs_all2all.py，E896/K8/H3584/Hp256，同四组 MaxVio，warmup20/iters100 | .29 | 两种实现每组完整 hidden / weights roundtrip 通过，计时完成；`all2all_final.log/csv` |
| bench_comm.py，S8192/K8/E896/H3584，warmup2/iters5 | .42 | 内核事件修复后数值及计时完成；`event_fix.log/csv`；发生在逐轮恢复改动前 |
| bench_comm.py，--prod，warmup3/iters10 | .92 | 最终恢复输入路径的完整随机校验与四方向计时通过；`prod_retry.log/csv` |
| bench_comm.py，普通及 --prod 的最终长循环 | .42、.122 | 存在停滞／超时，不能标为通过；`comm_final.log`、`comm_retry.log`、`prod_final.log`（该长循环停滞后主动终止） |
| bench_vs_all2all.py，README 形状 E384/K8/H7168/Hp256 | .29、.18 | 存在停滞／超时，不能标为全部完成；`readme_final.log`、`readme_retry.log` |
| bench_vs_all2all.py --percentiles，E896/K8/H3584/Hp256，warmup3/iters10 | .90、.116 | 早期 `percentiles.log/csv` 完成；最终路径复测存在停滞，见 `percentiles_final.log`、`percentiles_retry.log` |
| bench_combine_udma_maxvio.py | .92 等 | SKIP：仅 Ascend950 UDMA；`udma_skip.log` |

## 未解决项与历史数据

增加 host 同步并未消除全部间歇停滞；部分 traceback 停在 dispatch 后 `.cpu()`，部分停在 combine 输出拷回或 collective，尚不能断定是内核同步、运行时还是并发作业资源竞争。降低为 8 AIV 的 `comm_sms8.log` 仍超时，不能靠减少核心数确认解决。`.37` 尝试在 set_device 即因 HBM 不足失败（`comm_trace_final.log`），不能将其记为算法错误或成功。需要空闲节点上的重复运行进一步定位。

修复前的大 shape 随机校验曾有约 240 / 29360128 个 hidden 元素错误（`debug_serial.log`、`comm_prod_fixed.log`）；不能用四舍五入后的 0.0% 忽略。旧 guide 中的 bench_comm / all2all 数字不作为当前正确性的证据。当前只能确认上表通过的参数与路径，**不能宣称所有 benchmark 全部正确或稳定**。
