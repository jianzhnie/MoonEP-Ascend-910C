# MoonEP 适配 Ascend 910C：架构设计、实现与性能验证

MoE 专家并行的效率受到路由负载、数据布局和跨卡传输量的共同影响。MoonEP 通过动态复制热点专家均衡负载，将通信与重排融合，并减少发往同一目标 rank 的重复载荷。迁移到 Ascend 910C，需要围绕这些机制重新选择规划器、内存管理和同步方式。

本文基于当前代码梳理 **主机规划 + AscendC MTE + CANN shmem 普通对称堆** 的实现路径。性能数据来自单机 8 卡测试；文档修订未重新运行硬件测试。现有记录包含通过的数值检查和计时结果，也存在尚未定位的间歇停滞。

## 1. 从路由负载到通信布局

通用 all-to-all 提供数据交换原语，但专家分组、token 重排、重复载荷消除与负载均衡通常需要上层另行实现。热点专家可能同时增加其所在 rank 的接收量和计算量；动态接收布局还会引入容量管理与同步成本。

MoonEP 将热点专家的部分计算迁移到持有其临时副本的 rank，并规划对应的 token 落点。均衡路由条目数有助于稳定专家计算形状，但不能保证各 rank 的实际执行时间相同：权重搬运、链路竞争和算子效率仍会影响延迟。

### 1.1 数据流与接口语义

每个 token 选择 K 个专家，通信过程围绕同一份路由计划组织：

| 阶段 | 工作 | 910C 实现 |
| --- | --- | --- |
| planning | 根据全组专家计数决定 token 落点、冗余专家及去重关系 | C++ 主机 planner；benchmark 使用 reference golden plan |
| dispatch | 将 hidden 和路由权重写入目标 rank 的专家分组缓冲 | AscendC MTE put；去重传输后本地展开重复行 |
| prefetch | 为冗余专家复制权重 | MTE get |
| 专家计算 | 上层框架执行专家网络 | 由上层框架负责 |
| combine | 预归并重复行，再按源 token 的 K 条路由规约 hidden | MTE get + 本地归并；路由权重单独收集 |
| grad-reduce | 将冗余专家梯度累加到 home rank | 远端读取、本地累加与槽位清零 |

这里的 combine **对 hidden 求和，并独立收集路由权重**；内核不额外执行 hidden × 路由权重。模型需要的加权应由上层计算图安排，对比基线必须采用相同语义。

后文记每 rank token 数为 S、hidden 维度为 H、专家总数为 E、top-k 为 K、EP 大小为 R，`epn=E/R`，专家权重内维度为 Hp。每 rank 有 `S×K` 个路由条目，而非 `S×K` 个不同 token。当前 Ascend 通信容量为 `NvS=S×K`，`token_padding` 不改变该逻辑行数。

### 1.2 去重与零拷贝的边界

同一 token 的多个目标专家若落在同一目标 rank，dispatch 只需传输一份 hidden，再在目标端展开到各专家槽位；路由权重仍逐条发送。combine 在本地预归并重复组后收回主行，减少跨卡载荷。

直写专家布局减少了通信与重排之间的中间拷贝。API 是否返回共享视图则由参数决定：`Buffer.dispatch` 的 `zero_copy` 与 `router_weights_zero_copy` 默认均为 False。使用零拷贝视图时，调用方需要管理别名、数据生命周期和同步。

### 1.3 Plan 是通信算子的共同契约

plan 同时描述路由落点和本地布局，不能只将它理解为一张目标 rank 表。当前 rank 去重路径的关键字段如下：

| 字段 | 含义与用途 |
| --- | --- |
| `dst` | 每条路由的目标编码；非负值为 `目标 rank × NvS + 本地偏移`，负值用 `-raw_dst-1` 标记重复载荷 |
| `experts_to_copy` | 各 rank 的冗余专家列表，-1 表示空槽，供权重预取和梯度规约使用 |
| `cu_seqlens` | 专家分组的累计边界；由规划阶段返回，与 plan 配合使用 |
| `dup_groups` / `dup_loffs` / `dup_counts` | 主槽、重复槽列表及有效计数，供 dispatch 展开和 combine 预归并使用 |
| `zero_fill_ranges` | 规划得到的清零区间；具体布局需与调用方的专家计算一致 |

例如，一个 token 同时路由到目标 rank 上的两个专家，两个落点仍然存在，但只有一个条目触发 hidden 传输；另一个使用负编码，随后通过去重表在本地展开。负号表达的是载荷去重，不是丢弃该路由。

同一次路由的 plan 可在前向和反向通信中复用，路由改变后必须重建。复用 plan 的 dispatch 可能不返回新的 `cu_seqlens`，调用方应保留规划阶段的边界信息。**复用 plan 不等于复用未恢复的数据**：combine 的本地预归并会修改输入缓冲，重复运行前应重新填充有效载荷。

## 2. 适配的四个关键环节

### 2.1 按架构拆分构建目标

适配首先拆分设备能力。当前 a5 构建包含 SIMT planner 和 UDMA 内核；a2 构建保留 MTE 内核，并为不可用入口提供显式报错的桩函数。以下结论针对本仓库与已验证的 CANN 9.1 环境，不泛化为所有版本的硬件能力说明。

| 模块     | 原路径的依赖                  | 910C 的选择                               | 关键代码                                                     |
| -------- | ----------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| 构建     | a5 / SIMT 源码                | a2 / dav-c220，排除 SIMT 与 UDMA 源码     | [CMakeLists.txt](../npu_patch/native/CMakeLists.txt)         |
| 规划     | SIMT 设备 planner             | 主机 C++ planner；测试使用 reference plan | [planning_host.py](../npu_patch/planning_host.py)            |
| 数据传输 | UDMA/URMA                     | AscendC MTE put/get                       | `npu_patch/native/kernels/`                                  |
| 片上缓冲 | A5 staging 配置               | 按架构收缩 dispatch staging               | [moonep_dispatch.cpp](../npu_patch/native/kernels/moonep_dispatch.cpp) |
| 对称内存 | VMM user-buffer 共享          | shmem 普通堆                              | [buffer_c.py](../npu_patch/buffer_c.py)                      |
| 设备同步 | vector barrier / SyncAll 路径 | shmem barrier 与 GM flag 软核间同步       | [shmem_barrier.h](../npu_patch/native/include/ascend_moonep/kernels/shmem_barrier.h) |

Ascend950 的 SIMT / UDMA 分支继续保留，本次验证范围限于 910C。

### 2.2 使用普通对称堆管理通信内存

设置 `ASCEND_MOONEP_HEAP=shmem` 后，通过 shmem 分配器 `aclshmem_align` 获取对称 chunk。不能从 heap base 自建 bump 分配器，否则可能覆盖 shmem 的内部同步区。

堆记录进程组、设备、rank、world size；重复初始化必须属于同一范围。分配不再隐式使用 WORLD。非零根子组测试覆盖仅 ranks 2/3 初始化的场景。完整调用示例见 `test_npu/mte/example_a3_buffer_usage.py`。

普通堆由进程内运行时共享，chunk 不单独回收，切换配置需协调 finalize。当前分配器请求 2 MiB 对齐，而 Python 主要按申请字节数记账，容量预算还需预留对齐间隙与内部开销。

设备同步还有一项独立的平台适配：当前 a2 构建没有配置 SyncAll 所需的 FFTS 同步基址，因此 `shmem_barrier.h` 在引入 shmem 头文件时选择 GM flag 软核间同步，跨 rank 仍调用 shmem barrier。该机制与第 2.4 节的 UB 流水线事件分别解决跨核同步与缓冲依赖，不能相互替代。

### 2.3 明确主机规划的接入边界

910C 构建不提供设备侧 SIMT planner。上层汇总 `[R,E]` 计数后，由主机执行负载再均衡、复制专家选择和落点规划，再将结果传回设备。

- `native/csrc/host_planning.{h,cpp}` 实现主机算法。
- `native/csrc/pybind.cpp` 导出 `_C.plan_host`。
- `planning_host.py` 提供 `build_plan_host(counts_all, local_counts, route_ids, rank, epn, nvs)` 包装。
- 输入检查拒绝负计数、本地计数与全组计数不一致、路由直方图不一致及 int32 索引溢出。

`build_plan_host` 不会自动替换 `Buffer.dispatch` 的设备 planner。上层仍需负责计数汇总、plan 封装和完整去重元数据。benchmark 的 `--plan-source golden` 使用 `test_npu/planing_reference.py`，不能将它的计时外规划解释为训练中零规划成本。

主机规划仍引入逐层同步和数据传输。固定 buffer 形状不能消除该成本；相关测量见第 4.4 节。

### 2.4 建立正确的数据依赖与测试契约

| 问题 | 修复 | 验证 |
| --- | --- | --- |
| Golden 模式仅拷贝 dst，去重元数据不完整 | 保留完整 reference plan，并避免重复编码／重建 | 四组 MaxVio 全张量 roundtrip |
| combine 输出 UB 复用的依赖指向 MTE2 | Vector Cast 使用 MTE3_V，Scalar 写权重使用 MTE3_S | 生产形状随机输入及逐列变化输入全量检查 |
| all2all combine backward 额外乘权重 | hidden 求和与权重收集分离 | CPU 接口语义回归及 8 卡 roundtrip |
| 本地梯度以 `[E,H,Hp]` 分配 | 改为 `[epn,H,Hp]`，匹配本地专家所有权 | 梯度数值与槽位清零检查 |
| 重复 combine／grad-reduce 使用已被修改的输入 | 每轮计时前恢复输入，恢复操作放在事件区间外 | 计时恢复回归、14 项梯度案例 |
| float64 统计触发 910C MaxDim 不支持 | 计时统计使用 float32 | 实机 benchmark |
| 单 rank 校验失败，其余 rank 继续 collective | 汇总断言状态后统一失败 | 失败日志中可见全组退出 |

combine 的输出缓冲复用尤其需要区分流水线依赖。MTE3 写回尚未完成时，Vector Cast 和 Scalar 写入分别通过 `MTE3_V`、`MTE3_S` 等待，`MTE3_MTE2` 不能约束这两类覆盖操作。相关修复见 [moonep_combine.cpp](../npu_patch/native/kernels/moonep_combine.cpp)。

校验状态汇总只捕获 Python 断言失败，不能保证设备异常或 collective 超时后的全组退出。

## 3. 构建与运行

完整依赖及构建步骤见 [910C README](../README_910C.md#构建)。当前 `build_base.sh` 和 `build_a5.sh` 只接受 Ascend950；910C 使用显式 CMake a2 构建。shmem、内核库与 `_C` 扩展应使用匹配的工具链，torch / torch_npu 变化后需检查 ABI 并重编译扩展。

普通堆的调用顺序如下，各步骤使用同一 EP 进程组：

1. 设置本地 NPU，初始化 HCCL 进程组。
2. 构造 `Buffer(..., group=ep_group, use_udma=False)`；独立分配前也可显式调用 `ensure_shmem_heap(total_bytes, group=ep_group)`。
3. 调用 `ShmemRuntime.init_with_buffer(group=ep_group, enable_udma=False)`，构建完整 plan 后执行算子。
4. 同步设备，先 `ShmemRuntime.finalize()`，再销毁 Buffer 和进程组。

可运行示例见 [example_a3_buffer_usage.py](../test_npu/mte/example_a3_buffer_usage.py)。

在仓库根目录执行以下检查。生产随机联合测试的历史运行使用每 rank 8 GiB 堆；实际预算需按可用内存和分配需求调整。

```bash
export ASCEND_MOONEP_HEAP=shmem
export ASCEND_MOONEP_HEAP_BYTES=8589934592
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61031
export OMP_NUM_THREADS=2

python -m pytest -q test_npu/test_host_planning.py \
  test_npu/test_heap_scope.py test_npu/test_benchmark_contracts.py

python test_npu/mte/test_dispatch_combine_joint.py \
  --R 8 --prod-shape --max-checks 0 --random-hidden

torchrun --standalone --nproc_per_node=2 \
  test_npu/mte/example_a3_buffer_usage.py
```

联合测试由脚本自行创建进程，不再套 torchrun；Buffer 示例使用 torchrun。示例通过只说明调用链、形状与有限值检查通过，不等于训练收敛验证。

## 4. Benchmark：结果与口径

### 4.1 环境与计时方法

测试采用单机 8 × Ascend910_9392，环境为 CANN 9.1.0、Python 3.12、torch / torch_npu 2.12.0，未覆盖跨机器 EP。测试期间存在并发训练任务，下表用于记录执行表现，不作为稳定加速比或版本回退的依据。

主对比与生产形状使用 NPU events。公共计时器在 shmem 模式逐轮计时、禁用 graph，轮间进程组 barrier 位于事件区间外。combine 的输入恢复和 grad-reduce 的非零槽位填充也位于事件区间外。独立脚本可能采用自己的计时函数，不能将该口径推广到全部 benchmark。

| 指标          | 解释                                                         |
| ------------- | ------------------------------------------------------------ |
| `d_f` / `d_b` | 对比脚本的 dispatch；MoonEP 两者均包含 prefetch，`d_b` 使用保存的 plan 且不传路由权重 |
| `c_f` / `c_b` | hidden 归并；bwd 变体还收集路由权重                          |
| `plan_us=n/a` | golden plan 在计时外构建，不代表规划无开销                   |
| 逻辑带宽      | `S×K×H×2 / t`，不能当作链路实际传输带宽                      |
| MaxVio        | 全局专家负载的 `max/mean−1`，表中报告实际值                  |
| p50/p90       | 公共计时器对各 rank 分位数再聚合，并非全组样本合并分位数     |

通信表使用跨 rank 平均时间；默认 prefetch / grad-reduce suite 使用最慢 rank。主机规划微基准使用墙钟时间。下表不包含专家计算、完整训练步和 golden plan 生成。

### 4.2 MoonEP 生产形状通信延迟

S=4096、K=16、E=896、H=7168、ep=8，`--plan-source golden --warmup 3 --iters 10`。完整随机 hidden / route weights roundtrip 通过；独立 joint 测试也检查了每 rank 全部 4096 行。

| 方向 | 延迟（ms） | 逻辑带宽（GB/s） |
| --- | ---: | ---: |
| dispatch fwd | 4.418 | 212.7 |
| dispatch bwd | 5.207 | 180.4 |
| combine fwd | 5.507 | 170.6 |
| combine bwd | 5.348 | 175.7 |

### 4.3 MoonEP 与 all2all 主形状对比

S=8192、K=8、E=896、H=3584、Hp=256、ep=8，warmup20/iters100。两种实现每组均完成 hidden 和路由权重的全量往返校验。

单位为 ms。prefetch 独立测量，但已包含在 MoonEP 的 `d_f`、`d_b` 内，不应重复相加。all2all 基线没有专家权重池。

| 实际 MaxVio | 实现 | d_f | d_b | c_f | c_b | prefetch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.196 | moonep | 2.844 | 2.766 | 2.740 | 2.874 | 0.258 |
| 0.196 | all2all | 11.353 | 11.254 | 13.854 | 14.010 | n/a |
| 0.988 | moonep | 2.877 | 2.804 | 2.729 | 2.876 | 0.284 |
| 0.988 | all2all | 11.458 | 11.337 | 13.956 | 14.104 | n/a |
| 9.985 | moonep | 2.910 | 2.829 | 2.713 | 2.856 | 0.318 |
| 9.985 | all2all | 11.690 | 11.492 | 14.822 | 15.103 | n/a |
| 19.971 | moonep | 2.938 | 2.851 | 2.728 | 2.866 | 0.365 |
| 19.971 | all2all | 11.615 | 11.438 | 15.599 | 15.923 | n/a |

### 4.4 梯度规约与主机规划开销

梯度规约使用 E=896、K=8、H=3584、Hp=256，warmup 3 / iters 10，每轮恢复非零输入。

| 实际 MaxVio | grad-reduce（ms） | 最大接收 slots |
| ----------: | ----------------: | -------------: |
|       0.196 |             1.534 |              1 |
|       0.988 |             1.947 |              2 |
|       9.985 |             2.958 |              3 |
|      19.971 |             1.473 |              3 |

主机规划微基准在单进程中模拟 R=8、E=128、K=8，warmup 20 / iters 100，统计包装接口的平均墙钟耗时。CPU 列包含 Python 包装与张量转换；NPU 列还包含 D2H、H2D 和设备同步。它没有测量全组计数通信，也不是纯 C++ 内核耗时。

| 每 rank S | CPU 包装接口（μs） | 包含 NPU 传输与同步（μs） |
| --------: | -----------------: | ------------------------: |
|        64 |               66.0 |                     370.4 |
|      1024 |              164.9 |                     477.3 |
|      4096 |              558.1 |                     892.1 |
|      8192 |             1051.7 |                    1392.2 |

## 5. 验证范围与已知限制

已有验证记录包括 CPU 回归 10 项通过，dispatch、combine、joint 和 Buffer 示例通过。生产随机联合测试检查每 rank 全部 4096 行；默认 prefetch 与 grad-reduce suite 各 14 项通过。bf16 输出按误差阈值比较，路由权重精确比较。示例通过不等于训练收敛验证。

| 项目 | 当前证据 |
| --- | --- |
| 普通形状 bench_comm 长循环 | 间歇停滞，降低 AIV 数仍超时 |
| 生产形状 warmup 20 / iters 100 | 未完成，短循环结果不能替代长循环验证 |
| E384/H7168/K8/Hp256 对比 | 未完成全部四组，出现停滞或 SIGABRT |
| 最终 percentiles 路径 | 复测停滞 |
| prefetch / grad-reduce 大张量 pytest | 各 2 项未执行 |
| UDMA-only combine benchmark | 910C 不适用，显式跳过 |
| 跨机器 EP、Ascend950、训练收敛 | 本次未验证 |

停滞涉及 dispatch 后数据拷回、combine 输出读取和 collective，根因尚未确定。后续需要在受控负载下重复运行，区分内核同步、运行时与缓冲复用问题。设备初始化时的显存不足应单独归类，不计入有效性能结果。

## 6. 复现命令

在仓库根目录运行，预留堆容量与端口：

```bash
export ASCEND_MOONEP_HEAP=shmem ASCEND_MOONEP_HEAP_BYTES=4294967296
export HCCL_NPU_SOCKET_PORT_RANGE=65400-65431 OMP_NUM_THREADS=2
bash npu_patch/benchmarks/run_bench.sh bench_comm.py \
  "--prod --plan-source golden --warmup 3 --iters 10 --out ./prod.csv"
bash npu_patch/benchmarks/run_bench.sh bench_vs_all2all.py \
  "--libs moonep all2all --experts 896 --topk 8 --hidden 3584 --hp 256 \
   --plan-source golden --maxvios 0.2,1,10,20 --warmup 20 --iters 100 \
   --out ./vs.csv"
bash npu_patch/benchmarks/run_bench.sh bench_grad_reduce_maxvio.py \
  "--experts 896 --topk 8 --hidden 3584 --hp 256 --plan-source golden \
   --maxvios 0.2,1,10,20 --warmup 3 --iters 10 --out ./grad.csv"
python npu_patch/benchmarks/bench_host_plan.py --device cpu --out ./host_cpu.csv
python npu_patch/benchmarks/bench_host_plan.py --device npu --out ./host_npu.csv
```
