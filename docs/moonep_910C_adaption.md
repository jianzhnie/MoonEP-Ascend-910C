# MoonEP 适配 Ascend 910C：实现过程与 Benchmark

910C 适配采用 **主机规划器 + MTE 通信内核 + shmem 普通对称堆**。单机 8 卡生产形状的随机数据检查和主形状四组路由倾斜测试已通过；部分长循环仍间歇停滞，尚不能视为全部路径的稳定性认证。

## 1. MoonEP 的数据流

MoE 每层为每个 token 选择 K 个专家。专家分布在 R 个 rank 上，MoonEP 通过动态冗余专家均衡负载，并将通信与 token 重排合并。

| 阶段 | 工作 | 910C 实现 |
| --- | --- | --- |
| planning | 根据全组专家计数决定 token 落点、冗余专家及去重关系 | C++ 主机 planner；benchmark 使用 reference golden plan |
| dispatch | 将 hidden 和路由权重写入目标 rank 的专家分组缓冲 | AscendC MTE put；去重传输后本地展开重复行 |
| prefetch | 为冗余专家复制权重 | MTE get |
| 专家计算 | 上层框架执行专家网络 | 不包含在本文通信 benchmark 中 |
| combine | 预归并重复行，再按源 token 的 K 条路由规约 hidden | MTE get + 本地归并；路由权重单独收集 |
| grad-reduce | 将冗余专家梯度归还 home rank | MTE 通信与本地梯度累加 |

本库 combine 接口对 hidden 求和，不在内部乘路由权重。模型所需的加权应按上层计算图安排，不能在对比基线中额外乘权重。

后文中 S 为每 rank token 数，H 为 hidden 维度，E 为专家总数，K 为 top-k，`epn=E/R`，Hp 为专家权重内维度。Ascend 路径的固定通信行数为 `NvS=S×K`。MaxVio 为全局专家负载的 `max/mean−1`，与 `bench_comm` 的 `unbalance_ratio` 不同。

## 2. 适配过程

### 2.1 先拆分芯片能力与构建目标

原 Ascend950 路径依赖 SIMT planner、UDMA/URMA，以及 VMM user-buffer 跨进程共享。本次 910C/CANN 9.1 环境无法直接使用这些路径。

| 原依赖 | 遇到的问题 | 适配方式 | 主要文件 |
| --- | --- | --- | --- |
| a5 / dav-c310 构建 | SIMT 源码无法用于 a2 / dav-c220 | CMake 按架构选择源码 | `npu_patch/native/CMakeLists.txt` |
| SIMT planner / UDMA 入口 | 排除源码后缺少链接符号 | 提供明确报错的入口桩；运行时选 MTE | `native/kernels/non_simt_stubs.cpp`、`api.py` |
| VMM user-buffer 共享 | fabric handle／导入能力不匹配，曾报 507899 | 普通 shmem 堆作为通信内存 | `buffer_c.py`、`native/csrc/acl_shared_buffer.cpp` |
| 设备 barrier | SyncAll 缺 FFTS 配置，曾报 507015 | A2/A3 使用 GM flag 软核间同步 | `native/include/ascend_moonep/kernels/shmem_barrier.h` |
| UB staging 容量 | A5 配置不适用于 dav-c220 | 非 3510 目标采用较小 staging | `native/kernels/moonep_dispatch.cpp` |

表中 `native/`、`api.py`、`buffer_c.py` 均位于 `npu_patch/` 下。Ascend950 分支通过条件编译和 SoC 检测保留，本轮未在 950 上回归。

### 2.2 用普通对称堆替换 VMM 通信共享

设置 `ASCEND_MOONEP_HEAP=shmem` 后，通过 shmem 分配器 `aclshmem_align` 获取对称 chunk。不能从 heap base 自建 bump 分配器，否则可能覆盖 shmem 的内部同步区。

初始化需要所有 EP rank 协调参与，顺序为：

1. 设置本地 NPU，初始化 HCCL 进程组。
2. 构造 `Buffer(..., group=ep_group, use_udma=False)`，或在独立分配前调用 `ensure_shmem_heap(total_bytes, group=ep_group)`。
3. 调用 `ShmemRuntime.init_with_buffer(group=ep_group, enable_udma=False)`，再使用对称视图。
4. 结束时同步设备，先 `ShmemRuntime.finalize()`，再销毁 Buffer 和进程组。

堆记录进程组、设备、rank、world size；重复初始化必须属于同一范围。分配不再隐式使用 WORLD。非零根子组测试覆盖仅 ranks 2/3 初始化的场景。完整调用示例见 `test_npu/mte/example_a3_buffer_usage.py`。

普通堆按进程生命周期共享，chunk 销毁不单独回收；切换配置需协调 finalize。堆预算还需预留 2 MiB 对齐和运行时开销，不能只累加张量裸字节数。

### 2.3 将规划移到主机

910C 构建不提供设备侧 SIMT planner。主机路径汇总 `[R,E]` 计数，执行负载再均衡、复制专家选择和落点规划，再将 plan 送回设备。

- `native/csrc/host_planning.{h,cpp}` 实现主机算法。
- `native/csrc/pybind.cpp` 导出 `_C.plan_host`。
- `planning_host.py` 提供 `build_plan_host(counts_all, local_counts, route_ids, rank, epn, nvs)` 包装。
- 输入检查拒绝负计数、本地计数与全组计数不一致、路由直方图不一致及 int32 索引溢出。

`build_plan_host` 不会自动替换 `Buffer.dispatch` 的设备 planner。上层仍需负责计数汇总、plan 封装和完整去重元数据。benchmark 的 `--plan-source golden` 使用 `test_npu/planing_reference.py`，不能将它的计时外规划解释为训练中零规划成本。

主机规划仍引入逐层同步和数据传输。固定 buffer 形状不能消除该成本；本次测量见第 4 节主机规划表。

### 2.4 修复正确性与计时输入

| 问题 | 修复 | 验证 |
| --- | --- | --- |
| Golden 模式仅拷贝 dst，去重元数据不完整 | 保留完整 reference plan，并避免重复编码／重建 | 四组 MaxVio 全张量 roundtrip |
| combine 输出 UB 复用的依赖指向 MTE2 | Vector Cast 使用 MTE3_V，Scalar 写权重使用 MTE3_S | 生产形状随机输入及逐列变化输入全量检查 |
| all2all combine backward 额外乘权重 | hidden 求和与权重收集分离 | CPU 合同回归及 8 卡 roundtrip |
| 本地梯度以 `[E,H,Hp]` 分配 | 改为 `[epn,H,Hp]`，匹配本地专家所有权 | 梯度数值与槽位清零检查 |
| 重复 combine／grad-reduce 使用已被修改的输入 | 每轮计时前恢复输入，恢复操作放在事件区间外 | 计时恢复回归、14 项梯度案例 |
| float64 统计触发 910C MaxDim 不支持 | 计时统计使用 float32 | 实机 benchmark |
| 单 rank 校验失败，其余 rank 继续 collective | collective 汇总校验状态并统一失败 | 失败日志中可见全组退出 |


## 3. 构建与验证

使用匹配版本的 CANN、torch、torch_npu 和 shmem。构建 shmem、CMake a2 内核以及 `_C` 扩展的完整命令见 [构建说明](../README_910C.md#构建)。当前 `build_base.sh` / `build_a5.sh` 仅接受 Ascend950，910C 使用显式 CMake a2 步骤。torch / torch_npu 版本变化后需重编译 `_C`。

在仓库根目录设置：

```bash
export ASCEND_MOONEP_HEAP=shmem
export ASCEND_MOONEP_HEAP_BYTES=4294967296
export HCCL_NPU_SOCKET_PORT_RANGE=65400-65431 OMP_NUM_THREADS=2

python -m pytest -q test_npu/test_host_planning.py \
  test_npu/test_heap_scope.py test_npu/test_benchmark_contracts.py
python test_npu/mte/test_dispatch_combine_joint.py \
  --R 8 --prod-shape --max-checks 0 --random-hidden
```

堆容量按实际 shape 调整；生产随机 joint 的本轮运行使用 8 GiB 堆。端口需避开其他作业。数值比较考虑 bf16 中间舍入，权重按精确相等检查。

## 4. Benchmark

本轮采用单机 8 卡测试，未覆盖跨机器 EP。环境为 CANN 9.1.0、Python 3.12、torch / torch_npu 2.12.0，容器 `hyper-parallel-env`。


### 4.1 计时口径

通信表为跨 rank 平均延迟；prefetch / 默认 grad-reduce suite 报告最慢 rank。

- 使用 NPU events；Golden plan 在计时外构建，规划时间为 n/a，不是零开销。
- shmem 堆模式逐轮计时并禁用 graph，轮间进程组 barrier 位于事件区间外。
- combine 原地归并重复行，每轮在计时外重新 dispatch 恢复输入；grad-reduce
  每轮在计时外恢复非零输入，避免后续迭代只处理已清空的槽位。
- 通信时间不包含主机规划、输入恢复及训练中的专家计算。对比脚本 d_f/d_b
  包含权重 prefetch，不能直接与 bench_comm 的 dispatch 列比较。
- 逻辑带宽为 `S×K×H×2 / t`，不代表物理链路带宽。
- p50/p90 为各 rank 的分位数再按算子规则 mean/max 聚合，不是合并全 rank 样本。

### 4.2 MoonEP 生产形状

S=4096、K=16、E=896、H=7168、ep=8，
`--plan-source golden --warmup 3 --iters 10`。完整随机 hidden / route weights
roundtrip 通过；独立 joint 测试也检查了每 rank 全部 4096 行。

| 方向 | 延迟（ms） | 逻辑带宽（GB/s） |
| --- | ---: | ---: |
| dispatch fwd | 4.418 | 212.7 |
| dispatch bwd | 5.207 | 180.4 |
| combine fwd | 5.507 | 170.6 |
| combine bwd | 5.348 | 175.7 |

### 4.3 MoonEP 与 all2all 主形状对比

S=8192、K=8、E=896、H=3584、Hp=256、ep=8，
warmup20/iters100。两种实现每组均完成完整 hidden 和 route weights roundtrip。

| MaxVio（realized） | MoonEP d_f | all2all d_f | MoonEP c_f | all2all c_f | 时延比 d_f / c_f |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.196 | 2.79 ms (168 GB/s) | 11.44 ms (41 GB/s) | 2.59 ms (182 GB/s) | 13.80 ms (34 GB/s) | ×4.10 / ×5.34 |
| 0.988 | 2.83 ms (166 GB/s) | 11.60 ms (40 GB/s) | 2.58 ms (182 GB/s) | 13.90 ms (34 GB/s) | ×4.10 / ×5.39 |
| 9.985 | 2.85 ms (165 GB/s) | 12.08 ms (39 GB/s) | 2.58 ms (182 GB/s) | 14.91 ms (32 GB/s) | ×4.23 / ×5.77 |
| 19.971 | 2.89 ms (163 GB/s) | 12.92 ms (36 GB/s) | 2.58 ms (182 GB/s) | 15.71 ms (30 GB/s) | ×4.47 / ×6.08 |


### 4.4 专家权重、梯度与主机规划

`bench_prefetch.py` 默认 14 cases 的非零 payload 校验与计时通过：
warmup20/iters100（`prefetch_checked.log`），以及最终逐轮同步版本
warmup3/iters10（`prefetch_latest.log`）。
`bench_grad_reduce.py` 默认 14 cases 的非零梯度和 consumed slots 校验通过，
最终逐轮恢复输入版本在 warmup3/iters10 全部完成（`grad_restored.log`）。
早期 `grad_reduce_checked.log` 的计时未逐轮恢复，不能与最终计时混用。

梯度 MaxVio 扫描：E896/K8/H3584/Hp256、warmup3/iters10，
每轮恢复非零输入。

| 实际 MaxVio | grad-reduce（ms） | 最大接收 slots |
| ---: | ---: | ---: |
| 0.196 | 1.534 | 1 |
| 0.988 | 1.947 | 2 |
| 9.985 | 2.958 | 3 |
| 19.971 | 1.473 | 3 |

`bench_dispatch_maxvio.py` 在四组 MaxVio、warmup20/iters100 已完成计时
（`dispatch_maxvio.log/csv`），该脚本未逐组增加 full payload 检查。
`bench_barrier.py` 8 ranks 完成（`barrier.log`）。

主机规划器单进程测量，模拟逻辑 R8/E128/K8，warmup20/iters100，
**不是 8 rank collective 端到端耗时**。NPU 模式包含输入回传 CPU 和结果回传 NPU。

| 每 rank S | CPU 计算（μs） | 包含 NPU 传输（μs） |
| ---: | ---: | ---: |
| 64 | 66.0 | 370.4 |
| 1024 | 164.9 | 477.3 |
| 4096 | 558.1 | 892.1 |
| 8192 | 1051.7 | 1392.2 |


## 6. Benchmark 复跑命令

在仓库根目录运行，预留堆容量与端口：

```bash
export ASCEND_MOONEP_HEAP=shmem ASCEND_MOONEP_HEAP_BYTES=4294967296
export HCCL_NPU_SOCKET_PORT_RANGE=65400-65431 OMP_NUM_THREADS=2
bash npu_patch/benchmarks/run_bench.sh bench_comm.py \
  "--prod --plan-source golden --warmup 3 --iters 10 --out /tmp/prod.csv"
bash npu_patch/benchmarks/run_bench.sh bench_vs_all2all.py \
  "--libs moonep all2all --experts 896 --topk 8 --hidden 3584 --hp 256 \
   --plan-source golden --maxvios 0.2,1,10,20 --warmup 20 --iters 100 \
   --out /tmp/vs.csv"
bash npu_patch/benchmarks/run_bench.sh bench_grad_reduce_maxvio.py \
  "--experts 896 --topk 8 --hidden 3584 --hp 256 --plan-source golden \
   --maxvios 0.2,1,10,20 --warmup 3 --iters 10 --out /tmp/grad.csv"
python npu_patch/benchmarks/bench_host_plan.py --device cpu --out /tmp/host_cpu.csv
python npu_patch/benchmarks/bench_host_plan.py --device npu --out /tmp/host_npu.csv
```

