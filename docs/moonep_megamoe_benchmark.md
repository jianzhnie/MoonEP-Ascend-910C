# MoonEP 通信 benchmark：MegaMoE 同形状配置

本测试采用 MegaMoE README 表格中的八组路由专家配置：EP=8、top-k=6，Tokens 为每 rank token 数。环境为单机 8 × Ascend 910C、aarch64、Python 3.12.13、torch 2.12.0+cpu、torch_npu 2.12.0、CANN 9.1.0。

## 测试范围与方法

- 测量 BF16 hidden 的 dispatch 前向和 combine 前向，路由权重为 FP32；不包含 grouped GEMM、SwiGLU、shared expert、权重 prefetch 或梯度规约。Intermediate 仅保留参考配置，不影响本次通信算子。
- 路由按每 rank 独立标准正态分数取 top-k，种子为 `23000 + rank`，与 MegaMoE 测试的路由分布一致；路由权重使用 MoonEP 现有随机生成方式。CSV 的 `unbalance_ratio=1.00` 是沿用的配置字段，对 `random-topk` 不生效，实际不均衡以全组专家计数的 Max/Mean 为准。
- 使用 MTE、普通 shmem 堆，每 rank 堆预算 8 GiB；golden 主机 plan 在计时外生成，禁用 graph capture。每个配置独立启动 8 个进程，预热 5 次、计时 20 次，先计算各 rank 的迭代均值，再对 8 rank 求均值。
- 使用 NPU 事件计时，每轮结束在计时区外执行进程组同步。combine 修改输入缓冲，每轮在计时区外重新 dispatch 恢复输入。
- 每组计时前验证 dispatch→combine：hidden 应等于原始 hidden × top-k（BF16，`rtol=0.02, atol=0.125`），收回的路由权重应与输入完全一致。八组均输出 `roundtrip hidden + route weights PASS`。
- 带宽为 `Tokens × top-k × Hidden × 2 / latency`，是每 rank 的逻辑载荷带宽，未扣除本地路由、未按去重后的实际远端字节计算，也不包含路由权重字节，不等于物理链路带宽。

## 实测结果

延迟单位为 µs，带宽单位为 GB/s。D+C 是两个独立计时均值的算术和，不是一次完整前向调用的实测延迟。数据沿用原始 CSV 的有效位数。

| Experts | Hidden | Intermediate（参考） | Tokens/rank | Max/Mean | Dispatch | Combine | D+C | Dispatch GB/s | Combine GB/s | MoonEP GEMM TFLOPS | GEMM 算力利用率（%） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 5120 | 2304 | 64 | 1.469 | 122.8 | 119.7 | 242.5 | 32.03 | 32.84 | N/A | N/A |
| 96 | 5120 | 2304 | 256 | 1.188 | 198.1 | 200.1 | 398.2 | 79.41 | 78.62 | N/A | N/A |
| 96 | 7168 | 3072 | 64 | 1.469 | 157.4 | 164.8 | 322.2 | 34.97 | 33.41 | N/A | N/A |
| 96 | 7168 | 3072 | 256 | 1.188 | 353.5 | 320.9 | 674.4 | 62.29 | 68.61 | N/A | N/A |
| 384 | 5120 | 2304 | 4096 | 1.141 | 2272 | 2587 | 4859.0 | 110.7 | 97.28 | N/A | N/A |
| 384 | 5120 | 2304 | 16384 | 1.062 | 7004 | 7246 | 14250.0 | 143.7 | 138.9 | N/A | N/A |
| 384 | 7168 | 3072 | 4096 | 1.141 | 2432 | 2484 | 4916.0 | 144.9 | 141.8 | N/A | N/A |
| 384 | 7168 | 3072 | 16384 | 1.062 | 9976 | 11961 | 21937.0 | 141.3 | 117.8 | N/A | N/A |

八组各完成一次测试运行，全部通过数值检查，没有超时。96 experts 的小 token 配置带宽为约 32–79 GB/s；384 experts 的大 token 配置为约 97–145 GB/s。当前结果不包含独立多次运行的置信区间，不能据此判断细小性能差异。

## TFLOPS 与算力利用率

通信表的 N/A 保留：dispatch / combine 本身不执行专家 GEMM。完整前向实测和通信时间上的理论等效指标分别如下。

### 完整 BF16 前向实测

实测路径为 `dispatch → grouped GEMM(H→2I) → SwiGLU → grouped GEMM(I→H) → 路由加权 → combine → shared expert 输出相加`；实现中路由加权在第二次 GEMM 后执行，与 BF16 舍入顺序一致。shared expert 为一个本地 BF16 SwiGLU MLP。两个 routed GEMM 调用 `torch_npu.npu_grouped_matmul`，shared expert 使用普通 matmul，未做融合或通信计算重叠。权重初始化、主机 plan、专家副本分发不计时；使用实际活跃专家的常驻权重副本。

计时包含完整算子链、必要的跨 rank 发布同步、激活/输出分配及拷贝，排除每轮结束的进程组同步。预热 5 次、计时 20 次，以 **各 rank 平均延迟的最大值**作为同步前向时间。它是固定路由和常驻专家副本条件下的前向，不包含动态路由规划、权重搬运或训练反向。

每组在计时前检查所有输出有限，并在每个 rank 的第一个 token 上使用独立逐专家 matmul 参考计算验证 routed + shared 输出（`rtol=0.05, atol=0.05`）。这是抽样数值验证，不是全 token 独立参考验证。

利用率按 **376 TFLOPS / 逻辑设备的 BF16 稠密峰值假设**归一化。该值沿用现有训练配置，不是已核实的 Ascend910_9392 官方峰值；表中百分比是条件指标，不是 profiler 测得的 AICore 忙碌率。若采用不同峰值 P，百分比应乘以 `376/P`。EP8 的集群吞吐为表中每 rank TFLOPS 的 8 倍，集群峰值也必须同时乘以 8。

| Experts | Hidden | Intermediate | Tokens/rank | 完整前向 µs（max rank） | GEMM TFLOPS/rank | 利用率 %（假设峰值 376） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 5120 | 2304 | 64 | 1115.61 | 28.42 | 7.56 |
| 96 | 5120 | 2304 | 256 | 1603.06 | 79.12 | 21.04 |
| 96 | 7168 | 3072 | 64 | 1771.25 | 33.42 | 8.89 |
| 96 | 7168 | 3072 | 256 | 2676.94 | 88.44 | 23.52 |
| 384 | 5120 | 2304 | 4096 | 15106.74 | 134.34 | 35.73 |
| 384 | 5120 | 2304 | 16384 | 51884.92 | 156.45 | 41.61 |
| 384 | 7168 | 3072 | 4096 | 28667.25 | 132.14 | 35.14 |
| 384 | 7168 | 3072 | 16384 | 83616.41 | 181.22 | 48.20 |

乘加计为 2 FLOPs，两次 routed GEMM 加一个 shared expert 的平均运算量为 `F = 6×H×I×S×(K+1) = 42×S×H×I`；`TFLOPS/rank = F/(T_us×10^6)`，利用率为 `TFLOPS/rank / 376 × 100%`。此处统计 GEMM FLOPs，分母包含完整前向，因此反映前向整体对假设矩阵峰值的利用程度，不含激活及通信的 FLOPs。

### 基于通信时间的理论等效指标

以下沿用同一个理论运算量 F，但分母只有此前分别测得的 `dispatch + combine` 时间。专家 GEMM 和 shared expert 未在该通信计时中执行；这些数值 **不是实测 TFLOPS，也不是实际算力利用率**。比例超过 100% 不代表硬件超过峰值，只表示假想计算量与通信耗时的比值较大。

| Experts | Hidden | Intermediate | Tokens/rank | D+C µs | 理论等效 TFLOPS | 相对假设峰值比例 %（376） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 5120 | 2304 | 64 | 242.5 | 130.76 | 34.78 |
| 96 | 5120 | 2304 | 256 | 398.2 | 318.52 | 84.71 |
| 96 | 7168 | 3072 | 64 | 322.2 | 183.71 | 48.86 |
| 96 | 7168 | 3072 | 256 | 674.4 | 351.07 | 93.37 |
| 384 | 5120 | 2304 | 4096 | 4859.0 | 417.65 | 111.08 |
| 384 | 5120 | 2304 | 16384 | 14250.0 | 569.65 | 151.50 |
| 384 | 7168 | 3072 | 4096 | 4916.0 | 770.58 | 204.94 |
| 384 | 7168 | 3072 | 16384 | 21937.0 | 690.73 | 183.71 |

完整前向与纯通信是两次独立运行，聚合口径也不同（max rank 与 mean rank），不能简单相减作为专家 GEMM 的准确计时。

## 与 MegaMoE 表格的关系

参考 README 的 MegaMoE 数据来自 **Ascend 950DT、CANN 9.20**，并非本次 910C 环境。MegaMoE 表格测量 dispatch、两次 grouped GEMM、SwiGLU、combine 以及一个 shared expert 的融合计算，测试输入采用 FP8 数据及缩放因子；本表测量 MoonEP 的 BF16 通信算子。两者的输入精度、融合范围、远端字节量和带宽公式不同。MegaMoE 的通信带宽按远端 token 计数计算，而本表使用包含本地路由和重复路由的逻辑载荷。

本次八组的 D+C 均高于引用的 MegaMoE 整层延迟，当前数据不能证明 MoonEP 相对 MegaMoE 存在性能优势，也不能将两张表直接换算为相同精度与执行范围下的加速比。上面的通信测试不含专家计算；另表给出 BF16 完整前向实测，仍不能与 950DT 上的 FP8 融合实现直接归因比较。

## 复现

完成 910C 构建后，在配置好 CANN、torch 和 torch_npu 的环境中，从仓库根目录执行：

```bash
python npu_patch/benchmarks/run_megamoe_comm.py
```

单组示例：

```bash
export MOONEP_BACKEND=npu ASCEND_MOONEP_HEAP=shmem
export ASCEND_MOONEP_HEAP_BYTES=8589934592 OMP_NUM_THREADS=2
export HCCL_NPU_SOCKET_PORT_RANGE=62000-62031
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"
bash npu_patch/benchmarks/run_bench.sh bench_comm.py \
  "--tokens 64 --hidden 5120 --experts 96 --topk 6 \
   --routing random-topk --plan-source golden \
   --flows dispatch_fwd,combine_fwd --warmup 5 --iters 20 \
   --no-cudagraph --no-allow-stub --out /tmp/moonep_megamoe.csv"
```

`benchmark_results/megamoe/summary.csv` 与各配置日志、CSV 位于 `benchmark_results/megamoe/`。复现脚本会覆盖同名结果文件。日志中的主机名解析及分配器 padding 警告未导致本轮运行失败。

## 完整前向复现与原始数据

在同一 NPU 环境中，从仓库根目录运行：

```bash
python npu_patch/benchmarks/run_megamoe_forward.py
python npu_patch/benchmarks/update_megamoe_report.py
```

脚本会覆盖同名结果。`bench_moe_forward.py --peak-tflops` 支持调整峰值分母；修改该参数只改变归一化百分比，不改变实测 TFLOPS。

- `benchmark_results/megamoe/full_forward_summary.csv`（完整前向结果）
- `benchmark_results/megamoe/theoretical_equivalent.csv`（理论等效指标）
- [完整前向实现](../npu_patch/benchmarks/bench_moe_forward.py)

逐组验证信息与日志保存为 `benchmark_results/megamoe/full_*.log`。

原始结果保存在本地 `benchmark_results/`，该目录由 `.gitignore` 忽略，不随源码提交；本文表格保留本轮结果，复现脚本可重新生成日志与 CSV。
