# MoonEP + Ascend 910C

MoonEP 通过动态冗余专家均衡 MoE 的 token 负载，并融合跨卡通信与 token 重排。本适配在 Ascend 910C（A3）上使用 AscendC MTE 内核、CANN shmem 普通对称堆和主机侧规划器，支持 dispatch、combine、专家权重 prefetch 和梯度 reduce。

适配设计、关键修复与本轮 benchmark 见 [适配过程与性能验证](docs/moonep_ascend_910c_guide.md)。

## 支持范围

历史验证环境为单机 8 × Ascend910_9392、CANN 9.1.0、Python 3.12；已有 PyTorch / torch_npu 2.9.1 和 2.12.0 的运行记录。PyTorch 或 torch_npu 版本变化后必须重新编译 `_C` 扩展。

| 能力 | 910C 路径 |
| --- | --- |
| 通信内核 | MTE dispatch / combine / prefetch / grad_reduce |
| 对称内存 | `ASCEND_MOONEP_HEAP=shmem`，通过 shmem 分配器分配 |
| 规划器 | 主机 C++ planner；测试和 benchmark 使用 Python reference plan |
| SIMT planner、UDMA / URMA | 不支持，构建时排除；保留明确报错的入口桩 |
| Ascend950 | 保留原 SIMT / UDMA 分支，本次未在 950 上回归 |
| 多机集群 | 当前材料仅覆盖单机多卡；跨节点互联、地址可达性与性能需要独立验证 |

910C 的规划需要计数汇总、主机计算和 plan 回传，存在逐层同步开销。固定 token buffer 并不意味着 910C 已消除主机同步。

## 核心实现

- `npu_patch/native/CMakeLists.txt`、`kernels/non_simt_stubs.cpp`：按 a2 / a5 区分源码，避免在 910C 编译 SIMT 内核。
- `npu_patch/native/include/ascend_moonep/kernels/shmem_barrier.h`：910C 使用软核间同步，避免缺少 FFTS 配置时的 `SyncAll` 异常；950 保留 vector barrier。
- `npu_patch/buffer_c.py`、`api.py`、`shmem_runtime.py` 及 `native/csrc/acl_shared_buffer.*`：普通 shmem 堆初始化、UID 广播、对称 chunk 分配和生命周期管理。通过 shmem 分配器分配，避免覆盖其内部同步区。
- `npu_patch/planning_host.py`、`native/csrc/host_planning.*`、`pybind.cpp`、`setup.py`：导出 C++ 主机规划器，供上层框架接入。
- `npu_patch/benchmarks/`：支持 `--plan-source golden`、910C 计时兼容和去重 plan 复用。

## 构建

在安装了匹配版本的 torch / torch_npu、CANN 开发工具链和 C++ 编译依赖的环境中，从仓库根目录执行。先加载本机 CANN 的 `set_env.sh`，确认 `ASCEND_HOME_PATH` 指向该版本。

```bash
git submodule update --init --recursive
export SHMEM_HOME_PATH="$PWD/npu_patch/3rdparty/shmem/install"

# 在 910C 环境构建 shmem；切换 SoC 时不要复用 950 产物。
(cd npu_patch/3rdparty/shmem && bash scripts/build.sh)
cat "$SHMEM_HOME_PATH/version.info"
# 本机历史验证中 SOC Type 为 Ascend910B。

cmake -S npu_patch/native -B build_ascend_910c \
  -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" \
  -DASCEND_MOONEP_BISHENG_ARCH=a2 \
  -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH"
cmake --build build_ascend_910c --target ascend_moonep_kernels --parallel 8
mkdir -p npu_patch/lib
cp build_ascend_910c/lib/libascend_moonep_kernels.so npu_patch/lib/

(cd npu_patch && MOONEP_BUILD_BACKEND=npu \
  ASCEND_MOONEP_KERNEL_LIB_DIR="$PWD/lib" \
  python -m pip install -e . --no-build-isolation --no-deps)

python -c 'import torch, torch_npu; from npu_patch import _C; print(torch.__version__, torch_npu.__version__, _C.__file__)'
```

`build_a5.sh` 和当前 `build_base.sh` 只接受 Ascend950；910C 请使用上述显式 CMake 步骤。内核、shmem 和 Python 扩展应来自同一套工具链及依赖版本。

## 安装为 Python 包

完成上述 shmem 和 a2 内核构建后，可将 Python 接口、`npu_patch._C` 和 `npu_patch/lib/` 下的运行库打包为 wheel。在仓库根目录执行：

```bash
mkdir -p dist
(cd npu_patch && MOONEP_BUILD_BACKEND=npu \
  python -m pip wheel . --no-build-isolation --no-deps --wheel-dir ../dist)
python -m pip install --force-reinstall --no-deps dist/moonep-*.whl
```

Ascend 打包入口是 `npu_patch/`；仓库根目录的 `setup.py` 构建 CUDA 扩展。开发时可使用上一节的 editable 安装；wheel 安装后不需要保留源码目录，也不需要在业务代码中修改 `sys.path` 或设置指向源码的 `PYTHONPATH`。

wheel 不包含 PyTorch、torch_npu、CANN 和驱动。目标环境应使用匹配的 Python、CPU 架构和运行时版本；更换 torch / torch_npu 后需重新构建。当前打包验证环境为 aarch64、Python 3.12、torch / torch_npu 2.12.0、CANN 9.1.0。安装时使用 `--no-deps` 保留已配好的 NPU 环境。

该环境已生成 `moonep-0.0.1-cp312-cp312-linux_aarch64.whl`，完成独立目录安装、仓库外导入及随包动态库加载检查；普通安装后，从仓库外运行主机规划器契约测试，6 项通过。此验证不包含 NPU 通信或 benchmark 复测。

在任意外部代码目录中，先加载 CANN 环境，再导入：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export MOONEP_BACKEND=npu
export ASCEND_MOONEP_HEAP=shmem
python -c 'import moonep; from npu_patch import _C; print(moonep.__file__, _C.__file__)'
```

本次外部导入验证发现，容器的 `LD_LIBRARY_PATH` 末尾存在空项，可能从当前目录加载同名旧内核库。运行前应移除空项和旧 MoonEP 库路径；例如尾部空项可用 `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"` 移除，避免覆盖 wheel 内的运行库。

```python
from moonep import Buffer, MoonEPCommPlan, init_shmem
from npu_patch import ShmemRuntime
from npu_patch.planning_host import build_plan_host
```

导入成功只验证包与动态库加载。执行通信仍需设置本地 NPU、初始化 HCCL / shmem，并为 910C 显式提供主机 plan。`build_plan_host` 生成规划字段，调用方仍需汇总专家计数、组装 `MoonEPCommPlan` 和去重元数据；安装包不会自动启用 910C 设备规划器，也不依赖仓库中的测试模块。

## 运行与自检

```bash
export ASCEND_MOONEP_HEAP=shmem
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61031

python test_npu/mte/test_moonep_dispatch_with_plan.py \
  --mode kernel --R 8 --S 64 --K 4 --H 256 --E 8
python test_npu/mte/test_moonep_combine.py \
  --ep-size 2 --num-tokens 16 --hidden 128 --num-topk 4
python test_npu/mte/test_dispatch_combine_joint.py --R 8 --S 16 --K 4 --H 128
torchrun --standalone --nproc_per_node=2 test_npu/mte/example_a3_buffer_usage.py
```

前三项检查应输出 `ALL RANKS PASS`，示例应输出 `EXAMPLE PASS`。示例检查调用链、形状和有限值；数值正确性由前面的测试检查。2026-10-28 已完成复测；覆盖范围、失败尝试及证据说明见 [适配与验证文档](docs/moonep_ascend_910c_guide.md)。

示例展示完整调用顺序：初始化 HCCL、设置本地 NPU、构造 `Buffer(..., B=E//R, use_udma=False)`、初始化 `ShmemRuntime`、生成主机 plan、dispatch、combine，最后同步设备、调用 `ShmemRuntime.finalize()`，再销毁 Buffer 和进程组。路由变化时必须重建 plan。复用 plan 调用 dispatch 返回的 `cu_seqlens` 可能为 `None`，需保留规划阶段的值。

测试示例使用 `test_npu.planing_reference.reference_plan_and_cu_seqlens`。生产框架可调用 `npu_patch.planning_host.build_plan_host(counts_all, local_counts, route_ids, rank, epn, nvs)`：输入为 `[R,E]` 全组计数、`[E]` 本地计数和 `[S*K]` 路由专家编号，返回 dst、cu_seqlens、experts_to_copy 等字段。它提供规划数据，调用方仍需完成计数汇总、构造通信 plan 和去重元数据；不会自动替换 `Buffer.dispatch` 的设备 planner。

## Benchmark 与计时口径

```bash
python npu_patch/benchmarks/bench_comm.py --help
python npu_patch/benchmarks/bench_vs_all2all.py --help
```

910C 选择 `--plan-source golden`，禁用 UDMA，并根据堆容量降低 shape 或 `--hp`。benchmark 自行管理多进程启动，具体参数以脚本 `--help` 为准。

Golden 模式在计时区外构造主机 plan，不能把通信时间当作含规划的训练步延迟。对比时同时记录 shape、rank 数、实际 MaxVio、warmup / iters、prefetch 大小，以及是否计入去重展开和主机规划。带宽按逻辑载荷计算，不等于 HCCS 物理链路带宽。普通 shmem 堆模式使用逐轮 NPU 事件计时，每轮结束在计时区外同步进程组，避免下一轮覆盖远端仍在读取的缓冲区。combine 原地归并重复行，每轮在计时区外重新 dispatch 恢复输入，恢复时间不计入 combine 延迟；该模式禁用 graph capture。`--percentiles` 报告各 rank 分位数的 mean / max，取决于该算子的聚合方式，并非合并全部 rank 样本的全局分位数。

主机规划器可独立运行 `python npu_patch/benchmarks/bench_host_plan.py --help`；CPU 计算与包含 NPU→CPU→NPU 传输的时间需分开报告。

历史原始日志、CSV、失败记录和移植过程文档归档在仓库外，不作为核心源码提交的一部分；此 README 不将历史结果作为当前硬件复测结论。

## 限制与排错

| 现象 | 检查方式 |
| --- | --- |
| `planning_kernel_do is not built for this SoC` | 使用主机 plan；benchmark 指定 `--plan-source golden` |
| 507015 / barrier AICore 异常 | 确认 a2 构建和 `shmem_barrier.h` 生效，未复用 a5 动态库 |
| shmem 堆不足或初始化返回 -3 | 减小 shape / 专家权重池；必要时设置纯字节整数 `ASCEND_MOONEP_HEAP_BYTES` |
| HCCL error code 7 | 检查端口范围、端口占用和网络配置 |
| `_C` 导入缺少 C++ 符号 | 用当前 torch / torch_npu 环境重新编译扩展 |

历史环境普通堆可用容量约 8 GiB/rank，并非所有平台的固定上限。自动预算含 256 MiB 余量，但尚未完整记入各 chunk 的 2 MiB 对齐及内部开销；额外分配专家权重和梯度时应预留容量。独立分配前需显式调用 `ensure_shmem_heap(total_bytes, group=ep_group)`；不再隐式使用 WORLD 初始化。堆由进程内运行时共享，chunk 销毁不单独回收，完整生命周期需由调用方协调。
