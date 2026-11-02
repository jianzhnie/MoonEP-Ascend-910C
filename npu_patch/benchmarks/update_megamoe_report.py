"""Regenerate compute metric tables from recorded CSVs (no hardware execution)."""
import csv
from pathlib import Path

root=Path(__file__).resolve().parents[2]
out=root/'benchmark_results/megamoe'
rows=[]
for e,h,i,s in [(96,5120,2304,64),(96,5120,2304,256),(96,7168,3072,64),(96,7168,3072,256),(384,5120,2304,4096),(384,5120,2304,16384),(384,7168,3072,4096),(384,7168,3072,16384)]:
    path=out/f'full_e{e}_h{h}_s{s}.csv'
    if path.exists(): rows.extend(csv.DictReader(path.open()))
if len(rows)!=8: raise SystemExit(f'Expected 8 full forward results, found {len(rows)}')
with (out/'full_forward_summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
p=root/'docs/moonep_megamoe_benchmark.md'
s=p.read_text().split('\n## 完整前向复现与原始数据')[0]
if '参考 README 的 MegaMoE 数据来自' not in s:
    s=s.replace('MegaMoE 表格测量 dispatch', '参考 README 的 MegaMoE 数据来自 **Ascend 950DT、CANN 9.20**，并非本次 910C 环境。MegaMoE 表格测量 dispatch')
s=s.replace('MoonEP 本测试没有执行专家计算，因此不报告计算 TFLOPS 或专家权重 HBM 带宽。完整框架评估还需加入主机规划、专家计算和实际权重路径。', '上面的通信测试不含专家计算；另表给出 BF16 完整前向实测，仍不能与 950DT 上的 FP8 融合实现直接归因比较。')
start=s.index('## TFLOPS 与算力利用率')
end=s.index('## 与 MegaMoE 表格的关系',start)
s=s[:start]+'''## TFLOPS 与算力利用率

通信表的 N/A 保留：dispatch / combine 本身不执行专家 GEMM。完整前向实测和通信时间上的理论等效指标分别如下。

### 完整 BF16 前向实测

实测路径为 `dispatch → grouped GEMM(H→2I) → SwiGLU → grouped GEMM(I→H) → 路由加权 → combine → shared expert 输出相加`；实现中路由加权在第二次 GEMM 后执行，与 BF16 舍入顺序一致。shared expert 为一个本地 BF16 SwiGLU MLP。两个 routed GEMM 调用 `torch_npu.npu_grouped_matmul`，shared expert 使用普通 matmul，未做融合或通信计算重叠。权重初始化、主机 plan、专家副本分发不计时；使用实际活跃专家的常驻权重副本。

计时包含完整算子链、必要的跨 rank 发布同步、激活/输出分配及拷贝，排除每轮结束的进程组同步。预热 5 次、计时 20 次，以 **各 rank 平均延迟的最大值**作为同步前向时间。它是固定路由和常驻专家副本条件下的前向，不包含动态路由规划、权重搬运或训练反向。

每组在计时前检查所有输出有限，并在每个 rank 的第一个 token 上使用独立逐专家 matmul 参考计算验证 routed + shared 输出（`rtol=0.05, atol=0.05`）。这是抽样数值验证，不是全 token 独立参考验证。

利用率按 **376 TFLOPS / 逻辑设备的 BF16 稠密峰值假设**归一化。该值沿用现有训练配置，不是已核实的 Ascend910_9392 官方峰值；表中百分比是条件指标，不是 profiler 测得的 AICore 忙碌率。若采用不同峰值 P，百分比应乘以 `376/P`。EP8 的集群吞吐为表中每 rank TFLOPS 的 8 倍，集群峰值也必须同时乘以 8。

| Experts | Hidden | Intermediate | Tokens/rank | 完整前向 µs（max rank） | GEMM TFLOPS/rank | 利用率 %（假设峰值 376） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''+''.join('| '+' | '.join([r['experts'],r['hidden'],r['intermediate'],r['tokens'],f"{float(r['forward_us_max_rank']):.2f}",f"{float(r['tflops_per_rank']):.2f}",f"{float(r['utilization_pct_assumed_peak']):.2f}"])+' |\n' for r in rows)+'''
乘加计为 2 FLOPs，两次 routed GEMM 加一个 shared expert 的平均运算量为 `F = 6×H×I×S×(K+1) = 42×S×H×I`；`TFLOPS/rank = F/(T_us×10^6)`，利用率为 `TFLOPS/rank / 376 × 100%`。此处统计 GEMM FLOPs，分母包含完整前向，因此反映前向整体对假设矩阵峰值的利用程度，不含激活及通信的 FLOPs。

### 基于通信时间的理论等效指标

以下沿用同一个理论运算量 F，但分母只有此前分别测得的 `dispatch + combine` 时间。专家 GEMM 和 shared expert 未在该通信计时中执行；这些数值 **不是实测 TFLOPS，也不是实际算力利用率**。比例超过 100% 不代表硬件超过峰值，只表示假想计算量与通信耗时的比值较大。

| Experts | Hidden | Intermediate | Tokens/rank | D+C µs | 理论等效 TFLOPS | 相对假设峰值比例 %（376） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''+''.join('| '+' | '.join([r['E'],r['H'],r['intermediate_reference'],r['S'],r['dispatch_combine_sum_us'],f"{float(r['equivalent_tflops_over_comm_only']):.2f}",f"{float(r['equivalent_pct_of_assumed_peak']):.2f}"])+' |\n' for r in csv.DictReader((out/'theoretical_equivalent.csv').open()))+'''
完整前向与纯通信是两次独立运行，聚合口径也不同（max rank 与 mean rank），不能简单相减作为专家 GEMM 的准确计时。

'''+s[end:]
s+='''
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
'''
p.write_text(s)
