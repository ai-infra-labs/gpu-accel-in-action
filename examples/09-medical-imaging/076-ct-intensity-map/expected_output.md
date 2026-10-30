# 076 · 实测输出

**状态：CPU 检查已实测通过；CuPy / CUDA 部分未实测。**

## 1. CPU 检查（已实测，2026-10-28）

```
平台    ：macOS 27.0.1 (arm64, Apple M4)
Python  ：3.14.3
numpy   ：2.4.2
命令    ：python intensity_map_cupy.py --cpu-only --shape 64 64 64 --repeats 5
```

退出码 `0`，原始输出：

```
CPU检查通过：数学参考、布局转换、空/非有限/复数输入拒绝和索引分配；未验证CUDA。
```

这一行覆盖五件事：数学参考（`(x+57)/221` 再截断）、布局转换、
空数组拒绝、NaN/Inf 拒绝、复数拒绝与索引分配。它**不验证 CUDA**，
也不产生任何性能数字。

## 2. CuPy / CUDA 实验（未实测）

| 项 | 值 |
|---|---|
| CuPy 版本 / CUDA 版本 | 待填 |
| shape / repeats | 待填 |
| `numpy` 三段式 CPU us（median / p10 / p90） | 待填 |
| `cupy` 三段式：shape / dtype / max_abs_error | 待填 |
| `cupy` 三段式 GPU us（median / p10 / p90） | 待填 |
| `cupy` 融合版 GPU us | 待填 |
| `cuda_raw`：max_abs_error / GPU us | 待填 |
| `raw effective GB/s` | 待填 |

判读方式：

- 三段式每次都会分配临时数组，读一份、写两份中间结果；融合版只读写各一次。
  在 `128³` 的 float32（8 MiB）上，"少两趟往返"对应约 16 MiB 的显存流量，
  带宽越高、收益越明显。
- `raw effective GB/s` 是按 `8 * size` 估算的**有效带宽**（一次读 + 一次写 × 4 B），
  不是 DRAM 硬件计数器读数，不能拿它跟显卡规格表上的峰值带宽直接比。
- CPU 与 GPU 的时间来自不同计时机制（`perf_counter` vs CUDA 事件），
  可以比较量级，不能当成同一把尺子的精确比值。

## 3. 这份数据不能说明什么

- **不能说明端到端会快**：预处理只占整条流水线的一段，收益要看它占比多少。
- **不涉及精度**：归一化是逐元素线性变换，CPU 与 GPU 的 float32 结果应完全一致
  （容差 `1e-6` 只是浮点比较的习惯写法）。
