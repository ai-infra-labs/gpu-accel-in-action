# 002 · 实测输出

**状态：未实测。** 本机无 `nvcc`，无法编译 `SourceModule`。

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyCUDA / NumPy 版本：
N / M / BLOCK_SIZE：
WARMUP / ITERATIONS / LEARNING_RATE：
```

## 2. 运行结果

退出时应打印 `Validation: PASS`；若不是，脚本会抛 `RuntimeError`。

| 项 | 值 |
|---|---|
| Min (ms) | 待填 |
| Median (ms) | 待填 |
| P95 (ms) | 待填 |
| Max (ms) | 待填 |
| `dX` 是否正确 | 待填（PASS/FAIL） |
| `dX` 最大绝对误差 | 待填 |
| `dX` 最大相对误差 | 待填 |
| `W update` 是否正确 | 待填（PASS/FAIL） |
| `W update` 最大绝对误差 | 待填 |
| `W update` 最大相对误差 | 待填 |

判读方式：

- 容差为 `rtol=2e-5, atol=2e-5`。float32 上做 1024 项点积，误差在 `1e-5`
  量级属正常；若到 `1e-2`，先怀疑是不是拿更新后的权重复算了期望值。
- Median 与 P95 的差距反映抖动；若 P95 远超 Median（比如 3 倍），通常是
  第一次迭代或系统调度干扰混进了采样。脚本已 `WARMUP=5`，仍有差距就要看机器状态。

## 3. 这份数据能说明什么

- **单 Kernel 做完三件事**：`dX`、`dW`、权重更新。前提是单样本——`dW` 不需要
  跨样本求和，一个线程独占一列，天然无冲突。
- **Kernel 计时与训练一步是两回事**：这里只测 Kernel，不含任何数据准备与传输。
  真实训练一步还要加上 H2D、优化器状态读写等开销。

## 4. 踩坑记录

待填。至少记录：

- 每次迭代必须把权重重置回 `W0`，否则第二次跑的是"已经更新过的权重"，
  正确性对照会失败；重置放在计时窗口之外
- 这一版的 `dW` 是"一个线程负责一列"的串行归约，读 `x[row * M + col]` 时
  相邻线程地址相隔 $M$ 个 float，访存不合并——性能问题不等于算错
- `max_threads_per_block` 是从设备属性查出来的，不要硬编码 1024
