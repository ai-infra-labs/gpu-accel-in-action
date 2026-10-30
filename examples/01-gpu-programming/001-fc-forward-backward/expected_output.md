# 001 · 实测输出

**状态：未实测。** 本机无 `nvcc`，无法编译 `SourceModule`。

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyCUDA / NumPy 版本：
N / M / block_size：
```

## 2. 运行结果

| 项 | 值 |
|---|---|
| 前向 Kernel 执行时间 (ms) | 待填 |
| 反向 Kernels 执行时间 (ms) | 待填 |
| 前向输出是否通过 NumPy 对照 | 待填 |
| 输入梯度 `dX` 是否通过 | 待填 |
| 权重梯度 `dW` 是否通过 | 待填 |

判读方式：

- 前向 $N \times M = 1024 \times 512$，每个线程一次点积（512 次乘加）。
  时间应当在几十微秒量级；若到毫秒级，先看是不是把 H2D/D2H 算进了 Kernel 计时。
- 反向是两个 Kernel（`dX` 与 `dW`），时间通常显著高于前向单个 Kernel——
  因为它要写回 $N \times M$ 个梯度值。
- 三组对照只要有一项 `FAIL`，先查索引：行主序下 `input[row * M + col]`，
  `dW` 的累加方向与 `dX` 相反。

## 3. 踩坑记录

待填。至少记录：

- `block_size=256` 时 `grid_nm = ceil(1024*512/256) = 2048`，一维网格足够；
  若把 `grid_nm` 写成 `N*M//block_size`，非整除时会漏掉尾部元素
- `backward_weights` 的性能取决于访存：一个线程负责一列，读
  `input[row * M + col]` 时相邻线程的地址相隔 $M$ 个 float，整条 warp 的
  访问被打散。这是"算得一样多、跑得慢一截"的典型来源，不是正确性问题
- 权重梯度没有并发写冲突：每个线程独占 `grad_weights[col]`，因此不需要原子操作
- `cuda.Event().time_till()` 之前是否做了 `synchronize`
