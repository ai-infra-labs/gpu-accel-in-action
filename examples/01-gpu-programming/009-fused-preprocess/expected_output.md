# 009 · 实测输出

**状态：未实测。** 本机无 `nvcc`，无法编译 `SourceModule`。

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyCUDA / NumPy 版本：
批大小 / 输入尺寸 / 裁剪尺寸 / block：
```

脚本内置参数：`batch_size=8`、`height=width=256`、`crop_size=224`、
`mean=0.5`、`stddev=0.25`、`block=(16,16,1)`，因此
`grid=(14,14,8)`，共 8×224×224 = 401,408 个输出像素。

## 2. 运行结果

| 项 | 值 |
|---|---|
| Block | (16, 16, 1) |
| Grid | (14, 14, 8) |
| Kernel 时间 (ms) | 待填 |
| 结果正确（与 NumPy 参考比对） | 待填 |
| 端到端时间 (ms)（H2D + Kernel + D2H） | 待填 |

判读方式：

- `结果正确` 必须是 `True`。为 `False` 时先看中心裁剪的偏移量
  （`offset = (256-224)/2 = 16`），再看待写索引里的 `size_t` 转型。
- 端到端时间明显大于 Kernel 时间时，差额是 H2D 与 D2H——单张 8×256×256 的
  float32 输入只有 2 MiB，传输时间应当很小；若差额很大，检查是否混入了
  显存分配与首次启动。

## 3. 融合省下的往返（需要对照实验）

本篇主张融合省掉的是**一次显存往返**，而不是"少写几行代码"。要验证它，
必须把三个 Kernel 的版本也跑一遍：

| 路径 | Kernel 时间 (ms) | 端到端时间 (ms) |
|---|---|---|
| 融合（本脚本） | 待填 | 待填 |
| 裁剪 → 归一化，两个 Kernel | 待填 | 待填 |

只有两行都测出来，"少一次往返"才算被验证；只测融合版本说明不了任何事。

## 4. 踩坑记录

待填。至少记录：

- `SourceModule` 用了 `no_extern_c=True`，因为 Kernel 源码里已手写 `extern "C"`
- 显存分配放在计时窗口之外，否则测到的是 `cudaMalloc` 而不是 Kernel
- 预热调用（`cuda.Context.synchronize()` 之前那次）是否影响正式测量
