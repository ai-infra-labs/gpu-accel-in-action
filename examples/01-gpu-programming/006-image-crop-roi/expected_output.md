# 006 · 实测输出

**状态：未实测。** 本机无 `nvcc`，无法编译 `SourceModule`。

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyCUDA / NumPy 版本：
输入尺寸 / ROI / block / grid：
```

## 2. 运行结果

| 项 | 值 |
|---|---|
| Kernel latency median (ms) | 待填 |
| Kernel latency P95 (ms) | 待填 |
| 端到端 median (ms) | 待填 |
| 端到端 P95 (ms) | 待填 |
| 结果正确（逐字节一致） | 应为 `True`（脚本内有 `assert`） |

判读方式：

- 100×100 的 ROI 只有 1 万个像素，Kernel 时间通常在 10 微秒以内。
  这个尺度下 **端到端会明显大于 Kernel**：2 MiB 输入的 H2D 一个人就占了几十微秒。
  两者的差距就是"计算之外的开销"，别把它算到 Kernel 头上。
- 逐字节一致是硬要求。裁剪不涉及任何浮点运算，出现不一致一定是索引算错，
  不是精度问题。

## 3. 踩坑记录

待填。至少记录：

- 输入必须是 C 连续（脚本用 `assert input_image.flags["C_CONTIGUOUS"]` 守着）。
  非连续切片上传后，Kernel 按 `row * width + col` 读到的就是错位的像素
- `grid=(7,7)` 覆盖 112×112 个线程，比 100×100 多出来的线程必须靠边界判断挡掉
- 输出是独立连续的 `100×100` 数组，不是输入视图——这一点和 NumPy 切片语义不同
