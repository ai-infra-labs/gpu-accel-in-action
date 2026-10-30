# 004 · 实测输出

**状态：未实测。** 本机无 `nvcc`，无法编译 `SourceModule`。

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyCUDA / NumPy 版本：
批大小 / 裁剪尺寸：
```

## 2. 端到端

| 项 | 值 |
|---|---|
| 模拟端到端耗时 (s) | 待填 |
| 稳态吞吐量 (images/s) | 待填 |

判读方式：把"Kernel 时间 / 端到端时间"算出来。这个比值越小，说明瓶颈越不在
计算上，优化 Kernel 的收益越有限。

## 3. 踩坑记录

待填。至少记录：

- 第一批是否包含 context 初始化与 kernel 编译，导致吞吐被低估
- 主机侧准备数据（`fill_batch`）占用多少时间
- 传输是否与计算重叠，若不重叠，理论吞吐上限是多少
