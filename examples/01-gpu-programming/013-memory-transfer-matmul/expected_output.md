# 013 · 实测输出

**状态：未实测。**

本机（Apple Silicon，无 `nvcc`、无 NVIDIA GPU）无法运行 PyCUDA。
以下字段待填，**不得用推断值填充**。

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyCUDA / NumPy 版本：
BLOCK_SIZE：
矩阵规模 N：
```

## 2. 三段耗时

| 阶段 | 平均耗时 (s) |
|---|---|
| H2D（A 与 B） | 待填 |
| Kernel | 待填 |
| D2H（C） | 待填 |
| H2D + Kernel + D2H 端到端 | 待填 |

关键判断：三段占比是否符合"搬运与计算同量级"的预期。若 Kernel 段远小于
搬运段，说明优化方向应转向减少传输（合并批次、pinned memory、流水线），
而不是继续调 tile。

## 3. 正确性

| 项 | 结果 |
|---|---|
| 结果校验 | 待填（通过 / 失败） |
| 校验容差 | 待填 |

## 4. 踩坑记录

待填。至少记录：

- `SourceModule` 首次编译耗时是否被计入计时（会污染第一轮结果）
- `BLOCK_SIZE` 取值与共享内存上限的关系
- 计时是否每轮重新分配显存（影响端到端口径）
