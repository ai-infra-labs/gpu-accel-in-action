# 043 · 实测输出

**状态：未实测。**

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyTorch / CUDA build：
```

## 2. 四种模式耗时

| 模式 | 含义 | 耗时 (ms) |
|---|---|---|
| `A` | 只跑模型 A | 待填 |
| `B` | 只跑模型 B | 待填 |
| `serial` | 串行跑两个 | 待填 |
| `parallel` | 双 Stream 并行 | 待填 |

判读方式：

- `parallel ≈ max(A, B)` → 真正重叠了
- `parallel ≈ A + B` → 没有重叠，只是提交到了不同 Stream
- `parallel > serial` → 争用代价大于并行收益，这就是本篇要解释的反直觉现象

## 3. 踩坑记录

待填。至少记录：

- 两个模型大小相同时是否比不同大小时更差（同步长导致争用对齐）
- `torch.cuda.synchronize(device)` 的位置对计时的覆盖范围
- 显存占用是否随模式变化（理论上不变，实际要核对）
