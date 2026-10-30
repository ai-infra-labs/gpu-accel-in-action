# 028 · 实测输出

**状态：未实测（需 TensorRT 10.x + NVIDIA GPU）。**

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
TensorRT / torch / onnx / pycuda：
batch / 输入维度：
```

## 2. 运行结果

脚本按路径逐段打印（`[TensorRT FP32]`、`[TensorRT FP16]`），每段一个
`相对基线加速比`、一个 engine 文件大小、一个上下文显存需求。
下面的表是把这几段抄到一起，方便横向对比：

| 路径 | GPU 计算时间 (ms/次) | 相对基线 | 最大绝对误差 | 平均绝对误差 | allclose(1e-2) |
|---|---|---|---|---|---|
| PyTorch GPU FP32（基线） | 待填 | 1.00x | — | — | — |
| TensorRT FP32 | 待填 | 待填 | 待填 | 待填 | 待填 |
| TensorRT FP16 | 待填 | 待填 | 待填 | 待填 | 待填 |

另外记录：

| 项 | 值 |
|---|---|
| `mlp_fp32.plan` 大小 | 待填 |
| `mlp_fp16.plan` 大小 | 待填 |
| FP32 上下文设备内存 | 待填 |
| FP16 上下文设备内存 | 待填 |

判读方式：

- **FP16 的误差应该比 FP32 大一个量级**，但仍在 `1e-3` 以内（本脚本按 `1e-2` 判定，
  是教学容差）。如果 FP16 误差和 FP32 一模一样，多半是精度标志没生效
  （`builder.platform_has_fast_fp16` 为假时会静默回落，脚本已用 WARNING 提示）。
- **时间只含 GPU 计算**：H2D/D2H 在计时窗口外。拿这个数除以端到端时间，
  会高估 TensorRT 的实际收益。
- **batch=16 是固定的**。batch=1 时 TensorRT 的优势往往不明显，别把这张表
  当成"小 batch 也快 5 倍"的证据。

## 3. 这份数据不能说明什么

- **不是端到端加速**：预处理、后处理、数据传输都不在表里。
- **不可跨版本比较**：TensorRT 10.x 与 8.x 的 API 和优化策略不同，
  本篇数字不能和 029（8.x API）混着看。
- **单模型单形状**：MLP + 固定 batch，换形状要重新构建 engine。
