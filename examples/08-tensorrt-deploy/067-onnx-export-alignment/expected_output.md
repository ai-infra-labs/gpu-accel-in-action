# 067 · 实测输出

**状态：已实测通过**（CPU 路径，2026-10-27）。

## 1. 环境

```
平台      ：macOS 27.0.1 (arm64, Apple M4)
Python    ：3.14.3
torch     ：2.14.1
numpy     ：2.5.4
onnx      ：1.23.2
onnxscript：0.7.2
onnxruntime：1.31.1
```

本脚本不依赖 NVIDIA GPU，CPU 环境下的数值结论与 GPU 环境一致
（导出与 ONNX Runtime CPU 推理都不涉及 CUDA kernel）。

## 2. 运行结果

退出码 `0`，关键输出：

```
ONNX 输入: features [1, 256] tensor(float)
ONNX 输出: logits [1, 10] tensor(float)
通过 13 组数值核对；最大绝对误差=1.431e-06
已保存权重与参考数据: onnx_demo/small_classifier.pt
已保存输入与参考输出: onnx_demo/reference_inputs_outputs.npz
```

| 项 | 结果 |
|---|---|
| 输入名 / 形状 / dtype | `features` / `[1, 256]` / `tensor(float)` |
| 输出名 / 形状 / dtype | `logits` / `[1, 10]` / `tensor(float)` |
| 核对组数 | 13 |
| 最大绝对误差 | 1.431e-06 |
| 形状 / dtype / NaN 检查 | 全部通过 |

13 组用例的构成：1 组全零 + `scale ∈ {0.1, 1.0, 10.0}` 各 4 组随机输入
（固定种子 42）。因此 1.431e-06 是这 13 组的最大绝对误差，
**不是**该模型在所有输入上的误差上界。

## 3. 产物

```
onnx_demo/
├── small_classifier.onnx          71 KB
├── small_classifier.pt            69 KB
├── reference_inputs_outputs.npz   20 KB
└── onnx_export_2026-10-27_17-29-08-635841_success.md
```

## 4. 踩坑记录

**1. 缺 `onnxscript` 会在导出阶段失败，而不是导入阶段。**
`dynamo=True` 的新导出器把脚本翻译成 ONNX 图时才会 import 它，
所以 `import torch` 一切正常，跑到 `torch.onnx.export()` 才报错。

**2. 未安装 `torchvision` 会刷出 4 行 W 级日志，但不影响结果。**

```
W torch/onnx/_internal/exporter/_registration.py:107] torchvision is not installed. Skipping torchvision::nms
...（roi_align / roi_pool / deform_conv2d 各一行）
```

这是导出器在注册自定义算子时发现视觉算子缺失，跳过即可。看到 W 不要当成失败，
以退出码和最后的核对结果为准。

**3. `onnx_demo/` 每次运行会追加一个新的导出报告，不会覆盖旧的。**
仓库已用 `.gitignore` 忽略整个目录；如果要做版本对比，只保留最后一次的报告。
