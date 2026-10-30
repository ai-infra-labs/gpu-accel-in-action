# 067 · ONNX 导出成功，结果为什么变了

模型导出成功 ≠ 结果正确。导出流程最容易忽略的是**输入输出约定**：
输入叫什么、什么形状、什么 dtype、输出顺序是什么。任何一处对不上，
推理仍然能跑，只是算出来的数是错的。

本篇用同一份随机输入，分别跑 PyTorch 与 ONNX Runtime，逐组核对数值，
把容差、形状、dtype、NaN/Inf 一起检查，并把输入与参考输出存档，
供后续核对 TensorRT 复用。

## 前置

027（推理链路的概念）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `export_and_check.py` | 导出 ONNX + 两套执行结果数值对齐 | 博文 §05 代码块 |

## 运行

```bash
python -m pip install -r ../../requirements/python-cpu.txt
python export_and_check.py
```

**本脚本不需要 GPU**，纯 CPU 即可完成。它只验证导出链路与数值一致性，
不代表已经验证 TensorRT（TensorRT 校验是下一篇的事）。

运行后在工作目录生成 `onnx_demo/`（已被 `.gitignore` 忽略）：

```
onnx_demo/
├── small_classifier.onnx               # 导出的模型
├── small_classifier.pt                 # 权重存档
├── reference_inputs_outputs.npz        # 输入 + PyTorch 参考输出 + 容差
└── onnx_export_<时间戳>_success.md      # torch.onnx 导出的报告
```

## 环境要求

- 依赖层：`requirements/python-cpu.txt`
- 需要 `onnxscript`：`dynamo=True` 的新导出器依赖它，缺了会在导出阶段报错

## 观察点

- 13 组用例的最大绝对误差在 1e-6 量级；导出成功与数值一致是两件事。
- 类别翻转的例子要用 logits 比，不要比 softmax 后的概率——后者会把差异抹平。

## 验证状态

**已在本机实测通过**（Apple Silicon，CPU，Python 3.14.3）：

- [x] 语法检查
- [x] 端到端运行，退出码 0
- [x] 13 组输入数值核对全部通过，最大绝对误差 1.431e-06

数据见 `expected_output.md`。

## 出处

- 博文：[67-TensorRT推理加速入门-模型导出成功为什么结果却变了.md](../../../blog/067-TensorRT推理加速入门：模型导出成功为什么结果却变了.md)
- `export_and_check.py` ← 该文代码块 #2（122 行，md5 `b34707d2e91ddcdba07640c2f9cb6491`）

代码与博文逐字一致。博文里给出的文件名就是 `export_and_check.py`，此处沿用。
