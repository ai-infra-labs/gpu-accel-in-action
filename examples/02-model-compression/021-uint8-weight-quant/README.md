# 021 · 模型缩小近 4 倍，推理也会快 4 倍吗

把 fp32 权重存成 uint8，文件能小到 1/4。但"文件小"和"算得快"是两件事：
反量化（dequantize）发生在推理之前，卷积仍是 fp32——省的是显存与带宽，
不是乘加次数。

本篇跑一个 CIFAR-10 小 CNN 的完整对照：原始 fp32 → 逐层 uint8 存储 →
加载时反量化 → 测准确率与文件大小。

## 前置

020（模型压缩的全景，知道量化在整张图里的位置）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `uint8_weight_quant_demo.py` | 逐层 uint8 量化/反量化 + 训练 + 准确率与体积对照 | 博文代码块 #4 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python uint8_weight_quant_demo.py
```

首次运行会下载 CIFAR-10 到 `./data`（约 170 MB）。默认训练 5 轮、批大小写在
脚本内，无命令行参数。

## 环境要求

- 必须：NVIDIA GPU + `numba-cuda`（量化和反量化各是一个 Numba CUDA kernel，
  不是 torch 的 `.cuda()`，光有 CUDA 版 PyTorch 不够）
- 需要网络：`torchvision.datasets.CIFAR10(download=True)`
- 依赖层：`requirements/advanced.txt`

## 观察点

- 两个准确率必须完全相同：量化只改存储格式；差 0.01% 以上说明 scale/zero_point 用错了。
- 理论载荷≈1/4，实际文件远不到 1/4——两个数字要分开报，别混成「模型小了 4 倍」。

## 验证状态

**未实测。** 本机无 CUDA 版 torch，也未下载数据集；仅完成语法检查
（pyflakes 无未定义名称）。

需要记录：原始/反量化后准确率、理论张量载荷与实际文件大小。见 `expected_output.md`。

## 出处

- 博文：[21-模型缩小四倍推理也会快四倍吗_轻科技博文.md](../../../blog/021-模型缩小四倍推理也会快四倍吗.md)
- 来源代码块 #4，349 行，md5 `017b1fcb0282621a9522ae7506d01ca2`

代码逐字一致。**博文未指定文件名**，`uint8_weight_quant_demo.py` 为仓库新增命名。
