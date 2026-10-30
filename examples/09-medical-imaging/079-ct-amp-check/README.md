# 079 · CT 分割想跑得更快，能少用一些精度吗

把 FP32 换成 FP16/BF16，理论上显存与带宽都减半。但"能省"与"真省"之间隔着三件事：
格式本身的范围与舍入、损失缩放的动态范围、以及实际有没有落到 Tensor Core 上。

本篇分成两半，各自的边界写清楚：

| 文件 | 作用 | 环境 |
|---|---|---|
| `fp16_numpy_check.py` | 只验格式：舍入、FP16 范围、教学用缩放 | 纯 NumPy |
| `amp_probe.py` | 合成小块上的 AMP 训练连接 + 同权重推理探针 | **必须 CUDA** |

## 前置

078（一次训练更新的体检）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `fp16_numpy_check.py` | NumPy 浮点格式检查（不模拟 GPU 或 PyTorch AMP） | 博文代码块 #0 |
| `amp_probe.py` | `torch.amp` 训练检查 + 推理探针 + 报告 JSON | 博文代码块 #3 |

## 运行

```bash
# 格式检查：不需要 GPU
python fp16_numpy_check.py

# AMP 检查：必须 CUDA（脚本会显式拒绝 CPU）
python -m pip install -r ../../requirements/advanced.txt
python amp_probe.py --precision fp16 --roi 32 --steps 3 --out amp_report.json
python amp_probe.py --precision bf16 --benchmark --repeats 30
```

参数：`--precision {fp32,fp16,bf16}`、`--roi`（默认 32，须是不小于 8 的 4 的倍数）、
`--steps`（默认 3）、`--benchmark`、`--repeats`（默认 30）、`--out`。

## 环境要求

- `fp16_numpy_check.py`：只需 `numpy`（`requirements/python-cpu.txt`）
- `amp_probe.py`：`torch` + **CUDA**（`requirements/advanced.txt`）；无卡时脚本直接
  抛 `RuntimeError`，提示改跑 CPU 版格式演示

## 观察点

- FP16 的三条边界：下溢、舍入、溢出；三条都能在 CPU 上验证。
- AMP 的 optimizer_steps 必须等于 steps，被跳过说明梯度缩放出了问题。

## 验证状态

- `fp16_numpy_check.py`：**已实测通过**（CPU，2026-10-28）
- `amp_probe.py`：**未实测**（脚本自身要求 CUDA，本机无法启动）

## 出处

- 博文：[79-CT分割想跑得更快能少用一些精度吗.md](../../../blog/079-CT分割想跑得更快能少用一些精度吗.md)
- 代码块 #0，26 行，md5 `ff753db2e9dc906a1747b6199f8c8a24` → `fp16_numpy_check.py`
- 代码块 #3，173 行，md5 `5b5a86e8623bbe4360b63ab82a95852b` → `amp_probe.py`

代码逐字一致。**博文未指定文件名**；`amp_probe.py` 的 CUDA 守卫里提到的
`fp16_numbers.py` 是本篇格式检查脚本在博文中的叫法，仓库里命名为
`fp16_numpy_check.py`。
