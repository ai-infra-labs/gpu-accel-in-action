# 002 · 反向传播不是魔法：用 PyCUDA 拆开全连接层的 `dX`、`dW` 与 SGD

反向传播常被当成框架里的黑箱。拆开只有三件事：`dX = dY · Wᵀ`、
`dW = Xᵀ · dY`、`W ← W − lr · dW`。本篇把它们压进一个 Kernel：
一个线程负责一个权重列，先在寄存器里算出该列的 `dX` 与 `dW`，再原地更新权重。

这是"单样本"版本（`x` 是长度 N 的向量，不是矩阵），所以 `dW` 不需要跨样本归约，
也没有并发写冲突——这正是它能用单 Kernel 完成的原因。

## 前置

001（同样的 y=Xw，本篇只做反向与更新）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `fc_backprop_sgd.py` | 单样本反向传播 + 原地 SGD + 与 NumPy 期望值逐项对照 | 博文代码块 #34 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python fc_backprop_sgd.py
```

默认参数：`N=1024`、`M=512`、`BLOCK_SIZE=256`、`WARMUP=5`、`ITERATIONS=50`、
`LEARNING_RATE=0.01`，种子 42。脚本在模块级执行（无 `main()`）。

基准只测 Kernel：每次迭代前会把权重重置回 `W0`，**重置不计入计时**。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit（`SourceModule` 编译依赖 `nvcc`）
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- dX 与 W update 的误差都在 1e-5 量级；到 1e-2 先查是不是拿更新后的权重复算了期望值。
- 每次迭代必须把权重重置回 W0，否则第二轮的正确性对照会失败——重置不在计时窗口内。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查（pyflakes 无未定义名称）。

需要记录：Min / Median / P95 / Max 四个耗时，`dX` 与 `W update` 两组误差。
`expected_output.md` 里已按脚本的输出格式列好字段。

## 出处

- 博文：[02-pycuda-fully-connected-backprop-light-tech.md](../../../blog/002-PyCUDA全连接层反向传播实战.md)
- 来源代码块 #34，367 行，md5 `a1719689433064f0c2b7ba1eee91bf78`

代码逐字一致。**博文未指定文件名**，`fc_backprop_sgd.py` 为仓库新增命名。
