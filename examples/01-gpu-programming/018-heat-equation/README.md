# 018 · GPU 算了 1000 次，温度为什么只前进了一步

一维热传导的显式差分（FTCS）每步都读写两份缓冲区。如果忘了交换 `current` 与
`next`，Kernel 会一遍遍把同一个结果写回同一块内存——循环 2000 次，物理时间只
向前走了一步。本篇把"双缓冲 + 交换"这件事写进可运行的 PyCUDA 代码，并用 CPU
参考解做数值对照。

同时它也是一堂稳定性课：显式格式要求 $r = \alpha \Delta t / \Delta x^2 \le 0.5$，
脚本在入口处直接校验，越界就抛错。

## 前置

017（双 Kernel 串接与数值稳定性）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `heat_equation.py` | 一维热传导 Kernel + CPU 参考解 + 计时 + 曲线图 | 博文代码块 #15 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python heat_equation.py
```

默认参数：`n=2^16`、`timesteps=2000`、`r=0.4`、`block_size=256`。
脚本末尾调用 `plt.show()` 画"初始热点 / 扩散后"两条曲线，**无图形界面的环境
会卡住**，见 `expected_output.md` 的踩坑记录。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit（`SourceModule` 编译依赖 `nvcc`）
- 依赖层：`requirements/python-gpu.txt`（含 `matplotlib`）

## 观察点

- 关键看「最大绝对误差」：1e-6 说明缓冲区交换正确，1e-2 说明温度只前进了一步。
- r 超过 0.5 时脚本会在入口抛错——那是显式格式的稳定性边界，不是 bug。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：CPU 参考耗时、GPU Kernel 耗时、GPU 端到端耗时、加速比、
CPU/GPU 最大绝对误差。见 `expected_output.md`。

## 出处

- 博文：[18-pycuda-heat-equation-light-tech.md](../../../blog/018-PyCUDA热传导方程数值模拟.md)
- 来源代码块 #15，229 行，md5 `8766cc9a832834fe0fb149280a6092b8`

代码逐字一致。**博文未指定文件名**，`heat_equation.py` 为仓库新增命名。
