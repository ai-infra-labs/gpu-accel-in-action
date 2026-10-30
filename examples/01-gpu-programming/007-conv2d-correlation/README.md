# 007 · Kernel 能编译，不代表它在算卷积

卷积正确性不能靠"跑起来了"判断。边界怎么处理、核是否翻转、padding 补什么值，
任意一处错了，输出仍然是一张图，只是算的不是卷积。

本篇用 SciPy 的 `correlate2d` 作为独立参照，逐点比对 GPU 结果，并给出最大/平均
绝对误差。

## 前置

006（二维索引与边界）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `conv2d_correlation.py` | 2D 卷积 kernel + SciPy 参照 + 误差统计 | 博文代码块 #38 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python conv2d_correlation.py
```

打印 Kernel 执行时间、最大/平均绝对误差、一致性结论，以及输出左上角 5×5 数值。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit
- 依赖层：`requirements/python-gpu.txt`（含 scipy）
- 参照实现用 `scipy.signal.correlate2d`，注意它是**相关**不是**卷积**——
  核是否翻转的差异会直接体现在误差上

## 观察点

- 三个「反转」任一漏掉，结果都会整体偏移一格——先跑小图人眼对齐，再看误差。
- Kernel 能编译不等于在算卷积：拿一个 3×3 的手算例子逐格核对最省时间。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：Kernel 时间、最大/平均绝对误差、是否与 SciPy 一致。见 `expected_output.md`。

## 出处

- 博文：[07-pycuda-2d-convolution-correlation.md](../../../blog/007-PyCUDA二维卷积与互相关.md)
- 来源代码块 #38，121 行，md5 `a2859f02af1fa387dfb65a7309e650e7`

代码逐字一致。**博文未指定文件名**，`conv2d_correlation.py` 为仓库新增命名。
