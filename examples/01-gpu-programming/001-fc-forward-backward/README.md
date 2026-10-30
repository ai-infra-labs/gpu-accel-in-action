# 001 · GPU 不认识矩阵：它怎样用 1024 个线程算完一个线性层

GPU 里没有"矩阵"这个对象，只有一维的线程编号和一段连续的显存。一个
$N \times M$ 的输入乘上长度为 $M$ 的权重向量，要先把"二维坐标"翻译成
"一维地址"，再让每个线程各算一行点积。

本篇是系列第一课，把前向、`dX`、`dW` 三个 Kernel 放在同一份代码里，
每个都用 NumPy 对照。

## 前置

无。会写 NumPy 矩阵乘就能读；本篇是后面所有 Kernel 的骨架。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `fc_forward_backward.py` | 前向线性层 + 输入梯度 + 权重梯度，含计时与逐项对照 | 博文代码块 #45 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python fc_forward_backward.py
```

默认参数：`N=1024`、`M=512`、`block_size=256`，种子 42。
脚本在模块级执行（无 `main()`），跑完打印前向/反向 Kernel 时间与对照结论。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit（`SourceModule` 编译依赖 `nvcc`）
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- 前向 Kernel 时间应远小于反向的两个 Kernel——反向要写回 N×M 个梯度。
- 三组对照必须同时 PASS；只有一组失败时，先查 dW 的累加方向（X^T·dY 与 dY·X 差一次转置）。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查（pyflakes 无未定义名称，
唯一提示是 `pycuda.autoinit` 导入后未使用——这是它建立上下文的副作用导入）。

需要记录：前向 Kernel 时间、反向 Kernels 时间、三组对照是否通过。
见 `expected_output.md`。

## 出处

- 博文：[01-pycuda-fully-connected-forward-backward.md](../../../blog/001-PyCUDA全连接层前向与反向传播.md)
- 来源代码块 #45，306 行，md5 `f972483be612511b1de65faab3f0a348`

代码逐字一致。**博文未指定文件名**，`fc_forward_backward.py` 为仓库新增命名。
