# 005 · 一个线程处理一个像素：用 PyCUDA 手写图像水平翻转

图像翻转是"一维映射"问题：输出像素 (x, y) 与输入像素 (W-1-x, y) 一一对应。
这类问题的关键不在算法，而在**每个 warp 的 32 个线程是否读到连续地址**——
反向遍历时如果按 y 优先排列，就会把合并访存破坏掉。

本篇给出完整实现、CPU 参照、Kernel 计时与有效带宽。

## 前置

001（线程与索引的映射方式）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `image_hflip.py` | 水平翻转 kernel + 正确性核对 + 带宽计算 | 博文代码块 #17 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python image_hflip.py
```

打印 Grid/Block、Kernel 中位数与 P95、有效带宽、正确性、以及 H2D+Kernel+D2H
的端到端时间。脚本在模块级直接执行，没有 `main()`。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- 逐字节一致必须是 True：翻转没有浮点运算，不一致就是索引错，不是精度问题。
- 先写出映射 src_x = width - 1 - x 再读代码；写反了结果会左右镜像而不是报错。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：Kernel 时间（median/P95）、有效带宽、端到端时间、正确性结果。
见 `expected_output.md`。

## 出处

- 博文：[05-cuda-horizontal-image-flip-pycuda-light-tech.md](../../../blog/005-CUDA图像水平翻转（PyCUDA实战）.md)
- 来源代码块 #17，122 行，md5 `0f42c5848db3467598c99f7688822534`

代码逐字一致。**博文未指定文件名**，`image_hflip.py` 为仓库新增命名。
