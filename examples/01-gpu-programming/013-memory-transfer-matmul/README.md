# 013 · GPU 算得快，数据怎么搬也很关键

矩阵乘法本身很快，但 H2D（Host to Device）、Kernel、D2H（Device to Host）三段
经常是同一量级——数据搬运被忽略时，优化 Kernel 只是优化了总耗时的一部分。

本篇用 PyCUDA 把三段分别计时，看清内存流在哪一段被卡住。

## 前置

003（传输与计算的时间拆分）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `pycuda_memory_transfer_matmul.py` | 共享内存 tiled 矩阵乘法 + 三段计时 | 博文代码块 #7 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python pycuda_memory_transfer_matmul.py
```

脚本无命令行参数，直接打印矩阵规模、线程块、H2D / Kernel / D2H / 端到端耗时，
以及结果校验是否通过。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit（`SourceModule` 运行时编译依赖 `nvcc`）
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- 三段各占多少：H2D、Kernel、D2H。Kernel 只占小头时，优化 Kernel 是白费力气。
- 先看纯 Kernel 时间是否随规模线性增长，再谈「内存流」三段的比例。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：三段耗时各自的量级、`BLOCK_SIZE` 变化对 Kernel 段的影响、
结果校验的容差与是否通过。见 `expected_output.md`。

## 出处

- 博文：[13-pycuda-memory-transfer-matmul.md](../../../blog/013-PyCUDA内存传输与矩阵乘法.md)
- 来源代码块 #7，204 行，md5 `9469f211e2f56426c9cc628aecf9c6a1`

代码与博文逐字一致，未改写。
