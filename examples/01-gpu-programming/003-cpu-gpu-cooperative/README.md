# 003 · GPU 明明很快，程序为什么还在等？一次讲透 CPU–GPU 协同计算

GPU 的 Kernel 只要几毫秒，程序却慢得多，差额在协同：主机准备数据、H2D 传输、
Kernel 执行、D2H 回收，四段串起来才是用户感知的时间。

本篇做一个对照实验：同一份 1024×1024 数据，CPU 端用 Numba 多线程做矩阵加法，
GPU 端用 Kernel 做矩阵乘法，然后比较两种执行方式——

| 方式 | 时间构成 |
|---|---|
| 串行 | CPU 算完 → GPU 再算 |
| 重叠 | CPU 与 GPU 同时在各自硬件上跑，只取较慢的那一段 |

判据是 `max(CPU, GPU)`：重叠的理论上限就是这个值，达不到说明没有真正重叠。

## 前置

001、002（先知道单个 Kernel 多快，才看得出「等」在哪）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `cpu_gpu_cooperative.py` | Numba CPU 并行 + CUDA Kernel + 串行/重叠两种编排 + 正确性对照 | 博文代码块 #26 |

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt
python cpu_gpu_cooperative.py
```

默认 `N=1024`（方阵）、`SAMPLES=15`（取中位数），种子 42，`16×16` 线程块。
`cuda.is_available()` 为假时脚本只打印一行提示并返回，不报错。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit
- 依赖层：`requirements/advanced.txt`（`numba` 负责 CPU 侧 JIT 与 `prange`）

## 观察点

- 两条 [验证] 行各对应一个等式：CPU+GPU ≈ 串行、max(CPU,GPU) ≈ 重叠。
- 实际加速比明显低于理论上限时，先查传输是否还在默认流上——默认流的同步会抹掉并发。

## 验证状态

**未实测。** 本机无 `nvcc` 也无 NVIDIA GPU，仅完成语法检查。

需要记录：H2D / D2H 传输时间、CPU 与 GPU 单独时间、串行与重叠时间、
实际与理论加速比、两条正确性检查。见 `expected_output.md`。

## 出处

- 博文：[03-cpu-gpu-heterogeneous-computing-light-tech.md](../../../blog/003-CPU与GPU异构计算基础.md)
- 来源代码块 #26，349 行，md5 `ee6f4c373d45e916d61dd040cdde6a1e`

代码逐字一致。**博文未指定文件名**，`cpu_gpu_cooperative.py` 为仓库新增命名。
