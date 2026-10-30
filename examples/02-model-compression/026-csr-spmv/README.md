# 026 · 99.99% 都是零：从 CSR 到 CUDA

稀疏矩阵存不下不是因为零太多，而是因为**存储格式把零也存了**。
CSR（Compressed Sparse Row）只存非零值与列号，加上每行起点，把 40 GB 降到 8.4 MB。

本篇给出三份实现对同一份数据的比较：Numba CPU、Numba CUDA、SciPy。

## 前置

025（CSR 的由来；025 只有片段，读概念即可）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `csr_spmv_corrected.py` | CSR 构造 + CPU/CUDA 两个 SpMV kernel + SciPy 参考 | 博文代码块 #19 |

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt
python csr_spmv_corrected.py --rows 100000 --cols 100000 --density 0.0001
```

参数：`--rows` / `--cols` / `--density` 控制规模，`--runs` 控制重复次数。
默认十万行、十万列、密度万分之一（平均每行约 10 个非零）。

## 环境要求

- 依赖层：`requirements/advanced.txt`（numba + scipy）
- 无 GPU 时可走 CPU JIT 与模拟器路径；CUDA kernel 需要 NVIDIA GPU
- 新版 numba 把 CUDA 后端拆到了 `numba-cuda` 包，脚本会打印它的版本，
  缺这个包时 GPU 路径不可用

## 观察点

- 99.99% 稀疏度下，CSR 的体积由非零元个数决定，不是矩阵尺寸。
- SpMV 的速度取决于非零元分布；同一稀疏度换一种排布，耗时可能差几倍。

## 验证状态

**未实测。** 本机无 NVIDIA GPU，仅完成语法检查与未定义名称检查（均通过）。

需要记录：三种实现的误差量级、CSR 实际占用、CPU 与 GPU 各自耗时。见 `expected_output.md`。

## 出处

- 博文：[26-cuda-csr-spmv-guide.md](../../../blog/026-CUDA稀疏矩阵向量乘法（CSR-SpMV）指南.md)
- 来源代码块 #19，186 行，md5 `f5ef5d9ae320d94fded6b0df1d3160d3`

代码与博文逐字一致，未改写。
