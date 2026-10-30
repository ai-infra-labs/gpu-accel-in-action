# 076 · CT 预处理只有三行，为什么还要写 CUDA

CT 强度归一化在 NumPy 里就是三行：减窗位、除窗宽、截断到 `[0, 1]`。
三行的算力需求极低，但在流水线里它每例都要跑一遍，且**中间结果要落一次显存**——
`subtract → divide → clip` 三条表达式会生成两个临时数组，读写显存三趟。

本篇用 CuPy 做对照实验：CPU 三行、CuPy 三段式、CuPy 融合成一次，
以及原始 CUDA Kernel，看"省下的那一两趟显存往返"值多少。

## 前置

074（CT 强度与显存布局）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `intensity_map_cupy.py` | CPU 数学检查（可单独跑）+ CuPy/CUDA 三种实现的正确性与计时 | 博文代码块 #3 |

## 运行

```bash
# 只需要 NumPy：数学参考、布局转换、非法输入拒绝
python intensity_map_cupy.py --cpu-only

# 完整实验：额外需要 CuPy 与 CUDA
python -m pip install -r ../../requirements/advanced.txt
python intensity_map_cupy.py --shape 128 128 128 --repeats 50
```

参数：`--cpu-only`、`--shape`（默认 `128 128 128`）、`--repeats`（默认 50）。
窗口常量固定为 `LO=-57`、`SPAN=221`（对应 HU 的 [-57, 164] 窗）。

## 环境要求

- `--cpu-only`：只需 `numpy`（`requirements/python-cpu.txt`）
- 完整实验：`cupy` + CUDA（`requirements/advanced.txt`）

## 观察点

- 融合版与三段式的 GPU 时间差，应约等于省下的两趟显存往返。
- raw effective GB/s 是按 8 字节/元素估的，不能与规格表峰值带宽直接比。

## 验证状态

- `--cpu-only` 路径：**已实测通过**（CPU，2026-10-28）
- CuPy / CUDA 部分：**未实测**（本机无 CuPy）

## 出处

- 博文：[76-CT预处理只有三行为什么还要写CUDA.md](../../../blog/076-CT预处理只有三行为什么还要写CUDA.md)
- 来源代码块 #3，163 行，md5 `1be773041103729b750f5a3444543353`

代码逐字一致。**博文未指定文件名**，`intensity_map_cupy.py` 为仓库新增命名。
