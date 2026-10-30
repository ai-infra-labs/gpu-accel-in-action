# 011 · 从局部乘加到共享内存复用：用 PyCUDA 拆开 GPU 矩阵乘法

朴素 GEMM 的每个输出元素都要把一整行和一整列读一遍，同一批数据被不同线程
反复从全局内存搬。分块（tiling）做的事情很简单：把 $16\times16$ 的数据块搬进
共享内存，块内 256 个线程共用，全局内存的读取量按块大小成比例下降。

本篇把 v0（朴素）与 v1（分块）放在同一份代码里，先各自对 float64 参考结果
做验收，再在**同一口径**下比较。

另外附一个**不需要 GPU** 的主机端校验脚本：分块索引的边界逻辑（`ceil` vs `floor`、
越界补零）可以在 CPU 上算清楚，不必等显卡。

## 前置

010（共享内存与 __syncthreads）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `tiled_matmul.py` | v0/v1 两个 Kernel + 正确性验收 + median/p95/GFLOPS + 加速比 | 博文代码块 #13 |
| `validate_tiled_index.py` | 主机端索引与边界逻辑验证（NumPy，无需 GPU） | 博文代码块 #29 |

## 运行

```bash
# 不需要 GPU，几秒出结果，全部断言通过才打印结论
python validate_tiled_index.py

# 需要 N 卡：两个 Kernel 都跑 1024×1024，每轮 5 次预热 + 20 次采样
python -m pip install -r ../../requirements/python-gpu.txt
python tiled_matmul.py
```

`tiled_matmul.py` 默认 `N=1024`、`TILE_SIZE=16`、`block=(16,16,1)`、
`grid=(64,64,1)`，种子 42。它在模块级执行（无 `main()`），且**先验收后计时**：
任一 Kernel 的 `allclose` 失败会直接抛 `RuntimeError`。

## 环境要求

- `validate_tiled_index.py`：只需 `numpy`（`requirements/python-cpu.txt`）
- `tiled_matmul.py`：NVIDIA GPU + CUDA Toolkit（`requirements/python-gpu.txt`）

## 观察点

- validate_tiled_index.py 里 floor 版的 rel_err 必须明显大于 0（非整除用例），否则说明用例选错了。
- tiled 相对 naive 的 speedup 应随 N 增大；小 N 打平是正常的（启动开销主导）。

## 验证状态

- `validate_tiled_index.py`：**已实测通过**（CPU，2026-10-28）
- `tiled_matmul.py`：**未实测**（本机无 `nvcc`，仅语法检查）

## 出处

- 博文：[11-pycuda-tiled-matmul-shared-memory-light-tech.md](../../../blog/011-PyCUDA分块矩阵乘法与共享内存.md)
- 代码块 #13，298 行，md5 `2dc8aff50e27d449daf30afc61a85f4b` → `tiled_matmul.py`
- 代码块 #29，98 行，md5 `6369ba26ee1907f978d32919d8b62c84` → `validate_tiled_index.py`

代码逐字一致。`validate_tiled_index.py` 的文件名来自块首 docstring，
`tiled_matmul.py` 为仓库新增命名。
