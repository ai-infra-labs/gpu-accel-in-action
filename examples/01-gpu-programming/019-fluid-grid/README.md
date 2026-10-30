# 019 · 26 万个 CUDA 线程，如何推动一滴墨水

半拉格朗日平流（semi-Lagrangian advection）很反直觉：不是"把浓度推到邻居"，
而是**倒着找"这一步之前我在哪"**，再从那一点插值取回浓度。好处是步长不受
CFL 条件限制，代价是插值不严格守恒。

本篇跑两条路径，同一组初始场：

| 文件 | 角色 | 环境 |
|---|---|---|
| `fluid_grid.py` | 差异旋转速度场 + 平流/扩散 Kernel，`512×512 × 300` 步 | N 卡 + PyCUDA |
| `fluid_cpu_ref.py` | CPU 参考实现（只依赖 NumPy），检查有限性与非负 | 纯 CPU |

CPU 参考不是"陪跑"：它把同样的公式用一个不依赖 GPU 的写法独立实现一遍，
用来确认 GPU 版出现的 NaN、负浓度不是算法本身写错了。

## 前置

018（显式差分与双缓冲）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `fluid_grid.py` | 平流 + 扩散 Kernel + 300 步计时 + 数值断言 | 博文代码块 #35 |
| `fluid_cpu_ref.py` | NumPy 参考实现 + 逐帧有限性/非负断言 | 博文代码块 #36 |

## 运行

```bash
# CPU 参考，几秒出结果
python fluid_cpu_ref.py

# GPU 版，需要 N 卡
python -m pip install -r ../../requirements/python-gpu.txt
python fluid_grid.py
```

默认参数：GPU `N=512`、`steps=300`、`dt=0.02`、`alpha=0.02`；
CPU `N=64`、`steps=300`（同样的 `dt` 与 `alpha`）。

`alpha = D·dt/h²` 必须 ≤ 0.25 才稳定，脚本注释里写明了这个约束。

## 环境要求

- `fluid_cpu_ref.py`：只需 `numpy`（`requirements/python-cpu.txt`）
- `fluid_grid.py`：NVIDIA GPU + CUDA Toolkit（`requirements/python-gpu.txt`）

## 观察点

- 总浓度比 ≈ 0.969（CPU 参考 300 步）；显著更低时先查固定零边界。
- 平均每步内核时间应与「51 万次双线性插值」的量级相符；高十倍说明每步都在同步。

## 验证状态

- `fluid_cpu_ref.py`：**已实测通过**（CPU，2026-10-28）
- `fluid_grid.py`：**未实测**（本机无 `nvcc`，仅语法检查）

## 出处

- 博文：[19-pycuda-fluid-grid-light-tech.md](../../../blog/019-PyCUDA流体格子模拟.md)
- 代码块 #35，257 行，md5 `f6f3d4ad9a4ab26d84fe4eec8a2bdab6` → `fluid_grid.py`
- 代码块 #36，72 行，md5 `182dea670843d2fb0f689e5a99bc17c4` → `fluid_cpu_ref.py`

代码逐字一致。`fluid_cpu_ref.py` 的文件名来自块首注释，`fluid_grid.py` 为仓库新增命名。
