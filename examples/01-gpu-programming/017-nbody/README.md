# 017 · 1024 个 CUDA 线程一起模拟引力，算出的"宇宙"可能是错的

同一个时间步里，有的线程还在读旧位置，有的线程已经写下了新位置——两代坐标
混在一起，粒子会越飞越远。修法不是"加锁"，而是把"读世界"和"改世界"拆成
两个 Kernel：先算加速度，再统一积分位置。

本篇用 PyCUDA 实现朴素 N-body（$O(N^2)$ 两两相互作用），并给出与 CPU 参考结果
的一步误差核对。

## 前置

001、010（线程映射与归约）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `nbody.py` | 两 Kernel 串接的 N-body 模拟 + 单步 CPU 对照 + 计时 | 博文代码块 #24 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python nbody.py
```

默认参数：`n=1024`、`steps=100`、`dt=1e-3`、`softening2=1e-4`，种子 42，
无命令行参数。先跑 `verify_one_step()`（与 CPU 逐步对照），再跑 100 步计时。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit（`SourceModule` 编译依赖 `nvcc`）
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- 一步位置/速度误差应在 1e-6 量级；到 1e-2 先查「读旧写新」有没有拆成两个 Kernel。
- n 从 1024 到 4096，耗时增长应接近 16 倍（O(N²)）。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：单步位置/速度最大绝对误差、100 步 Kernel 总时间、函数端到端时间。
见 `expected_output.md`。

## 出处

- 博文：[17-pycuda-nbody-light-tech.md](../../../blog/017-PyCUDA N体模拟.md)
- 来源代码块 #24，241 行，md5 `c2ac92dd39769f83ec5631f444efda41`

代码逐字一致。**博文附录 A 未指定文件名**，`nbody.py` 为仓库新增命名。
