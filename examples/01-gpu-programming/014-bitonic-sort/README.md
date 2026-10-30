# 014 · 1024 个数，为什么这份朴素 CUDA 实现要启动 55 次 Kernel

双调排序（bitonic sort）把"排序"拆成一串固定的比较-交换阶段：先两两排成升序、
降序交替的双调序列，再成对归并。阶段数是
$\frac{\log N(\log N + 1)}{2}$，`N=1024` 时正好 55 次 Kernel 启动。

每一次启动只做一件很轻的事，开销几乎全在启动与同步上——这正是本篇要讲的
"Kernel 数量本身就是成本"。

## 前置

010（块内归约与同步）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `bitonic_sort.py` | 55 阶段排序 Kernel + 5 类输入 × 7 种规模的回归 + 50 次采样基准 | 博文代码块 #55 |

## 运行

```bash
python -m pip install -r ../../requirements/python-gpu.txt
python bitonic_sort.py
```

默认 `n=1024`、`block_size=256`，种子 42。流程是：主用例排序 → 回归
（5 类输入 × `n ∈ {1,2,4,8,32,256,1024}` = 35 组）→ 基准（warmup×5 + 50 次采样）。

## 环境要求

- 必须：NVIDIA GPU + CUDA Toolkit（`SourceModule` 带 `-O3`）
- 依赖层：`requirements/python-gpu.txt`

## 观察点

- Kernel stages 必须等于 55（n=1024：10×11/2）。数不对就是阶段循环写错。
- 基准测的是整串 55 次启动；把 n 减半后若耗时没有按比例下降，说明瓶颈在启动。

## 验证状态

**未实测。** 本机无 `nvcc`，仅完成语法检查。

需要记录：`Kernel stages`、回归组数、P50/P95/Min/Max 四个耗时。
见 `expected_output.md`。

## 出处

- 博文：[14-cuda-bitonic-sort-light-tech.md](../../../blog/014-CUDA双调排序.md)
- 来源代码块 #55，245 行，md5 `6913ccf0dde29388223f55526927f8e1`

代码逐字一致。**博文未指定文件名**，`bitonic_sort.py` 为仓库新增命名。
