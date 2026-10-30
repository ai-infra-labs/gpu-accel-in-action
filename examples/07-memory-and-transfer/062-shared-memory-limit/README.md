# 062 · 共享内存这么快，为什么不能全放进去？

共享内存是"每个 block 私有的小仓库"：延迟低，但容量只有几十 KiB，
而且它的占用会直接压低同时驻留的 block 数（occupancy）。本篇用同一道
矩阵乘法写两个版本——朴素版与 16×16 分块版——在同一台机器上对照。

## 前置

011（分块与同步）、061（合并访存）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `matmul_shared_limit.py` | naive / tiled 两个 kernel + 正确性核对 + 计时 | 块 #3 + #4，**拼接** |

单看块 #3 只有 import 与两个 kernel，单看块 #4 缺 `TILE` 与 `np`，
所以本文件是两块的顺序拼接（拼接规则见 `docs/PROVENANCE.json`）。
**拼接后的 `import` 出现在文件中段**，这是博文原文的顺序，不是笔误。

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt
python matmul_shared_limit.py
```

先跑三种小形状的正确性对照，再对 256³ / 512³ 计时。无命令行参数。

## 环境要求

- NVIDIA GPU + `numba-cuda`（两个 kernel 都是 `@cuda.jit`）
- 依赖层：`requirements/advanced.txt`

## 观察点

- 三个非整除形状的误差必须仍在 1e-6：越界位置要填 0，不能跳过赋值。
- tiled 的收益随规模变大；256³ 打平、512³ 才拉开，说明访存还没主导。

## 验证状态

**未实测。** 本机无 CUDA，仅完成语法检查。

需要记录：三个小形状下两个版本的 `max_abs_error`，以及两个大形状的 median 与区间。
见 `expected_output.md`。

## 出处

- 博文：[62-共享内存这么快为什么不能全放进去.md](../../../blog/062-共享内存限制与bank冲突.md)
- `matmul_shared_limit.py` = 块 #3 + 块 #4（各块单独都不完整）
