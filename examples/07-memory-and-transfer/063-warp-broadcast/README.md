# 063 · 一组线程读同一个数，GPU 有什么捷径？

一条 warp 的 32 个线程读同一地址时，硬件可以把它合并成一次访问（broadcast）；
而放在常量内存（constant memory）里的只读参数，还有专门的广播缓存。
本篇用两个实验把这件事量化：

| 实验 | 对比 | 关注点 |
|---|---|---|
| 5 抽头滤波 | 权重在全局内存 vs `cuda.const.array_like` | 同一权重被反复读 |
| 查表 | `u = 1 / 4 / 32` 三种 warp 内重复度 | 重复度越高，广播收益越大 |

## 前置

061（合并访存）、062（共享内存的边界）。

## 文件

| 文件 | 作用 | 来源 |
|---|---|---|
| `warp_broadcast_lab.py` | 滤波与查表两组对照 + 正确性 + 计时 | 块 #0 + #2，**拼接** |

块 #0 只有 import、常量与两个滤波 kernel，块 #2 才定义查表 kernel 与 main，
所以本文件是两块的顺序拼接（登记见 `docs/PROVENANCE.json`）。

## 运行

```bash
python -m pip install -r ../../requirements/advanced.txt
python warp_broadcast_lab.py
```

`n = 17, 257` 做正确性核对，`2^18` 与 `2^20` 做计时。无命令行参数。

## 环境要求

- NVIDIA GPU + `numba-cuda`
- 依赖层：`requirements/advanced.txt`

## 观察点

- 收益应随 u 增大而减小；u=32（warp 内全不同）时接近 global 版。
- 趋势不对时，先确认常量内存没有超限（本实验表只有 1 KiB，放得下）。

## 验证状态

**未实测。** 本机无 CUDA，仅完成语法检查。

需要记录：两组实验在各 `u` 下的 PASS 与 median/区间。
见 `expected_output.md`。

## 出处

- 博文：[63-一组线程读同一个数GPU有什么捷径.md](../../../blog/063-Warp级广播与Shuffle原语.md)
- `warp_broadcast_lab.py` = 块 #0 + 块 #2
