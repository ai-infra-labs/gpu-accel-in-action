# 063 · 实测输出

**状态：未实测（需 numba-cuda）。**

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
numba / numba-cuda / numpy：
THREADS / 采样设置（warmup=3, repeats=20, samples=5）：
```

## 2. 正确性

| 实验 | 用例 | global | constant |
|---|---|---|---|
| 5 抽头滤波 | n=17 | 待填 PASS/FAIL | 待填 |
| 5 抽头滤波 | n=257 | 待填 | 待填 |
| 查表 | u=1 | 待填 | 待填 |
| 查表 | u=4 | 待填 | 待填 |
| 查表 | u=32 | 待填 | 待填 |

参照值：滤波用 `np.convolve(..., mode="full")[:n]`，查表用
`host * TABLE[indices]`——两者都在 CPU 上按 float64 算，再与 GPU 的 float32 比
（`rtol=1e-5, atol=1e-6`）。**任何一个 FAIL 都不要往下看时间**。

## 3. 计时（n = 2^18 与 2^20）

| n | 实验 | global median (ms) | constant median (ms) | 区间 (ms) |
|---|---|---|---|---|
| 2^18 | filter | 待填 | 待填 | 待填 |
| 2^18 | lookup u=1 | 待填 | 待填 | 待填 |
| 2^18 | lookup u=4 | 待填 | 待填 | 待填 |
| 2^18 | lookup u=32 | 待填 | 待填 | 待填 |
| 2^20 | … | 待填 | 待填 | 待填 |

判读方式：

- **看 u 的趋势，不要只看单点**：`u=1` 时整条 warp 读同一个下标，
  `u=4` 时 8 个线程一组同值，`u=32` 时 32 个线程全不同。
  收益应当随 `u` 增大而减小——如果不是这个趋势，说明瓶颈在别处
  （比如数据量太小、或常数内存没有真正生效）。
- **constant memory 有容量上限**（通常 64 KiB），本实验的表只有 1 KiB。
  表变大到放不进常量内存时，这条路直接走不通。
- **2^18 与 2^20 的结论可能不同**：小规模看延迟，大规模看带宽。
  两个规模都测就是为了避免"一次测量定结论"。

## 4. 这份数据不能说明什么

- **不是通用优化建议**：只在"整条 warp 读同一地址"时成立，
  随机访问表反而会被常量内存的串行化惩罚。
- **不含 L1/L2 命中分析**：要判断是否真的走了广播缓存，得用 Nsight Compute
  看 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` 之类的指标。
