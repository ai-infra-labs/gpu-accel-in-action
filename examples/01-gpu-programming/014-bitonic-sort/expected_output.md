# 014 · 实测输出

**状态：未实测。** 本机无 `nvcc`，无法编译 `SourceModule`。

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
PyCUDA / NumPy 版本：
n / block_size：
```

## 2. 运行结果

| 项 | 值 |
|---|---|
| Correctness | 应为 `PASS`（脚本内 `assert`） |
| Kernel stages | 应为 `55`（`n=1024`） |
| Regression | 应为 `PASS (35 cases)` |
| P50 (ms) | 待填 |
| P95 (ms) | 待填 |
| Min (ms) | 待填 |
| Max (ms) | 待填 |

判读方式：

- `Kernel stages` 是可手算的：$\log_2 1024 = 10$，阶段数 $10 \times 11 / 2 = 55$。
  这个数与实际启动次数必须一致——不一致说明阶段循环写错了边界。
- 基准测的是"整串 Kernel 序列"，不是单次 Kernel。55 次启动的固定开销都会算进去。
  把 `n` 减半（512 → 45 次启动），P50 若没有按比例下降，说明瓶颈在启动而不在数据量。
- 回归覆盖 5 类输入（随机 / 已排序 / 逆序 / 大量重复 / 全相等）。`all_equal`
  能抓出"比较方向写错但不影响已排序输入"这类隐性 bug。

## 3. 踩坑记录

待填。至少记录：

- 双调排序要求 `n` 是 2 的幂；脚本的 `validate_input` 会先挡住非 2 的幂输入
- 方向参数（升 / 降）每一阶段都要翻转，漏翻会让归并后的序列只是"局部有序"
- 35 组回归里最容易漏的是 `n=1`：阶段数为 0，不应启动任何 Kernel
