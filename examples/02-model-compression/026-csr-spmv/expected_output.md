# 026 · 实测输出

**状态：未实测。**

语法与未定义名称检查已通过；三份实现的数值一致性与耗时全部待跑。

## 1. 环境

```
GPU 型号 / Compute Capability：
Driver / CUDA Toolkit：
Numba / numba-cuda / SciPy / NumPy 版本：
rows / cols / density / runs：
```

## 2. 存储

| 项 | 值 |
|---|---|
| 稠密存储理论占用 | 待填（rows × cols × 4 B） |
| CSR 实际占用（data + indices + indptr） | 待填 |
| 非零元个数 | 待填 |
| 压缩比 | 待填 |

## 3. 数值一致性

脚本以 SciPy 的 float64 结果为参考，逐个实现打印最大绝对误差。

| 实现 | 最大绝对误差 |
|---|---|
| Numba CPU（njit） | 待填 |
| Numba CUDA kernel | 待填 |

判断标准：GPU 版用 float32 累加，误差随每行非零数增长；若某行非零数上千，
float32 累加的误差会明显放大，这属于预期行为，不是 bug。

## 4. 性能

| 实现 | 耗时 |
|---|---|
| SciPy 参考 | 待填 |
| Numba CPU | 待填 |

注意：本实现的 CUDA kernel 是"一个线程管一行"。行内非零数不均时，
warp 内线程负载差异极大——密度均匀时看不出问题，真实图结构会退化。

## 5. 踩坑记录

待填。至少记录：

- `numba-cuda` 未安装时 GPU 路径的报错形态
- 打印版本号那一行是否因缺包而抛错（`version("numba-cuda")` 在旧环境会失败）
- 大规模 `--rows` 时 CSR 构造本身的内存峰值（构造阶段往往比计算更吃内存）
